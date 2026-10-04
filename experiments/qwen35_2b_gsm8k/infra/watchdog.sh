#!/usr/bin/env bash
# In-instance GPU-hours watchdog. Runs from a systemd timer every 5 min.
# Maintains s3://$ACR_S3_BUCKET/ledger/gpu_hours.json and enforces the
# 12h alert / 20h soft-limit / 24h hard-limit guardrails. MUST NEVER crash the
# timer -- every failure path is tolerated so the unit stays green.
set -uo pipefail
exec >> /var/log/agentcore-rl-watchdog.log 2>&1
echo "--- watchdog $(date -u) ---"

# Load env (bucket, thresholds). Fall back to sane defaults.
[ -f /etc/agentcore-rl.env ] && . /etc/agentcore-rl.env
BUCKET="${ACR_S3_BUCKET:-}"
REGION="${AWS_REGION:-us-west-2}"
EXP_TAG="${EXP_TAG:-qwen35-2b-gsm8k}"
ALERT="${GPU_HOURS_ALERT:-12}"
SOFT="${GPU_HOURS_SOFT_LIMIT:-20}"
HARD="${GPU_HOURS_HARD_LIMIT:-24}"
LEDGER_KEY="ledger/gpu_hours.json"

if [ -z "$BUCKET" ]; then echo "no bucket configured; exit 0"; exit 0; fi

imds_token() { curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null; }
imds() { local t; t=$(imds_token); curl -fsS -H "X-aws-ec2-metadata-token: $t" \
  "http://169.254.169.254/latest/$1" 2>/dev/null; }

IID="$(imds meta-data/instance-id || true)"
AZ="$(imds meta-data/placement/availability-zone || true)"
[ -z "$IID" ] && { echo "no instance-id; exit 0"; exit 0; }

# Launch time (session start) from EC2 API.
START="$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].LaunchTime' --output text 2>/dev/null || true)"
[ -z "$START" ] || [ "$START" = "None" ] && START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
LEDGER="$WORK/gpu_hours.json"

# Download or initialize the ledger.
if ! aws s3 cp "s3://${BUCKET}/${LEDGER_KEY}" "$LEDGER" --region "$REGION" 2>/dev/null; then
  echo '{"cumulative_hours":0,"sessions":[],"updated_at":null}' > "$LEDGER"
fi

# Upsert this instance's session and recompute cumulative via python3 (fallback jq).
CUMULATIVE=""
if command -v python3 >/dev/null 2>&1; then
  CUMULATIVE=$(python3 - "$LEDGER" "$IID" "$AZ" "$START" "$NOW" <<'PY'
import json, sys, datetime
path, iid, az, start, now = sys.argv[1:6]
def parse(t):
    return datetime.datetime.strptime(t.replace('+00:00','Z'), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
try:
    d = json.load(open(path))
except Exception:
    d = {"cumulative_hours":0,"sessions":[],"updated_at":None}
sess = d.get("sessions", [])
hours = round((parse(now)-parse(start)).total_seconds()/3600.0, 4)
found=False
for s in sess:
    if s.get("instance_id")==iid:
        s["az"]=az; s["start"]=start; s["end"]=now; s["hours"]=hours; found=True; break
if not found:
    sess.append({"instance_id":iid,"az":az,"start":start,"end":now,"hours":hours})
cum = round(sum(float(s.get("hours",0)) for s in sess), 4)
d["sessions"]=sess; d["cumulative_hours"]=cum; d["updated_at"]=now
json.dump(d, open(path,"w"), indent=2)
print(cum)
PY
)
else
  # jq fallback
  HOURS=$(python3 -c "import datetime" 2>/dev/null; echo "")   # noop guard
  CUMULATIVE=$(jq -r '.cumulative_hours // 0' "$LEDGER")
fi
[ -z "$CUMULATIVE" ] && CUMULATIVE=0
echo "cumulative_hours=${CUMULATIVE} (instance ${IID}, az ${AZ})"

# Upload updated ledger.
aws s3 cp "$LEDGER" "s3://${BUCKET}/${LEDGER_KEY}" --region "$REGION" 2>/dev/null || echo "ledger upload failed (tolerated)"

marker_absent() { ! aws s3 ls "s3://${BUCKET}/ledger/$1" --region "$REGION" >/dev/null 2>&1; }
write_marker() { echo "$2" | aws s3 cp - "s3://${BUCKET}/ledger/$1" --region "$REGION" 2>/dev/null || true; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0>=b+0)}'; }

# Detect a second running tagged trainer.
OTHERS=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Project,Values=${EXP_TAG}" "Name=tag:Role,Values=trainer" \
            "Name=instance-state-name,Values=running" \
  --query "Reservations[].Instances[?InstanceId!='${IID}'].InstanceId" --output text 2>/dev/null || true)
if [ -n "${OTHERS// /}" ] && [ "$OTHERS" != "None" ]; then
  echo "!!! multiple tagged trainers detected: ${OTHERS}"
  write_marker "ALERT_MULTI_INSTANCE" "instance ${IID} sees others: ${OTHERS} @ ${NOW}"
fi

# 12h alert (once).
if ge "$CUMULATIVE" "$ALERT" && marker_absent "ALERT_12H"; then
  echo "ALERT: >= ${ALERT}h"
  write_marker "ALERT_12H" "cumulative=${CUMULATIVE} @ ${NOW}"
fi

# 20h soft limit (once): alert + STOP flag + SIGTERM to training process group.
if ge "$CUMULATIVE" "$SOFT" && marker_absent "ALERT_20H"; then
  echo "SOFT LIMIT: >= ${SOFT}h -> STOP_TRAINING + SIGTERM"
  write_marker "ALERT_20H" "cumulative=${CUMULATIVE} @ ${NOW}"
  touch /data/STOP_TRAINING 2>/dev/null || true
  if [ -f /data/logs/train.pid ]; then
    PID="$(cat /data/logs/train.pid 2>/dev/null || true)"
    if [ -n "$PID" ]; then
      kill -TERM -- "-${PID}" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
    fi
  fi
fi

# 24h hard limit: final ckpt sync then self-terminate.
if ge "$CUMULATIVE" "$HARD"; then
  echo "HARD LIMIT: >= ${HARD}h -> final sync + self-terminate"
  SYNC="/data/repo/experiments/qwen35_2b_gsm8k/trainer/sync_ckpt.sh"
  if [ -x "$SYNC" ]; then
    "$SYNC" --once || echo "sync_ckpt.sh failed (tolerated)"
  else
    aws s3 sync /data/ckpts "s3://${BUCKET}/ckpt/" --region "$REGION" 2>/dev/null || echo "ckpt sync failed (tolerated)"
  fi
  aws s3 cp "$LEDGER" "s3://${BUCKET}/${LEDGER_KEY}" --region "$REGION" 2>/dev/null || true
  aws ec2 terminate-instances --instance-ids "$IID" --region "$REGION" 2>/dev/null || echo "self-terminate failed (tolerated)"
fi

echo "--- watchdog done ---"
exit 0
