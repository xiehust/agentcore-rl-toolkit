#!/usr/bin/env bash
# Local, read-only ledger inspector. Prints cumulative GPU hours, per-session
# breakdown, remaining budget to 24h, current tagged instances, and cost estimate.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
# EC2 ops target the trainer region (may differ from the ACR/S3 region)
export AWS_REGION="$TRAINER_REGION" AWS_DEFAULT_REGION="$TRAINER_REGION"

HARD="${GPU_HOURS_HARD_LIMIT:-24}"
RATE=2.63   # $/h p5.4xlarge spot (us-west-2), COST.md
LEDGER_KEY="ledger/gpu_hours.json"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
LEDGER="$WORK/gpu_hours.json"

echo "== GPU-hours ledger (s3://${ACR_S3_BUCKET}/${LEDGER_KEY}) =="
if aws s3 cp "s3://${ACR_S3_BUCKET}/${LEDGER_KEY}" "$LEDGER" --region "$AWS_REGION" 2>/dev/null; then
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$LEDGER" "$HARD" "$RATE" <<'PY'
import json, sys
path, hard, rate = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
d = json.load(open(path))
cum = float(d.get("cumulative_hours", 0))
print(f"cumulative_hours : {cum:.4f}")
print(f"remaining to {hard:.0f}h : {max(0.0, hard-cum):.4f}")
print(f"est. cost @ ${rate}/h : ${cum*rate:.2f}")
print(f"updated_at       : {d.get('updated_at')}")
print("\nsessions:")
print(f"  {'instance_id':22} {'az':14} {'hours':>8}  start -> end")
for s in d.get("sessions", []):
    print(f"  {s.get('instance_id',''):22} {s.get('az',''):14} {float(s.get('hours',0)):>8.3f}  {s.get('start')} -> {s.get('end')}")
PY
  else
    jq . "$LEDGER"
  fi
else
  echo "  (no ledger yet -- training has not started)"
fi

echo
echo "== Current tagged instances (Project=${EXP_TAG}) =="
aws ec2 describe-instances --region "$AWS_REGION" \
  --filters "Name=tag:Project,Values=${EXP_TAG}" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime,PublicIpAddress]' \
  --output table || echo "  (none)"
