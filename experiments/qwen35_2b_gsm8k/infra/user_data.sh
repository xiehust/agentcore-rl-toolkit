#!/usr/bin/env bash
# cloud-init user-data template (Ubuntu DLAMI). Rendered by launch_spot.sh via
# envsubst -- ${ACR_S3_BUCKET} ${EXP_TAG} ${GATEWAY_PORT} ${GPU_HOURS_ALERT}
# ${GPU_HOURS_SOFT_LIMIT} ${GPU_HOURS_HARD_LIMIT} ${AWS_REGION} are substituted
# at launch time. Runs as root on first boot.
set -uo pipefail
exec > >(tee -a /var/log/agentcore-rl-userdata.log) 2>&1
echo "=== agentcore-rl user-data start $(date -u) ==="

export DEBIAN_FRONTEND=noninteractive
REGION="${AWS_REGION}"
BUCKET="${ACR_S3_BUCKET}"

# --- tools: awscli v2 + jq if missing -------------------------------------
if ! command -v aws >/dev/null 2>&1; then
  echo "installing awscli v2"
  tmp=$(mktemp -d)
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o "$tmp/awscliv2.zip"
  (cd "$tmp" && unzip -q awscliv2.zip && ./aws/install --update)
  rm -rf "$tmp"
fi
command -v jq >/dev/null 2>&1 || { apt-get update -y && apt-get install -y jq; }

# --- environment file for the instance ------------------------------------
cat > /etc/agentcore-rl.env <<ENVEOF
ACR_S3_BUCKET=${ACR_S3_BUCKET}
EXP_TAG=${EXP_TAG}
GATEWAY_PORT=${GATEWAY_PORT}
AWS_REGION=${AWS_REGION}
AWS_DEFAULT_REGION=${AWS_REGION}
GPU_HOURS_ALERT=${GPU_HOURS_ALERT}
GPU_HOURS_SOFT_LIMIT=${GPU_HOURS_SOFT_LIMIT}
GPU_HOURS_HARD_LIMIT=${GPU_HOURS_HARD_LIMIT}
ENVEOF

# --- IMDSv2 helpers --------------------------------------------------------
imds_token() { curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 300"; }
imds() { local t; t=$(imds_token); curl -fsS -H "X-aws-ec2-metadata-token: $t" \
  "http://169.254.169.254/latest/$1"; }

# DataVolumeId comes from the instance tag (InstanceMetadataTags=enabled).
DATA_VOL_ID="$(imds meta-data/tags/instance/DataVolumeId 2>/dev/null || true)"
echo "DataVolumeId=${DATA_VOL_ID}"

# --- wait up to 10 min for the data volume device -------------------------
# NVMe device path: nvme-Amazon_Elastic_Block_Store_vol<volid without dash>
DEV=""
IID="$(imds meta-data/instance-id)"
# Prefer the authoritative source: the Role=data volume attached to this instance
# (the volume may be created/attached a minute after boot; IMDS tags lag too).
for _ in $(seq 1 120); do
  [ -z "${DATA_VOL_ID}" ] && DATA_VOL_ID="$(aws ec2 describe-volumes --region "$REGION" \
      --filters "Name=attachment.instance-id,Values=${IID}" "Name=tag:Role,Values=data" \
      --query 'Volumes[0].VolumeId' --output text 2>/dev/null | grep -v None || true)"
  [ -z "${DATA_VOL_ID}" ] && DATA_VOL_ID="$(imds meta-data/tags/instance/DataVolumeId 2>/dev/null || true)"
  if [ -n "${DATA_VOL_ID}" ]; then
    BY_ID="/dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_${DATA_VOL_ID/-/}"
    if [ -e "${BY_ID}" ]; then DEV="$(readlink -f "${BY_ID}")"; break; fi
  fi
  sleep 5
done
echo "DataVolumeId=${DATA_VOL_ID} device=${DEV}"
if [ -z "${DEV}" ]; then
  echo "!!! data volume device did not appear within 10 min" >&2
fi

# --- format if blank + mount /data ----------------------------------------
if [ -n "${DEV}" ]; then
  if [ -z "$(blkid -o value -s TYPE "${DEV}" 2>/dev/null || true)" ]; then
    echo "formatting ${DEV} as ext4 (label data)"
    mkfs.ext4 -L data "${DEV}"
  fi
  mkdir -p /data
  grep -q 'LABEL=data' /etc/fstab || echo 'LABEL=data /data ext4 defaults,nofail 0 2' >> /etc/fstab
  mount /data || mount -L data /data || true
  chown ubuntu:ubuntu /data
  mkdir -p /data/hf /data/repo /data/gsm8k /data/ckpts /data/logs /data/ledger
  chown -R ubuntu:ubuntu /data/hf /data/repo /data/gsm8k /data/ckpts /data/logs /data/ledger
fi

# --- fetch watchdog + spot interrupt watcher ------------------------------
mkdir -p /opt/agentcore-rl
aws s3 cp "s3://${BUCKET}/code/infra/watchdog.sh" /opt/agentcore-rl/watchdog.sh --region "$REGION" || true
aws s3 cp "s3://${BUCKET}/code/infra/spot_interrupt_watch.sh" /opt/agentcore-rl/spot_interrupt_watch.sh --region "$REGION" || true
chmod +x /opt/agentcore-rl/watchdog.sh /opt/agentcore-rl/spot_interrupt_watch.sh 2>/dev/null || true

# --- systemd: watchdog timer (every 5 min, first run 1 min after boot) -----
cat > /etc/systemd/system/agentcore-rl-watchdog.service <<'UNIT'
[Unit]
Description=AgentCore RL GPU-hours watchdog
After=network-online.target
[Service]
Type=oneshot
EnvironmentFile=/etc/agentcore-rl.env
ExecStart=/opt/agentcore-rl/watchdog.sh
UNIT

cat > /etc/systemd/system/agentcore-rl-watchdog.timer <<'UNIT'
[Unit]
Description=Run AgentCore RL watchdog every 5 minutes
[Timer]
OnBootSec=1min
OnUnitActiveSec=5min
AccuracySec=30s
[Install]
WantedBy=timers.target
UNIT

# --- systemd: spot interrupt watcher (long-running) ------------------------
cat > /etc/systemd/system/spot_interrupt_watch.service <<'UNIT'
[Unit]
Description=AgentCore RL spot interruption watcher
After=network-online.target
[Service]
Type=simple
EnvironmentFile=/etc/agentcore-rl.env
ExecStart=/opt/agentcore-rl/spot_interrupt_watch.sh
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now agentcore-rl-watchdog.timer || true
systemctl enable --now spot_interrupt_watch.service || true

echo "=== agentcore-rl user-data done $(date -u) ==="
