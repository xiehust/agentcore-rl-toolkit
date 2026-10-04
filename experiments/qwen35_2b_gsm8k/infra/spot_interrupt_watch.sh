#!/usr/bin/env bash
# Poll IMDSv2 spot/instance-action every 5s. On a 200 (interruption/termination
# notice), immediately sync checkpoints + run the watchdog once, then exit.
set -uo pipefail
exec >> /var/log/agentcore-rl-spot-watch.log 2>&1
echo "=== spot_interrupt_watch start $(date -u) ==="

[ -f /etc/agentcore-rl.env ] && . /etc/agentcore-rl.env
BUCKET="${ACR_S3_BUCKET:-}"
REGION="${AWS_REGION:-us-west-2}"

imds_token() { curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null; }

while true; do
  TOKEN="$(imds_token || true)"
  if [ -n "$TOKEN" ]; then
    CODE=$(curl -fsS -o /dev/null -w '%{http_code}' \
      -H "X-aws-ec2-metadata-token: $TOKEN" \
      "http://169.254.169.254/latest/meta-data/spot/instance-action" 2>/dev/null || echo 000)
    if [ "$CODE" = "200" ]; then
      echo "!!! spot interruption notice at $(date -u) -- syncing"
      SYNC="/data/repo/experiments/qwen35_2b_gsm8k/trainer/sync_ckpt.sh"
      if [ -x "$SYNC" ]; then
        "$SYNC" --once || true
      elif [ -n "$BUCKET" ]; then
        aws s3 sync /data/ckpts "s3://${BUCKET}/ckpt/" --region "$REGION" 2>/dev/null || true
      fi
      [ -x /opt/agentcore-rl/watchdog.sh ] && /opt/agentcore-rl/watchdog.sh || true
      echo "=== spot_interrupt_watch exit after interruption ==="
      exit 0
    fi
  fi
  sleep 5
done
