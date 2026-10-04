#!/usr/bin/env bash
# restore_ckpt.sh — reverse sync: pull checkpoints (and logs) from S3 back to the
# data volume, e.g. after a spot reclaim rebuilt the instance on a fresh volume.
# SPEC §6. verl's resume_mode=auto then continues from the newest local step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

: "${ACR_S3_BUCKET:?ACR_S3_BUCKET must be set (env.sh)}"

DATA=/data
CKPT_DST="$DATA/ckpts"
LOG_DST="$DATA/logs"
CKPT_SRC="s3://$ACR_S3_BUCKET/ckpt/"
LOG_SRC="s3://$ACR_S3_BUCKET/logs/"

mkdir -p "$CKPT_DST" "$LOG_DST"

echo "[restore $(date -u +%FT%TZ)] $CKPT_SRC -> $CKPT_DST"
aws s3 sync "$CKPT_SRC" "$CKPT_DST" --no-progress

# Logs are best-effort; --with-logs to also pull them.
if [ "${1:-}" = "--with-logs" ]; then
  echo "[restore $(date -u +%FT%TZ)] $LOG_SRC -> $LOG_DST"
  aws s3 sync "$LOG_SRC" "$LOG_DST" --no-progress || true
fi

echo "restore complete. Newest local checkpoint:"
ls -1 "$CKPT_DST"/*/*/ 2>/dev/null | tail -n 5 || echo "  (no checkpoints found under $CKPT_DST)"
