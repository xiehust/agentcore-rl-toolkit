#!/usr/bin/env bash
# sync_ckpt.sh — push checkpoints and logs to S3. SPEC §6.
#   --once   sync one time and exit (used by stop_train.sh and the spot-interrupt hook)
#   (none)   loop forever, syncing every SYNC_INTERVAL (default 300s)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

: "${ACR_S3_BUCKET:?ACR_S3_BUCKET must be set (env.sh)}"

DATA=/data
CKPT_SRC="$DATA/ckpts"
LOG_SRC="$DATA/logs"
CKPT_DST="s3://$ACR_S3_BUCKET/ckpt/"
LOG_DST="s3://$ACR_S3_BUCKET/logs/"
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"

mkdir -p "$CKPT_SRC" "$LOG_SRC"

do_sync() {
  echo "[sync $(date -u +%FT%TZ)] $CKPT_SRC -> $CKPT_DST"
  aws s3 sync "$CKPT_SRC" "$CKPT_DST" --no-progress || echo "  ckpt sync returned non-zero (will retry next cycle)"
  echo "[sync $(date -u +%FT%TZ)] $LOG_SRC -> $LOG_DST"
  aws s3 sync "$LOG_SRC" "$LOG_DST" --no-progress || echo "  log sync returned non-zero"
}

if [ "${1:-}" = "--once" ]; then
  do_sync
  exit 0
fi

echo "ckpt sync loop every ${SYNC_INTERVAL}s (Ctrl-C / SIGTERM to stop)"
trap 'echo "sync loop received signal, doing final sync"; do_sync; exit 0' TERM INT
while true; do
  do_sync
  sleep "$SYNC_INTERVAL"
done
