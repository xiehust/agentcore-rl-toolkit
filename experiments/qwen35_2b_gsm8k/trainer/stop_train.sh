#!/usr/bin/env bash
# stop_train.sh — graceful stop of the training run started by run_train.sh:
# SIGTERM the trainer's process group, wait up to 120s, SIGKILL survivors, then
# do one final checkpoint sync. SPEC §5.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

LOG_DIR=/data/logs
PID_FILE="$LOG_DIR/train.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "no $PID_FILE; nothing to stop. Running final sync anyway."
  "$SCRIPT_DIR/sync_ckpt.sh" --once || true
  exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
  echo "recorded pid ${PID:-<none>} is not alive; final sync only."
  rm -f "$PID_FILE"
  "$SCRIPT_DIR/sync_ckpt.sh" --once || true
  exit 0
fi

# run_train.sh launched the trainer with setsid, so PID heads its own process
# group. Signal the whole group (negative pid) to catch verl/ray/vllm children.
echo "sending SIGTERM to process group $PID"
kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true

echo "waiting up to 120s for graceful exit ..."
gone=0
for _ in $(seq 1 120); do
  if ! kill -0 "$PID" 2>/dev/null; then gone=1; break; fi
  sleep 1
done

if [ "$gone" -eq 0 ]; then
  echo "still alive after 120s; sending SIGKILL to process group $PID"
  kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
  sleep 2
fi

rm -f "$PID_FILE"
echo "trainer stopped. Running final checkpoint sync ..."
"$SCRIPT_DIR/sync_ckpt.sh" --once || true
echo "done."
