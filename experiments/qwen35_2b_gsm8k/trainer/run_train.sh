#!/usr/bin/env bash
# run_train.sh — launch train_qwen35_2b.sh detached (survives SSM shell exit),
# start the checkpoint-sync loop if not already running, and print how to follow
# the log. SPEC §5. All args are forwarded to the trainer (e.g. --force, SMOKE via
# env, Hydra overrides).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

LOG_DIR=/data/logs
mkdir -p "$LOG_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TRAIN_LOG="$LOG_DIR/train_${TS}.log"
PID_FILE="$LOG_DIR/train.pid"

# Refuse to double-start: if a recorded PID is still alive, bail.
if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "training already running (pid $OLD_PID, pidfile $PID_FILE). Stop it first." >&2
    exit 1
  fi
fi

# setsid -> new process group so the watchdog can SIGTERM the whole group at the
# 20h soft limit; nohup so it outlives the SSM shell.
setsid nohup "$SCRIPT_DIR/train_qwen35_2b.sh" "$@" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "started training: pid $TRAIN_PID  log $TRAIN_LOG"

# --- start the ckpt sync loop if not already up ------------------------------
SYNC_PID_FILE="$LOG_DIR/sync_ckpt.pid"
sync_running=0
if [ -f "$SYNC_PID_FILE" ]; then
  SPID="$(cat "$SYNC_PID_FILE" 2>/dev/null || true)"
  [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null && sync_running=1
fi
if [ "$sync_running" -eq 1 ]; then
  echo "ckpt sync loop already running (pid $(cat "$SYNC_PID_FILE"))"
else
  setsid nohup "$SCRIPT_DIR/sync_ckpt.sh" > "$LOG_DIR/sync_ckpt_${TS}.log" 2>&1 &
  SYNC_PID=$!
  echo "$SYNC_PID" > "$SYNC_PID_FILE"
  echo "started ckpt sync loop: pid $SYNC_PID"
fi

cat <<EOF

Follow the run:
  tail -f $TRAIN_LOG
Stop gracefully (final sync included):
  $SCRIPT_DIR/stop_train.sh
EOF
