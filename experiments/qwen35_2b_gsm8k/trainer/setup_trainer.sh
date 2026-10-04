#!/usr/bin/env bash
# setup_trainer.sh — one-time trainer-instance bring-up (idempotent).
# Run on the p5.4xlarge via SSM as the ubuntu user. Each step drops a marker in
# /data/.setup/ and is skipped on re-run. SPEC §1.
#
#   sudo -iu ubuntu bash -lc '/data/repo/experiments/qwen35_2b_gsm8k/trainer/setup_trainer.sh'
#
# Flags:
#   --refresh   re-download repo.tar.gz even if /data/repo exists
#   --force     re-run every step, ignoring /data/.setup markers
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# env.sh may not exist yet on the very first boot (repo not pulled). Source it if
# present; otherwise fall back to the minimum we need to fetch the repo, then
# re-source after the pull.
_ENV="$SCRIPT_DIR/../env.sh"
if [ -f "$_ENV" ]; then
  # shellcheck disable=SC1090
  source "$_ENV"
fi

REFRESH=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --refresh) REFRESH=1 ;;
    --force) FORCE=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

DATA=/data
SETUP_DIR="$DATA/.setup"
LOG_DIR="$DATA/logs"
export UV_CACHE_DIR="$DATA/uv-cache"
export HF_HOME="$DATA/hf"
mkdir -p "$SETUP_DIR" "$LOG_DIR" "$HF_HOME" "$DATA/gsm8k" "$DATA/ckpts" "$UV_CACHE_DIR"

SETUP_LOG="$LOG_DIR/setup.log"
log() { echo "[setup $(date -u +%FT%TZ)] $*" | tee -a "$SETUP_LOG"; }

# done_step NAME  -> true if the marker exists and --force not set
done_step() { [ "$FORCE" -eq 0 ] && [ -f "$SETUP_DIR/$1.done" ]; }
mark_step() { touch "$SETUP_DIR/$1.done"; }

REQUIRED_DRIVER="580.65.06"
# ver_ge A B -> true if dotted version A >= B
ver_ge() {
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

# --- a. GPU / driver check (never skipped: cheap and load-bearing) ------------
log "step a: nvidia-smi / driver check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "ERROR: nvidia-smi not found. On the DL Base AMI the driver ships preinstalled;"
  log "       if this is a rebuilt instance, reinstall the >= ${REQUIRED_DRIVER} driver"
  log "       (see PLAN §5 'environment rebuild')."
  exit 1
fi
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d '[:space:]')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)"
log "GPU: $GPU_NAME | mem: $GPU_MEM | driver: $DRIVER_VER"
if ! ver_ge "$DRIVER_VER" "$REQUIRED_DRIVER"; then
  log "ERROR: driver $DRIVER_VER < required $REQUIRED_DRIVER (CUDA 13 verl stack)."
  log "       Fix: reinstall the DL Base OSS Nvidia Driver AMI's driver or run"
  log "       'sudo apt-get install -y nvidia-driver-580' then reboot."
  exit 1
fi

# --- b. uv install + cache/HF env --------------------------------------------
if done_step uv; then
  log "step b: uv already installed (skip)"
else
  log "step b: install uv"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  # uv installs to ~/.local/bin; make it visible for the rest of this run.
  export PATH="$HOME/.local/bin:$PATH"
  uv --version | tee -a "$SETUP_LOG"
  mark_step uv
fi
export PATH="$HOME/.local/bin:$PATH"

# --- c. pull repo from S3 -----------------------------------------------------
: "${ACR_S3_BUCKET:?ACR_S3_BUCKET must be set (source env.sh or export it)}"
if [ "$REFRESH" -eq 1 ] || [ ! -d "$DATA/repo" ] || ! done_step repo; then
  log "step c: pull s3://$ACR_S3_BUCKET/code/repo.tar.gz -> $DATA/repo"
  TARBALL="$DATA/repo.tar.gz"
  aws s3 cp "s3://$ACR_S3_BUCKET/code/repo.tar.gz" "$TARBALL"
  rm -rf "$DATA/repo"
  mkdir -p "$DATA/repo"
  # coworker's upload_code.sh tars the repo contents; strip a leading dir if the
  # tar wraps everything in one top-level folder.
  tar -xzf "$TARBALL" -C "$DATA/repo"
  if [ ! -f "$DATA/repo/pyproject.toml" ]; then
    inner="$(find "$DATA/repo" -maxdepth 2 -name pyproject.toml -print -quit || true)"
    if [ -n "$inner" ]; then
      inner_dir="$(dirname "$inner")"
      if [ "$inner_dir" != "$DATA/repo" ]; then
        log "repo unpacked under $inner_dir; flattening into $DATA/repo"
        shopt -s dotglob
        mv "$inner_dir"/* "$DATA/repo"/
        shopt -u dotglob
      fi
    fi
  fi
  rm -f "$TARBALL"
  mark_step repo
else
  log "step c: repo present (skip; use --refresh to re-pull)"
fi
[ -f "$DATA/repo/pyproject.toml" ] || { log "ERROR: $DATA/repo/pyproject.toml missing after pull"; exit 1; }
# Re-source env.sh from the freshly pulled repo so later steps see all exports.
if [ -f "$DATA/repo/experiments/qwen35_2b_gsm8k/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$DATA/repo/experiments/qwen35_2b_gsm8k/env.sh"
fi

# --- d. uv sync --extra verl --------------------------------------------------
if done_step sync; then
  log "step d: uv sync already done (skip)"
else
  log "step d: uv sync --extra verl (Python 3.12; ~3.11 floor, 3.12 preferred per pyproject)"
  t0=$(date +%s)
  # Pin 3.12: the root pyproject requires-python >=3.11 and prefers 3.12 (verl
  # extra resolves on the default index; only the megatron group is hard-pinned
  # to 3.12, which we don't use). CUDA 13 wheels; flash-attn from the Astral GPU
  # index, so no local CUDA toolkit needed.
  ( cd "$DATA/repo" && uv sync --python 3.12 --extra verl ) 2>&1 | tee -a "$SETUP_LOG"
  t1=$(date +%s)
  log "uv sync took $((t1 - t0))s"
  mark_step sync
fi

# --- e. download model to /data/hf/Qwen3.5-2B --------------------------------
MODEL_DIR="$HF_HOME/Qwen3.5-2B"
if done_step model && [ -d "$MODEL_DIR" ]; then
  log "step e: model already at $MODEL_DIR (skip)"
else
  log "step e: download ${MODEL_ID:-Qwen/Qwen3.5-2B} -> $MODEL_DIR"
  # Qwen3.5-2B is public & ungated: no HF token needed and none is embedded.
  # Probe which downloader the synced env exposes; both land the same snapshot.
  if ( cd "$DATA/repo" && uv run hf --help >/dev/null 2>&1 ); then
    ( cd "$DATA/repo" && uv run hf download "${MODEL_ID:-Qwen/Qwen3.5-2B}" --local-dir "$MODEL_DIR" ) 2>&1 | tee -a "$SETUP_LOG"
  elif ( cd "$DATA/repo" && uv run huggingface-cli --help >/dev/null 2>&1 ); then
    ( cd "$DATA/repo" && uv run huggingface-cli download "${MODEL_ID:-Qwen/Qwen3.5-2B}" --local-dir "$MODEL_DIR" ) 2>&1 | tee -a "$SETUP_LOG"
  else
    log "ERROR: neither 'hf' nor 'huggingface-cli' available in the synced env."
    exit 1
  fi
  mark_step model
fi

# --- f. preprocess GSM8K + build subsets -------------------------------------
PRE="$DATA/repo/src/agentcore_rl_toolkit/backends/verl/examples/math_agent/preprocess_gsm8k.py"
if done_step gsm8k \
   && [ -f "$DATA/gsm8k/gsm8k_agent_test_200.parquet" ] \
   && [ -f "$DATA/gsm8k/gsm8k_agent_train_smoke.parquet" ]; then
  log "step f: gsm8k parquet + subsets present (skip)"
else
  log "step f: preprocess gsm8k -> $DATA/gsm8k"
  ( cd "$DATA/repo" && uv run python3 "$PRE" --output-dir "$DATA/gsm8k" ) 2>&1 | tee -a "$SETUP_LOG"
  # val subset (test first 200) + smoke train subset (train first 64) via pandas.
  ( cd "$DATA/repo" && uv run python3 - "$DATA/gsm8k" <<'PY' ) 2>&1 | tee -a "$SETUP_LOG"
import sys, pandas as pd
d = sys.argv[1]
test = pd.read_parquet(f"{d}/gsm8k_agent_test.parquet")
test.head(200).to_parquet(f"{d}/gsm8k_agent_test_200.parquet", index=False)
train = pd.read_parquet(f"{d}/gsm8k_agent_train.parquet")
train.head(64).to_parquet(f"{d}/gsm8k_agent_train_smoke.parquet", index=False)
print(f"wrote gsm8k_agent_test_200.parquet: {min(200, len(test))} rows")
print(f"wrote gsm8k_agent_train_smoke.parquet: {min(64, len(train))} rows")
PY
  mark_step gsm8k
fi

# --- g. summary ---------------------------------------------------------------
log "setup complete."
log "  GPU:     $GPU_NAME ($GPU_MEM), driver $DRIVER_VER"
log "  repo:    $DATA/repo"
log "  model:   $MODEL_DIR"
log "  data:    $DATA/gsm8k (full + _test_200 + _train_smoke)"
log "  markers: $SETUP_DIR"
cat <<EOF
Next steps:
  1. ./vllm_sanity.sh                    # verify vLLM can serve the model
  2. SMOKE=1 ./run_train.sh              # 2-step smoke of the full RL loop
  3. ./run_train.sh                      # the real run (TOTAL_STEPS default 60)
  4. ./stop_train.sh                     # graceful stop + final ckpt sync
Logs: $LOG_DIR   Checkpoints: $DATA/ckpts
EOF
