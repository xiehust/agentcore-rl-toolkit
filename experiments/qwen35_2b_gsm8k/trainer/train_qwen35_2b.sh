#!/usr/bin/env bash
# train_qwen35_2b.sh — single-GPU GRPO on GSM8K for Qwen3.5-2B, run in the
# FOREGROUND. Single-card adaptation of the repo's fsdp_fft_sync_grpo.sh, with
# every hyperparameter pinned to PLAN §4. run_train.sh backgrounds this.
#
# Usage:
#   ./train_qwen35_2b.sh [--force] [hydra.overrides=...] ...
#   SMOKE=1 ./train_qwen35_2b.sh          # 2-step link smoke (PLAN §4 smoke row)
#   TOTAL_STEPS=40 ./train_qwen35_2b.sh   # override step budget
#   OPT_OFFLOAD=True PARAM_OFFLOAD=True ./train_qwen35_2b.sh   # OOM fallbacks
#
# Extra CLI args are forwarded verbatim to main_ppo as Hydra overrides via "$@".
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

# --- pre-flight: --force flag + STOP sentinel --------------------------------
FORCE=0
PASS_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *) PASS_ARGS+=("$arg") ;;
  esac
done
set -- "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}"

STOP_FILE=/data/STOP_TRAINING
if [ -f "$STOP_FILE" ]; then
  if [ "$FORCE" -eq 1 ]; then
    echo "found $STOP_FILE; --force given, removing it and proceeding"
    rm -f "$STOP_FILE"
  else
    echo "REFUSING to start: $STOP_FILE exists (a prior stop/guard set it)." >&2
    echo "Pass --force to delete it and start anyway." >&2
    exit 1
  fi
fi

DATA=/data
export HF_HOME="$DATA/hf"
export UV_CACHE_DIR="$DATA/uv-cache"
export HYDRA_FULL_ERROR=1
# Ray's uv-run runtime_env hook crashes (path_or_uri None) when launched via `uv run`;
# we use the venv interpreter directly and disable the hook.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# flashinfer JIT-compiles the Qwen3.5 GDN kernels inside the vLLM worker and needs
# ninja (venv) + nvcc on PATH; Ray workers inherit this environment.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH="/data/repo/.venv/bin:${CUDA_HOME}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
# Register the agentcore_* trainer modes before main_ppo's process-local lookup.
export VERL_USE_EXTERNAL_MODULES=agentcore_rl_toolkit.backends.verl.trainer

# --- gateway host: resolve public IPv4 from IMDSv2 if unset ------------------
# ACR PUBLIC-mode containers dial the trainer's public IP (PLAN §2.1). Resolve it
# once here and export so agentcore_agent.yaml's ${oc.env:GATEWAY_PUBLIC_HOST}
# and downstream workers all agree.
if [ -z "${GATEWAY_PUBLIC_HOST:-}" ]; then
  TOKEN="$(curl -sf -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" || true)"
  if [ -n "$TOKEN" ]; then
    GATEWAY_PUBLIC_HOST="$(curl -sf -H "X-aws-ec2-metadata-token: $TOKEN" \
      "http://169.254.169.254/latest/meta-data/public-ipv4" || true)"
  fi
  if [ -z "${GATEWAY_PUBLIC_HOST:-}" ]; then
    echo "ERROR: could not resolve public IPv4 from IMDSv2 and GATEWAY_PUBLIC_HOST unset." >&2
    echo "       Export GATEWAY_PUBLIC_HOST=<reachable trainer IP> and retry." >&2
    exit 1
  fi
fi
export GATEWAY_PUBLIC_HOST

# --- env the agent-loop yaml interpolates (fail loudly if missing) -----------
: "${AGENT_RUNTIME_ARN:?AGENT_RUNTIME_ARN must be set (agent/runtime.env / env.sh)}"
: "${ACR_S3_BUCKET:?ACR_S3_BUCKET must be set (env.sh)}"
: "${EXP_ID:?EXP_ID must be set (env.sh)}"
: "${GATEWAY_PORT:?GATEWAY_PORT must be set (env.sh)}"
export AGENT_RUNTIME_ARN ACR_S3_BUCKET EXP_ID GATEWAY_PORT

AGENT_LOOP_CONFIG="$SCRIPT_DIR/agentcore_agent.yaml"
MODEL_PATH="$DATA/hf/Qwen3.5-2B"   # PLAN §4: pre-downloaded, use local path
[ -d "$MODEL_PATH" ] || { echo "model missing at $MODEL_PATH — run setup_trainer.sh" >&2; exit 1; }

# --- PLAN §4 fixed hyperparameters -------------------------------------------
MAX_MODEL_LEN=4096
PROMPT_LENGTH=2048
RESPONSE_LENGTH=4096

PROJECT_NAME=${PROJECT_NAME:-qwen35_2b_gsm8k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_2b_grpo}

# Logger: env-overridable, defaults to console-only (no wandb dep/account).
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}

# OOM-fallback offload switches (PLAN §3), default off.
OPT_OFFLOAD=${OPT_OFFLOAD:-False}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}

# Data files (full run). Val is the 200-row subset per PLAN §4.
TRAIN_FILE="$DATA/gsm8k/gsm8k_agent_train.parquet"
VAL_FILE="$DATA/gsm8k/gsm8k_agent_test_200.parquet"

# Batch / rollout knobs — full-run defaults, overridden below for SMOKE.
TRAIN_BATCH_SIZE=32
PPO_MINI_BATCH_SIZE=32
ROLLOUT_N=8
TOTAL_STEPS=${TOTAL_STEPS:-60}
VAL_BEFORE_TRAIN=true
TEST_FREQ=10
SAVE_FREQ=10

# --- SMOKE overrides (PLAN §4 smoke row) -------------------------------------
if [ "${SMOKE:-0}" = "1" ]; then
  echo "SMOKE=1: link-check configuration (2 steps, no baseline val)"
  TRAIN_FILE="$DATA/gsm8k/gsm8k_agent_train_smoke.parquet"
  TRAIN_BATCH_SIZE=8
  PPO_MINI_BATCH_SIZE=8
  ROLLOUT_N=4
  TOTAL_STEPS=2
  VAL_BEFORE_TRAIN=false
  TEST_FREQ=-1
  SAVE_FREQ=1
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_smoke"
fi

CKPTS_DIR="$DATA/ckpts/$PROJECT_NAME/$EXPERIMENT_NAME"
mkdir -p "$CKPTS_DIR"

train_files="['$TRAIN_FILE']"
val_files="['$VAL_FILE']"

# --- launch main_ppo ----------------------------------------------------------
cd "$DATA/repo"
/data/repo/.venv/bin/python -m verl.trainer.main_ppo \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=agentcore_sync \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$PROMPT_LENGTH" \
    data.max_response_length="$RESPONSE_LENGTH" \
    data.custom_cls.path=pkg://agentcore_rl_toolkit.backends.verl.dataset \
    data.custom_cls.name=PayloadDataset \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OPT_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.param_offload="$PARAM_OFFLOAD" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.prompt_length="$PROMPT_LENGTH" \
    actor_rollout_ref.rollout.response_length="$RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE:-1}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=agentcore_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.critic_warmup=0 \
    trainer.default_local_dir="$CKPTS_DIR" \
    trainer.resume_mode=auto \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.n_gpus_per_node="${N_GPUS:-1}" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    "$@"
