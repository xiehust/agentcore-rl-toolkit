# trainer/ — Stage 4/5 training scripts (Qwen3.5-2B × GSM8K GRPO)

Single-GPU (p5.4xlarge, 1× H100 80GB) GRPO fine-tuning of `Qwen/Qwen3.5-2B`,
rollouts on Bedrock AgentCore Runtime, token capture via the in-repo rollout
gateway. All scripts are bash `set -euo pipefail`, run non-interactively over SSM
as the `ubuntu` user, log to `/data/logs`, and `source ../env.sh`.

## Run order (on the instance)

```bash
cd /data/repo/experiments/qwen35_2b_gsm8k/trainer

# 1. one-time bring-up (idempotent; per-step markers in /data/.setup)
./setup_trainer.sh                # add --refresh to re-pull the repo, --force to redo all steps

# 2. verify vLLM can serve the model (Stage 5 gate)
./vllm_sanity.sh

# 3. smoke the full RL loop end to end (2 steps, no baseline val) — Stage 6 gate
SMOKE=1 ./run_train.sh

# 4. the real run (defaults to TOTAL_STEPS=60)
./run_train.sh                    # backgrounded; follow with tail -f /data/logs/train_*.log

# 5. graceful stop + final S3 sync (also runs automatically at the 20h watchdog limit)
./stop_train.sh
```

Restore after a spot reclaim on a fresh volume: `./restore_ckpt.sh` then
`./run_train.sh` (verl `resume_mode=auto` picks up the newest local step).

## Files

| script | role |
|---|---|
| `setup_trainer.sh` | driver check, uv, repo pull, `uv sync --extra verl`, model download, GSM8K preprocess + subsets |
| `vllm_sanity.sh` | `vllm serve` the local model, one tool-calling chat completion, teardown |
| `agentcore_agent.yaml` | `AgentCoreAgentLoop` kwargs (ARN/bucket/port/host, per-turn tokens, tps, `require_registered_sessions`) |
| `train_qwen35_2b.sh` | foreground `main_ppo` launch, all PLAN §4 hyperparameters, `SMOKE=1` overrides |
| `run_train.sh` | detached launch (`setsid nohup`), starts the ckpt-sync loop, writes `train.pid` |
| `stop_train.sh` | SIGTERM the process group, 120s grace, SIGKILL, final `sync_ckpt.sh --once` |
| `sync_ckpt.sh` | `--once` or 300s loop: `aws s3 sync /data/ckpts` + `/data/logs` → S3 |
| `restore_ckpt.sh` | reverse sync S3 → `/data/ckpts` |

## Locations

- **Logs:** `/data/logs/` — `setup.log`, `vllm_sanity.log`, `train_<ts>.log`,
  `sync_ckpt_<ts>.log`. PIDs: `train.pid`, `sync_ckpt.pid`.
- **Checkpoints:** `/data/ckpts/$PROJECT_NAME/$EXPERIMENT_NAME/` (default
  `qwen35_2b_gsm8k/qwen35_2b_grpo`; smoke adds a `_smoke` suffix), mirrored to
  `s3://$ACR_S3_BUCKET/ckpt/`.
- **Model:** `/data/hf/Qwen3.5-2B`. **Data:** `/data/gsm8k/` (full
  train/test + `gsm8k_agent_test_200.parquet` val + `gsm8k_agent_train_smoke.parquet`).

## OOM fallback switches (PLAN §3)

Training footprint is ~37 GB (bf16 weights + fp32 master + Adam + grads) plus
activations; gradient checkpointing is on by default. If you still OOM, escalate
in this order via env vars on `run_train.sh` / `train_qwen35_2b.sh`:

```bash
OPT_OFFLOAD=True ./run_train.sh                       # offload the Adam optimizer state
OPT_OFFLOAD=True PARAM_OFFLOAD=True ./run_train.sh     # also offload FSDP params (ref logprob relief)
```

Further fallback (config change, not a switch): drop to LoRA per PLAN §3
(`fsdp_lora_sync_grpo.sh`: `lora_rank=32`, `lr=2e-5`). `gpu_memory_utilization`
is 0.40 so rollout and training colocate without stacking; lowering it trades
vLLM KV-cache headroom for training memory.

## Metrics to watch (`trainer.logger=["console"]`)

- **`batching/total_real_rows`** — must be `> 0`. Zero means no trajectory rows
  materialized (gateway/agent contract broken).
- **`training/rollout_failure/total_missing_sessions`** — nominal rollouts
  (`train_batch_size * rollout.n`) minus sessions actually seen; should be ≈ 0.
  Sustained non-zero ⇒ ACR errors, gateway unreachable, or timeouts.
- **val reward** — `val_before_train` gives the base baseline; each `test_freq`
  (10) step re-evaluates the 200-row subset. This is the headline curve.
- **`critic/advantages/zero_mean`** (and `zero_pass_mean`) — fraction of GRPO
  groups whose advantages collapsed (all rollouts in a group scored identically);
  a high value means the group learns nothing that step.

## Token budget (why these lengths)

`max_model_len = 4096` is the engine context (GSM8K questions < 200 tok; with
calculator multi-turn, trajectories stay < ~1.5k). `prompt_length = 2048` is
verl's fixed storage width for each row's leading context; `response_length =
4096 = max_model_len` is the cumulative trajectory budget (must not exceed
`max_model_len`). `max_tokens_per_turn = 1024` (yaml) caps a single model call.
With `use_remove_padding=True`, the nominal padded row width is
`prompt_length + response_length`, but the gateway only ever emits up to
`max_model_len` valid tokens.

## Networking / security note

The gateway listens on `0.0.0.0:$GATEWAY_PORT` (18765) and advertises the
instance's public IPv4 (resolved from IMDSv2 at launch) so ACR PUBLIC-mode
containers can call back. The port is otherwise open to the internet, so
`require_registered_sessions: true` in `agentcore_agent.yaml` makes the gateway
accept only trainer-registered uuid4 sids (Bearer = one-time session token);
everything else gets 401. The port is only open while training runs and the
instance is destroyed within 24h.
