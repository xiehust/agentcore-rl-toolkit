# trainer/ 脚本规格（Stage 4）

依据（必读）：`../env.sh`、`../PLAN.md` §3/§4/§5（**§4 超参表为准**）、`../research/qwen35_2b_model.md`、
`src/agentcore_rl_toolkit/backends/verl/README.md`、`backends/verl/examples/math_agent/{fsdp_fft_sync_grpo.sh,agentcore_agent.yaml,preprocess_gsm8k.py}`、
`backends/verl/agent_loop.py`（`__init__` kwargs）、`backends/verl/gateway_host.py`、`docs/site/src/content/docs/guides/verl-backend-setup.md`、
根 `pyproject.toml`（verl extra、`[tool.uv]` conflicts）以确定安装命令与 Python 版本。

## 环境上下文
- 训练机 EC2 p5.4xlarge（1× H100 80GB，Ubuntu 24.04 DL Base OSS Nvidia Driver AMI）。
- 持久卷 `/data`：`/data/hf`（HF cache）、`/data/repo`（本仓库，来自 `s3://$ACR_S3_BUCKET/code/repo.tar.gz`，由同事的 `infra/upload_code.sh` 产出）、`/data/gsm8k`、`/data/ckpts`、`/data/logs`。
- 实例公网 IPv4 作为 `gateway_public_host`；gateway 端口固定 `$GATEWAY_PORT`（18765）。
- 同事正在给 `AgentCoreAgentLoop` 增加 kwarg `require_registered_sessions: bool`，yaml 里设 `true` 并注释。
- 实例使用 IAM 角色（无静态密钥）。`Qwen/Qwen3.5-2B` 公开无 gate：不需要也**不要嵌入** HF token。
- 无 SSH，脚本经 SSM shell 以 ubuntu 用户运行：非交互、日志到 `/data/logs`。
- 所有脚本：bash，`set -euo pipefail`，`bash -n` 通过，chmod +x，source `/data/repo/experiments/qwen35_2b_gsm8k/env.sh`（或相对路径等价）。

## 1. setup_trainer.sh（实例上运行一次；各步可通过 `/data/.setup/` 标记文件跳过）
a. `nvidia-smi` 可用且驱动 ≥ 580.65.06（解析比较；失败给出修复提示），打印 GPU 名/显存。
b. 缺则安装 uv；`export UV_CACHE_DIR=/data/uv-cache HF_HOME=/data/hf`。
c. 拉代码 `s3://$ACR_S3_BUCKET/code/repo.tar.gz` → `/data/repo`（`--refresh` 重新下载）。
d. `cd /data/repo && uv sync --extra verl`，Python 版本按仓库要求（≥3.11，优先 3.12），日志 `/data/logs/setup.log`，打印耗时。
e. 下载模型到 `/data/hf/Qwen3.5-2B`（`uv run hf download` 或 `huggingface-cli download`，运行时探测哪个存在）。
f. 预处理 GSM8K：仓库 `preprocess_gsm8k.py --output-dir /data/gsm8k`；再用 pandas 生成 `/data/gsm8k/gsm8k_agent_test_200.parquet`（test 前 200 行）与 `/data/gsm8k/gsm8k_agent_train_smoke.parquet`（train 64 行）。
g. 打印摘要与下一步。

## 2. vllm_sanity.sh
`uv run vllm serve /data/hf/Qwen3.5-2B --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.4`，并按 `research/qwen35_2b_model.md` 的 text-only 建议（如 `--language-model-only`）在 `vllm serve --help` 检测支持后有条件加上；等 `/health`；发一条带 calculator tool 定义的 chat completion（"What is 17*23? Use the tool."）；打印响应；杀掉服务。日志 `/data/logs/vllm_sanity.log`。

## 3. agentcore_agent.yaml
基于示例 yaml，按 PLAN §4：`agent_runtime_arn ${oc.env:AGENT_RUNTIME_ARN}`、`s3_bucket ${oc.env:ACR_S3_BUCKET}`、`exp_id ${oc.env:EXP_ID}`、`max_tokens_per_turn 1024`、`tps_limit 8`、`max_rollout_time 180`、`gateway_port` 来自 env（若需 int 用 `${oc.decode:${oc.env:GATEWAY_PORT}}`，在 scratch venv 用 omegaconf 验证）、`gateway_public_host ${oc.env:GATEWAY_PUBLIC_HOST}`、`require_registered_sessions: true`。逐项注释。

## 4. train_qwen35_2b.sh（前台运行）
fsdp_fft_sync_grpo.sh 的单卡改版，**严格按 PLAN §4 表**：model `/data/hf/Qwen3.5-2B`、MAX_MODEL_LEN 4096、prompt_length/max_prompt_length 2048、response_length/max_response_length 4096、train_batch_size 32、ppo_mini_batch_size 32、n 8、lr 5e-6、KL 0.001 low_var_kl、rollout_is token/2.0、use_dynamic_bsz + ppo_max_token_len_per_gpu 8192、seq-mean-token-sum、TP 1、gpu_memory_utilization 0.40、n_gpus_per_node 1、nnodes 1、save_freq 10、test_freq 10、val_before_train true、val = 200 行子集、val n=1 temp 0.6、`trainer.total_training_steps=${TOTAL_STEPS:-60}`、resume_mode=auto、default_local_dir `/data/ckpts/$PROJECT_NAME/$EXPERIMENT_NAME`、`TRAINER_LOGGER` env 默认 `'["console"]'`、`actor_rollout_ref.model.enable_gradient_checkpointing=True`、`actor.fsdp_config.optimizer_offload=${OPT_OFFLOAD:-False}` / `param_offload=${PARAM_OFFLOAD:-False}`、ref log_prob micro batch 1。
启动前：未设置则从 IMDSv2 `public-ipv4` 解析 `GATEWAY_PUBLIC_HOST`；导出 AGENT_RUNTIME_ARN / ACR_S3_BUCKET / EXP_ID / GATEWAY_PORT；存在 `/data/STOP_TRAINING` 则拒绝启动（`--force` 删除）。
`SMOKE=1`：train 文件 = smoke parquet、train_batch_size 8、ppo_mini_batch_size 8、n 4、total_training_steps 2、val_before_train false、test_freq -1、save_freq 1、EXPERIMENT_NAME 加 `_smoke` 后缀。
额外 Hydra 覆盖用 `"$@"`。在 `/data/repo` 下 `uv run python3 -m verl.trainer.main_ppo`；导出 VERL_USE_EXTERNAL_MODULES、HYDRA_FULL_ERROR=1、HF_HOME、TOKENIZERS_PARALLELISM=false。

## 5. run_train.sh / stop_train.sh
- run：`setsid nohup ./train_qwen35_2b.sh "$@" > /data/logs/train_<ts>.log 2>&1 &`，PID 写 `/data/logs/train.pid`（watchdog 在 20h 软限时对其进程组发 SIGTERM）；若 sync_ckpt.sh 循环未运行则启动（pid `/data/logs/sync_ckpt.pid`）；打印 tail 提示。
- stop：对 train.pid 进程组 SIGTERM，等 120 s 后 SIGKILL；最后 `sync_ckpt.sh --once`。

## 6. sync_ckpt.sh / restore_ckpt.sh
- `--once`：`aws s3 sync /data/ckpts s3://$ACR_S3_BUCKET/ckpt/` + `/data/logs` → `logs/`；无参数则每 300 s 循环。
- restore：反向同步。

## 7. trainer/README.md
实例上的运行顺序（setup → vllm_sanity → `SMOKE=1 ./run_train.sh` → `./run_train.sh` → stop/terminate）、日志与 ckpt 位置、OOM 回退开关、要盯的指标（`batching/total_real_rows`、`training/rollout_failure/total_missing_sessions`、val reward、`critic/advantages/zero_mean`）、token 预算说明。

## 约束
本机（aarch64、无 GPU）**不要**安装 verl/vllm；可在 `$KIROCREW_SCRATCH` 建临时 venv 做 omegaconf/pandas 验证。`bash -n` 全部脚本。只改 `experiments/qwen35_2b_gsm8k/trainer/`。汇报：文件、与 PLAN §4 的偏差及原因、无法验证的 flag。
