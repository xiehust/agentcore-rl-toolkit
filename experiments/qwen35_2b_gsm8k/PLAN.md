# 实验方案：Qwen3.5-2B × AgentCore Runtime × verl GRPO（GSM8K math agent）

> 目标：以最低成本在 Bedrock AgentCore Runtime (ACR) 上跑通一条完整的 Agent RL 链路
> （ACR 上的 Strands 数学 Agent 做 rollout → 训练机上的 rollout gateway 采集 token 级轨迹 →
> verl GRPO 更新 Qwen3.5-2B），并拿到 base vs. 训练后 的 GSM8K 对比结果。
> 约束：单台 p5 系列 spot 实例、GPU 累计运行 ≤ 24h、spot 被回收可续训。

调研依据见 `research/STAGE1_FINDINGS.md`。

---

## 1. 总体架构

```
                    us-west-2 (单区域，所有资源同区)
┌──────────────────────────────────────────────────────────────────────────────┐
│  EC2  p5.4xlarge  spot  (1× H100 80GB)          默认 VPC / 公网 IPv4          │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ verl main_ppo  (trainer.v1.trainer_mode=agentcore_sync, FSDP, 1 GPU)    │  │
│  │   ├─ vLLM rollout engine   (token-in/token-out, gpu_mem_util 0.40)      │  │
│  │   ├─ AgentLoopWorker ─► AgentCoreAgentLoop                              │  │
│  │   │     ├─ RolloutGateway  :18765  (OpenAI /v1 适配器, 固定端口)        │  │
│  │   │     └─ RolloutClient   ─► InvokeAgentRuntime (sid = uuid4)          │  │
│  │   └─ FSDP actor  (全参微调 2.27B, bf16 + fp32 master/Adam)              │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  /data (独立 gp3 EBS, 300GB): HF cache · repo+.venv · gsm8k parquet · ckpts  │
│  watchdog: GPU 小时账本 → S3, 20h 告警 / 24h 自毁; spot 2min 通知 → 立即同步 │
└───────────┬───────────────────────────────▲──────────────────┬───────────────┘
            │ ① InvokeAgentRuntime          │ ② chat/completions │ ③ ckpt / ledger sync
            │   payload + _rollout{base_url, │   (Bearer = sid)   │
            │   model_id, api_key=sid}       │                    ▼
┌───────────▼───────────────────────────────┴──────────┐  ┌─────────────────────┐
│  Bedrock AgentCore Runtime  (PUBLIC network mode)     │  │  S3  结果桶          │
│  arm64 容器: examples/strands_math_agent/rl_app.py    │  │  rollouts/<exp>/...  │
│  Strands Agent + calculator tool + GSM8KReward        │─►│  ckpt/  ledger/      │
│  @rollout_entrypoint → 后台执行 → 结果写 S3 (④)        │  └──────────▲──────────┘
└───────────────────────────────────────────────────────┘             │ ⑤ HEAD 轮询
                                                                      └── RolloutClient
```

**一次 rollout 的生命周期**
1. `AgentCoreAgentLoop.run()` 生成 `sid=uuid4`，`gateway.create_session(sid)`，`RolloutClient.invoke_async(payload, session_id=sid)`；`_rollout` 里携带 `base_url=http://<EC2公网IP>:18765/v1`、`model_id`、`api_key=sid`、`sampling_params`。
2. ACR 起一个 microVM 会话运行 `rl_app.py`，HTTP 立即返回 in-progress，Agent 在后台跑；Strands `OpenAIModel` 用 `api_key=sid` 调 gateway。
3. gateway 用 HF chat template（已命中 `qwen3_5 nothink` schema）渲染成 token ids → vLLM 采样 → 记录 token ids + logprobs + loss mask → 反解成 OpenAI 响应（含 XML tool call）。
4. Agent 完成后把 `{"rewards": 0/1}` 写到 S3；trainer HEAD 轮询到结果，`finish_session(sid)` 得到 `TraceRecord` → verl 训练行；reward 直接作为 `rm_scores`。
5. GRPO：同一 prompt 的 `n` 条 rollout 做组内标准化优势；单次策略更新 + KL loss + token 级 IS 校正作为信任域。

## 2. 网络方案

### 2.1 主方案：ACR PUBLIC 模式 + EC2 公网 IP + 固定 gateway 端口

| 项 | 设定 | 理由 |
|---|---|---|
| ACR `networkConfiguration` | `PUBLIC` | 无需 VPC/子网/ENI 配置，创建最快；容器可直接出公网访问 EC2 |
| gateway 监听 | `gateway_bind_host=0.0.0.0`，`gateway_port=18765`（固定） | 端口可预知，安全组只开一个端口；`gateway_public_host=<EC2 公网 IPv4>`（启动时从 IMDS 读取注入 env） |
| 安全组 inbound | `tcp/18765` from `0.0.0.0/0`；**不开 22** | ACR PUBLIC 出口 IP 不可枚举，只能全开该端口；见下方缓解 |
| 管理通道 | SSM Session Manager（实例角色附 `AmazonSSMManagedInstanceCore`）；备用：临时加 22 仅放本机出口 IP | 不暴露 SSH |
| 出站 | 默认全开 | 拉镜像/HF/ pip / S3 / ACR API |

**开放 18765 的风险缓解**（Stage 4 落实、Stage 6 验证）：
- **已核实**：当前 gateway **无鉴权** —— `adapters/common.py` 对请求里的 sid 用 `store.setdefault(sid, Session())` 隐式建会话，任何人拿到端口都能白嫖推理并往 trainer 内存里塞垃圾会话树。
- 对策（Stage 4 实现，小改动、默认关闭保持仓库行为不变）：给 `RolloutGateway`/adapter 加一个 `require_registered_sessions` 开关，开启时只接受 trainer 已 `create_session` 的 sid（uuid4，攻击者不可猜），其余返回 401；通过 `agentcore_agent.yaml` 的 kwarg 打开。这样 Bearer=sid 本身就成了一次性会话令牌。
- 端口只在训练进程存活期间监听；实例 24h 内销毁。
- 暴露面仅为一个只会返还 2B 模型采样文本的推理接口，无凭证、无数据。
- 不采用"SG 只放 AWS us-west-2 IP 段"：AMAZON 前缀数百条，超过 SG 规则配额。

### 2.2 加固备选：ACR VPC 模式
把 ACR runtime 的 `networkConfiguration` 设为 `VPC`（默认 VPC 的私有/公有子网 + 专用 SG），EC2 SG 的 18765 只放行 ACR 的 SG，`gateway_public_host` 用 EC2 **私网 IP**。代价：需要额外子网/NAT（PUBLIC 模式容器仍需出公网拉 S3？——ACR VPC 模式访问 S3 走 VPC endpoint 或 NAT），配置面变大。仅在主方案被安全评审否决时启用。

## 3. 计算与显存预算（单卡 H100 80GB）

verl 同步模式下 rollout 与训练 **colocate 在同一张卡**，训练期间 vLLM 引擎 sleep 并释放显存，两阶段基本不叠加：

| 阶段 | 占用估算 | 说明 |
|---|---|---|
| rollout（vLLM） | ≤ 32 GB（`gpu_memory_utilization=0.40`） | 权重 bf16 ≈ 4.6 GB，其余为 KV cache；2B 模型 4k 上下文足够并发数百会话 |
| training（FSDP 全参） | 权重 bf16 4.6 + fp32 master 9.1 + Adam 18.2 + grad 4.6 ≈ **37 GB** + 激活 | `use_dynamic_bsz` + `ppo_max_token_len_per_gpu=8192` 控制激活；开 gradient checkpointing |
| ref 模型 logprob | 复用 actor 权重前向（verl 在 1 GPU 时 ref 与 actor 顺序执行） | 若 OOM，`ref.fsdp_config.param_offload=true` |

回退梯度：① `actor.fsdp_config.optimizer_offload=true`；② 改 LoRA（`lora_rank=32`，`lr=2e-5`，参考 `fsdp_lora_sync_grpo.sh`）。

## 4. 训练超参（正式跑）

基于仓库已验证的 `fsdp_fft_sync_grpo.sh`（Qwen3-4B, 8×GPU）缩放到 1×GPU / 2B / 短上下文：

| 参数 | 值 | 说明 |
|---|---|---|
| `actor_rollout_ref.model.path` | `Qwen/Qwen3.5-2B` | 预下载到 `/data/hf`，用本地路径 |
| `MAX_MODEL_LEN` / `rollout.max_model_len` | **4096** | GSM8K 问题 <200 tok，含 calculator 多轮一般 <1.5k |
| `rollout.prompt_length` / `data.max_prompt_length` | 2048 | 训练行的前导上下文（可含多轮） |
| `rollout.response_length` / `data.max_response_length` | 4096 | 累计轨迹预算 = max_model_len |
| `max_tokens_per_turn`（yaml） | 1024 | 单次模型调用上限 |
| `data.train_batch_size` | **32** | 每 step 32 个 prompt |
| `rollout.n` | **8** | 每 step 256 条 rollout（ACR 并发 256 会话，远低于账户上限） |
| `actor.ppo_mini_batch_size` | 32 | = train_batch_size → 每 step 一次策略更新，KL 为唯一信任域 |
| `actor.optim.lr` | 5e-6 | 全参微调标度 |
| `actor.use_kl_loss` / `kl_loss_coef` / `kl_loss_type` | true / 0.001 / low_var_kl | 仓库验证的稳定配置 |
| `algorithm.rollout_correction.rollout_is` / threshold | token / 2.0 | 用 gateway 采到的 rollout logprob 校正 vLLM↔FSDP 概率差 |
| `actor.use_dynamic_bsz` / `ppo_max_token_len_per_gpu` | true / 8192 | |
| `actor.loss_agg_mode` | seq-mean-token-sum | `agentcore_sync` 强制要求 |
| `rollout.tensor_model_parallel_size` | 1 | 单卡 |
| `rollout.gpu_memory_utilization` | 0.40 | |
| `rollout.temperature` | 1.0（训练）/ 0.6（val） | |
| `rollout.agent.num_workers` | 1 | 一个 gateway 进程/端口 |
| `tps_limit`（yaml） | 8 | 账号 ACR 新会话 25/s 的 1/3；256 会话约 32s 提交完 |
| `max_rollout_time`（yaml） | 180 s | 数学题远低于此；卡死会话尽快失败 |
| `trainer.n_gpus_per_node` / `nnodes` | 1 / 1 | |
| `trainer.total_epochs` / 目标 step | 1 / **≤ 60 step**（7473/32 ≈ 233 step/epoch，不跑完） | 用 `trainer.total_training_steps=60` |
| `trainer.save_freq` / `test_freq` | 10 / 10 | 每 10 step 存 ckpt 并同步 S3 |
| `trainer.val_before_train` | true | 取得 base 基线 |
| val 集 | GSM8K test **前 200 题**子集（`gsm8k_agent_test_200.parquet`），`val_kwargs.n=1` | 全量 1319 题每次 ~5-8 min，200 题足够看趋势 |
| `trainer.resume_mode` | auto | spot 回收后从 `/data/ckpts` 最新 ckpt 续训 |
| `trainer.logger` | console（+ 本地 tensorboard 文件） | 不引入 wandb 依赖/账号 |

**冒烟配置（Stage 6）**：`train_batch_size=8, n=4, ppo_mini_batch_size=8, total_training_steps=2, val_before_train=false, save_freq=1`，验证链路后立即停。

**预计节拍**：每 step ≈ rollout 2–3 min（含 ACR 冷启动、tool 多轮）+ 训练 <1 min ≈ **3–4 min/step**；60 step ≈ 3.5–4 h；每 10 step 一次 200 题 val ≈ +1.5 min。

## 5. Spot 回收容错设计

| 机制 | 设计 |
|---|---|
| 持久化数据卷 | 独立 gp3 EBS 300 GB（tag `Project=qwen35-2b-gsm8k, Role=data`），挂 `/data`：`hf/`（模型缓存）、`repo/`（仓库 + `.venv`）、`gsm8k/`、`ckpts/`、`logs/`。root 卷仅放系统，随实例销毁 |
| spot 请求类型 | `one-time`（不用 persistent，避免 AWS 自动重拉实例绕过我们的 24h 账本） |
| AZ 绑定 | EBS 卷绑定 AZ；`resume.sh` 优先在同 AZ 重申请。若该 AZ 无 spot 容量：对数据卷做快照 → 在有容量的 AZ 从快照建新卷 → 启动。首次选 AZ 时按 `describe-spot-placement-scores`/当前价格挑最便宜且容量分高的 |
| checkpoint 同步 | `sync_ckpt.sh` 后台循环：每 5 min `aws s3 sync /data/ckpts s3://<bucket>/ckpt/` ；spot 2 分钟中断通知（IMDS `spot/instance-action`）触发立即同步 |
| 续训 | verl `trainer.resume_mode=auto` 读取 `default_local_dir=/data/ckpts/...` 最新 step；若卷丢失则先 `aws s3 sync` 回本地 |
| 环境重建 | 实例启动 user-data：挂卷 → `source /data/repo/.venv/bin/activate` → 启动 watchdog；不重装依赖（venv 在卷上）。仅 AMI 驱动版本不匹配时重装（脚本检测 `nvidia-smi` ≥ 580） |
| 一键脚本 | `infra/launch_spot.sh`（首启，建卷）→ `infra/resume.sh`（复用卷）→ `infra/terminate.sh`（销毁实例，保留卷/可选快照后删卷） |

## 6. 成本护栏（详见 COST.md）

- **GPU 小时账本**：实例内 `watchdog.sh` 每 5 min 把 `{instance_id, boot_time, session_hours, cumulative_hours}` 写入 `s3://<bucket>/ledger/gpu_hours.json`；启动时读回累计值。**≥ 20h 发告警（写 S3 + 控制台日志），≥ 24h 自动 `terminate-instances` 自身**（实例角色权限用 `ec2:ResourceTag/Project` 条件收窄）。
- 本机 `infra/gpu_hours.sh`：读账本 + `describe-instances` 汇总当前累计与剩余额度。
- 实例硬约束：所有启动脚本先 `describe-instances` 确认无同 tag 的 running/pending 实例，否则拒绝启动（**最多 1 台**）。
- 每个阶段结束都要显式记录 GPU 小时消耗。

## 7. 实验目录布局

```
experiments/qwen35_2b_gsm8k/
├── README.md              # 索引 + 快速开始
├── PLAN.md                # 本文件
├── COST.md                # 成本模型与阈值
├── research/              # Stage 1 调研报告
├── agent/                 # Stage 3: 镜像构建/ECR 推送/ACR 创建/冒烟调用
├── infra/                 # Stage 4: IAM、SG、spot 启动/续训/销毁、watchdog、账本
├── trainer/               # Stage 4: setup_trainer.sh、train_qwen35_2b.sh、agentcore_agent.yaml、smoke 覆盖
├── eval/                  # Stage 8: base vs. trained 评估脚本
└── REPORT.md              # Stage 8: 结果、成本、踩坑
```

## 8. 阶段门禁（每阶段结束必须满足）

| Stage | 门禁 |
|---|---|
| 3 | 本地 docker 起 rl_app 容器：`/ping` 200；ACR runtime 状态 READY；一次 `invoke` 用外部 OpenAI 兼容端点跑通并在 S3 看到 `{"rewards": ...}` |
| 4 | 所有脚本 `bash -n` 通过；IAM/SG 已创建；`launch_spot.sh --dry-run` 输出正确参数；未启动任何 GPU 实例 |
| 5 | `nvidia-smi` 驱动 ≥ 580.65.06；`uv sync --extra verl` 成功；vLLM 单独 serve Qwen3.5-2B 一次 chat completion 成功；账本已开始计时 |
| 6 | 冒烟 2 step 完成：`batching/total_real_rows>0`、`rollout_failure/total_missing_sessions≈0`、ckpt 落盘并同步 S3；GPU 累计 ≤ 4h |
| 7 | ≥ 40 step 或达到预算阈值；reward 曲线记录；ckpt 在 S3 |
| 8 | 200 题 val：base vs trained 对比；GPU 累计 ≤ 24h；实例已 terminate；REPORT.md 完成 |
