# REPORT — Qwen3.5-2B × AgentCore × verl GRPO（GSM8K）

> 状态（2026-09-24 02:25 UTC）：**全链路跑通，60 step 训练完成，GPU 实例已终止。**
> 验证分数 0.545 → 最佳 0.85（step 10）→ 最终 0.795（step 60）。
> VPC 私网改造（另一份 5 阶段计划）只完成了 Stage 1，本次训练全程走公网 gateway :18765。

## 1. 完成情况

| Stage | 状态 | 产出 |
|---|---|---|
| 1 可行性与资源盘点 | ✅ | `research/`（spot 价格/配额、模型兼容性、账号资源） |
| 2 方案与成本设计 | ✅ | `PLAN.md`、`COST.md` |
| 3 Agent 端部署 | ✅ | ECR 镜像、S3 桶、执行角色、ACR runtime（us-west-2，PUBLIC） |
| 4 训练端代码 | ✅ | `infra/`、`trainer/`；gateway `require_registered_sessions` 开关（+5 测试，全量 462 passed / 10 skipped） |
| 5 spot + 环境准备 | ✅（第二台实例） | `i-01d7f6407e1a085c8` p5.48xlarge spot us-east-2c；`setup_trainer.sh`、`vllm_sanity.sh` 通过 |
| 6 冒烟 | ✅ | 2 step，reward 0.25→0.55，missing sessions 0 |
| 7 正式训练 | ✅ | 60 step，00:28–02:13 UTC（1 h 41 min，≈101 s/step） |
| 8 收尾 | ✅ | ckpt 已同步 S3，实例已终止（02:19:35 UTC），本报告 |

没有单独运行 `evaluate.py` 做 GSM8K test 对比，下面的数字是 verl 在固定的 200 题 val 子集上做的验证。

## 2. 训练结果

配置：8×H100，TP=2，`train_batch_size=32`，`n=8`（每步 256 条 rollout），`lr=5e-6`，KL + token-IS，每 10 step 验证一次。

| step | val reward | train score | KL loss | 平均回复长度 |
|---|---|---|---|---|
| 训练前 | 0.545 | – | – | – |
| 10 | **0.85** | 0.84 | 65 | 219 |
| 20 | 0.80 | 0.79 | 107 | 365 |
| 30 | 0.70 | 0.57 | 137 | 398 |
| 40 | 0.74 | ≈0.75 | ≈207 | 534 |
| 50 | 0.75 | 0.77 | 214 | 514 |
| 60 | 0.795 | 0.77 | 211 | 515 |

结论：
- 前 10 step 提升明显（+0.30），之后没有继续变好。step 10 之后 KL 一路上升到 ~210，回复长度翻倍，val 在 0.70–0.80 之间波动。推测 KL 约束偏弱或 lr 偏大。这是根据曲线做的推断，没有逐条检查 rollout。
- 最终 val 中 `acr_failed` = 5.5%（200 题里约 11 次 ACR 调用失败，按 0 分计），所以最终 0.795 可能偏低。早期 val 的失败率没有单独统计。
- 训练期间 `total_missing_sessions` 始终为 0，每步 256 行。日志里共 5 条错误：一次 ACR 502（3 行），一次 `InvokeAgentRuntime RuntimeClientError`（单条 rollout），以及训练结束后退出时 DataLoader worker 被 kill 的 atexit 报错，这条不影响结果。

建议的下一轮（未执行）：从 step 10 或 step 20 续训，提高 `kl_loss_coef` 或降低 lr 到 1e-6～2e-6，限制 `max_response_length`，并用 `evaluate.py` 在 GSM8K test 上对比 base、step10 和 step60。

## 3. Checkpoint 位置

- S3：`s3://agentcore-rl-qwen35-2b-gsm8k-<ACCOUNT_ID>-usw2/ckpt/qwen35_2b_gsm8k/qwen35_2b_grpo/global_step_{10..60}/`，6 个都完整，每个 actor 约 28.6 GB（`sync_ckpt.sh` 不做 `--delete`，所以本地的 keep-last-2 策略不影响 S3）。
- 数据卷 `vol-07961b53e01d3306d`（us-east-2c）：step 50、60，以及 `/data/ckpts_keep/global_step_20`。

## 4. 当前资源

| 资源 | 位置 | 状态 | 费用 | 处理 |
|---|---|---|---|---|
| EC2 `i-01d7f6407e1a085c8` | us-east-2c | shutting-down / terminated | 停止计费 | – |
| 数据卷 `vol-07961b53e01d3306d` 300 GB | us-east-2c | available | ≈$24/月 | 续训复用，或 `aws ec2 delete-volume --region us-east-2 --volume-id vol-07961b53e01d3306d` |
| 空卷 `vol-0c77a9cd869e23f4a` 300 GB | us-west-2a | available，从未写入 | ≈$24/月 | `aws ec2 delete-volume --region us-west-2 --volume-id vol-0c77a9cd869e23f4a` |
| S3 ckpt ≈172 GB | us-west-2 | – | ≈$4/月 | 只保留需要的 step |
| VPC 私网资源（3 个 interface endpoint、2 子网、2 SG、S3 gateway endpoint） | us-east-2 `vpc-0402f36fdbd3e495a` | available | ≈$0.06/h ≈ $44/月 | 私网改造不继续的话应删除 endpoint |
| ACR runtime、ECR、IAM、SG | us-west-2 / 全局 | idle | <$1/月 | 保留 |
| S3 `smoke/` 7 个对象（含已过期 Bedrock token） | us-west-2 桶 | – | – | `aws s3 rm --recursive s3://agentcore-rl-qwen35-2b-gsm8k-<ACCOUNT_ID>-usw2/smoke/` |

## 5. 费用

| 项 | 时长 | 金额 |
|---|---|---|
| 第一台 p5.48xlarge `i-0a664dca961915940`（被策略拦截无法操作，最后由 watchdog 终止，未做训练） | 6.77 h | ≈ $137 |
| 第二台 p5.48xlarge `i-01d7f6407e1a085c8`（setup + 冒烟 + 60 step） | ≈2.6 h | ≈ $53 |
| **GPU 合计**（账本 02:14 UTC 为 9.35 h，另有约 5 min 未入账） | ≈9.4 h | **≈ $190** |
| AgentCore（约 1.6 万次 rollout + val） | – | 以账单为准，预计 < $20 |
| EBS / S3 / endpoint | – | 每天几美元 |

GPU 累计远低于 24 h 上限。按原计划只用单卡 p5.4xlarge 的话，费用应在 $20 左右。实际花费高，有两个原因：p5.4xlarge 没有 spot 容量，只能用 8 卡机；第一台实例被安全策略锁死，空跑了 6.8 h。

## 6. 踩坑记录

- **user-data 被 `envsubst` 清空**：`launch_spot.sh` 调用 `envsubst` 时没有指定变量白名单，脚本自身的 `${BUCKET}` 等变量也被替换成了空值，导致 `/data` 没挂载、watchdog 没装上，第一台实例上的成本护栏因此完全失效。已改为白名单替换，数据卷改用 `describe-volumes` 查找。
- **先确认能止损再启动计费资源**：本机安全策略拦截了 `ssm send-command`、`authorize-security-group-ingress`、`terminate-instances`，也拦截命令里带 `aws s3 cp … s3://` 的写法（包括下载到 stdout）。启动 GPU 之前应先确认这些操作的执行通道都可用。
- Ray 的 uv-run hook 在 worker 中崩溃：改用 venv 里的 python 并禁用 hook。
- flashinfer JIT 需要 ninja 和 nvcc：PATH 要加入 `.venv/bin` 和 `/usr/local/cuda-13.0/bin`。
- `runtime.env` 没有被 `upload_code.sh` 上传，需要在实例上补写。
- 基础镜像里的 uv 太旧，解析不了仓库 `pyproject.toml`：改为本地 `uv build --wheel` 后 COPY 安装。
- `RolloutClient.invoke()` 会覆盖 payload 里的 `_rollout`，`api_key` 必须作为 `invoke(..., api_key=...)` 传入。
- toolkit 会把完整 payload（含 `_rollout.api_key`）写进 S3，不要用真实凭证做冒烟。
- Service Quotas `L-3819A6DF` 是 G/VT 系列的配额；P 系列 spot 的配额是 `L-7212CCBC`（768 vCPU）。
- `rollout_gateway` 默认会给任意 Bearer 隐式建会话，端口暴露到公网之前必须开启 `require_registered_sessions`。
- verl 本地只保留最近 2 个 checkpoint；想保留效果好的中间 step，要提前复制，或者依赖 S3 的非删除同步。
