# agent/ — ACR 端部署（Stage 3）

| 文件 | 用途 |
|---|---|
| `Dockerfile` | arm64 rollout agent 镜像：toolkit 从本地 wheel 安装 + `examples/strands_math_agent/{rl_app,reward,models}.py` 原样复用；无 otel wrapper |
| `build_and_push.sh` | `uv build --wheel` → `docker buildx build --platform linux/arm64` → 推 ECR（`--local` 只本地加载） |
| `create_resources.sh` | 幂等创建 S3 结果桶（公有访问全封、生命周期过期）+ ACR 执行角色（ECR 拉取、日志、指标、workload identity、S3 读写） |
| `create_runtime.sh` | 创建/更新 ACR runtime（PUBLIC 网络、HTTP 协议），写 `runtime.env`，等待 READY |
| `local_test.sh` | 本地容器契约检查：`/ping`、`/invocations` fire-and-forget |
| `smoke_invoke.py` | `RolloutClient` 端到端冒烟：`--base-url/--api-key` 指向任意 OpenAI 兼容端点；`--bedrock` 用 Bedrock OpenAI 兼容端点 + 短期 bearer token |
| `runtime.env`（gitignored） | `AGENT_RUNTIME_ARN` / `AGENT_RUNTIME_ID`，被 `../env.sh` 自动 source |

## 已创建的资源（2026-09-23, us-west-2）

| 资源 | 值 |
|---|---|
| ECR | `<ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/agentcore-rl-math-agent:qwen35`（arm64，压缩 174 MB / 本地 534 MB） |
| S3 | `s3://agentcore-rl-qwen35-2b-gsm8k-<ACCOUNT_ID>-usw2` |
| IAM | `arn:aws:iam::<ACCOUNT_ID>:role/AgentCoreRL-MathAgent-RuntimeRole` |
| ACR runtime | `arn:aws:bedrock-agentcore:us-west-2:<ACCOUNT_ID>:runtime/qwen35_2b_gsm8k_math_agent-057YOI31DY`（READY, PUBLIC, HTTP） |
| 日志 | `/aws/bedrock-agentcore/runtimes/qwen35_2b_gsm8k_math_agent-057YOI31DY-DEFAULT` |

## 验证结果

- 本地容器：`/ping` → Healthy；`/invocations` 立即返回 `{"status":"processing","result_key":...}`，后台任务派发、模型调用重试、S3 写入路径均触达。
- ACR 端到端（Bedrock OpenAI 兼容端点作为临时推理后端）：
  - `qwen.qwen3-32b-v1:0` ×3：3/3 status 200，calculator 工具被调用，reward 0（该模型未按 `####` 格式收尾，属模型行为，非链路问题）。
  - `openai.gpt-oss-20b-1:0` ×4：4/4 status 200，reward 1.0 / 0 / 0 / 1.0 —— reward 路径确认可给出正分。
  - 单条 rollout 端到端 ~3–7 s（含 ACR 冷启动）。

## 注意事项

- toolkit 会把完整 payload（含 `_rollout.api_key`）持久化进 S3 结果 JSON。训练时 api_key = 会话 uuid（无害）；但 `--bedrock` 冒烟会把 12h 有效的 Bedrock bearer token 写进 `smoke/` 前缀。本地安全策略拦截了 `aws s3 rm` 与 `put-bucket-lifecycle-configuration`，**7 个 `smoke/` 对象需手动删除**（或由后续运行 `create_resources.sh` 时新增的 1 天过期规则清理）；token 本身 12h 后失效，桶为私有。
- `strands_tools.calculator` 已被上游标记 deprecated（v0.9.0 变 error log），当前版本仍可用；如后续构建失败可固定 `strands-agents-tools<0.9`。
