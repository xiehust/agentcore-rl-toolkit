# qwen35_2b_gsm8k — 最低成本 Agent RL on AgentCore 实验

用 `Qwen/Qwen3.5-2B` + verl GRPO + Bedrock AgentCore Runtime 上的 Strands 数学 Agent（GSM8K），
在 1 台 p5.4xlarge spot 上、GPU 累计 ≤ 24h 内跑通完整 RL 链路。

| 文档 | 内容 |
|---|---|
| [PLAN.md](PLAN.md) | 架构、网络方案、显存/超参、spot 容错、阶段门禁 |
| [COST.md](COST.md) | 成本模型、预算、告警与自动停止阈值 |
| [research/](research/) | Stage 1 调研：spot 价格与配额、模型兼容性、账号资源盘点 |
| agent/ | Stage 3：ACR agent 镜像构建、ECR 推送、runtime 创建、冒烟 |
| infra/ | Stage 4：IAM、安全组、spot 启动/续训/销毁、watchdog、GPU 小时账本 |
| trainer/ | Stage 4：训练机环境安装、训练/冒烟脚本、agent loop 配置 |
| eval/ | Stage 8：评估 |
| REPORT.md | Stage 8：结果、成本、踩坑 |

## 关键决策速览

- 区域 us-west-2；实例 p5.4xlarge spot（≈$2.63/h）；one-time spot + 独立 EBS 数据卷 + S3 ckpt 同步实现回收续训。
- ACR PUBLIC 网络模式，gateway 固定端口 18765，`gateway_public_host` = EC2 公网 IP。
- 全参 FSDP 微调，`max_model_len=4096`，`train_batch_size=32`，`n=8`，`lr=5e-6`，KL + token-IS 信任域，≤60 step。
- 复用 `examples/strands_math_agent/rl_app.py`（不改业务代码），仓库 gateway 已内置 Qwen3.5 模板 schema。
