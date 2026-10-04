# Stage 1 — 可行性与资源盘点结论（2026-09-23）

详细报告：`spot_gpu_survey.md`、`qwen35_2b_model.md`、`aws_inventory.md`。

## 结论：方案可行，关键决策已定

| 项目 | 结论 |
|---|---|
| 训练机 | **p5.4xlarge spot（1× H100 80GB，16 vCPU）**。us-west-2 ≈ $2.63/h（4 个 AZ 均有）；us-east-2b ≈ $2.52/h 最便宜。24h 上限 ≈ **$60–65** |
| 8 卡机 | p5.48xlarge / p5en.48xlarge **被 spot 配额挡住**：All P Spot = 64 vCPU（三区一致），48xlarge 需 192 vCPU。不申请提额，单卡足够 2B |
| 区域 | **us-west-2**（ACR、ECR、S3 同区，避免跨区流量与延迟；spot 差价仅 $0.1/h） |
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) `ami-07d69ce07bfe5628f`（需在 Stage 5 核实驱动 ≥ 580.65.06，否则手动升级驱动） |
| 模型 | `Qwen/Qwen3.5-2B` 公开、无 gate，~2.27B 参数（bf16 ≈ 4.55 GB）。架构 `Qwen3_5ForConditionalGeneration`（VL 外壳 + 混合 Gated DeltaNet 线性注意力 3:1 + MTP 头），思考模式**默认关闭** |
| Gateway 兼容 | chat template sha256 = `273d8e0e…` **精确命中** `qwen3_5 (nothink)` schema，XML tool-call 可解析，**无需改 gateway** |
| 引擎兼容 | 仓库 verl extra 已固定 `flash-linear-attention>=0.5.2`，且仓库已用同一架构类（Qwen3.6-27B）跑通 Megatron+LoRA；vLLM 0.24 原生支持 GDN。单卡 FSDP 不用 CP，megatron-bridge 补丁不涉及 |
| ACR | 账号内已有大量 demo runtime，**无可复用的 RL rollout runtime** → 新建。要求 **linux/arm64** 镜像（本机 aarch64 原生构建）、镜像 < 2 GB、新 session 创建 25/s、请求超时 15 min |
| ECR / S3 / IAM | 新建专用 ECR repo、专用 S3 结果桶、带 S3 读写的 ACR 执行角色；EC2 侧新建 trainer 实例角色（S3 + ECR + `bedrock-agentcore:InvokeAgentRuntime`） |
| 现存 GPU 成本 | **无运行中的 GPU 实例**（g6e/g7e 各 1 台处于 stopped），无 spot 请求；账号无 AWS Budgets |
| 本机 | docker 25 (aarch64)、uv 0.12.6、aws-cli 2.33、HF token 已设置；python 3.9（训练环境在 EC2 上用 uv 建 3.11+） |

## 风险与对策

1. **ACR → EC2 gateway 回连**：ACR 默认 PUBLIC 网络模式，容器需能访问 EC2 公网 IP:gateway_port → 安全组开放该端口（来源无法精确限定为 ACR 出口，用高位随机端口 + 会话 key 兜底），`gateway_public_host` 设为公网 IP，`gateway_port` 固定。
2. **单卡显存**：vLLM（`gpu_memory_utilization≈0.4`）与 FSDP 训练同卡 colocate；2B 全参 + AdamW 约 2.27B×16B ≈ 36 GB 峭值以外还有激活，需 `use_dynamic_bsz` + 小 `max_model_len`（4096）+ offload 备选（LoRA 作为回退）。
3. **spot 回收**：模型/venv/数据放独立 EBS 数据卷，checkpoint 定期同步 S3，`resume.sh` 重建实例并挂卷续训。
4. **成本护栏**：实例内 watchdog 记录累计运行小时到 S3，达 22h 告警、24h 自动 terminate。
5. **admin 用户缺 `bedrock-agentcore:GetAgentRuntime`**：新建 runtime 后需核实 IAM 是否能 describe；若不能，用 list 输出 + 调用测试代替。
