# 成本模型与护栏（us-west-2，2026-09-23 报价）

价格来源：`research/spot_gpu_survey.md`（spot 价为调研当刻近 6h 观测值，会浮动）与 AWS 公开价目。

## 1. GPU 计算（主成本）

| 项 | 单价 | 说明 |
|---|---|---|
| p5.4xlarge spot（1×H100 80GB, 16 vCPU, 256 GB RAM, 3.8 TB NVMe） | **≈ $2.63 / h**（us-west-2，4 AZ 相近） | on-demand $6.88/h，spot 折扣 ~62% |
| 备选 us-east-2b | ≈ $2.52 / h | 仅省 $0.1/h，但跨区访问 ACR/S3 增加延迟与流量费，**不采用** |
| p5.48xlarge / p5en.48xlarge spot | 被 64 vCPU 配额阻断 | 不在预算内 |

**GPU 硬上限：累计 24 实例小时 → 24 × $2.63 ≈ $63**（spot 价上浮 30% 也 < $85）。

### 24h 预算分配（计划值 / 上限）

| 阶段 | 计划 | 上限 | 内容 |
|---|---|---|---|
| Stage 5 环境准备 | 1.0 h | 2 h | 驱动检查、`uv sync --extra verl`（CUDA13 wheel 约 10 GB）、下载 2B 模型、GSM8K 预处理、vLLM 单测 |
| Stage 6 冒烟 | 0.5 h | 2 h | 2 step 小 batch，排错缓冲 |
| Stage 7 正式训练 | 4–5 h | 12 h | 60 step × ~4 min + 6 次 val；spot 回收重建各次 +0.5 h |
| Stage 8 评估 | 0.5 h | 1 h | base/trained 各 200 题（base 基线在 val_before_train 已获得，可省） |
| 预留 | — | 7 h | 意外重装、二次冒烟 |
| **合计** | **≈ 6–7 h ≈ $18** | **24 h ≈ $63** | |

## 2. 存储

| 项 | 用量 | 单价 | 估算 |
|---|---|---|---|
| gp3 EBS 数据卷 | 300 GB × ~5 天 | $0.08/GB·月 | ≈ $4 |
| gp3 root 卷 | 100 GB，随实例存活 | $0.08/GB·月 | < $0.5 |
| EBS 快照（跨 AZ 迁移时才做） | ~60 GB 增量 | $0.05/GB·月 | ≈ $0.5/次 |
| S3 | ckpt 2.27B×(bf16+fp32 optim) ≈ 30 GB/ckpt × 保留最近 3 个 ≈ 90 GB；rollout 结果 JSON 可忽略 | $0.023/GB·月 | ≈ $2/月；PUT/HEAD 请求 ~30 万次 ≈ $1.5 |
| ECR | 镜像 ~500 MB | $0.10/GB·月 | < $0.1 |

> 训练结束保留 S3 最终 ckpt（仅 actor bf16 权重约 4.6 GB）+ 删数据卷，长期成本 < $0.2/月。

## 3. AgentCore Runtime

计费为按会话 vCPU-秒 + GB-秒（仅活跃时计，微 VM idle 不计费 CPU）。

| 项 | 估算 |
|---|---|
| 会话数 | 训练 60 step × 256 + val 7 次 × 200 + 冒烟 ≈ **17 000 会话** |
| 单会话 | ~20–40 s 活跃（多数时间等 LLM），1 vCPU / 2 GB 级别 |
| 费用 | 参考当前 MTD $37 对应的 demo 用量，估算 **$10–20**；上限按 $30 计 |

## 4. 网络

- ACR → EC2 公网 IP 的入向流量免费；EC2 → ACR 的响应出向按互联网出流量 $0.09/GB。每次 chat completion 响应 <5 KB，17 000 会话 × ~5 轮 ≈ 0.5 GB → **< $0.1**。
- 公网 IPv4 地址 $0.005/h × 24 h ≈ $0.1。
- 模型/wheel 下载为入向流量免费。

## 5. 总预算

| 类别 | 计划 | 上限 |
|---|---|---|
| GPU spot | $18 | $63（24 h 硬上限）|
| AgentCore | $15 | $30 |
| EBS + S3 + ECR | $8 | $12 |
| 网络 | <$1 | $1 |
| **合计** | **≈ $42** | **≈ $106** |

## 6. 告警与自动停止阈值

| 触发 | 动作 |
|---|---|
| GPU 累计 ≥ 12 h（Stage 7 上限） | watchdog 写 `ledger/ALERT_12H`，控制台/对话告警，人工决定是否继续 |
| GPU 累计 ≥ 20 h | watchdog 写 `ledger/ALERT_20H`，训练脚本收到 SIGTERM 保存 ckpt 后退出，实例保留 1 h 供收尾 |
| GPU 累计 ≥ 24 h | watchdog 强制 `terminate-instances` 自身（先 `s3 sync` ckpt） |
| spot 2 min 中断通知 | 立即 `s3 sync` ckpt + 账本 |
| 检测到第 2 台同 tag 实例 | 启动脚本拒绝；watchdog 告警 |
| AgentCore MTD 费用 > $30（手工 `ce get-cost-and-usage` 检查，每阶段一次） | 降低 `rollout.n` 或提前停 |

实现载体：`infra/watchdog.sh`（实例内，systemd timer 5 min）+ `infra/gpu_hours.sh`（本机查账）。账本格式：

```json
{"cumulative_hours": 6.42, "sessions": [{"instance_id": "i-...", "az": "us-west-2a", "start": "...", "end": "...", "hours": 3.1}], "updated_at": "..."}
```

## 7. 省钱要点

1. **不把 GPU 实例当开发机**：所有脚本/配置在本机写好、`bash -n` 校验，再上机执行；环境安装期间就是在烧钱，所以 wheel/模型缓存全部落在持久卷，重建实例零重装。
2. **先冒烟再正式跑**：2 step 冒烟捕获链路问题，避免正式配置下反复重试。
3. **小 val 集**：200 题而非 1319 题。
4. **训练一结束就 terminate**，评估用已保存的结果（`val_before_train` 拿 base，最后一次 test 拿 trained），不额外起 vLLM 服务评估；仅当需要 `evaluate.py` 独立评估时才多花 ~0.5 h。
5. 每阶段结束更新 `REPORT.md` 的 GPU 小时与费用表。
