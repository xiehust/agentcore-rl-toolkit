# infra/ — Stage 4: IAM、安全组、Spot 启动/续训/销毁、账本

单区域 us-west-2，账号 <ACCOUNT_ID>。所有资源打 `Project=qwen35-2b-gsm8k` 标签，
清理时一条 tag 查询即可定位。所有脚本 `set -euo pipefail`、幂等、`source ../env.sh`。

## 脚本一览

| 脚本 | 位置 | 说明 |
|---|---|---|
| `create_trainer_iam.sh` | 本机 | 创建 trainer 角色 + 同名 instance profile（trust ec2），内联最小权限策略（S3 读写、bedrock-agentcore Invoke/Stop/GetRuntime、EC2 Describe/Terminate/CreateTags/AttachVolume 以 `Project` tag 收窄），附加 `AmazonSSMManagedInstanceCore`。 |
| `create_sg.sh` | 本机 | 默认 VPC 内 SG，仅入向 `tcp/18765` from `0.0.0.0/0`，**不开 22**（走 SSM）。写 `sg.env`。 |
| `create_vpc_private.sh` | 本机 | 私网路径（ACR VPC 模式）：训练机 VPC 内 2 个私有子网（无 IGW/NAT）+ 私有路由表、ACR/endpoint 两个 SG、ecr.api/ecr.dkr/logs interface endpoint、S3 gateway endpoint（策略限 ECR layer 桶 + 本区结果桶）。幂等，写 `vpc.<region>.env`。 |
| `create_vpc_private_rules.sh` | 本机（操作员） | 上述 SG 的最小权限规则；`--lock-trainer` 额外撤销训练机 18765 的 `0.0.0.0/0`（VPC runtime 上线后再用）。因本地策略拦截 `authorize-security-group-*`，需人工执行。 |
| `launch_spot.sh` | 本机 | 启动**恰好一台** `p5.4xlarge` one-time spot。硬护栏、AZ 选择、数据卷创建/迁移、`--dry-run`、`--resume`。写 `instance.env`。 |
| `resume.sh` | 本机 | `exec launch_spot.sh --resume "$@"`；要求数据卷已存在。 |
| `terminate.sh` | 本机 | 终止 tag 实例、取消 open spot 请求；默认保留数据卷；`--snapshot`、`--delete-volume`、`--yes`。 |
| `user_data.sh` | 模板→实例 | cloud-init（Ubuntu DLAMI）：装 awscli/jq、写 `/etc/agentcore-rl.env`、等挂数据卷、拉 watchdog/spot 脚本、装 systemd timer/service。 |
| `watchdog.sh` | 实例内 | systemd timer 每 5 min：更新 `ledger/gpu_hours.json`，12h 告警 / 20h 软限（SIGTERM+STOP flag）/ 24h 硬限（同步 ckpt 后自毁），多实例告警。 |
| `spot_interrupt_watch.sh` | 实例内 | 每 5s 轮询 IMDSv2 `spot/instance-action`；200 → 同步 ckpt + 跑 watchdog 后退出。 |
| `gpu_hours.sh` | 本机 | 只读查账：累计/分会话/剩余额度/费用估算/当前实例。 |
| `upload_code.sh` | 本机 | 打包仓库（排除 `.venv`、`agent/dist`）→ S3；上传 watchdog/spot 脚本到 `code/infra/`。 |

> `trainer/` 目录由另一位同事并行编写，本目录脚本仅**引用** `trainer/sync_ckpt.sh`，不创建它。

## 执行顺序

```bash
cd experiments/qwen35_2b_gsm8k/infra
./create_trainer_iam.sh          # IAM 角色 + instance profile
./create_sg.sh                   # 安全组，写 sg.env
./launch_spot.sh --dry-run       # 校验参数，不启动任何实例
./launch_spot.sh                 # 首次启动（建数据卷）——会烧 GPU 钱！
#   ...训练... spot 被回收后：
./resume.sh                      # 复用数据卷续训
./gpu_hours.sh                   # 随时查账
./terminate.sh --snapshot        # 收尾：快照后终止，保留数据卷
```

## SSM 连接（不暴露 SSH）

```bash
aws ssm start-session --target <instance-id>
```

实例角色附了 `AmazonSSMManagedInstanceCore`，SG 未开 22。

## 安全护栏

- **最多 1 台实例**：`launch_spot.sh` 启动前 `describe-instances` / `describe-spot-instance-requests`，存在同 tag 实例或 open/active spot 请求即 `exit 2`。
- **GPU 24h 硬上限**：watchdog 累计到 `GPU_HOURS_HARD_LIMIT=24` 自动同步 ckpt 后 `terminate-instances` 自身；20h SIGTERM 收尾；12h 告警。账本在 `s3://<bucket>/ledger/gpu_hours.json`。
- **one-time spot**：不用 persistent，避免 AWS 自动重拉绕过账本。
- **IMDSv2 required** + 实例元数据 tags（watchdog 从 tag 读 `DataVolumeId`）。
- **网关 18765 全开的缓解**：见 `../PLAN.md` §2.1（gateway `require_registered_sessions`，sid=uuid4 一次性令牌；端口仅训练期监听；实例 24h 内销毁；暴露面仅 2B 采样接口，无凭证无数据）。
- IAM EC2 写操作以 `aws:ResourceTag/Project=qwen35-2b-gsm8k` 条件收窄。
