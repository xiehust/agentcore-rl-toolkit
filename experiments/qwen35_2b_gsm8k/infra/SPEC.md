# infra/ 脚本规格（Stage 4）

所有脚本：bash，`set -euo pipefail`，幂等，`bash -n` 通过，chmod +x，`source "$(dirname "$0")/../env.sh"`。
账号 <ACCOUNT_ID>，us-west-2，默认凭证。所有资源打 `Project=$EXP_TAG` 标签。
依据：`../env.sh`、`../PLAN.md` §2/§5/§6、`../COST.md` §6、`../research/spot_gpu_survey.md`（AZ、AMI、key pair、默认 VPC `vpc-0edf3a4e323c23b22`）。

## 1. create_trainer_iam.sh
角色 + 同名 instance profile `$TRAINER_ROLE_NAME`，trust ec2.amazonaws.com。内联策略：
- S3 rw（ListBucket/Get/Put/DeleteObject）on `$ACR_S3_BUCKET` 与 `/*`（DeleteObject 供 `s3 sync --delete`）。
- bedrock-agentcore：grep `src/agentcore_rl_toolkit/client.py` 里所有 `self.agentcore_client.<method>` 对应的 IAM action（至少 InvokeAgentRuntime、StopRuntimeSession）+ GetAgentRuntime，资源 `arn:aws:bedrock-agentcore:us-west-2:<ACCOUNT_ID>:runtime/*`。
- ec2 Describe{Instances,Volumes,Tags,SpotInstanceRequests} on `*`；ec2:TerminateInstances / CreateTags / AttachVolume 以 `aws:ResourceTag/Project=$EXP_TAG` 条件限制。
- 附加托管策略 AmazonSSMManagedInstanceCore。

## 2. create_sg.sh
默认 VPC 中的 SG `$TRAINER_SG_NAME`：入向仅 `tcp/$GATEWAY_PORT` from `0.0.0.0/0`（ACR PUBLIC 出口 IP 不可枚举；gateway 自身拒绝未注册 sid）；**不开 22**（走 SSM）。按 group-name 幂等。写 `sg.env`（TRAINER_SG_ID）。

## 3. launch_spot.sh —— 恰好一台 p5.4xlarge one-time spot
1. 硬护栏：任何 `Project=$EXP_TAG` 的实例处于 pending/running/stopping/stopped，或存在 open/active 的同 tag spot 请求 → 打印并 `exit 2`。
2. AZ：若存在数据卷（tags Project、Role=data；state available）用其 AZ；否则按 `describe-spot-price-history` 最新 p5.4xlarge 价格在 us-west-2a–d 选最便宜；`--az` 可覆盖。
3. 数据卷：该 AZ 无则创建 gp3 `$DATA_VOLUME_GB` GB（3000 IOPS / 250 MBps，tags Project、Role=data、Name=qwen35-2b-gsm8k-data）。若卷只在其他 AZ 而 `--az` 强制：快照 → 目标 AZ 建新卷 → 旧卷改 tag Role=data-old 并提示日后删除。
4. `run-instances`：`--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}'`，AMI `$TRAINER_AMI`（env 可覆盖），类型 `$TRAINER_INSTANCE_TYPE`；遇 InsufficientInstanceCapacity/SpotMaxPriceTooLow 先换其它 AZ，再回退 p5.48xlarge / p5en.48xlarge 并**大声警告**（需提配额、约 20 倍成本）。root gp3 100 GB delete-on-termination；instance profile `$TRAINER_ROLE_NAME`；SG 来自 sg.env；该 AZ 默认子网；分配公网 IP；key pair `4344-us-west-2`（env 可覆盖）；IMDSv2 required + `InstanceMetadataTags=enabled`；tags Project、Name=qwen35-2b-gsm8k-trainer、Role=trainer、DataVolumeId。user-data = 渲染后的 user_data.sh（envsubst/sed）。
5. 等待 running → `attach-volume` `/dev/sdf` → 打印 id/AZ/公网 IP → 写 `instance.env`（TRAINER_INSTANCE_ID、TRAINER_PUBLIC_IP、TRAINER_AZ、DATA_VOLUME_ID）。
6. `--dry-run`：只做只读查询，打印所有解析出的参数与 run-instances JSON。启动前调用 `upload_code.sh`。

## 4. resume.sh
薄封装：`exec launch_spot.sh --resume "$@"`；`--resume` 要求数据卷已存在，否则报错。

## 5. terminate.sh
终止所有 tag Project=$EXP_TAG、Role=trainer 的实例（无 `--yes` 则确认），取消 open 的同 tag spot 请求，默认保留数据卷；`--snapshot`（先快照）、`--delete-volume`；等待终止完成。

## 6. user_data.sh（cloud-init 模板，Ubuntu DLAMI）
- 缺则装 awscli v2、jq；写 `/etc/agentcore-rl.env`（bucket、exp tag、gateway port、GPU_HOURS_* 阈值）。
- 最多等 10 min 数据卷出现：`/dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_vol<去掉横线的卷 id>`，卷 id 从 IMDSv2 实例 tag `DataVolumeId` 读；`blkid` 为空则 `mkfs.ext4 -L data`；挂 `/data`（fstab by LABEL），chown ubuntu，`mkdir /data/{hf,repo,gsm8k,ckpts,logs,ledger}`。
- 从 `s3://$ACR_S3_BUCKET/code/infra/` 下载 watchdog.sh、spot_interrupt_watch.sh；安装 systemd timer `agentcore-rl-watchdog`（每 5 min，开机 1 min 后首跑）与 `spot_interrupt_watch.service`。
- 日志 `/var/log/agentcore-rl-userdata.log`。

## 7. watchdog.sh（实例内，IMDSv2）
账本 `s3://$ACR_S3_BUCKET/ledger/gpu_hours.json`：`{cumulative_hours, sessions:[{instance_id,az,start,end,hours}], updated_at}`。下载/初始化 → upsert 本实例会话（start=launch time）→ end=now → cumulative=sum → 上传（python3/jq）。阈值来自 `/etc/agentcore-rl.env`：
- ≥ GPU_HOURS_ALERT(12)：写 `ledger/ALERT_12H`（一次）。
- ≥ GPU_HOURS_SOFT_LIMIT(20)：写 `ALERT_20H`，`touch /data/STOP_TRAINING`，对 `/data/logs/train.pid` 的进程组发 SIGTERM（一次）。
- ≥ GPU_HOURS_HARD_LIMIT(24)：存在则执行 `/data/repo/experiments/qwen35_2b_gsm8k/trainer/sync_ckpt.sh --once`，否则 `aws s3 sync /data/ckpts s3://$ACR_S3_BUCKET/ckpt/`；上传账本；`terminate-instances` 自身。
- 发现另一台 running 的同 tag trainer → 写 `ledger/ALERT_MULTI_INSTANCE`。
- 绝不让 timer 崩溃；日志 `/var/log/agentcore-rl-watchdog.log`。

## 8. spot_interrupt_watch.sh
每 5 s 轮询 IMDSv2 `spot/instance-action`；200 → ckpt 同步 + 运行 watchdog.sh，退出。

## 9. gpu_hours.sh（本机，只读）
打印账本累计/分会话表、距 24 h 剩余、当前同 tag 实例（id、type、state、AZ、launch、公网 IP）、按 $2.63/h 估算费用。

## 10. upload_code.sh（本机）
在仓库根 `git ls-files -co --exclude-standard`，排除 `.venv` 与 `experiments/qwen35_2b_gsm8k/agent/dist`，打包到 `$KIROCREW_SCRATCH/repo.tar.gz` → `s3://$ACR_S3_BUCKET/code/repo.tar.gz`；同时上传 infra/watchdog.sh、spot_interrupt_watch.sh 到 `code/infra/`。（trainer/ 目录由另一位同事并行编写，勿创建。）

## 11. infra/README.md
脚本说明、顺序（create_trainer_iam → create_sg → launch_spot --dry-run → launch_spot → … → terminate）、SSM 连接（`aws ssm start-session --target <id>`）、安全护栏。

## 执行要求
`bash -n` 全部脚本；**真实运行** `create_trainer_iam.sh`、`create_sg.sh`（便宜、非 GPU）与 `launch_spot.sh --dry-run`。**不要**真实运行 launch_spot / 建卷 / 起实例。若 AWS 命令被 Kiro Crew 安全策略拦截，**不要绕过**（不改写、不包装），如实报告被拦截的命令。临时文件放 `$KIROCREW_SCRATCH`。
