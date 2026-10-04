# AWS Inventory — Bedrock AgentCore RL Training

**Account:** <ACCOUNT_ID> (`arn:aws:iam::<ACCOUNT_ID>:user/admin`)
**aws-cli:** 2.33.15, Python 3.9.25, Linux aarch64
**Region focus:** us-west-2 (primary), us-east-1 (glance)
**Generated:** 2026-09-23 (READ-ONLY inventory)

---

## 1. Bedrock AgentCore Runtime

`bedrock-agentcore-control list-agent-runtimes` works. **Note:** the `admin` user is **NOT authorized for `bedrock-agentcore:GetAgentRuntime`** — so per-runtime `containerUri / roleArn / networkConfiguration / protocol` **could not be retrieved** (AccessDeniedException). Only list metadata (name/ARN/version/status) is available.

### us-west-2 (all `READY`; partial list — API paginates)
| Name | Version | Last Updated |
|---|---|---|
| unified_obs_probe_0728_1049_42f500 | 2 | 2026-07-28 |
| studio_skill_e2e_dbe162 | 1 | 2026-07-11 |
| studio_canvas_e2e_e73406 | 4 | 2026-07-11 |
| strands_claude6 | 1 | 2025-07-17 |
| strands_claude2 | 1 | 2025-07-17 |
| srpool_runtime | 6 | 2026-09-08 |
| skillopt_exec_worker | 3 | 2026-08-17 |
| shopagent | 4 | 2026-08-25 |
| shared_runtime_multiuser_m7g | 1 | 2026-08-13 |
| shared_runtime_multiuser | 5 | 2026-08-13 |
| research_graph | 1 | 2025-10-08 |
| product_ingestion_agent | 18 | 2026-08-24 |
| produc_selection_agent_7711c3 | … | … |

### us-east-1 (all `READY`; partial)
`yilong_hl_xhs_duoyuan`, `yilong_hl_xhs_ces`, `yilong_hl_chatbot_v2`, `yilong_cs_chatbot_rt`, `yifei_lab_hr_assistant`, `qingy_v2_coldstart_3rounds`, `prod_claude_native_final`, `prod_byoc_zip_0920`, `prod_byoc_docker_0920`, `leo_test`, `launchpad_skill_lab_worker`, `lab_hr_assistant`, …

**Verdict:** All existing runtimes are chatbot/agent-demo/skill workers. **None** is an RL-training or rollout-worker runtime for a math agent. A new dedicated runtime should be created for the Qwen3.5-2B GSM8K experiment (the runtime is the *rollout/inference actor*, not the trainer).

---

## 2. ECR (us-west-2)

~40 repos; agentcore/RL/qwen/strands-relevant ones:

| Repo | Notes |
|---|---|
| `sagemaker/easyr1` | **RL training image** (EasyR1 / verl-family). Tags `latest`, `0.3.1`. **~14.6 GB** — GPU RL trainer. Most relevant existing RL asset. |
| `fsdp` | `pytorch2.2-cpu`, 3.7 GB — FSDP training base (CPU tag only). |
| `bedrock-agentcore-qwen_3_agentcore_stream3` | Qwen3 agent runtime image, `latest`, ~370 MB, pushed 2025-09-29. |
| `bedrock-agentcore-qwen_no_stream` | Qwen3 non-stream variant, `latest`, ~370 MB. |
| `strands-agent-qwen3-stream` | Qwen3 strands agent; **has native `arm64` tags** + x86_64. ~95 MB. |
| `bedrock-agentcore-*` (many) | Per-agent runtime images auto-created by the AgentCore SDK/CodeBuild. |

**Verdict:** `sagemaker/easyr1` is a reusable RL-trainer image if EasyR1/verl is the chosen framework. The `qwen_3_agentcore_stream` images are reusable *starting points* for the rollout-agent container (rebuild for arm64 + GSM8K tool logic). A **new** repo (e.g. `agentcore-gsm8k-rollout`) is cleaner for the experiment's agent image.

---

## 3. S3

Candidate reusable buckets (all confirmed **us-west-2**):

| Bucket | Purpose |
|---|---|
| `bedrock-agentcore-runtime-<ACCOUNT_ID>-us-west-2-kb0977l9tg` | AgentCore runtime artifacts |
| `bedrock-agentcore-code-<ACCOUNT_ID>-us-west-2` | AgentCore code deploy source |
| `bedrock-agentcore-codebuild-sources-<ACCOUNT_ID>-us-west-2` | CodeBuild sources |
| `agentcore-us-west-2-<ACCOUNT_ID>` | Generic agentcore bucket |
| `hyperpod-eks-bucket-<ACCOUNT_ID>-us-west-2` | HyperPod/EKS GPU training data |
| `skillopt-agentcore-<ACCOUNT_ID>-us-west-2` | SkillOpt agentcore workspace |
| `srpool-workspaces-<ACCOUNT_ID>-us-west-2` | Runtime-pool workspaces |
| `sagemaker-us-west-2-<ACCOUNT_ID>` | Default SageMaker bucket |

No bucket named for `rollout / verl / checkpoint / gsm8k`. **Verdict:** reuse `sagemaker-us-west-2-<ACCOUNT_ID>` or `hyperpod-eks-bucket-...` for datasets/checkpoints, or create a dedicated `rollout`/`checkpoint` bucket for the experiment. AgentCore code buckets are managed by the SDK — don't repurpose.

---

## 4. IAM roles & instance profiles

**Runtime execution roles** (`AmazonBedrockAgentCoreSDKRuntime-us-west-2-*`, ~15 of them, incl. `-prod`) are auto-minted per deploy. Inspected `AmazonBedrockAgentCoreSDKRuntime-us-west-2-prod` (inline `AgentCoreRuntimePolicy`):

- **ECR:** `BatchGetImage`, `GetDownloadUrlForLayer`, `GetAuthorizationToken` (pull image ✅)
- **Logs:** CreateLogGroup/Stream, PutLogEvents, Describe* ✅
- **X-Ray + CloudWatch:** PutTraceSegments, PutMetricData ✅
- **AgentCore:** GetWorkloadAccessToken*, GetResourceOauth2Token ✅
- **No S3 access** in this policy → a rollout runtime that reads GSM8K data / writes rollouts from S3 needs an **augmented role** (add `s3:GetObject`/`PutObject` on the data bucket).

Other notable roles: `AgentCoreColdstartRole`, `SkillOptAgentCoreExecRole`, `AmazonBedrockAgentCoreRuntimeDefaultServiceRole`, capacity-provider service roles, and 3 service-linked roles (`...RuntimeInstances`, `...RuntimeIdentity`, `...Network`).

**Instance profiles** (for EC2 trainers): `gpu-eks-cluster_*` (×3 — GPU EKS), `comfyUI-sd-dev-ComfyInstanceProfile`, `AmazonBedrockAgentCoreCapacityProviderDefaultInstanceRole_w2pfl`, `jfsbench-*`. There are GPU-EKS instance profiles usable for a GPU trainer node; **no** dedicated `verl/trainer` profile — create one if running the trainer on EC2/EKS.

**Verdict:** reuse the SDK-minted runtime role pattern for the rollout agent, but **create a role with S3 read/write** for training-data/rollout/checkpoint access. Trainer GPU compute can reuse a `gpu-eks-cluster_*` profile or a new dedicated one.

---

## 5. Service quotas — bedrock-agentcore (us-west-2)

| Quota | Value | Adjustable |
|---|---|---|
| Rate of new Runtime session creation | 25/s | yes |
| Rate of Runtime data plane APIs (incl. InvokeAgentRuntime) | 1000/s | yes |
| Rate of Runtime control plane mutation APIs | 50/s | no |
| Rate of Runtime control plane Get APIs | 150/s | no |
| Rate of Runtime control plane List APIs | 25/s | no |
| Max payload size | 100 MB | no |
| Max Docker image size in AgentCore Runtime | 2048 MB (2 GB) | **no** |
| Request timeout | 15 min | no |

**No explicit "concurrent runtime sessions per account" or "number of runtimes" quota is surfaced** by `list-service-quotas` for the runtime dimension (the visible concurrency caps are for browser/code-interpreter = 1000 each). **Critical constraint for RL:** the **2 GB image-size cap** — a rollout container packing a local model + heavy deps may not fit; keep the agent image thin and call the model via API, or host the policy model separately. The 15-min request timeout and 100 MB payload bound per-invocation rollout length.

---

## 6. Running EC2 / spot (GPU cost check)

No active/open **spot** requests in any region. **No running GPU instances anywhere** — good.

### ⚠️ us-west-2 GPU instances — BOTH **stopped** (no compute cost, EBS only):
- 🛑 `i-0412845665caca364` **g6e.2xlarge** "comfyui" — **stopped**
- 🛑 `i-03ac34e9183d5f202` **g7e.2xlarge** "g7e" — **stopped**

Running in us-west-2 (all non-GPU Graviton/x86): `agentcore_eva_simulation` (t4g.xlarge), `development_server` (r7g.2xlarge), `skillopt` (m7g.xlarge), 2× AnthropicProxy (t4g.large), `jfsbench-gateway` (m7g.large).
Running us-east-1: `demo_dev_server` (m6a.2xlarge), `agentops_launchpad` (m8g.large). us-east-2: 2× t3.micro.

**Verdict:** **No stray running GPU.** Two GPU boxes (g6e, g7e) exist but are stopped — confirm they stay stopped; they incur only EBS cost. No GPU currently billing.

---

## 7. Cost (MTD 2026-09-01 → 09-23) & Budgets

Top services by unblended cost:
| Service | USD |
|---|---|
| Amazon Bedrock Service | 4,098 |
| EC2 - Other | 968 |
| EC2 - Compute | 720 |
| Amazon Bedrock | 639 |
| RDS | 389 |
| S3 | 370 |
| VPC | 340 |
| ELB | 198 |
| ECS | 192 |
| ECR | 100 |
| CloudWatch | 66 |
| **Amazon Bedrock AgentCore** | **36.6** |
| SageMaker | 33 |

AgentCore spend is tiny (~$37 MTD). **Budgets:** `describe-budgets` returned **null → no AWS Budgets configured**. Consider creating one before GPU RL training ramps cost.

---

## 8. Container architecture requirement

**Confirmed: Bedrock AgentCore Runtime REQUIRES `linux/arm64` container images** (AWS docs: "Amazon Bedrock AgentCore requires ARM64 architecture for all deployed agents"; HTTP protocol contract: "Platform: ARM64 container — Required"). Agent must expose `POST /invocations` + `GET /ping`.

**Local machine is aarch64 (arm64)** → native `docker build` produces arm64 with no emulation/buildx needed. Do **not** target `linux/amd64` for the rollout runtime image. (The GPU **trainer** side is normal x86/arm GPU — arm64 requirement applies only to the AgentCore Runtime agent container.)

---

## Reuse vs Create — Verdict

**REUSE:**
- **RL trainer image:** `sagemaker/easyr1` ECR repo (EasyR1/verl, `0.3.1`) if EasyR1 is the framework.
- **Rollout-agent base:** `bedrock-agentcore-qwen_3_agentcore_stream3` / `strands-agent-qwen3-stream` (has arm64 tags) as a starting Dockerfile — rebuild arm64 with GSM8K tool logic.
- **S3:** `sagemaker-us-west-2-<ACCOUNT_ID>` or `hyperpod-eks-bucket-...-us-west-2` for datasets/checkpoints.
- **Runtime role pattern:** SDK-minted `AmazonBedrockAgentCoreSDKRuntime-us-west-2-*` (ECR+logs+identity already covered).
- **GPU compute:** stopped `g7e.2xlarge`/`g6e.2xlarge` or `gpu-eks-cluster_*` instance profile.

**CREATE:**
- A **dedicated AgentCore Runtime** for the GSM8K rollout agent (arm64 image, `/invocations`+`/ping`).
- A **new ECR repo** (e.g. `agentcore-gsm8k-rollout`) for the thin arm64 rollout image (stay < 2 GB image cap).
- An **augmented IAM role** adding `s3:GetObject/PutObject` on the data/checkpoint bucket (SDK runtime role has no S3).
- Optionally a dedicated **checkpoint/rollout S3 bucket** and an **AWS Budget** (none exist).

**Blockers/flags:** `admin` lacks `GetAgentRuntime` (can't inspect existing runtime network/protocol — request the permission if needed). 2 GB runtime image cap + 15-min request timeout constrain rollout design. No stray running GPU. No budget guardrail set.
