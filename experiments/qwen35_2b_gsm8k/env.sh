# Shared settings for the qwen35_2b_gsm8k experiment. Source from any script:
#   source "$(dirname "$0")/../env.sh"
# Everything lives in one region; all names carry the experiment tag so cleanup is a
# tag query away.

export AWS_REGION=us-west-2
export AWS_DEFAULT_REGION=$AWS_REGION
export AWS_ACCOUNT=${AWS_ACCOUNT:-$(aws sts get-caller-identity --query Account --output text)}

export EXP_TAG=qwen35-2b-gsm8k            # Project tag value on every resource
export EXP_ID=qwen35-2b-gsm8k             # RolloutClient exp_id -> S3 prefix

# --- Agent side (Stage 3) ---
export ECR_REPO_NAME=agentcore-rl-math-agent
export IMAGE_TAG=${IMAGE_TAG:-qwen35}
export ECR_IMAGE_URI=${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}
export ACR_S3_BUCKET=agentcore-rl-${EXP_TAG}-${AWS_ACCOUNT}-usw2
export ACR_ROLE_NAME=AgentCoreRL-MathAgent-RuntimeRole
export ACR_ROLE_ARN=arn:aws:iam::${AWS_ACCOUNT}:role/${ACR_ROLE_NAME}
export ACR_RUNTIME_NAME=qwen35_2b_gsm8k_math_agent
# Filled in by agent/create_runtime.sh (written to agent/runtime.env)
_RUNTIME_ENV="$(dirname "${BASH_SOURCE[0]}")/agent/runtime.env"
[ -f "$_RUNTIME_ENV" ] && source "$_RUNTIME_ENV"

# --- Trainer side (Stage 4/5) ---
export GATEWAY_PORT=18765
export TRAINER_ROLE_NAME=AgentCoreRL-Trainer-InstanceRole
export TRAINER_SG_NAME=agentcore-rl-trainer-sg
export TRAINER_INSTANCE_TYPE=${TRAINER_INSTANCE_TYPE:-p5.4xlarge}
# Region for the GPU trainer EC2 only (ACR/S3/ECR stay in AWS_REGION). Override when
# us-west-2 has no p5.4xlarge spot capacity: TRAINER_REGION=us-east-2 ./launch_spot.sh
_TGT="$(dirname "${BASH_SOURCE[0]}")/infra/target.env"; [ -f "$_TGT" ] && source "$_TGT"   # persisted trainer target (region/type)
export TRAINER_REGION=${TRAINER_REGION:-us-west-2}
export TRAINER_AMI_NAME='Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*'
# Pinned id for us-west-2; other regions are resolved by name in launch_spot.sh
if [ "$TRAINER_REGION" = us-west-2 ]; then export TRAINER_AMI=${TRAINER_AMI:-ami-07d69ce07bfe5628f}; fi
export DATA_VOLUME_GB=300
export GPU_HOURS_HARD_LIMIT=24
export GPU_HOURS_SOFT_LIMIT=20
export GPU_HOURS_ALERT=12
export MODEL_ID=Qwen/Qwen3.5-2B

# --- 8-GPU fallback (p5.48xlarge / p5en.48xlarge spot, ~$21/h vs $2.6/h) ---
# Used when p5.4xlarge spot has no capacity anywhere. Hour caps are tightened so the
# dollar cap stays roughly where the 24h x $2.63 single-GPU plan put it (~$63-125).
case "$TRAINER_INSTANCE_TYPE" in
  p5.48xlarge|p5en.48xlarge|p5e.48xlarge)
    export GPU_HOURS_HARD_LIMIT=${GPU_HOURS_HARD_LIMIT_8GPU:-12}
    export GPU_HOURS_SOFT_LIMIT=${GPU_HOURS_SOFT_LIMIT_8GPU:-11}
    export GPU_HOURS_ALERT=${GPU_HOURS_ALERT_8GPU:-9}
    export N_GPUS=8 TP_SIZE=2 ;;
esac
