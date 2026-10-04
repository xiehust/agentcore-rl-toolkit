#!/usr/bin/env bash
# Create the trainer EC2 instance role + instance profile (idempotent).
# Trust ec2.amazonaws.com; inline least-privilege policy; managed SSM core policy.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

ROLE="$TRAINER_ROLE_NAME"
PROFILE="$TRAINER_ROLE_NAME"   # instance profile shares the role name
POLICY_NAME=trainer-inline
RUNTIME_ARN="arn:aws:bedrock-agentcore:${AWS_REGION}:${AWS_ACCOUNT}:runtime/*"
BUCKET_ARN="arn:aws:s3:::${ACR_S3_BUCKET}"

TRUST=$(cat <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}
  ]
}
JSON
)

# Inline policy. bedrock-agentcore actions derived from
# src/agentcore_rl_toolkit/client.py self.agentcore_client.<method> calls:
#   invoke_agent_runtime -> InvokeAgentRuntime
#   stop_runtime_session -> StopRuntimeSession
# plus GetAgentRuntime (spec).
POLICY=$(cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ReadWrite",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["${BUCKET_ARN}"]
    },
    {
      "Sid": "S3Objects",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": ["${BUCKET_ARN}/*"]
    },
    {
      "Sid": "AgentCore",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:InvokeAgentRuntime",
        "bedrock-agentcore:StopRuntimeSession",
        "bedrock-agentcore:GetAgentRuntime"
      ],
      "Resource": ["${RUNTIME_ARN}"]
    },
    {
      "Sid": "Ec2Describe",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeVolumes",
        "ec2:DescribeTags",
        "ec2:DescribeSpotInstanceRequests"
      ],
      "Resource": ["*"]
    },
    {
      "Sid": "Ec2MutateTagged",
      "Effect": "Allow",
      "Action": [
        "ec2:TerminateInstances",
        "ec2:CreateTags",
        "ec2:AttachVolume"
      ],
      "Resource": ["*"],
      "Condition": {
        "StringEquals": {"aws:ResourceTag/Project": "${EXP_TAG}"}
      }
    }
  ]
}
JSON
)

echo "==> Ensuring role ${ROLE}"
if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  echo "    role exists; updating trust policy"
  aws iam update-assume-role-policy --role-name "$ROLE" --policy-document "$TRUST"
else
  aws iam create-role --role-name "$ROLE" \
    --assume-role-policy-document "$TRUST" \
    --tags "Key=Project,Value=${EXP_TAG}" \
    --description "AgentCore RL trainer EC2 instance role (${EXP_TAG})"
fi

echo "==> Putting inline policy ${POLICY_NAME}"
aws iam put-role-policy --role-name "$ROLE" \
  --policy-name "$POLICY_NAME" --policy-document "$POLICY"

echo "==> Attaching AmazonSSMManagedInstanceCore"
aws iam attach-role-policy --role-name "$ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

echo "==> Ensuring instance profile ${PROFILE}"
if aws iam get-instance-profile --instance-profile-name "$PROFILE" >/dev/null 2>&1; then
  echo "    instance profile exists"
else
  aws iam create-instance-profile --instance-profile-name "$PROFILE" \
    --tags "Key=Project,Value=${EXP_TAG}"
fi

# Add role to instance profile if not already a member (idempotent).
if aws iam get-instance-profile --instance-profile-name "$PROFILE" \
     --query "InstanceProfile.Roles[?RoleName=='${ROLE}'] | length(@)" --output text | grep -q '^0$'; then
  echo "==> Adding role to instance profile"
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE" --role-name "$ROLE"
else
  echo "    role already in instance profile"
fi

ROLE_ARN=$(aws iam get-role --role-name "$ROLE" --query 'Role.Arn' --output text)
PROFILE_ARN=$(aws iam get-instance-profile --instance-profile-name "$PROFILE" \
  --query 'InstanceProfile.Arn' --output text)
echo "==> Done"
echo "    ROLE_ARN=${ROLE_ARN}"
echo "    INSTANCE_PROFILE_ARN=${PROFILE_ARN}"
