#!/usr/bin/env bash
# Idempotently create the agent-side AWS resources: S3 result bucket + ACR execution role.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"

# ---------- S3 result bucket ----------
if aws s3api head-bucket --bucket "$ACR_S3_BUCKET" 2>/dev/null; then
  echo "S3 bucket $ACR_S3_BUCKET exists"
else
  echo "Creating S3 bucket $ACR_S3_BUCKET"
  aws s3api create-bucket --bucket "$ACR_S3_BUCKET" \
    --create-bucket-configuration LocationConstraint="$AWS_REGION" >/dev/null
  aws s3api put-public-access-block --bucket "$ACR_S3_BUCKET" \
    --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-tagging --bucket "$ACR_S3_BUCKET" --tagging "TagSet=[{Key=Project,Value=$EXP_TAG}]"
  # rollout result JSONs are throwaway: expire after 14 days; ckpt/ and ledger/ are kept.
  # smoke/ results may embed a short-lived Bedrock bearer token (the toolkit persists the
  # full payload incl. _rollout.api_key) -> expire after 1 day.
  aws s3api put-bucket-lifecycle-configuration --bucket "$ACR_S3_BUCKET" --lifecycle-configuration '{
    "Rules": [{"ID": "expire-rollout-results", "Status": "Enabled",
               "Filter": {"Prefix": "'"$EXP_ID"'/"}, "Expiration": {"Days": 14}},
              {"ID": "expire-smoke", "Status": "Enabled",
               "Filter": {"Prefix": "smoke/"}, "Expiration": {"Days": 1}}]}'
fi

# ---------- ACR execution role ----------
TRUST=$(cat <<EOF
{"Version": "2012-10-17", "Statement": [{
  "Effect": "Allow",
  "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
  "Action": "sts:AssumeRole",
  "Condition": {
    "StringEquals": {"aws:SourceAccount": "$AWS_ACCOUNT"},
    "ArnLike": {"aws:SourceArn": "arn:aws:bedrock-agentcore:$AWS_REGION:$AWS_ACCOUNT:*"}
  }}]}
EOF
)
POLICY=$(cat <<EOF
{"Version": "2012-10-17", "Statement": [
  {"Sid": "ECRImageAccess", "Effect": "Allow",
   "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
   "Resource": "arn:aws:ecr:$AWS_REGION:$AWS_ACCOUNT:repository/$ECR_REPO_NAME"},
  {"Sid": "ECRToken", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
  {"Sid": "Logs", "Effect": "Allow",
   "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams", "logs:DescribeLogGroups"],
   "Resource": ["arn:aws:logs:$AWS_REGION:$AWS_ACCOUNT:log-group:/aws/bedrock-agentcore/runtimes/*",
                "arn:aws:logs:$AWS_REGION:$AWS_ACCOUNT:log-group:*"]},
  {"Sid": "Metrics", "Effect": "Allow", "Action": "cloudwatch:PutMetricData", "Resource": "*",
   "Condition": {"StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}}},
  {"Sid": "Xray", "Effect": "Allow",
   "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords", "xray:GetSamplingRules", "xray:GetSamplingTargets"],
   "Resource": "*"},
  {"Sid": "WorkloadIdentity", "Effect": "Allow",
   "Action": ["bedrock-agentcore:GetWorkloadAccessToken", "bedrock-agentcore:GetWorkloadAccessTokenForJWT", "bedrock-agentcore:GetWorkloadAccessTokenForUserId"],
   "Resource": ["arn:aws:bedrock-agentcore:$AWS_REGION:$AWS_ACCOUNT:workload-identity-directory/default",
                "arn:aws:bedrock-agentcore:$AWS_REGION:$AWS_ACCOUNT:workload-identity-directory/default/workload-identity/*"]},
  {"Sid": "RolloutResults", "Effect": "Allow",
   "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
   "Resource": ["arn:aws:s3:::$ACR_S3_BUCKET", "arn:aws:s3:::$ACR_S3_BUCKET/*"]}
]}
EOF
)
if aws iam get-role --role-name "$ACR_ROLE_NAME" >/dev/null 2>&1; then
  echo "IAM role $ACR_ROLE_NAME exists; refreshing inline policy"
else
  echo "Creating IAM role $ACR_ROLE_NAME"
  aws iam create-role --role-name "$ACR_ROLE_NAME" --assume-role-policy-document "$TRUST" \
    --tags Key=Project,Value="$EXP_TAG" >/dev/null
fi
aws iam put-role-policy --role-name "$ACR_ROLE_NAME" --policy-name AgentCoreRuntimeAccess --policy-document "$POLICY"
echo "Role: $ACR_ROLE_ARN"
echo "Bucket: s3://$ACR_S3_BUCKET"
