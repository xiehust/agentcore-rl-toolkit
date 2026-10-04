#!/usr/bin/env bash
# Build the arm64 ACR agent image from the local checkout and push it to ECR.
#   ./build_and_push.sh            # build + push
#   ./build_and_push.sh --local    # build + load locally only (for local_test.sh)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../../.." && pwd)
source "$HERE/../env.sh"

MODE=push
[[ "${1:-}" == "--local" ]] && MODE=local

# toolkit wheel from the local checkout (see Dockerfile header for why a wheel)
rm -rf "$HERE/dist" && (cd "$REPO_ROOT" && uv build --wheel -o "$HERE/dist" >/dev/null)
ls "$HERE"/dist/agentcore_rl_toolkit-*.whl

if [[ $MODE == push ]]; then
  if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" >/dev/null 2>&1; then
    echo "Creating ECR repository $ECR_REPO_NAME"
    aws ecr create-repository --repository-name "$ECR_REPO_NAME" \
      --image-scanning-configuration scanOnPush=false \
      --tags Key=Project,Value="$EXP_TAG" >/dev/null
  fi
  aws ecr get-login-password | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
  OUT=(--push -t "$ECR_IMAGE_URI")
else
  OUT=(--load -t "agentcore-rl-math-agent:local")
fi

docker buildx build --platform linux/arm64 \
  --build-context example="$REPO_ROOT/examples/strands_math_agent" \
  -f "$HERE/Dockerfile" "${OUT[@]}" "$HERE"

if [[ $MODE == push ]]; then
  aws ecr describe-images --repository-name "$ECR_REPO_NAME" --image-ids imageTag="$IMAGE_TAG" \
    --query 'imageDetails[0].{tags:imageTags,pushed:imagePushedAt,MB:imageSizeInBytes}' --output table
  echo "Pushed $ECR_IMAGE_URI"
fi
