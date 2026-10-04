#!/usr/bin/env bash
# Create (or update the image of) the AgentCore Runtime for the math agent.
# Writes agent/runtime.env with AGENT_RUNTIME_ARN / AGENT_RUNTIME_ID for other scripts.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"

existing=$(aws bedrock-agentcore-control list-agent-runtimes \
  --query "agentRuntimes[?agentRuntimeName=='$ACR_RUNTIME_NAME'] | [0]" --output json)

if [[ "$existing" == "null" || -z "$existing" ]]; then
  echo "Creating runtime $ACR_RUNTIME_NAME -> $ECR_IMAGE_URI"
  out=$(aws bedrock-agentcore-control create-agent-runtime \
    --agent-runtime-name "$ACR_RUNTIME_NAME" \
    --description "GSM8K Strands math agent for RL rollouts (agentcore-rl-toolkit, $EXP_TAG)" \
    --agent-runtime-artifact "containerConfiguration={containerUri=$ECR_IMAGE_URI}" \
    --role-arn "$ACR_ROLE_ARN" \
    --network-configuration networkMode=PUBLIC \
    --protocol-configuration serverProtocol=HTTP \
    --tags Project="$EXP_TAG" \
    --output json)
  RUNTIME_ARN=$(echo "$out" | python3 -c 'import json,sys; print(json.load(sys.stdin)["agentRuntimeArn"])')
  RUNTIME_ID=$(echo "$out" | python3 -c 'import json,sys; print(json.load(sys.stdin)["agentRuntimeId"])')
else
  RUNTIME_ARN=$(echo "$existing" | python3 -c 'import json,sys; print(json.load(sys.stdin)["agentRuntimeArn"])')
  RUNTIME_ID=$(echo "$existing" | python3 -c 'import json,sys; print(json.load(sys.stdin)["agentRuntimeId"])')
  echo "Runtime exists ($RUNTIME_ID); updating image -> $ECR_IMAGE_URI"
  aws bedrock-agentcore-control update-agent-runtime \
    --agent-runtime-id "$RUNTIME_ID" \
    --agent-runtime-artifact "containerConfiguration={containerUri=$ECR_IMAGE_URI}" \
    --role-arn "$ACR_ROLE_ARN" \
    --network-configuration networkMode=PUBLIC \
    --protocol-configuration serverProtocol=HTTP >/dev/null
fi

cat > "$HERE/runtime.env" <<EOF
export AGENT_RUNTIME_ARN=$RUNTIME_ARN
export AGENT_RUNTIME_ID=$RUNTIME_ID
EOF
echo "AGENT_RUNTIME_ARN=$RUNTIME_ARN"

# wait for READY
for i in $(seq 1 60); do
  status=$(aws bedrock-agentcore-control get-agent-runtime \
    --agent-runtime-id "$RUNTIME_ID" --query status --output text)
  echo "  status=$status"
  case "$status" in
    READY) exit 0 ;;
    *FAILED*) echo "runtime failed"; exit 1 ;;
  esac
  sleep 10
done
echo "timed out waiting for READY"; exit 1
