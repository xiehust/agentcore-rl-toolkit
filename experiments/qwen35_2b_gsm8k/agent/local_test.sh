#!/usr/bin/env bash
# Contract check of the locally built image: /ping + fire-and-forget /invocations.
# No AWS credentials are given to the container, so the background S3 write is expected
# to fail (logged) — we only verify the HTTP contract and that the entrypoint dispatches.
set -euo pipefail
IMG=${IMG:-agentcore-rl-math-agent:local}
PORT=${PORT:-18080}
NAME=rl-agent-local-test

docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME -p 127.0.0.1:$PORT:8080 "$IMG" >/dev/null
trap 'docker logs $NAME 2>&1 | tail -40; docker rm -f $NAME >/dev/null' EXIT

for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/ping" >/dev/null 2>&1; then break; fi
  sleep 1
done
echo "--- /ping"; curl -sS "http://127.0.0.1:$PORT/ping"; echo

echo "--- /invocations (fire-and-forget)"
curl -sS -X POST "http://127.0.0.1:$PORT/invocations" \
  -H 'Content-Type: application/json' \
  -H "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: local-test-session-0000000000000000001" \
  -d '{"prompt": "What is 2+3?", "answer": "5",
       "_rollout": {"exp_id": "local", "input_id": "0", "s3_bucket": "no-such-bucket-local",
                    "base_url": "http://127.0.0.1:9/v1", "model_id": "dummy", "api_key": "sid-test"}}'
echo
sleep 3
echo "--- /ping while busy (expect HealthyBusy or Healthy after fast failure)"; curl -sS "http://127.0.0.1:$PORT/ping"; echo
echo "--- container logs (tail)"
