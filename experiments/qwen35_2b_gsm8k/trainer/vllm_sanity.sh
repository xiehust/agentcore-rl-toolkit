#!/usr/bin/env bash
# vllm_sanity.sh — stand up vLLM on the local model once, prove it serves a
# tool-calling chat completion, then tear it down. SPEC §2 / PLAN gate Stage 5.
# Nothing here touches ACR or the gateway; it only checks that the engine can
# load Qwen3.5-2B and generate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

DATA=/data
export HF_HOME="$DATA/hf"
export UV_CACHE_DIR="$DATA/uv-cache"
MODEL_DIR="$HF_HOME/Qwen3.5-2B"
PORT=8001
LOG="$DATA/logs/vllm_sanity.log"
mkdir -p "$DATA/logs"

[ -d "$MODEL_DIR" ] || { echo "model not found at $MODEL_DIR — run setup_trainer.sh first" >&2; exit 1; }

# Detect whether this vLLM build supports the VL text-only flag. Qwen3.5-2B is a
# Qwen3_5ForConditionalGeneration (carries a vision tower even for text); the
# research doc recommends --language-model-only to skip the vision encoder. Only
# pass it if `vllm serve --help` advertises it, so an older/newer build that
# renamed or dropped the flag doesn't fail to start.
EXTRA_ARGS=()
if ( cd "$DATA/repo" && uv run vllm serve --help 2>/dev/null ) | grep -q -- '--language-model-only'; then
  EXTRA_ARGS+=(--language-model-only)
  echo "vllm supports --language-model-only; enabling (text-only, skips vision tower)"
else
  echo "vllm build has no --language-model-only; serving without it"
fi

# Tool-call requests to a bare `vllm serve` are rejected (400) unless auto tool
# choice + a parser are enabled. Only relevant for this standalone sanity check:
# during training the rollout gateway does tool/reasoning parsing token-side and
# vLLM never sees the tools schema.
EXTRA_ARGS+=(--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3)

echo "starting vllm serve on :$PORT (log: $LOG)"
( cd "$DATA/repo" && uv run vllm serve "$MODEL_DIR" \
    --port "$PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.4 \
    "${EXTRA_ARGS[@]}" ) > "$LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "stopping vllm (pid $VLLM_PID)"
  kill "$VLLM_PID" 2>/dev/null || true
  # give it a moment, then hard-kill any survivor
  for _ in $(seq 1 20); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for /health ..."
ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vllm process died during startup; tail of $LOG:" >&2
    tail -n 40 "$LOG" >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
[ "$ready" -eq 1 ] || { echo "ERROR: vllm did not become healthy in time" >&2; tail -n 40 "$LOG" >&2; exit 1; }
echo "vllm healthy."

echo "sending tool-calling chat completion (calculator tool) ..."
RESP="$(curl -sS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$MODEL_DIR"'",
    "messages": [{"role": "user", "content": "What is 17*23? Use the tool."}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
          "type": "object",
          "properties": {"expression": {"type": "string", "description": "e.g. 17*23"}},
          "required": ["expression"]
        }
      }
    }],
    "max_tokens": 256,
    "temperature": 0.6
  }')" || { echo "ERROR: chat completion request failed" >&2; tail -n 40 "$LOG" >&2; exit 1; }

echo "=== vllm response ==="
echo "$RESP"
echo "====================="
echo "vllm_sanity OK"
