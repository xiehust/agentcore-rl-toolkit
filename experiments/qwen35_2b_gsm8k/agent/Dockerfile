# ACR rollout agent for the qwen35_2b_gsm8k experiment.
#
# The toolkit is installed from a wheel built from the LOCAL checkout (build_and_push.sh
# runs `uv build --wheel -o dist/` first) so the agent's @rollout_entrypoint / S3 result
# contract stays in lockstep with the trainer-side RolloutClient + gateway in this repo
# instead of whatever PyPI has. (Installing the source tree inside the image fails: the
# base image's uv is older than the repo's pyproject `override-dependencies` syntax.)
#
#   docker buildx build --platform linux/arm64 \
#     --build-context example=<repo root>/examples/strands_math_agent \
#     -f Dockerfile -t <uri> --load .          # see build_and_push.sh
#
# Differences from examples/strands_math_agent/.bedrock_agentcore/*/Dockerfile:
#   - toolkit from local wheel instead of PyPI
#   - no opentelemetry-instrument wrapper: shorter cold start, fewer deps; ACR
#     observability is not needed for RL rollouts
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 \
    AWS_REGION=us-west-2 \
    AWS_DEFAULT_REGION=us-west-2

# 1. toolkit wheel from the local checkout + the agent's deps
COPY dist/agentcore_rl_toolkit-*.whl /tmp/
RUN uv pip install /tmp/agentcore_rl_toolkit-*.whl \
      "strands-agents[openai]>=1.18.0" "strands-agents-tools>=0.2.16" \
    && rm -f /tmp/*.whl

# 2. the example agent (unmodified)
COPY --from=example rl_app.py reward.py models.py /app/

RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 8080
CMD ["python", "-m", "rl_app"]
