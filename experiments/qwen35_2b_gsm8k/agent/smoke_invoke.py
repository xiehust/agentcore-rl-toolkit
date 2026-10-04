#!/usr/bin/env python3
"""End-to-end smoke test of the deployed ACR math agent via RolloutClient.

Points the agent at any OpenAI-compatible endpoint and checks the full path:
InvokeAgentRuntime -> agent runs (tool calls) -> {"rewards": ...} lands in S3 -> client
polls it back.

  # against the trainer's rollout gateway (Stage 6)
  python smoke_invoke.py --base-url http://<ec2-ip>:18765/v1 --model-id Qwen/Qwen3.5-2B --api-key <sid>

  # against Bedrock's OpenAI-compatible endpoint with a short-lived bearer token
  # (no gateway/GPU needed; validates ACR + S3 + reward path today)
  python smoke_invoke.py --bedrock --model-id openai.gpt-oss-20b-1:0 -n 3

Env: AGENT_RUNTIME_ARN, ACR_S3_BUCKET (source ../env.sh).
"""

import argparse
import json
import os
import sys
import time

from agentcore_rl_toolkit import RolloutClient

QUESTIONS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
     "How many clips did Natalia sell altogether in April and May?", "72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
     "How much did she earn?", "10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. "
     "Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. "
     "How much more money does Betty need to buy the wallet?", "5"),
    ("Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as "
     "many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should "
     "she read?", "42"),
]


def bedrock_token(region: str) -> str:
    try:
        from aws_bedrock_token_generator import provide_token
    except ImportError:
        sys.exit("pip install aws-bedrock-token-generator  (needed for --bedrock)")
    return provide_token(region=region)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--bedrock", action="store_true", help="use Bedrock OpenAI-compatible endpoint + short-lived token")
    ap.add_argument("-n", type=int, default=2, help="number of questions to send")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    region = os.environ.get("AWS_REGION", "us-west-2")
    arn = os.environ["AGENT_RUNTIME_ARN"]
    bucket = os.environ["ACR_S3_BUCKET"]

    if args.bedrock:
        args.base_url = f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1"
        args.api_key = bedrock_token(region)
    if not args.base_url:
        sys.exit("--base-url required unless --bedrock")

    client = RolloutClient(
        agent_runtime_arn=arn,
        s3_bucket=bucket,
        exp_id=os.environ.get("EXP_ID", "smoke"),
        base_url=args.base_url,
        model_id=args.model_id,
        sampling_params={"max_completion_tokens": args.max_tokens, "temperature": 0.6},
    )

    t0 = time.time()
    futures = []
    for i, (q, a) in enumerate(QUESTIONS[: args.n]):
        # api_key rides in _rollout via per-invocation override (a payload-level
        # "_rollout" key would be replaced by the client's own config).
        f = client.invoke(payload={"prompt": q, "answer": a}, input_id=f"smoke-{i}", api_key=args.api_key)
        futures.append((f, a))
        print(f"[{time.time()-t0:5.1f}s] submitted smoke-{i} session={f.session_id}")

    ok = 0
    for f, a in futures:
        try:
            r = f.result(timeout=args.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"[{time.time()-t0:5.1f}s] {f.input_id}: FAILED {type(e).__name__}: {e}")
            continue
        status = r.get("status_code")
        print(f"[{time.time()-t0:5.1f}s] {f.input_id}: status={status} rewards={r.get('rewards')} "
              f"(gt={a}) key={r.get('result_key')}")
        if status != 200:
            print("   error:", json.dumps({k: v for k, v in r.items() if k in ('error', 'error_type', 'message')})[:500])
        else:
            ok += 1
    print(f"done: {ok}/{len(futures)} successful rollouts in {time.time()-t0:.1f}s")
    sys.exit(0 if ok == len(futures) else 1)


if __name__ == "__main__":
    main()
