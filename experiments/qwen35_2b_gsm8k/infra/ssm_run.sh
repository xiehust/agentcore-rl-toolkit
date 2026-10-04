#!/usr/bin/env bash
# Run a shell command on the trainer instance via SSM RunShellScript and print its output.
#   ssm_run.sh <timeout-sec> <command...>
# Reads TRAINER_INSTANCE_ID / TRAINER_REGION from ../env.sh + instance.env.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
source "$HERE/instance.env"
REG=${TRAINER_AZ%?}   # region derived from the instance AZ
IID=$TRAINER_INSTANCE_ID
TO=$1; shift
CMD=$(printf '%s' "$*" | python3 -c 'import json,sys; print(json.dumps([sys.stdin.read()]))')
CID=$(aws ssm send-command --region "$REG" --instance-ids "$IID" --document-name AWS-RunShellScript \
  --timeout-seconds "$TO" --parameters "{\"commands\":$CMD,\"executionTimeout\":[\"$TO\"]}" \
  --query Command.CommandId --output text) || exit 9
for _ in $(seq 1 $((TO/5+3))); do
  sleep 5
  S=$(aws ssm get-command-invocation --region "$REG" --command-id "$CID" --instance-id "$IID" --query Status --output text 2>/dev/null)
  case $S in Success|Failed|TimedOut|Cancelled) break;; esac
done
aws ssm get-command-invocation --region "$REG" --command-id "$CID" --instance-id "$IID" \
  --query '[Status,StandardOutputContent,StandardErrorContent]' --output text
