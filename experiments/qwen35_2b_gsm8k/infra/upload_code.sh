#!/usr/bin/env bash
# Local helper: package the repo (tracked + untracked, respecting .gitignore,
# excluding .venv and the agent dist dir) and upload to S3, plus the two
# in-instance infra scripts to code/infra/.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
SCRATCH="${KIROCREW_SCRATCH:-/tmp}"
TARBALL="${SCRATCH}/repo.tar.gz"

echo "==> Repo root: ${REPO_ROOT}"
cd "$REPO_ROOT"

# git ls-files -co --exclude-standard = tracked + untracked-not-ignored.
# Exclude .venv and experiments/qwen35_2b_gsm8k/agent/dist.
FILELIST="${SCRATCH}/repo.filelist"
git ls-files -co --exclude-standard \
  | grep -Ev '(^|/)\.venv(/|$)' \
  | grep -Ev '^experiments/qwen35_2b_gsm8k/agent/dist(/|$)' \
  > "$FILELIST"

echo "==> Packaging $(wc -l < "$FILELIST") files -> ${TARBALL}"
tar -czf "$TARBALL" -T "$FILELIST"

echo "==> Uploading repo tarball to s3://${ACR_S3_BUCKET}/code/repo.tar.gz"
aws s3 cp "$TARBALL" "s3://${ACR_S3_BUCKET}/code/repo.tar.gz" --region "$AWS_REGION"

echo "==> Uploading in-instance infra scripts to code/infra/"
aws s3 cp "${HERE}/watchdog.sh"             "s3://${ACR_S3_BUCKET}/code/infra/watchdog.sh"             --region "$AWS_REGION"
aws s3 cp "${HERE}/spot_interrupt_watch.sh" "s3://${ACR_S3_BUCKET}/code/infra/spot_interrupt_watch.sh" --region "$AWS_REGION"

echo "==> Done"
