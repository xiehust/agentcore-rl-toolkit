#!/usr/bin/env bash
# Thin wrapper: resume onto the existing data volume. --resume enforces that a
# data volume already exists (launch_spot.sh errors otherwise).
set -euo pipefail
exec "$(dirname "$0")/launch_spot.sh" --resume "$@"
