#!/usr/bin/env bash
# Terminate the tagged trainer instance(s), cancel open spot requests.
# By default the data volume is KEPT. --snapshot snapshots the data volume first;
# --delete-volume deletes it after (optionally after snapshot).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
# EC2 ops target the trainer region (may differ from the ACR/S3 region)
export AWS_REGION="$TRAINER_REGION" AWS_DEFAULT_REGION="$TRAINER_REGION"

ASSUME_YES=0
DO_SNAPSHOT=0
DELETE_VOLUME=0

usage() {
  cat <<EOF
Usage: $0 [--yes] [--snapshot] [--delete-volume]
  --yes            Skip the confirmation prompt.
  --snapshot       Snapshot the data volume before anything else.
  --delete-volume  Delete the data volume after terminating (implies confirm).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --yes)           ASSUME_YES=1; shift ;;
    --snapshot)      DO_SNAPSHOT=1; shift ;;
    --delete-volume) DELETE_VOLUME=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

log()  { echo "==> $*"; }
warn() { echo "!!! $*" >&2; }

INSTANCE_IDS=$(aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=${EXP_TAG}" "Name=tag:Role,Values=trainer" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text || true)

DATA_VOLUME_ID=$(aws ec2 describe-volumes \
  --filters "Name=tag:Project,Values=${EXP_TAG}" "Name=tag:Role,Values=data" \
  --query 'Volumes[0].VolumeId' --output text 2>/dev/null || true)
[ "$DATA_VOLUME_ID" = "None" ] && DATA_VOLUME_ID=""

if [ -z "${INSTANCE_IDS:-}" ]; then
  log "No tagged trainer instances found."
else
  log "Trainer instances to terminate: ${INSTANCE_IDS}"
fi
log "Data volume: ${DATA_VOLUME_ID:-<none>} (kept unless --delete-volume)"

if [ "$ASSUME_YES" -eq 0 ]; then
  read -r -p "Proceed with termination? [y/N] " ans
  case "$ans" in y|Y|yes|YES) ;; *) log "Aborted."; exit 0 ;; esac
fi

# Snapshot the data volume first if requested.
if [ "$DO_SNAPSHOT" -eq 1 ] && [ -n "$DATA_VOLUME_ID" ]; then
  log "Snapshotting data volume ${DATA_VOLUME_ID}"
  SNAP=$(aws ec2 create-snapshot --volume-id "$DATA_VOLUME_ID" \
    --description "qwen35-2b-gsm8k data snapshot before terminate" \
    --tag-specifications "ResourceType=snapshot,Tags=[{Key=Project,Value=${EXP_TAG}},{Key=Role,Value=data-backup}]" \
    --query 'SnapshotId' --output text)
  log "Waiting for snapshot ${SNAP}"
  aws ec2 wait snapshot-completed --snapshot-ids "$SNAP"
  log "Snapshot ${SNAP} complete"
fi

# Cancel open/active spot requests for this project.
OPEN_SPOT=$(aws ec2 describe-spot-instance-requests \
  --filters "Name=tag:Project,Values=${EXP_TAG}" "Name=state,Values=open,active" \
  --query 'SpotInstanceRequests[].SpotInstanceRequestId' --output text || true)
if [ -n "${OPEN_SPOT:-}" ]; then
  log "Cancelling spot requests: ${OPEN_SPOT}"
  aws ec2 cancel-spot-instance-requests --spot-instance-request-ids $OPEN_SPOT >/dev/null
fi

# Terminate instances.
if [ -n "${INSTANCE_IDS:-}" ]; then
  log "Terminating: ${INSTANCE_IDS}"
  aws ec2 terminate-instances --instance-ids $INSTANCE_IDS >/dev/null
  log "Waiting for termination"
  aws ec2 wait instance-terminated --instance-ids $INSTANCE_IDS
  log "Instances terminated."
fi

# Delete volume if requested.
if [ "$DELETE_VOLUME" -eq 1 ] && [ -n "$DATA_VOLUME_ID" ]; then
  log "Deleting data volume ${DATA_VOLUME_ID}"
  aws ec2 delete-volume --volume-id "$DATA_VOLUME_ID"
  log "Data volume deleted."
elif [ -n "$DATA_VOLUME_ID" ]; then
  log "Data volume ${DATA_VOLUME_ID} retained."
fi

log "Done."
