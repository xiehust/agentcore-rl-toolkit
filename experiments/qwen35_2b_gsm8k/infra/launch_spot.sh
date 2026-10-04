#!/usr/bin/env bash
# Launch exactly ONE p5.4xlarge one-time spot instance for the qwen35_2b_gsm8k
# experiment. Hard guardrail: max 1 tagged trainer at a time. Idempotent-safe:
# refuses to launch a second instance. Supports --dry-run (read-only) and --resume.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
# EC2 ops target the trainer region (may differ from the ACR/S3 region)
export AWS_REGION="$TRAINER_REGION" AWS_DEFAULT_REGION="$TRAINER_REGION"

HERE="$(cd "$(dirname "$0")" && pwd)"

DRY_RUN=0
RESUME=0
FORCE_AZ=""
KEY_PAIR="${TRAINER_KEY_PAIR:-4344-us-west-2}"
AMI="${TRAINER_AMI:-}"
if [ -z "$AMI" ]; then
  AMI=$(aws ec2 describe-images --owners amazon \
    --filters "Name=name,Values=${TRAINER_AMI_NAME}" "Name=architecture,Values=x86_64" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
  { [ -n "$AMI" ] && [ "$AMI" != None ]; } || { echo "no DLAMI found in ${AWS_REGION}" >&2; exit 1; }
fi
# key pair only if it exists in this region (management is via SSM anyway)
if ! aws ec2 describe-key-pairs --key-names "$KEY_PAIR" >/dev/null 2>&1; then KEY_PAIR=""; fi
INSTANCE_TYPE="${TRAINER_INSTANCE_TYPE}"

usage() {
  cat <<EOF
Usage: $0 [--dry-run] [--resume] [--az us-west-2X]
  --dry-run   Read-only: resolve all parameters and print run-instances JSON; launch nothing.
  --resume    Require an existing data volume; error if none exists.
  --az AZ     Force a specific AZ (overrides price/volume-based selection).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --resume)  RESUME=1; shift ;;
    --az)      FORCE_AZ="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

log()  { echo "==> $*" >&2; }
warn() { echo "!!! $*" >&2; }

# ---------------------------------------------------------------------------
# 1. Hard guardrail: at most one tagged trainer / no open spot request.
# ---------------------------------------------------------------------------
log "Guardrail: checking for existing tagged instances / spot requests"
EXISTING=$(aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=${EXP_TAG}" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text || true)
if [ -n "${EXISTING:-}" ]; then
  warn "A tagged instance already exists (max 1 allowed):"
  echo "$EXISTING" >&2
  exit 2
fi

OPEN_SPOT=$(aws ec2 describe-spot-instance-requests \
  --filters "Name=tag:Project,Values=${EXP_TAG}" "Name=state,Values=open,active" \
  --query 'SpotInstanceRequests[].[SpotInstanceRequestId,State]' --output text || true)
if [ -n "${OPEN_SPOT:-}" ]; then
  warn "An open/active tagged spot request already exists:"
  echo "$OPEN_SPOT" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# 2. AZ selection.
# ---------------------------------------------------------------------------
# Find an existing available data volume (tags Project, Role=data).
VOL_JSON=$(aws ec2 describe-volumes \
  --filters "Name=tag:Project,Values=${EXP_TAG}" "Name=tag:Role,Values=data" \
            "Name=status,Values=available" \
  --query 'Volumes[0].[VolumeId,AvailabilityZone]' --output text || true)
DATA_VOLUME_ID=""
DATA_VOLUME_AZ=""
if [ -n "${VOL_JSON:-}" ] && [ "$VOL_JSON" != "None" ]; then
  DATA_VOLUME_ID=$(echo "$VOL_JSON" | awk '{print $1}')
  DATA_VOLUME_AZ=$(echo "$VOL_JSON" | awk '{print $2}')
  [ "$DATA_VOLUME_ID" = "None" ] && DATA_VOLUME_ID=""
fi

if [ "$RESUME" -eq 1 ] && [ -z "$DATA_VOLUME_ID" ]; then
  warn "--resume requires an existing data volume (tags Project=${EXP_TAG}, Role=data) but none is available."
  exit 3
fi

pick_cheapest_az() {
  # Latest spot price per AZ across us-west-2a..d; echo cheapest AZ.
  aws ec2 describe-spot-price-history \
    --instance-types "$INSTANCE_TYPE" \
    --product-descriptions "Linux/UNIX" \
    --start-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --query 'SpotPriceHistory[].[AvailabilityZone,SpotPrice]' --output text 2>/dev/null \
    | sort -k2 -g | awk 'NR==1{print $1}'
}

if [ -n "$FORCE_AZ" ]; then
  AZ="$FORCE_AZ"
  log "AZ forced to ${AZ}"
elif [ -n "$DATA_VOLUME_AZ" ]; then
  AZ="$DATA_VOLUME_AZ"
  log "Using existing data volume AZ ${AZ} (volume ${DATA_VOLUME_ID})"
else
  AZ="$(pick_cheapest_az)"
  [ -z "${AZ:-}" ] && AZ="${AWS_REGION}a"
  log "No data volume; cheapest spot AZ selected: ${AZ}"
fi

# ---------------------------------------------------------------------------
# 3. Data volume handling.
# ---------------------------------------------------------------------------
# If the data volume lives in a different AZ than the chosen (forced) AZ, we must
# snapshot -> create new volume in target AZ -> retag old volume Role=data-old.
NEED_CREATE_VOLUME=0
CROSS_AZ_MIGRATE=0
if [ -z "$DATA_VOLUME_ID" ]; then
  NEED_CREATE_VOLUME=1
elif [ -n "$FORCE_AZ" ] && [ "$DATA_VOLUME_AZ" != "$AZ" ]; then
  CROSS_AZ_MIGRATE=1
fi

create_data_volume() {
  local az="$1"
  log "Creating gp3 data volume ${DATA_VOLUME_GB}GB in ${az} (3000 IOPS / 250 MBps)"
  aws ec2 create-volume \
    --availability-zone "$az" \
    --size "$DATA_VOLUME_GB" \
    --volume-type gp3 --iops 3000 --throughput 250 \
    --tag-specifications "ResourceType=volume,Tags=[{Key=Project,Value=${EXP_TAG}},{Key=Role,Value=data},{Key=Name,Value=qwen35-2b-gsm8k-data}]" \
    --query 'VolumeId' --output text
}

migrate_data_volume() {
  # snapshot old -> new volume in $AZ -> retag old Role=data-old.
  local old_vol="$1" old_az="$2" new_az="$3"
  log "Cross-AZ migrate: snapshot ${old_vol} (${old_az}) -> new volume in ${new_az}"
  local snap
  snap=$(aws ec2 create-snapshot --volume-id "$old_vol" \
    --description "qwen35-2b-gsm8k data migrate ${old_az}->${new_az}" \
    --tag-specifications "ResourceType=snapshot,Tags=[{Key=Project,Value=${EXP_TAG}},{Key=Role,Value=data-migrate}]" \
    --query 'SnapshotId' --output text)
  log "Waiting for snapshot ${snap} to complete"
  aws ec2 wait snapshot-completed --snapshot-ids "$snap"
  local newvol
  newvol=$(aws ec2 create-volume --availability-zone "$new_az" \
    --snapshot-id "$snap" --volume-type gp3 --iops 3000 --throughput 250 \
    --tag-specifications "ResourceType=volume,Tags=[{Key=Project,Value=${EXP_TAG}},{Key=Role,Value=data},{Key=Name,Value=qwen35-2b-gsm8k-data}]" \
    --query 'VolumeId' --output text)
  log "Retagging old volume ${old_vol} Role=data-old (delete later manually)"
  aws ec2 create-tags --resources "$old_vol" --tags "Key=Role,Value=data-old"
  warn "Old volume ${old_vol} in ${old_az} kept as Role=data-old -- delete it manually once ${newvol} is verified."
  echo "$newvol"
}

# ---------------------------------------------------------------------------
# 4/6. Resolve run-instances parameters (needed for both dry-run and real run).
# ---------------------------------------------------------------------------
# Security group id from sg.${AWS_REGION}.env.
if [ -f "${HERE}/sg.${AWS_REGION}.env" ]; then
  # shellcheck disable=SC1091
  source "${HERE}/sg.${AWS_REGION}.env"
fi
if [ -z "${TRAINER_SG_ID:-}" ]; then
  warn "TRAINER_SG_ID not set (run create_sg.sh first to produce sg.${AWS_REGION}.env)."
  [ "$DRY_RUN" -eq 0 ] && exit 4
fi

# Default subnet in the chosen AZ.
SUBNET_ID=$(aws ec2 describe-subnets \
  --filters "Name=availability-zone,Values=${AZ}" "Name=default-for-az,Values=true" \
  --query 'Subnets[0].SubnetId' --output text 2>/dev/null || true)
[ "$SUBNET_ID" = "None" ] && SUBNET_ID=""
if [ -z "${SUBNET_ID:-}" ]; then
  warn "No default subnet found for ${AZ}."
  [ "$DRY_RUN" -eq 0 ] && exit 4
fi

# Render user-data from template.
USER_DATA_FILE="${HERE}/user_data.sh"
RENDERED_USER_DATA="${KIROCREW_SCRATCH:-/tmp}/user_data.rendered.sh"
if [ -f "$USER_DATA_FILE" ]; then
  ACR_S3_BUCKET="$ACR_S3_BUCKET" EXP_TAG="$EXP_TAG" GATEWAY_PORT="$GATEWAY_PORT" \
  GPU_HOURS_ALERT="$GPU_HOURS_ALERT" GPU_HOURS_SOFT_LIMIT="$GPU_HOURS_SOFT_LIMIT" \
  GPU_HOURS_HARD_LIMIT="$GPU_HOURS_HARD_LIMIT" AWS_REGION="$AWS_REGION" \
    envsubst '${ACR_S3_BUCKET} ${EXP_TAG} ${GATEWAY_PORT} ${GPU_HOURS_ALERT} ${GPU_HOURS_SOFT_LIMIT} ${GPU_HOURS_HARD_LIMIT} ${AWS_REGION}' < "$USER_DATA_FILE" > "$RENDERED_USER_DATA"
else
  warn "user_data.sh template missing at ${USER_DATA_FILE}"
  [ "$DRY_RUN" -eq 0 ] && exit 4
fi

# Build run-instances argument JSON pieces.
MARKET_OPTS='{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}'
BLOCK_DEV='[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]'
TAG_SPEC="ResourceType=instance,Tags=[{Key=Project,Value=${EXP_TAG}},{Key=Name,Value=qwen35-2b-gsm8k-trainer},{Key=Role,Value=trainer}]"
METADATA_OPTS='HttpTokens=required,HttpEndpoint=enabled,InstanceMetadataTags=enabled'

print_run_plan() {
  cat <<EOF
--------------------------------------------------------------------------
Resolved launch parameters:
  Region              : ${AWS_REGION}
  AZ                  : ${AZ}
  Subnet              : ${SUBNET_ID:-<unresolved>}
  Instance type       : ${INSTANCE_TYPE}
  AMI                 : ${AMI}
  Key pair            : ${KEY_PAIR}
  Security group      : ${TRAINER_SG_ID:-<unresolved>}
  Instance profile    : ${TRAINER_ROLE_NAME}
  Data volume id      : ${DATA_VOLUME_ID:-<none, will create ${DATA_VOLUME_GB}GB>}
  Need create volume  : ${NEED_CREATE_VOLUME}
  Cross-AZ migrate    : ${CROSS_AZ_MIGRATE}
  Market options      : ${MARKET_OPTS}
  Block device (root) : ${BLOCK_DEV}
  Metadata options    : ${METADATA_OPTS} (IMDSv2 required + tags)
  User-data (rendered): ${RENDERED_USER_DATA}
--------------------------------------------------------------------------
EOF
}

if [ "$DRY_RUN" -eq 1 ]; then
  log "DRY-RUN: read-only queries only; no volume creation, no run-instances."
  print_run_plan
  DVID="${DATA_VOLUME_ID:-DATA_VOLUME_ID_PLACEHOLDER}"
  cat <<EOF
run-instances JSON (preview):
{
  "ImageId": "${AMI}",
  "InstanceType": "${INSTANCE_TYPE}",
  "KeyName": "${KEY_PAIR}",
  "MaxCount": 1, "MinCount": 1,
  "InstanceMarketOptions": ${MARKET_OPTS},
  "BlockDeviceMappings": ${BLOCK_DEV},
  "IamInstanceProfile": {"Name": "${TRAINER_ROLE_NAME}"},
  "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": "${SUBNET_ID:-<subnet>}", "Groups": ["${TRAINER_SG_ID:-<sg>}"], "AssociatePublicIpAddress": true}],
  "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled", "InstanceMetadataTags": "enabled"},
  "TagSpecifications": [{"${TAG_SPEC}"}],
  "Placement": {"AvailabilityZone": "${AZ}"},
  "UserData": "<base64 of ${RENDERED_USER_DATA}>",
  "Tags(add DataVolumeId post-attach)": "${DVID}"
}
EOF
  log "DRY-RUN complete. No resources created."
  exit 0
fi

# ---------------------------------------------------------------------------
# REAL LAUNCH (only reached when NOT --dry-run).
# ---------------------------------------------------------------------------
log "Uploading code before launch"
"${HERE}/upload_code.sh"

# Create / migrate the data volume as needed.
if [ "$CROSS_AZ_MIGRATE" -eq 1 ]; then
  DATA_VOLUME_ID="$(migrate_data_volume "$DATA_VOLUME_ID" "$DATA_VOLUME_AZ" "$AZ")"
fi
# When creating a fresh volume, defer creation until an AZ with spot capacity is
# found (the instance launch decides the AZ); the DataVolumeId tag is added after.

launch_in_az() {
  local az="$1" subnet="$2" itype="$3"
  aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type "$itype" \
    ${KEY_PAIR:+--key-name "$KEY_PAIR"} \
    --count 1 \
    --instance-market-options "$MARKET_OPTS" \
    --block-device-mappings "$BLOCK_DEV" \
    --iam-instance-profile "Name=${TRAINER_ROLE_NAME}" \
    --network-interfaces "DeviceIndex=0,SubnetId=${subnet},Groups=${TRAINER_SG_ID},AssociatePublicIpAddress=true" \
    --metadata-options "$METADATA_OPTS" \
    --tag-specifications "${TAG_SPEC%]}${DATA_VOLUME_ID:+,{Key=DataVolumeId,Value=${DATA_VOLUME_ID}\}}]" \
    --user-data "fileb://${RENDERED_USER_DATA}" \
    --query 'Instances[0].InstanceId' --output text
}

# Try chosen AZ; on capacity/price errors, try other us-west-2 AZs; then fall
# back to bigger boxes with a LOUD warning.
INSTANCE_ID=""
CANDIDATE_AZS=("$AZ")
for z in $(aws ec2 describe-availability-zones --query "AvailabilityZones[?State=='available'].ZoneName" --output text | tr "\t" "\n" | sed "s/^${AWS_REGION}//"); do
  [ "${AWS_REGION}${z}" != "$AZ" ] && CANDIDATE_AZS+=("${AWS_REGION}${z}")
done

for cand in "${CANDIDATE_AZS[@]}"; do
  # Only try AZs compatible with the data volume (same AZ) unless creating fresh.
  if [ "$cand" != "$AZ" ] && [ -n "$DATA_VOLUME_ID" ] && [ "$NEED_CREATE_VOLUME" -eq 0 ]; then
    continue
  fi
  csub=$(aws ec2 describe-subnets \
    --filters "Name=availability-zone,Values=${cand}" "Name=default-for-az,Values=true" \
    --query 'Subnets[0].SubnetId' --output text 2>/dev/null || true)
  [ "$csub" = "None" ] && continue
  log "Attempting run-instances in ${cand} (${INSTANCE_TYPE})"
  if INSTANCE_ID=$(launch_in_az "$cand" "$csub" "$INSTANCE_TYPE" 2>"${KIROCREW_SCRATCH:-/tmp}/ri.err"); then
    AZ="$cand"; SUBNET_ID="$csub"
    break
  fi
  if grep -qE 'InsufficientInstanceCapacity|SpotMaxPriceTooLow' "${KIROCREW_SCRATCH:-/tmp}/ri.err"; then
    warn "Capacity/price issue in ${cand}; trying next AZ."
    INSTANCE_ID=""
    continue
  fi
  cat "${KIROCREW_SCRATCH:-/tmp}/ri.err" >&2
  exit 5
done

if [ -z "$INSTANCE_ID" ]; then
  warn "############################################################"
  warn "# NO p5.4xlarge spot capacity in any us-west-2 AZ.         #"
  warn "# Falling back to 8-GPU boxes requires a QUOTA INCREASE    #"
  warn "# (192 vCPU > 64 vCPU spot quota) AND costs ~20x more.     #"
  warn "# NOT launching automatically. Request quota, then rerun   #"
  warn "# with TRAINER_INSTANCE_TYPE=p5.48xlarge (or p5en.48xlarge)#"
  warn "############################################################"
  exit 6
fi

log "Instance ${INSTANCE_ID} requested in ${AZ}; waiting for running"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

if [ "$NEED_CREATE_VOLUME" -eq 1 ]; then
  DATA_VOLUME_ID="$(create_data_volume "$AZ")"
  log "Created data volume ${DATA_VOLUME_ID} in ${AZ}"
  aws ec2 wait volume-available --volume-ids "$DATA_VOLUME_ID"
  aws ec2 create-tags --resources "$INSTANCE_ID" --tags "Key=DataVolumeId,Value=${DATA_VOLUME_ID}"
fi

log "Attaching data volume ${DATA_VOLUME_ID} as /dev/sdf"
aws ec2 attach-volume --volume-id "$DATA_VOLUME_ID" --instance-id "$INSTANCE_ID" --device /dev/sdf

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

log "Writing instance.env"
cat > "${HERE}/instance.env" <<EOF
# Generated by launch_spot.sh -- $(date -u +%Y-%m-%dT%H:%M:%SZ)
export TRAINER_INSTANCE_ID=${INSTANCE_ID}
export TRAINER_PUBLIC_IP=${PUBLIC_IP}
export TRAINER_AZ=${AZ}
export DATA_VOLUME_ID=${DATA_VOLUME_ID}
EOF

echo "==> Launched"
echo "    INSTANCE_ID = ${INSTANCE_ID}"
echo "    AZ          = ${AZ}"
echo "    PUBLIC_IP   = ${PUBLIC_IP}"
echo "    DATA_VOLUME = ${DATA_VOLUME_ID}"
echo "    Connect: aws ssm start-session --target ${INSTANCE_ID}"
