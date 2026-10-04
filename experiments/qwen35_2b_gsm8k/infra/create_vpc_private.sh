#!/usr/bin/env bash
# Private network path ACR (VPC mode) -> trainer gateway, in the trainer's VPC.
# Idempotent: every resource is looked up by its Name tag before creation.
#
# Creates (trainer region, default VPC of the trainer):
#   - 2 private subnets (no IGW/NAT route) in ACR-supported AZ IDs; one in the
#     trainer's AZ so rollout traffic stays in-AZ
#   - a private route table associated with both subnets (local route only +
#     the S3 gateway endpoint prefix list)
#   - SG agentcore-rl-acr-sg     (attached to ACR runtime ENIs)
#   - SG agentcore-rl-vpce-sg    (attached to interface endpoints)
#   - interface endpoints ecr.api, ecr.dkr, logs (private DNS on)
#   - S3 gateway endpoint on the private route table, policy scoped to the ECR
#     layer bucket + the ACR result bucket in this region
#
# SG *rules* are NOT created here: authorize-security-group-* is blocked by the
# local agent security policy. They live in create_vpc_private_rules.sh, which
# the operator runs (or the agent runs once the policy allows it).
#
# Writes vpc.${TRAINER_REGION}.env.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../env.sh"
export AWS_REGION="$TRAINER_REGION" AWS_DEFAULT_REGION="$TRAINER_REGION"
source "${HERE}/sg.${TRAINER_REGION}.env"          # TRAINER_SG_ID, TRAINER_SG_VPC_ID

VPC_ID="$TRAINER_SG_VPC_ID"
REGION_SHORT=$(echo "$TRAINER_REGION" | sed -E 's/^us-east-/use/; s/^us-west-/usw/')
ACR_S3_BUCKET_PRIVATE="${ACR_S3_BUCKET_PRIVATE:-agentcore-rl-${EXP_TAG}-${AWS_ACCOUNT}-${REGION_SHORT}}"
# Two subnets: "<az-name>=<cidr>". Default: trainer AZ us-east-2c (use2-az3) + us-east-2a (use2-az1).
PRIVATE_SUBNETS="${PRIVATE_SUBNETS:-us-east-2c=172.31.48.0/24 us-east-2a=172.31.49.0/24}"
# ACR VPC-mode supported AZ IDs (docs: agentcore-vpc.html#supported-availability-zones)
SUPPORTED_AZ_IDS="${SUPPORTED_AZ_IDS:-use2-az1 use2-az2 use2-az3}"

tagspec() { echo "ResourceType=$1,Tags=[{Key=Project,Value=${EXP_TAG}},{Key=Name,Value=$2}]"; }
by_name() {  # by_name <describe-cmd> <query-root> <id-field> <name>
  aws ec2 "$1" --filters "Name=tag:Name,Values=$4" "Name=vpc-id,Values=${VPC_ID}" \
    --query "$2[0].$3" --output text 2>/dev/null | sed 's/^None$//'
}

echo "==> VPC ${VPC_ID} (${TRAINER_REGION})"
[ "$(aws ec2 describe-vpc-attribute --vpc-id "$VPC_ID" --attribute enableDnsHostnames \
  --query EnableDnsHostnames.Value --output text)" = "True" ] \
  || { echo "ERROR: VPC needs enableDnsHostnames for endpoint private DNS" >&2; exit 1; }

# --- route table (private) ---------------------------------------------------
RT_NAME="agentcore-rl-private-rt"
RT_ID=$(by_name describe-route-tables RouteTables RouteTableId "$RT_NAME")
if [ -z "$RT_ID" ]; then
  RT_ID=$(aws ec2 create-route-table --vpc-id "$VPC_ID" \
    --tag-specifications "$(tagspec route-table "$RT_NAME")" --query RouteTable.RouteTableId --output text)
  echo "==> Created route table ${RT_ID}"
else echo "==> Route table exists ${RT_ID}"; fi

# --- private subnets ----------------------------------------------------------
SUBNET_IDS=()
for pair in $PRIVATE_SUBNETS; do
  AZ="${pair%%=*}"; CIDR="${pair#*=}"
  AZ_ID=$(aws ec2 describe-availability-zones --zone-names "$AZ" --query 'AvailabilityZones[0].ZoneId' --output text)
  case " $SUPPORTED_AZ_IDS " in *" $AZ_ID "*) ;; *) echo "ERROR: ${AZ} (${AZ_ID}) not ACR-supported" >&2; exit 1;; esac
  NAME="agentcore-rl-private-${AZ}"
  SN=$(by_name describe-subnets Subnets SubnetId "$NAME")
  if [ -z "$SN" ]; then
    SN=$(aws ec2 create-subnet --vpc-id "$VPC_ID" --availability-zone "$AZ" --cidr-block "$CIDR" \
      --tag-specifications "$(tagspec subnet "$NAME")" --query Subnet.SubnetId --output text)
    echo "==> Created subnet ${SN} ${AZ}/${AZ_ID} ${CIDR}"
  else echo "==> Subnet exists ${SN} ${AZ}/${AZ_ID}"; fi
  # Never hand out public IPs; bind to the private route table (idempotent).
  aws ec2 modify-subnet-attribute --subnet-id "$SN" --no-map-public-ip-on-launch
  CUR_RT=$(aws ec2 describe-route-tables --filters "Name=association.subnet-id,Values=${SN}" \
    --query 'RouteTables[0].RouteTableId' --output text | sed 's/^None$//')
  if [ "$CUR_RT" != "$RT_ID" ]; then
    aws ec2 associate-route-table --route-table-id "$RT_ID" --subnet-id "$SN" >/dev/null
    echo "    associated ${SN} -> ${RT_ID}"
  fi
  SUBNET_IDS+=("$SN")
done

# Guard: the private route table must not reach the internet.
if aws ec2 describe-route-tables --route-table-ids "$RT_ID" \
     --query 'RouteTables[0].Routes[].[GatewayId,NatGatewayId]' --output text | grep -qE 'igw-|nat-'; then
  echo "ERROR: ${RT_ID} has an IGW/NAT route; refusing to treat it as private" >&2; exit 1
fi

# --- security groups (rules in create_vpc_private_rules.sh) --------------------
ensure_sg() {  # ensure_sg <name> <description>
  local id
  id=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$1" "Name=vpc-id,Values=${VPC_ID}" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null | sed 's/^None$//')
  if [ -z "$id" ]; then
    id=$(aws ec2 create-security-group --group-name "$1" --description "$2" --vpc-id "$VPC_ID" \
      --tag-specifications "$(tagspec security-group "$1")" --query GroupId --output text)
    echo "==> Created SG ${1} ${id}" >&2
  else echo "==> SG exists ${1} ${id}" >&2; fi
  echo "$id"
}
ACR_SG_ID=$(ensure_sg agentcore-rl-acr-sg "AgentCore RL ACR runtime ENIs (${EXP_TAG}); egress to trainer gateway + VPC endpoints")
VPCE_SG_ID=$(ensure_sg agentcore-rl-vpce-sg "AgentCore RL interface endpoints (${EXP_TAG}); 443 from ACR SG")

# --- interface endpoints ------------------------------------------------------
ensure_iface_ep() {  # ensure_iface_ep <service-suffix>
  local svc="com.amazonaws.${TRAINER_REGION}.$1" id
  id=$(aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=${VPC_ID}" "Name=service-name,Values=${svc}" \
    "Name=vpc-endpoint-state,Values=pending,available" --query 'VpcEndpoints[0].VpcEndpointId' --output text | sed 's/^None$//')
  if [ -z "$id" ]; then
    id=$(aws ec2 create-vpc-endpoint --vpc-id "$VPC_ID" --vpc-endpoint-type Interface --service-name "$svc" \
      --subnet-ids "${SUBNET_IDS[@]}" --security-group-ids "$VPCE_SG_ID" --private-dns-enabled \
      --tag-specifications "$(tagspec vpc-endpoint "agentcore-rl-vpce-$1")" \
      --query VpcEndpoint.VpcEndpointId --output text)
    echo "==> Created endpoint ${svc} ${id}" >&2
  else echo "==> Endpoint exists ${svc} ${id}" >&2; fi
  echo "$id"
}
VPCE_ECR_API=$(ensure_iface_ep ecr.api)
VPCE_ECR_DKR=$(ensure_iface_ep ecr.dkr)
VPCE_LOGS=$(ensure_iface_ep logs)

# --- S3 gateway endpoint (private route table only) ---------------------------
S3_POLICY=$(cat <<EOF
{"Statement":[
 {"Sid":"EcrLayers","Effect":"Allow","Principal":"*","Action":["s3:GetObject"],
  "Resource":["arn:aws:s3:::prod-${TRAINER_REGION}-starport-layer-bucket/*"]},
 {"Sid":"RolloutResults","Effect":"Allow","Principal":"*",
  "Action":["s3:GetObject","s3:PutObject","s3:ListBucket","s3:GetBucketLocation"],
  "Resource":["arn:aws:s3:::${ACR_S3_BUCKET_PRIVATE}","arn:aws:s3:::${ACR_S3_BUCKET_PRIVATE}/*"]}
]}
EOF
)
S3_SVC="com.amazonaws.${TRAINER_REGION}.s3"
VPCE_S3=$(aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=${VPC_ID}" "Name=service-name,Values=${S3_SVC}" \
  "Name=vpc-endpoint-type,Values=Gateway" "Name=tag:Name,Values=agentcore-rl-vpce-s3" \
  --query 'VpcEndpoints[0].VpcEndpointId' --output text | sed 's/^None$//')
if [ -z "$VPCE_S3" ]; then
  VPCE_S3=$(aws ec2 create-vpc-endpoint --vpc-id "$VPC_ID" --vpc-endpoint-type Gateway --service-name "$S3_SVC" \
    --route-table-ids "$RT_ID" --policy-document "$S3_POLICY" \
    --tag-specifications "$(tagspec vpc-endpoint agentcore-rl-vpce-s3)" \
    --query VpcEndpoint.VpcEndpointId --output text)
  echo "==> Created S3 gateway endpoint ${VPCE_S3}"
else
  aws ec2 modify-vpc-endpoint --vpc-endpoint-id "$VPCE_S3" --policy-document "$S3_POLICY" >/dev/null
  echo "==> S3 gateway endpoint exists ${VPCE_S3} (policy refreshed)"
fi

cat > "${HERE}/vpc.${TRAINER_REGION}.env" <<EOF
# Generated by create_vpc_private.sh -- $(date -u +%Y-%m-%dT%H:%M:%SZ)
export PRIVATE_VPC_ID=${VPC_ID}
export PRIVATE_RT_ID=${RT_ID}
export PRIVATE_SUBNET_IDS="${SUBNET_IDS[*]}"
export ACR_SG_ID=${ACR_SG_ID}
export VPCE_SG_ID=${VPCE_SG_ID}
export VPCE_ECR_API=${VPCE_ECR_API}
export VPCE_ECR_DKR=${VPCE_ECR_DKR}
export VPCE_LOGS=${VPCE_LOGS}
export VPCE_S3=${VPCE_S3}
export ACR_S3_BUCKET_PRIVATE=${ACR_S3_BUCKET_PRIVATE}
EOF
echo "==> Wrote vpc.${TRAINER_REGION}.env"
