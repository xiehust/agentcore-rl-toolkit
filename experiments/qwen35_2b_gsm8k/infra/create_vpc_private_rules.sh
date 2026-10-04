#!/usr/bin/env bash
# Least-privilege SG rules for the private ACR -> trainer path. Run AFTER
# create_vpc_private.sh. Idempotent (duplicate-rule errors are ignored).
#
#   ./create_vpc_private_rules.sh               # ACR SG egress + endpoint SG ingress + trainer 18765 from ACR SG
#   ./create_vpc_private_rules.sh --lock-trainer # additionally REVOKE trainer tcp/18765 from 0.0.0.0/0
#                                                # (only after the VPC-mode runtime is live -- the
#                                                #  PUBLIC-mode runtime stops working at that point)
#
# Kept separate because authorize-security-group-* is blocked by the local agent
# security policy; the operator may need to run this by hand.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../env.sh"
export AWS_REGION="$TRAINER_REGION" AWS_DEFAULT_REGION="$TRAINER_REGION"
source "${HERE}/sg.${TRAINER_REGION}.env"
source "${HERE}/vpc.${TRAINER_REGION}.env"
LOCK=0; [ "${1:-}" = "--lock-trainer" ] && LOCK=1

ok_dup() { grep -qE 'InvalidPermission.Duplicate|already exists' && return 0 || return 1; }
run() {  # run <aws args...>; tolerate duplicate-rule errors only
  local out
  if ! out=$(aws "$@" 2>&1 >/dev/null); then
    echo "$out" | ok_dup || { echo "$out" >&2; exit 1; }
  fi
}
S3_PL=$(aws ec2 describe-prefix-lists --filters "Name=prefix-list-name,Values=com.amazonaws.${TRAINER_REGION}.s3" \
  --query 'PrefixLists[0].PrefixListId' --output text)

echo "==> ACR SG ${ACR_SG_ID}: egress 18765->trainer SG, 443->endpoint SG, 443->S3 prefix list"
run ec2 authorize-security-group-egress --group-id "$ACR_SG_ID" --ip-permissions \
  "IpProtocol=tcp,FromPort=${GATEWAY_PORT},ToPort=${GATEWAY_PORT},UserIdGroupPairs=[{GroupId=${TRAINER_SG_ID},Description=trainer-gateway}]" \
  "IpProtocol=tcp,FromPort=443,ToPort=443,UserIdGroupPairs=[{GroupId=${VPCE_SG_ID},Description=vpc-endpoints}]" \
  "IpProtocol=tcp,FromPort=443,ToPort=443,PrefixListIds=[{PrefixListId=${S3_PL},Description=s3-gateway}]"
# Drop the default allow-all egress (no-op if already gone).
aws ec2 revoke-security-group-egress --group-id "$ACR_SG_ID" \
  --ip-permissions 'IpProtocol=-1,IpRanges=[{CidrIp=0.0.0.0/0}]' >/dev/null 2>&1 || true

echo "==> Endpoint SG ${VPCE_SG_ID}: ingress 443 from ACR SG + trainer SG"
# Private DNS on the interface endpoints applies VPC-wide, so the trainer's own
# ecr/logs calls also resolve to them -- allow the trainer too.
run ec2 authorize-security-group-ingress --group-id "$VPCE_SG_ID" --ip-permissions \
  "IpProtocol=tcp,FromPort=443,ToPort=443,UserIdGroupPairs=[{GroupId=${ACR_SG_ID},Description=acr-runtime},{GroupId=${TRAINER_SG_ID},Description=trainer}]"

echo "==> Trainer SG ${TRAINER_SG_ID}: ingress ${GATEWAY_PORT} from ACR SG"
run ec2 authorize-security-group-ingress --group-id "$TRAINER_SG_ID" --ip-permissions \
  "IpProtocol=tcp,FromPort=${GATEWAY_PORT},ToPort=${GATEWAY_PORT},UserIdGroupPairs=[{GroupId=${ACR_SG_ID},Description=acr-runtime-private}]"

if [ "$LOCK" = 1 ]; then
  echo "==> Trainer SG ${TRAINER_SG_ID}: REVOKE ${GATEWAY_PORT} from 0.0.0.0/0 (public path closed)"
  aws ec2 revoke-security-group-ingress --group-id "$TRAINER_SG_ID" --ip-permissions \
    "IpProtocol=tcp,FromPort=${GATEWAY_PORT},ToPort=${GATEWAY_PORT},IpRanges=[{CidrIp=0.0.0.0/0}]" >/dev/null 2>&1 || true
fi
echo "==> Done"
