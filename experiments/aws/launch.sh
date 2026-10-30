#!/usr/bin/env bash
# Launch ONE GPU instance for plantpipeline. NOT idempotent; every run creates a billable instance.
# Usage:
#   ./launch.sh                 # interactive confirm; on-demand g6.xlarge, fallback g5.xlarge on no capacity
#   YES=1 ./launch.sh           # no prompt
# Env knobs (all optional):
#   INSTANCE_TYPE  g6.xlarge (default) | g5.xlarge | ...      FALLBACK_TYPE  g5.xlarge (default; "" = no fallback)
#   SPOT=1         one-time Spot request (needs L-3819A6DF quota > 0; currently 0 -> request first)
#   MAX_PRICE      spot max price USD/h
#   AMI            default: DLAMI PyTorch 2.12 Ubuntu 24.04 (us-east-1, 20261024)
#   KEY_NAME       key pair name (default icp-flow-mac); KEY_FILE local private key (default ~/.ssh/icp-flow-aws)
#                  (verified 2026-10-30: fingerprints match, SHA256:RecdPdyAo53Ds+ZP91AJVZ54+gTCW38lF79jpMyWca0)
#   ROOT_GB        gp3 root size (default 200);  SUBNET_ID  subnet/AZ (default: the us-east-1a default subnet, see below)
#   NAME           short instance name (or first arg): writes .instances/<NAME> and tags Name=plantpipeline-<NAME>.
#                  Without NAME: writes .current_instance (the MonoGrow box) and tags Name=plantpipeline-gpu.
# Record file line: "<id> <ip> <type> <launch_time_utc>". Waits for SSH.
set -euo pipefail

REGION=${REGION:-us-east-1}
INSTANCE_TYPE=${INSTANCE_TYPE:-g6.xlarge}
FALLBACK_TYPE=${FALLBACK_TYPE-g5.xlarge}
SPOT=${SPOT:-0}
MAX_PRICE=${MAX_PRICE:-}
AMI=${AMI:-ami-055bccd3bd826922c}
KEY_NAME=${KEY_NAME:-icp-flow-mac}
KEY_FILE=${KEY_FILE:-$HOME/.ssh/icp-flow-aws}
ROOT_GB=${ROOT_GB:-200}
SUBNET_ID=${SUBNET_ID:-subnet-0312921865d45e2e6}   # default subnet us-east-1a (all boxes in one AZ); SUBNET_ID="" = let AWS pick
SG_NAME=${SG_NAME:-plantpipeline-ssh}
NAME=${NAME:-${1:-}}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -n "$NAME" ]; then TAG_NAME="plantpipeline-$NAME"; REC="$HERE/.instances/$NAME"; mkdir -p "$HERE/.instances"
else TAG_NAME=plantpipeline-gpu; REC="$HERE/.current_instance"; fi
if [ -s "$REC" ]; then echo "ERROR: $REC exists ($(cat "$REC")); pick another NAME or remove it" >&2; exit 1; fi

# --- 0. quota sanity: vCPUs of running G/VT instances + this one must fit L-DB2E81BA ---
VCPU=$(aws ec2 describe-instance-types --region "$REGION" --instance-types "$INSTANCE_TYPE" \
        --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text)
if [ "$SPOT" = 1 ]; then QCODE=L-3819A6DF; else QCODE=L-DB2E81BA; fi
QUOTA=$(aws service-quotas get-service-quota --region "$REGION" --service-code ec2 --quota-code $QCODE \
        --query 'Quota.Value' --output text)
USED=$(aws ec2 describe-instances --region "$REGION" \
        --filters Name=instance-state-name,Values=pending,running "Name=instance-type,Values=g*,vt*" \
        --query 'Reservations[].Instances[].CpuOptions.[CoreCount,ThreadsPerCore]' --output text \
        | awk '{s+=$1*$2} END{print s+0}')
echo "quota $QCODE=${QUOTA%.*} vCPU, in use=$USED, requested=$VCPU"
if (( USED + VCPU > ${QUOTA%.*} )); then
  echo "ERROR: would exceed quota (AWS would refuse with VcpuLimitExceeded)." >&2
  echo "  Either stop one of YOUR OWN other G instances, or request an increase (free, often approved in minutes):" >&2
  echo "  aws service-quotas request-service-quota-increase --region $REGION --service-code ec2 --quota-code L-DB2E81BA --desired-value 32" >&2
  aws ec2 describe-instances --region "$REGION" --filters Name=instance-state-name,Values=pending,running "Name=instance-type,Values=g*,vt*" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,Tags[?Key==`Name`].Value|[0]]' --output text >&2
  exit 1
fi

# --- 1. security group: SSH from current public IP only ---
MYIP=$(curl -s --max-time 5 https://checkip.amazonaws.com)
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC_ID" \
         --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG_ID" ] || [ "$SG_ID" = None ]; then
  SG_ID=$(aws ec2 create-security-group --region "$REGION" --group-name "$SG_NAME" --vpc-id "$VPC_ID" \
           --description "plantpipeline SSH" --query GroupId --output text)
  aws ec2 create-tags --region "$REGION" --resources "$SG_ID" --tags Key=Project,Value=plantpipeline
fi
# add-only: the laptop is behind a carrier NAT that picks the source IP PER DESTINATION (37.203.155.169-174 seen), so a new
# box may be unreachable although others work; then add the next /32 of that pool (see INSTANCE.md). Never revoke the others.
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$MYIP/32" 2>/dev/null || true

# --- 2. market options ---
MARKET=(); SHUTDOWN=(--instance-initiated-shutdown-behavior stop)
if [ "$SPOT" = 1 ]; then
  OPTS='{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"'
  [ -n "$MAX_PRICE" ] && OPTS="$OPTS,\"MaxPrice\":\"$MAX_PRICE\""
  MARKET=(--instance-market-options "$OPTS}}"); SHUTDOWN=()   # one-time spot cannot use shutdown-behavior=stop
fi
SUBNET=(); [ -n "$SUBNET_ID" ] && SUBNET=(--subnet-id "$SUBNET_ID")

# --- 3. launch (retry once with FALLBACK_TYPE on InsufficientInstanceCapacity) ---
echo "Launching $INSTANCE_TYPE (spot=$SPOT) ami=$AMI key=$KEY_NAME root=${ROOT_GB}GB sg=$SG_ID ssh-from=$MYIP"
if [ "${YES:-0}" != 1 ]; then read -r -p "Proceed? This starts billing. [y/N] " ok; [ "$ok" = y ] || exit 1; fi
launch() {
  aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$1" --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" ${SUBNET[@]+"${SUBNET[@]}"} ${MARKET[@]+"${MARKET[@]}"} ${SHUTDOWN[@]+"${SHUTDOWN[@]}"} \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$ROOT_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --user-data "file://$HERE/bootstrap.sh" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=plantpipeline},{Key=Name,Value=$TAG_NAME}]" \
                         "ResourceType=volume,Tags=[{Key=Project,Value=plantpipeline}]" \
    --query 'Instances[0].InstanceId' --output text
}
TYPE=$INSTANCE_TYPE
if ! IID=$(launch "$TYPE" 2>"$HERE/.launch_err"); then
  if grep -q InsufficientInstanceCapacity "$HERE/.launch_err" && [ -n "$FALLBACK_TYPE" ]; then
    echo "no $TYPE capacity, retrying with $FALLBACK_TYPE"; TYPE=$FALLBACK_TYPE; IID=$(launch "$TYPE")
  else cat "$HERE/.launch_err" >&2; exit 1; fi
fi
T0=$(date -u +%FT%TZ)
echo "instance: $IID ($TYPE) launched $T0"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$IID $IP $TYPE $T0" > "$REC"
echo -n "waiting for ssh at $IP "
for i in $(seq 1 60); do
  ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes ubuntu@"$IP" true 2>/dev/null && break
  echo -n .; sleep 5
done; echo
echo "ssh -i $KEY_FILE ubuntu@$IP     # bootstrap (apt, uv, cron) ~2 min: tail -f /var/log/cloud-init-output.log"
echo "next: INSTANCE=$NAME ./sync_data.sh && INSTANCE=$NAME ./setup_env.sh   (record: $REC)"
