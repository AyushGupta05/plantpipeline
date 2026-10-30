#!/usr/bin/env bash
# Stop (not terminate) running instances tagged Project=plantpipeline, after confirmation. Compute billing stops; the
# 200 GB gp3 root volume stays (~$16/month each). Usage: ./stop_all.sh [name ...]   (default: all tagged; names = .instances/<name>|monogrow)
set -euo pipefail
REGION=${REGION:-us-east-1}; HERE=$(cd "$(dirname "$0")" && pwd)
if [ $# -gt 0 ]; then IDS=""; for n in "$@"; do f="$HERE/.instances/$n"; [ "$n" = monogrow ] && f="$HERE/.current_instance"; IDS="$IDS $(awk '{print $1}' "$f")"; done
else IDS=$(aws ec2 describe-instances --region "$REGION" --filters Name=tag:Project,Values=plantpipeline Name=instance-state-name,Values=running \
      --query 'Reservations[].Instances[].InstanceId' --output text); fi
[ -z "${IDS// /}" ] && { echo "nothing running with Project=plantpipeline"; exit 0; }
aws ec2 describe-instances --region "$REGION" --instance-ids $IDS --filters Name=tag:Project,Values=plantpipeline \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,PublicIpAddress,State.Name,Tags[?Key==`Name`].Value|[0]]' --output table
read -r -p "stop these? [y/N] " ok; [ "$ok" = y ] || exit 1
aws ec2 stop-instances --region "$REGION" --instance-ids $IDS --output table
echo "note: public IPs change on restart; after 'aws ec2 start-instances' update .instances/<name> / .current_instance"
