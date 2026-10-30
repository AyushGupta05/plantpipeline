#!/usr/bin/env bash
# TERMINATE instances tagged Project=plantpipeline (root volumes are DeleteOnTermination: everything on them is lost;
# run ./pull_results.sh first). Usage: ./terminate_all.sh [name ...]  (default: all tagged, incl. the MonoGrow box!)
set -euo pipefail
REGION=${REGION:-us-east-1}; HERE=$(cd "$(dirname "$0")" && pwd)
if [ $# -gt 0 ]; then IDS=""; for n in "$@"; do f="$HERE/.instances/$n"; [ "$n" = monogrow ] && f="$HERE/.current_instance"; IDS="$IDS $(awk '{print $1}' "$f")"; done
else IDS=$(aws ec2 describe-instances --region "$REGION" --filters Name=tag:Project,Values=plantpipeline Name=instance-state-name,Values=running,stopped,pending \
      --query 'Reservations[].Instances[].InstanceId' --output text); fi
[ -z "${IDS// /}" ] && { echo "no plantpipeline instances"; exit 0; }
# tag filter applied again on the resolved ids: never touch an untagged instance
IDS=$(aws ec2 describe-instances --region "$REGION" --instance-ids $IDS --filters Name=tag:Project,Values=plantpipeline --query 'Reservations[].Instances[].InstanceId' --output text)
aws ec2 describe-instances --region "$REGION" --instance-ids $IDS \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,PublicIpAddress,State.Name,Tags[?Key==`Name`].Value|[0]]' --output table
read -r -p "type 'terminate' to confirm (data on root volumes is lost): " ok; [ "$ok" = terminate ] || exit 1
aws ec2 terminate-instances --region "$REGION" --instance-ids $IDS --output table
for id in $IDS; do for f in "$HERE"/.instances/* "$HERE/.current_instance"; do [ -f "$f" ] && grep -q "^$id " "$f" && rm -f "$f" && echo "removed record $f"; done; done; true
