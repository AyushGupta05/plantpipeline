#!/usr/bin/env bash
# Terminate all plantpipeline instances (root volume is DeleteOnTermination -> all data on it is lost).
set -euo pipefail
REGION=${REGION:-us-east-1}
IDS=$(aws ec2 describe-instances --region $REGION --filters Name=tag:Project,Values=plantpipeline Name=instance-state-name,Values=running,stopped,pending \
      --query 'Reservations[].Instances[].InstanceId' --output text)
[ -z "$IDS" ] && { echo "no plantpipeline instances"; exit 0; }
echo "will TERMINATE: $IDS (data on root volume is lost - rsync results back first)"
read -r -p "type 'terminate' to confirm: " ok; [ "$ok" = terminate ] || exit 1
aws ec2 terminate-instances --region $REGION --instance-ids $IDS --output table
rm -f "$(dirname "$0")/.current_instance"
