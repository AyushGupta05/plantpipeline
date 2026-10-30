#!/usr/bin/env bash
# Stop (not terminate) all plantpipeline instances. Stopped instances bill only EBS (200 GB gp3 = $16/month).
# One-time Spot instances cannot be stopped; use terminate.sh for those.
set -euo pipefail
REGION=${REGION:-us-east-1}
IDS=$(aws ec2 describe-instances --region $REGION --filters Name=tag:Project,Values=plantpipeline Name=instance-state-name,Values=running \
      --query 'Reservations[].Instances[].InstanceId' --output text)
[ -z "$IDS" ] && { echo "nothing running with Project=plantpipeline"; exit 0; }
echo "stopping: $IDS"; aws ec2 stop-instances --region $REGION --instance-ids $IDS --output table
