#!/bin/zsh
cd /Users/ayush/dev/plantpipeline/experiments/aws
ID=i-0bcc5b885b6b1001b
for i in $(seq 1 40); do
  st=$(aws ec2 describe-instances --region us-east-1 --instance-ids $ID --query 'Reservations[0].Instances[0].State.Name' --output text)
  echo "$(date -u) attempt $i state=$st"
  if [ "$st" = "running" ]; then break; fi
  if [ "$st" = "stopped" ]; then aws ec2 start-instances --region us-east-1 --instance-ids $ID --output text 2>&1 | grep -oE 'InsufficientInstanceCapacity|pending' | head -1; fi
  sleep 120
done
aws ec2 wait instance-running --region us-east-1 --instance-ids $ID || { echo "NOT RUNNING, giving up"; exit 1; }
ip=$(aws ec2 describe-instances --region us-east-1 --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
read -r id _ type t0 < .instances/gf3; echo "$id $ip $type $t0" > .instances/gf3; echo "$(date -u) gf3 running at $ip"
for i in $(seq 1 30); do ./ssh.sh gf3 'sudo touch /var/tmp/no-idle-stop && sudo pkill -f idle-check; echo IDLE-STOP-DISABLED' 2>/dev/null | grep -q IDLE && { echo "$(date -u) idle-stop disabled"; break; }; sleep 10; done
for i in 1 2 3 4 5; do ./pull_results.sh > pull_gf3_$i.out 2>&1; if ! grep -q 'gf3 FAILED' pull_gf3_$i.out; then echo "$(date -u) PULL GF3 OK"; grep 'finished runs' pull_gf3_$i.out; exit 0; fi; echo "$(date -u) pull attempt $i failed"; sleep 30; done
echo "PULL GF3 FAILED"
