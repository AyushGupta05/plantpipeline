#!/usr/bin/env bash
# ssh to an instance: ./ssh.sh [name] [ssh args...]  or INSTANCE=name ./ssh.sh [ssh args...]; no name = .current_instance
# (MonoGrow box). e.g. ./ssh.sh gf1 -t tmux attach -t train
HERE=$(cd "$(dirname "$0")" && pwd); KEY_FILE=${KEY_FILE:-$HOME/.ssh/icp-flow-aws}
INSTANCE=${INSTANCE:-}   # instance name (.instances/<name>) or empty = .current_instance (MonoGrow box); also accepted as first arg
if [ -z "$INSTANCE" ] && [ -n "${1:-}" ] && [ -f "$HERE/.instances/$1" ]; then INSTANCE=$1; shift; fi
if [ -n "$INSTANCE" ]; then REC="$HERE/.instances/$INSTANCE"; else REC="$HERE/.current_instance"; fi
read -r IID IP _ < "$REC"
exec ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 ubuntu@"$IP" "$@"
