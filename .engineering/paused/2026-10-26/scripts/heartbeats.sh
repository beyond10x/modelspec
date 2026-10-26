#!/usr/bin/env bash
# heartbeats.sh: heartbeat the coordinator leases every 15 minutes for 4 hours; stops when a tree is gone
set -u
[ $# -ge 1 ] || { echo "usage: heartbeats.sh <tree-id>:<session> ..."; exit 1; }
R=$HOME/.local/state/worktree/trees/b10x/modelspec
for i in $(seq 1 16); do
  for t in "$@"; do
    p=$R/${t%%:*}
    [ -d "$p" ] && worktree hook heartbeat --path "$p" --session "${t##*:}" 2>&1 | tail -1
  done
  date +%H:%M
  timeout 900 tail -f /dev/null
done
echo EXIT=0
