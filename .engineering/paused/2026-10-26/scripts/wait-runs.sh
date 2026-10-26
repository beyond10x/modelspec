#!/usr/bin/env bash
# wait-runs.sh <sha>: polls the workflow runs of a commit every 60 s until none is queued or in progress (at most 90 min)
set -u
S=$1
out=""
for i in $(seq 1 90); do
  out=$(gh run list -R beyond10x/modelspec --commit "$S" --json name,status,conclusion,databaseId --limit 20 | jq -r '.[] | "\(.name)\t\(.status)\t\(.conclusion)\t\(.databaseId)"')
  if [ -n "$out" ] && ! echo "$out" | grep -qE "queued|in_progress|waiting|pending|requested"; then
    echo "$out"; echo "EXIT=done"; exit 0
  fi
  timeout 60 tail -f /dev/null
done
echo "$out"; echo "EXIT=timeout"
