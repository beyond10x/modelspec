#!/usr/bin/env bash
set -u
T=$1
cd "$T" || exit 1
B=$(git branch --show-current)
[ -n "$B" ] || { echo "REFUSE: detached"; exit 1; }
case "$B" in main|master) echo "REFUSE: $B"; exit 1;; esac
H=$(git rev-parse HEAD)
b10x-gates bot --repository beyond10x/modelspec -- push origin "${H}:refs/heads/${B}"
echo "push rc=$?"
git fetch -q origin "$B"
echo "$B local ${H:0:7} origin $(git rev-parse --short "origin/${B}" 2>/dev/null || git rev-parse --short FETCH_HEAD)"
echo EXIT=0
