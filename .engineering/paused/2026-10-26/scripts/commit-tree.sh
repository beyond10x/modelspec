#!/usr/bin/env bash
set -u
T=$1
MSG=$2
cd "$T" || exit 1
git branch --show-current
git add -A
if [ -f Cargo.toml ] && ! cargo fmt --check >/dev/null 2>&1; then echo "REFUSE: cargo fmt --check"; exit 1; fi
git diff --cached --stat | tail -3
if git diff --cached --name-only | grep -E '(^|/)target/|^\.engineering/drafts/'; then echo "REFUSE: target or drafts staged"; exit 1; fi
if git diff --cached | grep -n -i -E "^\+.*/"'home/[a-z]+/'; then echo "REFUSE: forbidden literal"; exit 1; fi
b10x-gates bot --repository beyond10x/modelspec -- commit -q -F "$MSG"
echo "rc=$?"
git log -1 --format='%H %an %s'
git status --short | wc -l
echo EXIT=0
