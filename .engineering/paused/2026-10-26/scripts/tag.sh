#!/usr/bin/env bash
# tag.sh <release-tree> <version>: fast-forward main to release/<version>'s head, tag <version> by the bot, push the tag
set -u
[ $# -eq 2 ] || { echo "usage: tag.sh <release-tree> <version>"; exit 1; }
T=$1; V=$2
cd "$T" || exit 1
H=$(git rev-parse "release/$V")
git fetch -q origin main
git merge-base --is-ancestor origin/main "$H" || { echo "REFUSE: main not an ancestor of $H"; exit 1; }
if [ "$(git rev-parse origin/main)" != "$H" ]; then
  b10x-gates bot --repository beyond10x/modelspec -- push origin "${H}:refs/heads/main" 2>&1 | grep -v '^remote:' | tail -2
  git fetch -q origin main
fi
[ "$(git rev-parse origin/main)" = "$H" ] || { echo "REFUSE: main is $(git rev-parse --short origin/main), not ${H:0:7}"; exit 1; }
echo "main = ${H:0:7}"
b10x-gates bot --repository beyond10x/modelspec -- tag -a "$V" -m "ModelSpec $V" "$H" 2>&1 | tail -2
git cat-file -t "$V"
b10x-gates bot --repository beyond10x/modelspec -- push origin "$V" 2>&1 | grep -v '^remote:' | tail -2
git ls-remote --tags origin "$V"
echo EXIT=0
