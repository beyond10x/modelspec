#!/usr/bin/env bash
set -eu
[ $# -eq 4 ] || { echo "usage: mkpr.sh <tree> <body.md> <title> <head-branch>"; exit 1; }
D=$(mktemp -d "$HOME/.cache/claude-tmp/pr-bench-page.XXXXXX"); chmod 700 "$D"
P=$HOME/beyond10x/gates-policy/policy.json
B=$2
cd "$1"
b10x-gates scan-text --repo . --policy "$P" --repository beyond10x/modelspec --gitleaks "$(command -v gitleaks)" "$B"
jq -n --rawfile body "$B" --arg title "$3" --arg head "$4" '{title:$title, head:$head, base:"main", body:$body}' > "$D/pr.json"
b10x-gates api --method POST --path /repos/beyond10x/modelspec/pulls --input "$D/pr.json" --output "$D/pr-out.json" --repository beyond10x/modelspec --policy "$P"
jq -r '.number, .html_url, .user.login' "$D/pr-out.json"
echo EXIT=0
