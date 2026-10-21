#!/usr/bin/env bash
set -u
label="$1"
bin="$2"
S=~/.cache/modelspec-wave-cpu-perf-1/cpu-matmul-row-order/scratch
cd "$S" || exit 1
echo "loadavg: $(cat /proc/loadavg)"
taskset -c 3 perf record -q -g -o "$S/perf-$label.data" "$bin" > "$S/perf-$label-run.log" 2>&1
echo "EXIT=$?"
tail -3 "$S/perf-$label-run.log"
perf report -i "$S/perf-$label.data" --no-children --stdio --sort symbol -g none 2>/dev/null \
  | grep -v '^#' | grep -v '^$' | sed "s/  \+/ /g" | head -15 > "$S/perf-$label-report.txt"
cat "$S/perf-$label-report.txt"
