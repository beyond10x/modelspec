#!/usr/bin/env bash
set -u
label="$1"
W=~/.local/state/worktree/trees/b10x/modelspec/modelspec-impl-cpu-matmul-row-order
S=~/.cache/modelspec-wave-cpu-perf-1/cpu-matmul-row-order/scratch
cd "$W" || exit 1
for n in 1 2 3; do
  log="$S/bench-$label-$n.log"
  {
    echo "date: $(date -Is)"
    echo "loadavg: $(cat /proc/loadavg)"
    echo "commit: $(git rev-parse HEAD); interpreter.rs blake3-ish: $(sha256sum crates/modelspec-runtime/src/cpu/interpreter.rs)"
    echo "command: taskset -c 3 b10x-build-slot cargo bench -p modelspec-runtime --bench mnist_shape"
    taskset -c 3 b10x-build-slot cargo bench -p modelspec-runtime --bench mnist_shape 2>&1
    echo "EXIT=$?"
  } > "$log"
  grep -h 'mnist_shape:\|loadavg\|EXIT' "$log"
done
