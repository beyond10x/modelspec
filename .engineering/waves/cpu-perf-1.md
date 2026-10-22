# Wave cpu-perf-1

Serves `epic:pytorch-10x-mnist`, CPU lane.

## Approval

On 2026-10-21 the operator approved this wave (option A): the unit commit of
`story:cpu-matmul-row-order` on `impl/cpu-matmul-row-order` through the bot, the adversary's cases
on that branch, the merge into `main`, and one push of `main` together with wave `observatory-1`.
No tag or release.

## Units

| # | story | branch | tree id | head | stage |
|---|---|---|---|---|---|
| 1 | cpu-matmul-row-order | impl/cpu-matmul-row-order | modelspec-impl-cpu-matmul-row-order | ab9ed27 | paused; WIP committed ddf282b (test, bench, CHANGELOG line; loop change not made) |

## Paused 2026-10-21

The operator paused modelspec on 2026-10-21. `impl/cpu-matmul-row-order` is pushed at `ddf282b`.

- Done: `tests/matmul_row_order.rs` (242 shapes, f32 and f64; green on the base loop, 2 of 4
  cases red with a zero-started accumulator planted); `benches/mnist_shape.rs` and its
  `Cargo.toml` entry; the `CHANGELOG.md` line. Base bench: 2,823 / 3,967 / 3,427 images/s on
  one core, median 3,427, weights hash `805f64bd…02df` on all three runs. `perf`:
  `interpreter::run::<f32>` 98.15% of base time.
- Open: the `i, k, j` loop change, its three bench runs, the gate, the adversary pass, the merge.
- First step on resume: start a new managed tree from `impl/cpu-matmul-row-order`, apply the
  `i, k, j` loop with each output row started from the `k = 0` products, then run
  `scratch/bench.sh change` and check the weights hash matches the base. The brief, `bench.sh`,
  `profile.sh` and the base logs are copied to `.engineering/paused/cpu-matmul-row-order/`.

## Closed 2026-10-22

- `impl/cpu-matmul-row-order` at `ed41930`, merged into `main` as `c1c47c7`; landed through
  https://github.com/beyond10x/modelspec/pull/8 (`task ci`, docs build and security checks green,
  run 39706801381), `main` fast-forwarded from `16d3dd2`.
- Adversary pass 1: `review-result:adversary-cpu-matmul-row-order-pass-1`, outcome no-op. Its
  NaN-payload cases are kept outside the repository; they fail under `--release` on base and head.
- Bench, base `ab9ed27` and change interleaved on core 3, 7 pairs: median 18.6 vs 1.226 ms/step
  (5,389 vs 81,544 images/s), 15.2×. The earlier 23.7× and 7.4× were taken under load 16–37 with
  base and change run apart.
- `story:cpu-matmul-row-order` moved to implemented. Next: `story:cpu-matmul-avx2-blocking`
  (draft), wave cpu-perf-2.

## Scratch

`~/.cache/modelspec-wave-cpu-perf-1/cpu-matmul-row-order/scratch/` holds the brief, the logs and
the benchmark runs.
