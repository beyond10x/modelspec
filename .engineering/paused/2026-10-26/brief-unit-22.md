# Unit 22: story:transformer-block-benchmark — the block example, the Rust bench, the PyTorch opponent, the counting rules

```
story:     story:transformer-block-benchmark   (aep plan artifact show …, inside the worktree)
worktree:  WT_PLACEHOLDER
branch:    impl/transformer-block-benchmark   (plan commit on main = release 0.20.0)
scratch:   <wave-root>/transformer-block/scratch   (everything untracked; never /tmp)
lease:     worktree hook session-start --path <worktree> --session impl-transformer-block; heartbeat every 20 min; session-end before the report
venv:      <venv>/bin/python   (torch 2.14.1+cu130, no numpy; stays outside the repository)
```

The story's § Acceptance, the epic's § Measurement rules (`aep plan artifact show
epic:transformer-vs-pytorch-cuda`) and coordinator decision D81 (`.engineering/waves/bert-1.md`)
are the specification. Read `AGENTS.md` (§ Rules on opponents, the generated files, the gates),
`examples/bert_tiny.mspec` (the two encoder blocks this unit copies), `crates/modelspec-runtime/
benches/mlp_sweep.rs` with `benches/mlp_sweep/{sweep,runs}.rs` and `tests/mlp_sweep.rs` (the
bench this unit mirrors: the point source, warm-up, calibration, 3 timed runs, the JSON result
line, `train_repeatedly`), `opponents/mlp_sweep/mlp_sweep.py` and its README (the opponent this
unit mirrors), `crates/modelspec-runtime/src/vulkan/profile.rs` (`DispatchCount`, the counting
rows, `uncounted`), `crates/modelspec-runtime/src/train/checkpoint.rs` (the `.msckpt` layout:
56-byte header, JSON manifest, aligned little-endian `f32` payload), `crates/modelspec-cli/src/
train.rs` (`--profile`, `--checkpoint`, `--steps`, the `step <k> loss <v>` lines), `spec/domains/
benchmark.yaml` (`BenchmarkRun`, `FlopRule`) and `crates/modelspec-docs/src/benchmarks.rs`
(`read`'s `MatmulOnly` check at about line 199, the declared workload `transformer-block`, the
section a first transformer run gets).

## What this unit does

1. `examples/transformer_block.mspec`: the two BERT-tiny encoder blocks of `bert_tiny.mspec`
   (hidden 128, 2 heads, feed-forward 512, sequence 128, post-layer-norm, GELU, no dropout) on a
   random `f32` input `x: [B·128, 128]` with an all-ones key mask `[B, 128]`, an MSE loss against
   a random target of the block's output shape, SGD, batch B = 8 in the file. No token or
   position table, no MLM head, no AdamW, no `--data`: the data is the runner's synthetic
   batch, as the sweep's `Point::source` declares it (read how `mse` on a synthetic input and
   target is declared there and in `tests/`; if a random target of shape `[B·128, 128]` needs a
   teacher the runner does not have, say so in the report and use what the runner offers). The
   file's header comment says what it is and what it leaves out, in the register of
   `bert_tiny.mspec`. It passes `modelspec check`, trains on the CPU and on `--backend vulkan`,
   and a CPU test (`crates/modelspec-cli/tests/`, beside the `bert_cpu` tests) plus an
   `#[ignore]`d device test hold that the two runs' step losses agree over 10 steps within the
   DESIGN.md § 9 run tolerance (`GPU_RUN_ATOL` 1e-4, `GPU_RUN_RTOL` 1e-3, `tests/common/mod.rs`).
2. Counting rules: `train --backend vulkan --profile` on the example prints no `uncounted`
   dispatch of the step plan. Add the counting row for every kind it still prints as
   `uncounted`, with a unit test per added row in `profile.rs`, and the DESIGN.md § 3.1 counting
   rule text if a row is new. Report the `profile step_total` line.
3. `crates/modelspec-runtime/benches/transformer_block.rs` with `[[bench]] name =
   "transformer_block", harness = false, required-features = ["vulkan"]` in
   `crates/modelspec-runtime/Cargo.toml` (after `mlp_sweep`), and its pure module
   `benches/transformer_block/block.rs` included by a new `tests/transformer_block.rs` without a
   device: the three points batch 8, 32, 128 sequences; per point a 20-step warm-up, calibration
   until a run lasts at least `--min-seconds` (default 5), 3 timed runs, the median; the point's
   source formatted from the example's text with the batch substituted (never a second copy of
   the model: read `examples/transformer_block.mspec` at run time or `include_str!` it and
   substitute B, and a test holds that the bench's source at B = 8 equals the file). One JSON
   line per point on stdout in the sweep's shape: `side: "modelspec-vulkan"`, `batch`, `seq`
   128, `tokens_per_step` B·128, `flop_per_step`, `flop_rule: "matmul-only"`, `device`, `build`
   (`commit`, `dirty`, `version`), `status`, `steps`, `runs` (per run `us_per_step`, `seconds`,
   `span_unix_ms`), `median_us_per_step`, `tokens_per_s` (B·128·1e6 / median µs), `gflops`.
   `--point B` (repeatable) and `--min-seconds` as the sweep has them. No `--crossover`.
4. The FLOP rule of the block, `matmul-only`: the closed form in B of the dense and attention
   matmuls of one step (forward and backward, the five dense matmuls' pattern of the sweep applied
   to each `matmul` of the block, plus the two batched attention matmuls `QKᵀ` and `PV` and
   their adjoints, per head; no elementwise, layer-norm, softmax, loss or update work). Derive
   it in `block.rs` with the derivation as its doc comment, a test that evaluates it at B = 8,
   and the same formula in the opponent; `spec/domains/benchmark.yaml`'s `FlopRule.MatmulOnly`
   summary gains the block's formula beside the sweep's, and `benchmarks.rs` `read` checks a
   `MatmulOnly` count per workload (`mlp-784-H-10` against 3196·B·H as now, `transformer-block`
   against the block's formula at its batch) and refuses any other workload stating
   `MatmulOnly`; regenerate the spec outputs the repository generates (`task check` says which).
5. `opponents/transformer_block/transformer_block.py` and `README.md`: the same two blocks in
   PyTorch (`torch.nn.functional` calls written out, not `nn.TransformerEncoderLayer`, so GELU
   (the tanh form if ModelSpec's is tanh: read `gelu` in the CPU backend), layer-norm eps, the
   head split, the mask and the MSE match the file), `f32`, TF32 off, SGD at the file's lr, eager
   and `torch.compile(dynamic=False)` after a dynamo reset, the same warm-up, calibration, 3
   timed runs, `torch.cuda.synchronize` around the span, one JSON line per point and mode in the
   bench's shape (`side: "pytorch-eager"` / `"pytorch-compile"`), progress on stderr, a refusal
   line on failure. `--checkpoint <path.msckpt>` loads ModelSpec's initial parameters (parse the
   header and manifest with `struct` and `json`, the payload with `array('f')` or
   `torch.frombuffer`; no numpy) by parameter path into the module, and `--loss-check
   <path.msckpt>` prints `step 0 loss <v>` for the first step on those parameters and on the
   batch ModelSpec draws (the runner's synthetic batch at seed 42: say in the README how the
   opponent reproduces it, or, if it cannot, read the batch from a file ModelSpec writes; in that
   case add the smallest `train` or bench flag that writes the first batch as little-endian `f32`
   and document it). The README has the sweep README's sections. No ModelSpec build, test, gate
   or `task` target runs the opponent.
6. Draft, under `.engineering/drafts/verification-report-transformer-block.md` (git-ignored,
   the coordinator creates the artifact), the report's skeleton in the form of
   `verification-report:mlp-size-sweep-2026-10-25` (`aep plan artifact show`): the Commands
   block (the bench, the opponent, the `nvidia-smi` sampler, the checkpoint export, the loss
   check), the Host table, a Results table with the columns batch, tokens per step, FLOP per
   step, modelspec µs / tokens per s / GFLOP/s, PyTorch eager and compile the same, and a
   First-step loss table (ModelSpec CPU, ModelSpec Vulkan, PyTorch, the § 9 tolerance), the
   figures left as `TBD`. The coordinator runs the device runs and fills it.
7. `benchmarks/runs.yaml` gets no entry in this pass (the figures do not exist yet); the page's
   `transformer-block` section generator must already render a run with `seq`/`tokens_per_step`
   if the schema gains them: add `tokens_per_step: Optional<PositiveInteger>` to `BenchmarkRun`
   and a tokens-per-second column to the block's table, with a fixture test; regenerate.
8. CHANGELOG.md `### Added` under Unreleased: the example, the bench, the opponent, the counting
   rows, in the form of the other entries. AGENTS.md: the opponent in the list where
   `opponents/mlp_sweep` is named, if there is one.
9. Gates, each exit 0 (quote the last lines): `B10X_BUILD_REPO_BUDGET_G=90 b10x-build-slot cargo
   test --workspace --features vulkan` (device tests stay `#[ignore]`d), `… cargo test -p
   modelspec-runtime --features vulkan --test transformer_block`, `… cargo bench -p
   modelspec-runtime --features vulkan --bench transformer_block -- --point 8 --min-seconds 1`
   (a smoke run, not a figure: one point, the GPU is shared with nothing of yours), `… cargo run
   -p modelspec-cli --features vulkan -- train examples/transformer_block.mspec --backend vulkan
   --profile --steps 10`, `… task docs-check`, `… task check`, `… cargo fmt --all --check`,
   `… cargo clippy --workspace --all-targets --features vulkan -- -D warnings`. Run the opponent
   once with the venv: `--point 8 --min-seconds 1 --modes eager` and the `--loss-check` on a
   checkpoint `train … --steps 0 --checkpoint` writes (if `--steps 0` is refused, the smallest
   change that lets `train` write the initial checkpoint, documented; say which).

Timed figures are not this unit's to produce: the coordinator starts every timed device run
(epic rule "Device runs are the operator's to start"). Your device runs are smoke runs of one
point with `--min-seconds 1`, and the 10-step agreement test. No edits under `.engineering/`
except the draft. No git commit, add, stash, checkout, branch, push.

## Repository invariants (modelspec)

* Running code is Rust. Python only under `opponents/`, never run by any build, test, gate or
  `task` target. No `python3 -c`, no heredoc feeding a stdin reader; write files with your Write
  tool and run them by path.
* Every cargo or task invocation goes through `B10X_BUILD_REPO_BUDGET_G=90 b10x-build-slot …`;
  never set `CARGO_TARGET_DIR`; never delete `target/` by hand; `df -h /` first (the figure
  free at dispatch).
* Nothing under `/tmp` or `/dev/shm`. No home-directory path in any tracked file. No name of
  any other organization or customer. No comments in config or manifest files.
* Generated docs, schemas and UI types are never edited by hand.
* The GitLab API is never called with a personal token. `gh` is read-only. Never push.
* Never lock GPU clocks. A replay server on port 7878 belongs to the coordinator; leave it.

## The report header

```
unit: story:transformer-block-benchmark
verdict: green | red | blocked
cases: executed <before>→<after>, red 0
origin: n/a
wrote-outside-worktree: <paths> | none
needs-coordinator: <what you could not do and who owns it> | none
```

Then: the FLOP formula and its value at B = 8, 32, 128; the `profile step_total` line and any
row added; the smoke run's JSON line for both sides; the first-step losses of the loss check
(CPU, Vulkan, PyTorch) and the tolerance; the exact commands the coordinator runs for the timed
runs (bench, opponent, sampler, checkpoint export, loss check), ready to paste; the gates' last
lines, `git status --short` and `git diff --stat | tail -1`, every deviation from this brief.
