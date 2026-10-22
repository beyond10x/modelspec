---
format: aep.planning-md/3
id: story:m1-checkpoint-cpu
kind: story
status: implemented
title: 'Checkpoints on the CPU: .msckpt save and resume, and modelspec infer --backend cpu'
summary: The .msckpt writer and validating reader, train --checkpoint, --checkpoint-every, --resume and --steps, byte-equal resume on the CPU, infer on a checkpoint, E-CKPT-001 to 004
relations:
- serves: vision:modelspec
- depends_on: epic:m1-vulkan-execution
- depends_on: story:m1-kernel-plan
- depends_on: story:observatory-live-event-stream
- depends_on: story:m1-vulkan-elementwise
scope:
- confidence: cited
  path: CHANGELOG.md
- confidence: cited
  path: Cargo.lock
- confidence: cited
  path: Cargo.toml
- confidence: cited
  path: DESIGN.md
- confidence: inferred
  path: crates/modelspec-cli/src/infer.rs
- confidence: cited
  path: crates/modelspec-cli/src/lib.rs
- confidence: cited
  path: crates/modelspec-cli/src/train.rs
- confidence: inferred
  path: crates/modelspec-cli/tests/checkpoint_cpu.rs
- confidence: cited
  path: crates/modelspec-docs/src/cli.rs
- confidence: cited
  path: crates/modelspec-docs/src/diagnostics.rs
- confidence: cited
  path: crates/modelspec-runtime/Cargo.toml
- confidence: inferred
  path: crates/modelspec-runtime/src/train/checkpoint.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/mod.rs
- confidence: cited
  path: crates/modelspec-runtime/src/train/runner.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/checkpoint.rs
- confidence: inferred
  path: crates/modelspec-runtime/tests/checkpoint_adversary.rs
- confidence: cited
  path: website/docs/reference/cli.md
- confidence: cited
  path: website/docs/reference/diagnostics.md
revision: 9
transitions:
- {from: "draft", to: "proposed", at: "2026-10-22T07:42:34Z", actor: "human:timo", revision: 7, decided_on: {"recorded":{"review_outcome":4}}}
- {from: "proposed", to: "active", at: "2026-10-22T07:42:34Z", actor: "human:timo", revision: 8, decided_on: {"recorded":{"review_outcome":4}}}
- {from: "active", to: "implemented", at: "2026-10-22T08:41:16Z", actor: "human:timo", revision: 9, decided_on: {"recorded":{"test_result":1,"review_outcome":6}}}
---
# Checkpoints on the CPU: `.msckpt` save and resume, and `modelspec infer --backend cpu`

## Context

M1 checkbox 8 of DESIGN.md § 9: "Checkpoint save/load and inference work with the same parameter
identities". § 6.2 fixes the `.msckpt` package (header with magic `MSCK`, format version, endian
marker, manifest size and blake3 checksum; a JSON manifest; an aligned little-endian payload; an
optional payload hash), says to validate every offset and size against the file before mapping or
allocating, and that a checkpoint loads into the CPU backend whichever backend wrote it. § 9 gate 7
requires that an uninterrupted and a resumed CPU run be equivalent under fixed settings. § 6 says
`steps` counts the steps of one invocation, so a run resumed at step `s` ends at `s + steps`.

`spec/domains/checkpoint.yaml` types `Checkpoint`, which owns `ParameterRecord`, with
`schema_hash`, `optimizer`, `optimizer_slots` (empty for SGD) and `rng`; it has no UNMAPPED
marker. `story:m1-kernel-plan` defines the `schema_hash` a checkpoint is matched against. M0 has
no checkpoint and no `infer`. This story does both on the CPU, so the GPU story only adds device
read-back.

## What it adds

- DESIGN.md § 6.2: the header byte layout (field order, widths, the endian marker value), a
  payload alignment of 64 bytes, and the payload hash always written in M1.
- `crates/modelspec-runtime/src/train/checkpoint.rs`: write and read `.msckpt`. The reader checks,
  in order, the header, the checksum, `manifest_size` against the file size, the manifest
  invariants of `spec/domains/checkpoint.yaml`, every offset and length against `file_size` and
  `alignment`, and the payload hash, before it allocates a tensor. The manifest is JSON through
  `serde` and `serde_json`.
- `modelspec train ... --checkpoint <path>` writes the state at the end of the run;
  `--checkpoint-every <n>` also writes `<stem>-<k>.msckpt` beside it at every step `k` that is a
  multiple of `n` before the end; `--resume <path>` continues from a checkpoint; `--steps <n>`
  overrides the `train` block. `rng` records the seed and counter 0 (M0 SGD draws no number per
  step) and `optimizer.schedule` is absent.
- Resume refuses a checkpoint whose `schema_hash`, parameter paths, dtypes, shapes or optimizer
  differ from the compiled file, naming the first difference.
- `modelspec infer <file> --backend cpu --checkpoint <path>` runs the forward of the compiled file
  with the checkpoint parameters on the synthetic batch of its `train` block and prints a
  `prediction` line and a `loss` line. M1 takes no `--input`: the data interface is open
  (DESIGN.md § 12).
- E-CKPT-001 (not a readable checkpoint: magic, version, endian marker, checksum), E-CKPT-002 (an
  offset, length or size outside the file or misaligned), E-CKPT-003 (does not match the compiled
  file) and E-CKPT-004 (payload hash differs).

## Domain relations

- Checkpoint owns ParameterRecord (`parameters`, `spec/domains/checkpoint.yaml:127`).
  Inferable from ess/1.
- A checkpoint loads into an artifact only with equal `schema_hash` (line 100) and equal paths,
  dtypes and shapes; identity is the `ParameterPath`. Inferable from ess/1.
- `optimizer_slots` is empty for SGD (line 112). Inferable from ess/1.
- A resumed run names its checkpoint (TrainingRun `resumed_from`,
  `spec/domains/training.yaml:142`). Inferable from ess/1; this story writes no TrainingRun.

## Scope

- `crates/modelspec-runtime/src/train/checkpoint.rs` (new), `crates/modelspec-runtime/src/train/mod.rs`,
  `crates/modelspec-runtime/src/train/runner.rs`.
- `crates/modelspec-runtime/Cargo.toml` (`serde`, `serde_json`); its bench entry from
  `story:cpu-matmul-row-order` is the only other edit to this file, and that story merges first.
  `Cargo.toml` (`serde` and `serde_json` in `[workspace.dependencies]` if wave observatory-1 did not
  add them) and `Cargo.lock`.
- `crates/modelspec-runtime/tests/checkpoint.rs` (new),
  `crates/modelspec-runtime/tests/checkpoint_adversary.rs` (new).
- `crates/modelspec-cli/src/train.rs`, `crates/modelspec-cli/src/infer.rs` (new),
  `crates/modelspec-cli/src/lib.rs`, `crates/modelspec-cli/tests/checkpoint_cpu.rs` (new).
- `train/runner.rs`, `cli/src/train.rs`, root `Cargo.toml` and `Cargo.lock` are also edited by
  `story:observatory-live-event-stream` (wave observatory-1); this story depends on it and is scoped
  against `main` after that wave merges.
- `crates/modelspec-docs/src/diagnostics.rs`, `crates/modelspec-docs/src/cli.rs`, and through
  `task docs-generate` `website/docs/reference/cli.md` and `website/docs/reference/diagnostics.md`.
- `DESIGN.md` § 6.2 and § 8. The revision line, the § 8 verb table and `CHANGELOG.md` are
  append-only surfaces: this story adds its entry, and the wave coordinator assigns the revision
  number at merge.
- `CHANGELOG.md`.
- Ordering in this set: this story depends on `story:m1-vulkan-elementwise`, which depends on
  `story:m1-inspect-device`. Both edit root `Cargo.toml`, `Cargo.lock` and
  `crates/modelspec-runtime/Cargo.toml`; inspect-device also edits `crates/modelspec-cli/src/lib.rs`
  and `crates/modelspec-docs/src/diagnostics.rs` and adds `serde` and `serde_json` to
  `[workspace.dependencies]`, so this story adds them only to the crates that still lack them.

## Acceptance

A CPU run saved and resumed ends with the same bytes as one that never stopped, `infer` reproduces
the final loss of the run from a checkpoint, and every malformed checkpoint is refused with a stated
code before anything is allocated.

1. First: DESIGN.md § 6.2 states the header byte layout, the 64-byte alignment and the payload
   hash.
2. `cargo test -p modelspec-cli --test checkpoint_cpu resume_equals_uninterrupted`, in `task ci`,
   on `examples/linear_regression.mspec`: `--steps 500 --checkpoint a.msckpt`, and `--steps 250
   --checkpoint h.msckpt` then `--resume h.msckpt --steps 250 --checkpoint b.msckpt`, give `a` and
   `b` byte-equal, and the `step` lines from 250 to 500 equal byte for byte.
3. The same test file: `infer examples/linear_regression.mspec --backend cpu --checkpoint
   a.msckpt` exits 0 and its `loss` value equals the `step 500` loss byte for byte;
   `--checkpoint-every 100` with 500 steps writes exactly 4 extra files, at steps 100 to 400.
4. `cargo test -p modelspec-runtime --test checkpoint_adversary`, in `task ci`: the reader refuses
   each of eight altered files with the code stated and without a panic: a truncated file (002), a
   changed checksum byte (001), the big-endian marker (001), an offset past the file (002), an
   offset not a multiple of 64 (002), a length that is not shape times dtype size (002),
   `manifest_size` of `u64::MAX` (002) and a changed payload byte (004). The test binary installs a
   counting `#[global_allocator]` over `std::alloc::System` that records, per thread, the largest
   single allocation; for every case, the largest allocation made while the reader runs is at most
   the file size plus 4,096 bytes.
5. Resume of a `two_layer_relu` checkpoint into `examples/linear_regression.mspec` exits 1 with
   E-CKPT-003 naming the first differing parameter path; a changed learning rate exits 1 with
   E-CKPT-003 naming the optimizer.
6. `task ci` exits 0, `task docs-check` included, with E-CKPT-001 to 004 on the diagnostics page and
   `infer` on the CLI page.

## Out of scope

- Checkpoints of a GPU run and `infer --backend vulkan`: `story:m1-vulkan-checkpoint-infer`.
- `infer --input` and any data file (DESIGN.md § 12 "Data interface").
- Optimizer slots (AdamW, `story:bert-adamw-warmup`) and parameter-path migration adapters.
- A TrainingRun record on disk.
