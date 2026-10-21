# M1 decomposition, revision 1: coordinator decisions

All four plan critics returned `needs-revision`. Their verdicts are recorded verbatim as
`review-result:m1-scope-round-1`, `m1-acceptance-round-1`, `m1-design-round-1` and
`m1-parallel-safety-round-1` (read each with `aep plan artifact show`; copies in this directory as
`m1-critic-<lens>-round-1.md`). Address every finding. Where a finding offers a choice, the
coordinator's choice is below; take it and do not reopen it.

## R1. DESIGN.md baseline: the 10× patch lands first

The DESIGN patch at `~/.cache/modelspec-wave-cpu-perf-1/design-10x/design-10x.patch` (ADR-007,
§ 5.4 bit-for-bit vs tolerance classes, § 6.3 whole-run execution) is applied to `main` before any
M1 story starts, after wave observatory-1 merges; its revision number becomes 0.9 there. Write
every M1 draft against DESIGN with that patch applied. Where a draft edits a sentence the patch
also edits (§ 6, § 6.3), say the edit applies on top of it, and § 6.3 keeps only the M2
increments (device gather, parameter ping-pong, fused kernels).

## R2. DESIGN.md ownership (design finding on checkpoint-cpu, parallel-safety finding on reduce-matmul)

Each story keeps editing the DESIGN sections it changes, in the same change as its code
(AGENTS.md rule). The revision line, the § 8 verb table and `CHANGELOG.md` are append-only
surfaces: the story adds its entry, and the number on the revision line is assigned by the wave
coordinator at merge. Say so in one line in each story's Scope that touches them. No ordering
edges are added for these surfaces alone.

## R3. Numerics contract: one owner per class

- `story:m1-vulkan-elementwise` states gate 5's reference `r` for elementwise kernels only.
- `story:m1-vulkan-reduce-matmul` owns the reduction rule and extends gate 5 to reductions and
  matmul. Word the rule as "an order fixed by shape alone; the workgroup size is a compile-time
  constant, not read from DeviceProfile"; the CPU order is one admissible order. It edits the
  `deterministic` summary at `spec/domains/artifact.yaml:188` and runs `ess specify validate`.
  It states which kernels are in § 5.4's bit-for-bit class in M1 (the one-invocation row-major
  loss sum and the non-split-K matmul) and that M1 emits the RTE, denorm-preserve and
  signed-zero-preserve execution modes for them and asserts bit-equality against the CPU on the
  operator run. Parallel reductions are in the tolerance class.

## R4. The plan represents two-stage reductions (design blocker on kernel-plan)

`story:m1-kernel-plan` lowers a reduce to stage 1, an `Allocate` of a partials scratch buffer, a
barrier and stage 2, in the verified plan; its acceptance replaces "exactly one Dispatch per
computing op" with the rule that covers this. `ParameterSpec` carries its views so one view is the
in-place case and two is ping-pong (M2 uses two; M1 uses one).

## R5. compile-artifact off the critical path (scope and design warnings)

Move the emit-every-kernel entry into `story:m1-vulkan-reduce-matmul` (`spirv/emit.rs`).
`story:m1-vulkan-train` depends on reduce-matmul for it and no longer on compile-artifact.
`story:m1-compile-artifact` keeps the `compile` verb, the on-disk format, the reader and the
E-ARTIFACT codes; its body names the parent's `compile` verb and DESIGN § 7.1 and § 9 gate 8 as
what it serves. Its tamper cases say which ones re-seal `artifact_hash` (code 003) and which do
not (code 002).

## R6. mlp-example stays (scope warning)

The body says checkboxes 1–3 are already met by M0 and that the story exists for the GPU stories'
shapes and `epic:pytorch-10x-mnist`; relate it `serves` or `informed_by` that epic as the store
allows.

## R7. inspect-device (scope blocker, design and acceptance warnings)

- It owns feature negotiation: the Vulkan version and features the runtime requires, enabled at
  logical-device creation, and a refusal with a stated code when one is missing.
- `spec/domains/device.yaml` and the probe gain `shaderRoundingModeRTEFloat32`,
  `shaderDenormPreserveFloat32` and `shaderSignedZeroInfNanPreserveFloat32`.
- The fixture order is explicit: the operator device run produces `rtx3090.device.json`; the
  coordinator commits it; the CI items that read it are checked after that commit. The story is
  done only after the operator run.
- Claim `crates/modelspec-core/Cargo.toml` and `Cargo.lock`.

## R8. A host without Vulkan is produced, not assumed (acceptance blockers)

Every acceptance item about a missing loader uses `MODELSPEC_VULKAN_LIBRARY` pointing at a path
that does not exist, as `story:m1-vulkan-checkpoint-infer` item 5 does: compile-artifact item 5,
vulkan-train item 6. `compile --backend vulkan` in `task ci` is always given `--device-profile` or
the missing-loader variable, so `task ci` never creates a Vulkan instance.

## R9. Acceptance precision (acceptance warnings)

- checkpoint-cpu item 4: a counting global allocator in that test observes the largest allocation.
- vulkan-profile item 1: the count is per dispatch site (one line per dispatch in the plan), so
  two dispatches sharing a content-hashed module show as two lines; say so.

## R10. Ordering edges (parallel-safety blockers and warnings)

- `story:m1-vulkan-train` depends_on `story:m1-checkpoint-cpu` (both edit `train/runner.rs` and
  `cli/src/train.rs`; the CPU checkpoint lands first).
- `story:m1-vulkan-checkpoint-infer` depends_on `story:m1-vulkan-profile` (both edit
  `cli/src/train.rs` and the Vulkan runner loop).
- `story:m1-checkpoint-cpu` and `story:m1-vulkan-train` depend_on
  `story:observatory-live-event-stream` if the store holds that story here; if it does not, the
  body says both are scoped against `main` after wave observatory-1 merges.
- `story:m1-compile-artifact` and `story:m1-checkpoint-cpu`: add the edge compile-artifact →
  depends_on checkpoint-cpu (shared `cli/src/lib.rs`, docs `cli.rs`, docs `diagnostics.rs`).
- `story:m1-vulkan-elementwise` depends_on `story:m1-mlp-example` (`Taskfile.yml`, `AGENTS.md`).
- Every story that adds a dependency claims `Cargo.lock`; `crates/modelspec-runtime/Cargo.toml`
  overlaps with `story:cpu-matmul-row-order` only through its bench entry, which merges first.
- Scope entries under `website/docs/reference/` name files, not the directory.

## Limits

Draft only: create nothing new except edges and body changes on the ten drafts you created, move
nothing, relay every `aep` refusal verbatim, run `aep plan artifact validate` at the end and quote
it. Return per finding what changed (one line each), plus any finding you could not address and
why.
