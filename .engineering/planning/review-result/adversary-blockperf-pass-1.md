---
format: aep.planning-md/3
id: review-result:adversary-blockperf-pass-1
kind: review-result
status: active
title: Adversary pass 1 on story:matmul-block-kernel-throughput
relations:
- reviews: story:matmul-block-kernel-throughput
revision: 1
---
Only my files differ, so formatting all is safe.
unit: story:matmul-block-kernel-throughput
verdict: red
cases: executed 1210→1213, red 3
origin: introduced
wrote-outside-worktree: <wave-root>/blockperf/adv1-scratch/ (base/, mut/, bin/, orig/, mutants/, mutant-logs/, probe/, default-unchanged/, *.sh, *.log, mlp_sweep_b8192_h4096.mspec)
needs-coordinator: decide whether ADV-BLOCKPERF-1/2/3 get fixed in this unit or accepted. Only a hand-written profile or manifest reaches them; no real device does.

The verdict is red because three cases I added fail on code this unit introduced. All three are warnings, none a blocker. The kernel itself holds up: same bits at every boundary, 13 % faster, default compile unchanged.

**1. Diff stat.** `git diff --stat` is empty: nothing tracked changed. `git status --short`:
```
?? benchmarks/raw/2026-10-25-blockperf/adversary/   (47 files, 236K)
?? crates/modelspec-cli/tests/adversary_blockperf_compile.rs
?? crates/modelspec-core/tests/adversary_blockperf_limits.rs
?? crates/modelspec-runtime/tests/adversary_blockperf_device.rs
```

**2. Red cases.** Each is `#[ignore]`d with its finding id.

| id | test | red output |
|---|---|---|
| ADV-BLOCKPERF-1 | `a_manifest_breaking_the_matmul_limits_invariants_is_refused` | resealed limits {1024, 8192} and {1024, 0} are accepted by `read`; expected E-ARTIFACT-003 |
| ADV-BLOCKPERF-2 | `a_profile_below_128_invocations_is_not_reported_as_a_compiler_bug` | `E-INTERNAL-002: compiler bug: the execution plan does not verify / kernel ... has 128 invocations per workgroup, outside 1..=64`. The base binary exits 0 on the same input. |
| ADV-BLOCKPERF-3 | `no_artifact_records_matmul_limits_below_the_vulkan_minimum` | exit 0, manifest records `{"invocations":1024,"shared_bytes":8192}` |

Raw records: `case-adv1-limits-invariants.txt`, `case-adv2-adv3-compile.txt`.

**3. Suite run**
- **Host suites:** core, cli and runtime with the `vulkan` feature: 1213 passed, 0 failed, 125 ignored. My host cases add 3 executed and 6 ignored.
- **Device tests with validation layers on:** my 3 tests, and 5 existing device test files: pipeline statistics, profile wired, the two 256-tile files, plan sites. All green, 0 validation messages. Records: `case-device-validation.txt`, `case-validation-suites.txt`.
- **Checks:** `fmt --check` (only my files needed formatting), `clippy -D warnings`, `docs generate --check` and `ess specify validate` all exit 0.
- **D16, compile without a profile:** byte-identical between base and head for the sweep, mnist_mlp and linear_regression, with no `matmul_limits` in any manifest (`case-d16-default-unchanged.txt`).

**Attack 4: measurement.** GPU alone: the apps csv list only our own binaries. Clocks were never locked.

Profile A/B, median of 3 interleaved rounds (`profile/r{1,2,3}-{base,head}.txt`):

| | site 0 (µs) | site 14 (µs) | steady (µs) |
|---|---|---|---|
| base | 2926.1 | 3631.6 | 8132.6 |
| head | 2672.1 (−8.7 %, claim −8.9 %) | 2827.9 (−22.1 %, claim −22.6 %) | 7006.1 (−13.9 %) |

mlp_sweep b8192 h4096, median of the bench's 3 timed runs per round (`sweep-hot/`, `sweep-cool/`):

| series | base r1/r2/r3 | head r1/r2/r3 | head vs base | head / 7,290.5 |
|---|---|---|---|---|
| hot (83–86 °C peak) | 8338.9 / 8591.7 / 8461.2 | 7335.0 / 7475.3 / 7376.9 | −12.8 % | 1.012 |
| cool (each run started at ≤50 °C, 210 MHz, P8) | 8875.8 / 9170.6 / 9092.7 | 7894.7 / 7652.5 / 8247.1 | −13.2 % | 1.083 |

The 7,504 median is not flattered by heat: hot runs are faster than cool ones, and the speedup holds at about −13 % in both. The ratio to 7,290.5 depends on the GPU's state (1.012 hot against 1.083 starting cold). I don't know which state the 7,290.5 reference was measured in. The clock went back an hour (summer time ended) during the cool series, so its timestamps step from 02:59 to 02:01.

**Attack 8: mutants** (`case-mutants.txt`)

| mutant | caught by |
|---|---|
| M1 b stride 128 not 132 | killed on host: `the_block_stages_k_major...`, the `adversary_tile256_rule` and `adversary_tile256_spirv` files, 2 emit lib tests |
| M2 3 vec4 reads | survives the whole host suite; killed on device by `vulkan_matmul_tile256_same_bits`, `vulkan_fma_oracle`, `adversary_tile256_device` and mine |
| M3 E-ARTIFACT-005 check dropped | killed: `artifact_matmul_limits` (2 tests), `vulkan_profile_wired` |
| M4 limits not hashed | survives the implementor's suite; killed only by my hash test |

**4. Other judgement findings**
- A failing statistics query aborts the whole profiled run (`?` at `runner.rs:527`).
- The `--profile` row in cli.md (line 191, generated from the doc comment at `train.rs:207`) does not mention the `profile pipeline` lines. DESIGN.md and CHANGELOG do.

**5. What I could not break**
- Same bits as the oracle and the 128 tile: 17 boundary shapes × 4 flag sets, 251,184,240 elements.
- Signed zeros and cancellation: the block matches the tile bit for bit. The device flushes subnormals to zero, which was already the case and falls in the documented tolerance class.
- Fast-math gives the same bits; its cache keys differ and module ids are shared.
- The hash covers `matmul_limits`. The generated schema keeps the field optional, with both invariants as `minimum`.
- DESIGN §5.2 figures match diagnosis.md.
- NaN in the padding: I could not stage it without editing the emitter. Reading emit.rs, padding columns 128..131 are never read.

**6. Outside paths:** see the header. I cleaned both scratch build directories (`base/target` 405 MiB, `mut/target` 7.7 GiB) with `cargo clean`. The lease is ended.

```findings
- file: crates/modelspec-core/src/artifact.rs
  line: 488
  category: boundary
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: The reader accepts resealed matmul_limits below the spec invariants (invocations >= 128, shared_bytes >= 16384), e.g. {1024, 0} (ADV-BLOCKPERF-1).
- file: crates/modelspec-core/src/compile.rs
  line: 114
  category: correctness
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: A device profile below 128 invocations is reported as E-INTERNAL-002 compiler bug instead of a user diagnostic, where base compiled the same input (ADV-BLOCKPERF-2).
- file: crates/modelspec-core/src/compile.rs
  line: 116
  category: boundary
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: compile writes the profile's limits into the manifest unchecked, so an artifact records shared_bytes 8192, below the 16384 invariant (ADV-BLOCKPERF-3).
- file: crates/modelspec-core/src/artifact.rs
  line: 1469
  category: test-quality
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: No implementor test catches matmul_limits being left out of the manifest hash (mutant M4 survives; only adversary_blockperf_limits kills it).
- file: crates/modelspec-runtime/src/spirv/emit.rs
  line: 1710
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: Loading one a vec4 instead of two per k (mutant M2) passes the whole host suite and is caught only by device tests.
- file: crates/modelspec-runtime/src/vulkan/runner.rs
  line: 527
  category: correctness
  severity: note
  verdict: NEEDS-CHANGE
  origin: introduced
  message: 'An error from the pipeline statistics query aborts the whole profiled run instead of degrading to "profile pipeline statistics: not reported".'
- file: crates/modelspec-cli/src/train.rs
  line: 207
  category: docs
  severity: note
  verdict: NEEDS-CHANGE
  origin: introduced
  message: The --profile doc (cli.md line 191) omits the profile pipeline lines that DESIGN.md and CHANGELOG describe.
- file: benchmarks/raw/2026-10-25-blockperf/adversary/sweep-cool/r1-head.txt
  line: 1
  category: measurement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: The speedup reproduces at -12.8 % hot and -13.2 % cool, but head/PyTorch is 1.012 hot and 1.083 cold, so the 1.029 claim depends on GPU state.
```
