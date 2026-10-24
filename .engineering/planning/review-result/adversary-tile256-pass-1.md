---
format: aep.planning-md/3
id: review-result:adversary-tile256-pass-1
kind: review-result
status: active
title: Adversary pass 1 on matmul-tile-from-device-invocation-limit
relations:
- reviews: story:matmul-tile-from-device-invocation-limit
revision: 1
---
Now I'll end the lease before I report.
unit: story:matmul-tile-from-device-invocation-limit
verdict: red
cases: executed 1658→1662, red 4
origin: introduced
wrote-outside-worktree: <wave-root>/tile256/adv1-scratch (base copy 998M target, mutant copy 829M target, logs, scripts, profile, sources)
needs-coordinator: amend acceptance `default-unchanged` and `tile-from-profile` text or the code (ADV-T256-4, -1); `cargo clean --target-dir` the two scratch targets at merge

The findings cover head 00a2eed plus my untracked tests. The block kernel and the padding gave the CPU oracle's exact bits in every case I ran. The red cases are acceptance-text drift, test quality and edge states. Three of the red cases are states nobody reaches.

**1. `git --no-pager diff --stat`**: empty. Every file I touched is untracked and is a test:
- `crates/modelspec-core/tests/adversary_tile256_rule.rs`
- `crates/modelspec-runtime/tests/adversary_tile256_spirv.rs`
- `crates/modelspec-runtime/tests/adversary_tile256_device.rs`
- `crates/modelspec-cli/tests/adversary_tile256_artifact_hash.rs`

`?? .engineering/planning/story/matmul-block-kernel-throughput.md` is yours, and I did not touch it. `task ui` also wrote the gitignored `ui/node_modules` and `ui/dist`.

**2. Cases** (each run alone before the suite; reds kept with `#[ignore = "ADV-T256-n: …"]`)

| Case | Attack | Now | Output when written |
|---|---|---|---|
| rule `acceptance_text_an_output_of_128_blocks…` | 1 / acceptance | red, ignored | `[16384, 32] x [32, 1024]: tile workgroup_size 128 cols 64 … depth 16` (also 8192·10·4096, 2048·63·2048) |
| rule `the_staged_bytes_…_by_hand` | 4 | green | Literal bytes: block 16384/20480/16640/20736; wide 12288/14336/12544/14592; square 16384/18432/16640/18688. At 20,736 B the block is taken, at 20,735 B the wide tile |
| rule `every_tile_a_profile_gets_is_staged_within_its_memory…` | 4 | green | About 35k shapes × 4 limits × 7 memory sizes × 4 flags: always fits, the block never drops its padding, no `b` padded twice |
| rule `a_plan_scheduled_for_a_profile_passes_the_verifier…` | 5 | red, ignored | `invocations 64, x 64: kernel bf68bc3f… has 128 invocations per workgroup, outside 1..=64` (same at 1024/x 64) |
| spirv `the_block_module_has_the_arithmetic_of_the_wide_module…` | 2 | green | 7 shapes × 4 flags: arithmetic identical op for op to the 64×128 tile, all `NoContraction`, `LocalSize` 256, array length as computed by hand, `spirv-val` passes |
| spirv `the_padded_and_the_unpadded_square_module…` | 3 | green | Square tile at 48 KiB (padded) and 16 KiB (padding dropped): same arithmetic, lengths 4608/4672/4672 against 4096 |
| spirv `a_module_that_takes_a_value_from_the_profiles_workgroup_memory…` | 6 | red, ignored | `two modules of 81712 words and 81704 words under one cache key, ir_hash 1bd971d4…` |
| device `the_256_invocation_block_gives_the_cpu_oracle_bits…` | 1 | green | `validation true; 8 block shapes x 4 flags, 59058124 elements` |
| device `every_padded_layout_and_its_dropped_padding…` | 3 | green | `validation true; 14 shapes, 69085680 elements` (depths 2–64, both profiles) |
| cli `a_compile_with_a_128_invocation_profile…` | 6 | red, ignored | `left: "e14feec713ae…" right: "772ea38704fe…"` (the right value is the base 6203b9b release build) |

**3. Suite** (run after the cases existed):
- `cargo test --no-fail-fast --workspace`: 1229 passed, 0 failed, 110 ignored, exit 0.
- `cargo test --no-fail-fast -p modelspec-runtime --features vulkan`: 433 passed, 0 failed, 65 ignored, exit 0.
- My files account for 4 executed cases, which gives "before" 1658.
- `cargo fmt --check`: clean. Clippy `-D warnings` on core, runtime (vulkan) and cli: exit 0.

**4. Findings**

| id | file:line | verdict | severity | what reaches it |
|---|---|---|---|---|
| ADV-T256-4 | DESIGN/story `default-unchanged`; measured at `matmul_transposed`-padded `emit.rs` | CONFIRMED | warning | `compile --device-profile` with 128 invocations: the hash differs from base for all 5 sources, linear_regression included. Only the module of the transposed matmul differs, under the same module id. You accepted this; the acceptance text still says otherwise. |
| ADV-T256-1 | story `tile-from-profile` | CONFIRMED | note | The rule also requires `k ≥ 64`, as DESIGN § 5.2 says; the acceptance text omits it. |
| ADV-T256-5 | `adversary_transposed_spirv.rs:265` | CONFIRMED | warning | A mutant in a scratch copy with the padded `a` stride 8 wider passes all 14 tests of the unit (`matmul_tile_from_profile` 6, `matmul_tile256_spirv` 3, `adversary_transposed_spirv` 1, emit unit tests 4). Only my literal-value cases catch it (`mutant.log`). The rewritten assertion compares the emitter's output with `layout_within`, the function the emitter itself calls. |
| ADV-T256-6 | `matmul_tile256_spirv.rs:147` | CONFIRMED | note | It compares the head build against the head build. Its header claims `compile` with a 128 profile "writes the same artifact", and it passes while ADV-T256-4 is red. |
| ADV-T256-3 | `emit.rs:235` | INFEASIBLE | note | The matmul padding depends on the profile's workgroup memory, but `ir_hash` does not record it, unlike the loss sum's ` chunk`. It is reached only by two profiles with the same vendor, device and driver but different memory, which I constructed. |
| ADV-T256-2 | `schedule.rs:313` | INFEASIBLE | note | The doc says the result passes `verify_plan_for`. It does not for a profile under 128 invocations, and no conformant device reports that. |
| ADV-T256-7 | `kernel_plan.rs:530` | CONFIRMED | note | The substring check still passes while schedule.rs picks tiles from `MatMulLimits`, which is taken from a profile. |
| ADV-T256-8 | `kernel.rs:328` | CONFIRMED | note | Dead mechanism: `for_profile`, the `Some` path of `for_limits`, `schedule_for`, `verify_plan_for`, `verify_recorded_back_to_back_for` and `MatMulLimits::of`, and with them `MATMUL_LARGE_BLOCK`, are reached only from tests. `artifact.rs:442` caps every loaded artifact at 128 invocations, so a block artifact is refused on every device. Clippy flags none of it because the items are `pub`. |
| ADV-T256-9 | `spec/domains/artifact.yaml:206` | CONFIRMED | note | The narrowed sentence is weaker than the truth: every shipped artifact's matmul workgroup size is also a constant. Suggested wording: "workgroup sizes are fixed in the plan at compile time; no shipped command reads them from a DeviceProfile, and no reduction's depends on one". |
| ADV-T256-10 | `DESIGN.md:275` | CONFIRMED | note | § 5.2 states the 6,052/5,774 and 5,024/4,707 µs measurement twice in one paragraph. |

**5. Attacked and not broken**
- The block at partial blocks and partial tiles, the prefetch, and the masking of a short last tile: correct bits.
- Accumulation order, both against the wide tile and padded against unpadded: unchanged.
- No double padding, because `b_pad` is nonzero only on one-column tiles.
- A tile one byte short of the memory falls to the next tile; nothing is refused.
- Goldens: `git diff 6203b9b...HEAD -- goldens` is empty. A fresh `ir --stage kernel` of all 5 sources is byte-equal to the base build.
- Verifier cap: tested in the implementor's suite.
- Measurements: every quoted median matches the raw files. 4.8 %, 6.7 % and 8.8 % are correct, and 12,414→12,314 and 9,305→9,226 match `median_us_per_step`. One aside: the bench median of padded+block at 11,879 µs is lower than padonly's 12,314, which the profile does not show. It is within the 5 % spread.
- The 10 device failures: with `task ui` built, all 8 observatory UI cases pass. ADV-FM-1 and ADV-FM-2 still fail and also fail on the base copy, so neither belongs to this unit. That makes 37/40 pass; the third failure is my ADV-T256-4. The `train_fast_math` sweep CPU case ran for 13+ minutes in debug, so I stopped it and reran those binaries in release: 17/17 ok.

**6. Written outside the worktree**:
- The scratch directory `<wave-root>/tile256/adv1-scratch`, holding:
  - `base/` and `mutant/`, each with its own `target/`
  - `out/`, `src/`, `profile128.json`
  - `base-build.log`, `cli-ignored.log`, `cli-ignored-release.log`, `mutant.log`, `suite.log`
  - `compare.sh`, `cli-ignored.sh`, `cli-ignored-release.sh`, `mutant.sh`, `suite.sh`
- Tests also wrote under `<worktree>/target/tmp`.

**7.**
```findings
- file: crates/modelspec-cli/tests/adversary_tile256_artifact_hash.rs
  line: 65
  category: correctness
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: compile with a 128-invocation profile no longer writes the base artifact hash for any source with a transposed matmul, against acceptance default-unchanged
- file: crates/modelspec-core/tests/adversary_tile256_rule.rs
  line: 64
  category: docs
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the acceptance text of tile-from-profile omits the k >= 64 bound the rule and DESIGN apply
- file: crates/modelspec-runtime/tests/adversary_transposed_spirv.rs
  line: 265
  category: test-quality
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: no test of the unit pins a padded stride or staged byte count to a literal, so a mutant with a wider padded a stride passes all of them
- file: crates/modelspec-runtime/tests/matmul_tile256_spirv.rs
  line: 147
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the default-unchanged test compares the head build with itself and passes while the artifact hash differs from base
- file: crates/modelspec-runtime/src/spirv/emit.rs
  line: 235
  category: determinism
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: the matmul module's padding depends on the profile's workgroup memory but its ir_hash does not record it, unlike the loss sum's chunk
- file: crates/modelspec-core/src/schedule.rs
  line: 313
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: schedule_for under a profile below 128 invocations yields a plan verify_plan_for of that profile refuses, contrary to its doc
- file: crates/modelspec-core/tests/kernel_plan.rs
  line: 530
  category: test-quality
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the DeviceProfile substring check still passes while schedule.rs selects tiles from MatMulLimits, a projection of a profile
- file: crates/modelspec-core/src/ir/kernel.rs
  line: 328
  category: correctness
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the profile path and the 256-invocation block are reached only from tests, and the artifact reader caps every loaded plan at 128
- file: spec/domains/artifact.yaml
  line: 206
  category: docs
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the narrowed determinism sentence is weaker than the truth, since every shipped matmul workgroup size is also a compile-time constant
- file: DESIGN.md
  line: 275
  category: docs
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: section 5.2 states the block measurement twice in one paragraph
```
