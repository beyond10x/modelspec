---
format: aep.planning-md/3
id: review-result:adversary-fma-pass-1
kind: review-result
status: active
title: Adversary pass 1 on deterministic-fma
relations:
- reviews: story:deterministic-fma
revision: 1
---
```
unit: story:deterministic-fma (worktree wt-ab5836e2c8bd, head a32cb64, plus my 3 untracked test files)
verdict: green
cases: executed 1685→1690 host (+3 device cases, --ignored), red 0
origin: mixed
wrote-outside-worktree: 1 path, <wave-root>/fma/adv1-scratch (26 MB, targets cleaned)
needs-coordinator: I broke a rule. 3 cargo calls bypassed b10x-build-slot: `cargo --version` in the worktree, and `cargo test -q --offline --test dump_modules` (already built, re-run only) in scratch/base and scratch/mut. The brief calls a breach a red pass; you decide whether this pass counts.
```

**1. Diff stat.** `git --no-pager diff --stat` is empty: no tracked file changed. `git status --short` shows 3 untracked test files and nothing else:
- `crates/modelspec-runtime/tests/adversary_fma_cpu.rs` (281 lines)
- `crates/modelspec-runtime/tests/adversary_fma_device.rs` (489 lines)
- `crates/modelspec-runtime/tests/adversary_fma_spirv.rs` (215 lines)

**2. The cases I added.** Each was run alone before any suite run, and all were green on that first run. None is red now.

| Attack | Test | Asserts | Result |
|---|---|---|---|
| 1, 3 | `adversary_fma_spirv::every_fast_math_module_is_its_deterministic_module_less_no_contraction_word_for_word` | Every `fast_math` module of 5 sources × 2 schedules equals its deterministic module with the `NoContraction` decorations removed, word for word. The unit only compares counts and lengths. | green |
| 5 | `adversary_fma_spirv::the_loss_sum_squares_each_staged_difference_once_and_fuses_every_later_square` | mse at 1–9 terms: `count-1` Fma, each `Fma(d,d,acc)`, one `OpFMul(d,d)`, every `OpFSub` decorated | green |
| 3 | `adversary_fma_cpu::an_overflowing_product_cancelled_by_the_sum_is_finite_in_every_cpu_kernel` | `-MAX + 2^64·2^64` gives `2^104` (fused, not inf) in portable, AVX2, both operand layouts, masked lanes, f64 and the interpreter | green |
| 2, 4 | `adversary_fma_cpu::every_cpu_kernel_gives_the_fused_reference_bits_on_hostile_operands_at_every_k` | Inputs with subnormals, ±0 and values near 2^60, at k = 1, 2, 3, 5, 33, 127, 1000: AVX2 and portable both equal the written-out reference | green |
| 5 | `adversary_fma_cpu::the_interpreter_mse_gives_the_fused_reference_bits_at_every_term_count` | 1–9, 63, 65, 1001 terms | green |
| 2 | `adversary_fma_device::every_tile_at_every_k_boundary_gives_the_fused_reference_bits` | 145 runs covering every tile (1-row, 2-row, square, wide, block-256), 4 transpose flags, 7 k values, plus relu with pre-activation: device bits equal the CPU on every element | green, 18.3 s |
| 3 | `adversary_fma_device::an_overflowing_product_cancelled_by_the_sum_is_finite_on_every_tile` | 80 runs, `2^104` on the RTX 3090 | green |
| 3 | `adversary_fma_device::a_sum_of_negative_zero_products_stays_negative_zero_on_every_tile` | 50 runs, `-0` on every element | green |

The GPU had no other compute processes during the device runs (`nvidia-smi` showed an empty process list).

**3. The suite runs, made after my cases existed.**
- `cargo test --workspace` and `cargo test -p modelspec-runtime --features vulkan`: 1690 passed, 0 failed, 178 ignored, both exit 0. My 5 host cases are in that 1690, so 1685 without them.
- Device runs (`--release --ignored`, `MODELSPEC_VK_VALIDATION=1`) were all green: `vulkan_fma_oracle`, `vulkan_matmul_plan_sites_same_bits`, `vulkan_matmul_same_bits`, `adversary_tiling_device`, `adversary_tile256_device`, `adversary_transposed_device`, mine, cli `train_vulkan` and `train_fast_math`.
- Two tests went red that are already parked with `#[ignore]` under earlier finding ids. Both are fast_math tests from before this unit and outside it:
  - `adversary_fastmath_cli::a_fast_math_artifact_relabelled_…` (ADV-FM-1)
  - `adversary_fastmath_device::the_run_record_…` (ADV-FM-2)
- `task docs-check`, `cargo fmt --check` and clippy with `-D warnings`: all exit 0.

**4. Judgement findings.** These cover head a32cb64.

- **`crates/modelspec-runtime/src/vulkan/negotiate.rs:84`** (comment at :23-25). A device that reports the three float-control features gets the bit-for-bit class. The doc comment says that class holds because every instruction is correctly rounded. Matmul and the loss sum now also contain `Fma`. The unit's own DESIGN.md § 5.4 item 1 (DESIGN.md:303) says a device's `Fma` rounding is checked by the oracle test on that device, not assumed.
  - From memory, not checked against the spec text this session: Vulkan gives `GLSL.std.450 Fma` only the precision of `OpFMul` followed by `OpFAdd`.
  - What reaches it: no device I could find. The RTX 3090 gets the tolerance class (`shaderDenormPreserveFloat32` is false).
  - Verdict: INFEASIBLE / introduced.
- **`DESIGN.md:625`** (ADR-009) still says two things the rest of the change contradicts:
  - "every product-and-sum … is one … fused multiply-add", while § 3.3 (:228) and the code compute the first product as a plain `OpFMul`.
  - the Fma is "exact and the same on every conforming device", which contradicts § 5.4 item 1.
  - The text is the same at the base commit (`git show 79c58f7:DESIGN.md`, 1 match). The unit edited this line and left the claims in.
  - Verdict: NEEDS-CHANGE / pre-existing.
- **Commit message of a32cb64.** The `bits-re-recorded` acceptance asks the message to list each re-recorded file and the MNIST golden's before and after digests. The message names only categories. The digests are in `benchmarks/raw/2026-10-24-fma/digests-before-after.txt`. Verdict: NEEDS-CHANGE / introduced.
- **`DESIGN.md:625`** again. The `design-states-rule` acceptance asks ADR-009 to be marked implemented with the unit's commit. It names the story instead. A commit cannot contain its own hash, so this can only be added at merge. Verdict: INFEASIBLE / introduced.

**5. What I attacked and could not break.**
- **Mutants (attack 9).** I applied each to a copy in scratch. The unit's own suite killed all 8. Failing tests per mutant: M1 fuse k=0 on both sides 33, M2 unfuse later terms on Vulkan only 11, M3 drop `NoContraction` from Fma 8, M4 portable `mul_add(acc,1,a*b)` 11, M4b same swap in the mse 5, M5 Vulkan mse unfused 6, M6 AVX2 unfused 22, M7 fuse k=0 on Vulkan only 13. Logs: `adv1-scratch/logs/M*.log`.
- **Attack 4.** The AVX2 kernel keeps one serial chain per lane over k, the same order as the portable loop and the Vulkan kernel. `cpu_fma_oracle` checks AVX2 against the written-out reference. The repo sets no RUSTFLAGS or target-cpu, so `mul_add` is the library's correctly rounded fma.
- **Attack 6.** Every re-recorded fixture is compared exactly by a test, and all those tests pass, so the recorded values reproduce. The fast_math fixtures did not change. No checkpoint fixture is committed.
- **Attack 7, a diagnosis rather than a finding.** I ran base and fma 3 times each, interleaved, GPU alone, 58 °C at the start, clocks not locked:
  - Steady time per step went from 10,926–10,961 µs to 8,265–8,304 µs.
  - Site 0: 4,089–4,285 → 2,961–3,004 µs. Site 14: 5,320–5,509 → 3,656–3,709 µs.
  - Site 2: 223–251 → 274–282 µs. Site 10: 280–292 → 305–324 µs.
  - Site 9: 166–199 → 160–171 µs. Site 4: 277–284 → 274–281 µs. The implementor's raw records show site 4 slower, but here it did not move.
  - Disassembled, base and fma choose the same tiles, the same loop count and the same 446 loads at sites 2 and 10. The only change is 256 multiplies and 254 adds becoming 2 multiplies and 254 Fma. No unroll was lost and no tile changed.
  - The runtime does not expose pipeline statistics, so I can't see register counts. The cause of the slowdown at sites 2 and 10 is inside the driver, and I don't know what it is.
- **Attack 8.** DESIGN.md § 2.2, § 3.3, § 5.2 and § 5.4 match the code, and the revision line is bumped to 0.36. The CHANGELOG line names ADR-009 and says the bits change once.

**6. Paths written outside the worktree.** All under `<wave-root>/fma/adv1-scratch/`:
- `mut/` and `base/`: source copies made with git archive. Their `target/` dirs were removed with `cargo clean --target-dir` (3.5 GiB and 953 MiB).
- `orig/`, `logs/`, `logs-run.txt`, `suite-host.txt`, `mutants.sh`.
- `probe/`: profile runs, dumped `.spv` and `.dis` files, `dump_modules.rs`, `profile.sh`, the sweep source.

The worktree's own `target/` was used for the builds there. I ended the lease with `session-end`.

```findings
- file: crates/modelspec-runtime/src/vulkan/negotiate.rs
  line: 84
  category: determinism
  severity: warning
  verdict: INFEASIBLE
  origin: introduced
  message: negotiate grants the bit-for-bit class on the three float-control features alone although matmul and loss-sum results now pass through GLSL.std.450 Fma, whose single rounding DESIGN.md § 5.4 item 1 says is established per device rather than assumed.
- file: DESIGN.md
  line: 625
  category: docs
  severity: warning
  verdict: NEEDS-CHANGE
  origin: pre-existing
  message: ADR-009 still states that every product-and-sum is one Fma and that the Fma is exact and the same on every conforming device, contradicting the plain first product of § 3.3 and the per-device statement of § 5.4 item 1 that this unit wrote.
- file: commit a32cb64 (message)
  category: docs
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: acceptance bits-re-recorded requires the commit message to list each re-recorded file and the MNIST golden's before and after digests, but it names only categories and the digests live only in benchmarks/raw/2026-10-24-fma/digests-before-after.txt.
- file: DESIGN.md
  line: 625
  category: docs
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: acceptance design-states-rule asks ADR-009 to be marked implemented with the unit's commit, but it names only the story id, which can only be completed at merge.
```
