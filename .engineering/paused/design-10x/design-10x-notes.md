# design-10x.patch notes

Base: `DESIGN.md` of the `modelspec-plan-ui` tree, sha256 `06dd0773d541f28246568306ff1469c7ad5c0d5a6074d49cdb3b7f592884977a`, revision 0.7. Patch sets revision 0.8.

## Apply check

`git apply --check` did not run. The session's worktree hook refuses any git command aimed at a tree other than `modelspec-highlight`, whether it uses `-C` or runs from a scratch directory. In its place: `patch --dry-run -p1` against a byte-identical copy of the base (`cmp` matched) exited 0. Run `git -C <plan-ui tree> apply --check design-10x.patch` once to confirm.

## What changed

| section | change |
|---|---|
| header | revision 0.8 entry |
| § 1 M2 row | whole-run execution (§ 6.3), ADR-007 gate |
| § 4 | GPU results checked within tolerance; § 5.4 adds the bit-for-bit checks |
| § 5.2 M2 | no split-K in `deterministic` mode, named epilogue fusions, operand gather in the matmul load |
| § 5.4 (new) | which GPU results are compared bit for bit and which within tolerance |
| § 6, § 6.1 | no per-step block from M2; bit-for-bit exception |
| § 6.3 (new) | resident state, device gather, parameter ping-pong, chunked recording, read-back, K1–K4 schedule, 2-dispatch fallback, budget table |
| § 9 | one sentence pointing at ADR-007; gates and tolerances unchanged |
| § 12 | ADR-007, proposed |

## Calls taken

- ADR-007 is "proposed". DESIGN.md gives no rule that lets an author accept an ADR. Its status line reads "M1 onwards proposed".
- Bit for bit is added on top of gate 5. It replaces nothing. It holds only in `deterministic` mode and only when the device reports `shaderRoundingModeRTEFloat32`, `shaderDenormPreserveFloat32` and `shaderSignedZeroInfNanPreserveFloat32` (invariant 6). Basis: `ops.rs:273` folds the `mse` gradient scale into a compile-time `Scalar`, so no gradient goes through `OpFDiv`.
- The gated configuration has `deterministic` on and `fast_math` off, with no FMA. This matches the baseline (deterministic, TF32 off).

## Open, not settled by the patch

1. § 9 gate 5 sets no numeric tolerance, only "documented tolerances". The epic's "weights within the tolerance of § 9" has no number behind it. The patch adds no number.
2. I don't know where the PyTorch baseline kept its data set (device or host). The verification report doesn't say. ADR-007 leaves ModelSpec's upload out of the timed region. If PyTorch gathered from host memory inside its timed loop, the two timings don't measure the same work. Check `torch_bench.py`.
3. Nobody has checked whether the RTX 3090 driver reports the three float-control features. There were no device runs. The M1 `inspect-device` reports them.
4. The Vulkan precision claims (add, sub and mul correctly rounded, comparisons exact, `OpFDiv` 2.5 ULP, `NoContraction`, default rounding implementation-defined, denormals may flush) come from Reference 1, docs.vulkan.org `spirvenv`, read 2026-10-21. "Vulkan does not guarantee that the workgroups of one dispatch run concurrently" is from memory and was not re-read.

## Estimates in the patch (not measurements)

1. Per-step budgets: K1 12 µs, K2 3 µs, K3 3 µs, K4 12 µs.
2. Dispatch boundaries: 3 µs each, 12 µs for four.
3. Host recording, submission and waits on the critical path: 2 µs per step.
4. Loss sum on reported steps: 0.5 µs amortized. Read-back at the end: 0.5 µs amortized.
5. Reserve 6.9 µs. This is the remainder of the estimates above, not an independent figure.
6. "About 37 of 1,800" reported steps. The exact count depends on how § 8 indexes steps.
7. Usable f32 peak without FMA, about 17.8 TFLOP/s: half of the vendor's 35.6 TFLOP/s, which was not read from a file.
8. K1's 784-addition chain takes about 1.8 µs, assuming 4 cycles per addition at 1.7 GHz.
9. Hypothesis, not measured: the dispatch boundaries are the largest unknown, and the 2-dispatch recompute schedule keeps the bit-for-bit class.
10. Derived from the estimates: K1 and K4 run at about 0.84 TFLOP/s.

Computed from shapes or read from files (not estimates): 20.45 MFLOP per step, 10.04, 0.13 and 0.26 MFLOP per kernel, 0.256 MFLOP per recompute workgroup, 394 GFLOP/s needed. Byte counts: 188,160,000 images, 720,000 permutation, 313,600 `X_b`, 200,704 `W1`, 203,264 parameters. The gate is 51.9 µs and 1,927,410 images/s. The baseline is 519 µs, 192,741 images/s, 933 µs, accuracy 0.9088, driver 610.57.04 and PyTorch `2.14.1+cu130`.
