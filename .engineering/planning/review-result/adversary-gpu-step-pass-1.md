---
format: aep.planning-md/3
id: review-result:adversary-gpu-step-pass-1
kind: review-result
status: active
title: Adversary pass 1 on bert-vulkan-kernels-attention-loss-adamw
relations:
- reviews: story:bert-vulkan-kernels-attention-loss-adamw
revision: 1
---
unit: story:bert-vulkan-kernels-attention-loss-adamw (worktree wt-7d2aa11d7db8, unit commit d99c0ce)
verdict: findings 6 (blocker 0, warning 1, note 5)
cases: executed 2638→2640, red 3 (device-ignored cases: 17 run with `--include-ignored` per file)
origin: introduced 6 / pre-existing 0 / undecided 0
wrote-outside-worktree: `<wave-root>/gpu-step/adv1-scratch/` (probe, mutant copy with its target cleaned, logs, scripts)
needs-coordinator: f2 (whether the run tolerance of DESIGN § 9 covers the attention key biases, whose exact gradient is zero), and the two departures from D73 (n4, n5)

## Cases added

| file | cases | now |
|---|---|---|
| `crates/modelspec-runtime/tests/adversary_gpu_step.rs` | 5 device cases: mask and head edges (one-key, all-ones, prefix and suffix masks, heads 1/2/4/8, S = 1, d = 1; worst 7.153e-7); uniform attention bit-equal (pins the summation order); cross-entropy ties, labels 0 and V-1, one weight, fractional weights, V = 8192; `Sqrt` and `Div` within 1 ulp over 4096 wide-range inputs (0 and 43 of 4096 not bit-equal); f1 Div overflow | 4 green, 1 red (f1) |
| `crates/modelspec-cli/tests/adversary_gpu_step_cli.rs` | 12: attention over 512 keys is E-TARGET-011 at its call (S = 512 compiles; 513, 1024, 2×513 refused), `examples/bert_tiny.mspec` schedules the eight kernels (both in the default suite); f3 help text; f2 run tolerance at 80, 160 and 800 steps; 8 device cases (profile and stats, dropout SGD with resume and divergence, drift at 10/20/40 steps, 3 runs plus fast-math, eval every 3 steps, a diverging twin, dense data, 8 seeds) | 10 green, 2 red (f2, f3) |

## Red cases (verbatim)

```
f1: cpu [inf, -inf, inf, inf]
    gpu [NaN, NaN, NaN, inf]
    panicked at crates/modelspec-runtime/tests/adversary_gpu_step.rs:590:9: assertion `left == right` failed: [0]: gpu NaN cpu inf  left: 2147483647 right: 2139095040
f3: panicked at crates/modelspec-cli/tests/adversary_gpu_step_cli.rs:312:5:
    the --profile help names no refusal:
              With --backend vulkan, time every dispatch with GPU timestamp queries and print one `profile dispatch` line per dispatch site, ...
f2: steps | max |g-c| | bound | ratio | parameter [i] | cpu | gpu | elements outside | worst ratio outside the key biases
    80 | 2.307e-4 | 1.000e-4 | 2.306 | @BertTiny.l1_bk [31] | 0.000044065036 | -0.00018666025 | 4 | 0.002
    160 | 2.017e-4 | 1.000e-4 | 2.017 | @BertTiny.l0_bk [4] | 0.000031045896 | -0.00017067326 | 3 | 0.003
    800 | 1.099e-4 | 1.000e-4 | 1.098 | @BertTiny.l1_bk [25] | 0.00003584877 | -0.00007402513 | 3 | 0.009
    panicked at crates/modelspec-cli/tests/adversary_gpu_step_cli.rs:527:5:
    "80 steps, param @BertTiny.l0_bk [6]: gpu -0.000100810896 cpu 0.000070023714", ...
```

## Suite run after the cases existed

| command | result | exit |
|---|---|---|
| `cargo fmt --check`; clippy workspace and runtime `--features vulkan`, `-D warnings` | clean | 0 |
| `cargo test --workspace --no-fail-fast` | 1903 passed, 0 failed, 180 ignored | 0 |
| `cargo test -p modelspec-runtime --features vulkan --no-fail-fast` | 737 passed, 0 failed, 120 ignored | 0 |
| device run, `--include-ignored` per `--test` file, `MODELSPEC_VK_VALIDATION=1` | 17 adversary cases plus the unit's device tests; 3 red (f1, f2, f3) | 101 |

An earlier run with `MODELSPEC_VK_VALIDATION=1` failed two pre-existing tests (`adversary2_inspect_pre_1_2`, `adversary2_create_device_old`) with "Unable to load create_debug_utils_messenger_ext" and a non-unwinding panic; the unit changes `vulkan/mod.rs` by a doc line and a `mod` line only, and the base was not run, so the origin is unknown and it is not a finding of this unit.

## Hypotheses

| # | probe | observation | verdict |
|---|---|---|---|
| drift | twin runs at 10 to 800 steps | bounded, not linear; the bound breaks by 80 steps, only in `l0_bk` and `l1_bk` | f2 |
| drift cause | kernels read and compared bit for bit | the same order as the CPU (uniform case bit-equal); AdamW amplifies rounding noise in the key biases, whose exact gradient is 0 | explains f2 |
| 512 keys | CLI at S = 512/513/1024/2×513, bert_tiny | clean E-TARGET-011 at the call; the limit is per sequence length; bert_tiny schedules | holds |
| masks, ties | runtime cases | within tolerance; cross-entropy 0.0 diff | holds |
| moments | stepwise.rs read; mutant e | persistent device buffers; AdamW with start > 0 refused; `--checkpoint`/`--resume` refused for adamw; `lr_t` from the CPU's `adamw.scalars` | holds |
| dropout | dropout_train CLI case; mutant g | stdout byte-identical, resume works, a constant token is killed | holds |
| E-TRAIN-001 | lr = 100, lr = 1e30 | the same step and stderr as the CPU; RunFailed on both | holds as probed |
| determinism | 3 runs; `--fast-math` | byte-identical; fast-math accepted with different bits, worst ratio 0.499 | holds |
| lifted refusals | the 7 updated tests read | A8 now checks compile and ir only; the diagnostics page does not name "over 512 keys" | no finding |
| `--profile` | CLI | one line, exit 1 as DESIGN § 8 says; `--stats` prints; `--help` silent | f3 |
| no device | enabled.rs read | no disable switch; default suite green; full `task ci` not run | holds as run |
| Div overflow | constructed inputs | NaN on the device, inf on the CPU; nothing reaches it | f1 |
| "bit-equal" Div | wide range | 43 of 4096 differ by 1 ulp; the docs allow 1 ulp | no finding |

## Findings

| id | file:line | severity / verdict / origin | what |
|---|---|---|---|
| f1 | `src/spirv/emit/attention.rs:297` (`quotient`) | note / INFEASIBLE / introduced | on overflow `q` is inf, the residual is -inf and `q + step` is NaN; nothing found reaches it |
| f2 | `DESIGN.md:598` (run tolerance, 0.53) | warning / CONFIRMED / introduced | the twin leaves 1e-4 in `l*_bk` by 80 steps (ratio 2.306); every other parameter stays at or under 0.009 of its bound |
| f3 | `crates/modelspec-cli/src/train.rs:270` | note / CONFIRMED / introduced | the `--profile` doc and the generated CLI page do not mention the stepwise refusal |
| n4 | `src/vulkan/stepwise.rs:547` | note / CONFIRMED / introduced | evaluation runs inside the step's command buffer with 0 extra host waits, against D73(d)'s one wait per evaluated step |
| n5 | `crates/modelspec-core/src/schedule.rs:1309` | note / CONFIRMED / introduced | cross-entropy has no 512-class cap; V = 8192 schedules and runs green, against D73(b) |
| f6 | `tests/vulkan_bert_attention_loss_adamw.rs:386` | note / CONFIRMED / introduced | the unit's suite survives mutants b and d3; the adversary's uniform and ulp cases kill them |

## Mutants (scratch copy, device, validation on)

| mutant | unit suite | adversary cases |
|---|---|---|
| a: no 1/√d scale | killed | killed |
| b: descending key order | survived | `uniform_attention_is_bit_equal`, drift cases |
| c: CE grad without the weight division | killed | killed |
| d: eps inside the sqrt | killed | killed |
| d2: `root` without the Newton step | survived | survived (the device sqrt is bit-equal on all 4096 inputs) |
| d3: `quotient` without the correction | survived | `sqrt_and_div_within_one_ulp` (797/4096 not bit-equal) |
| e: bias correction at step k | killed | killed |
| f: eval with dropout on | killed | killed |
| g: dropout token at step 0 | killed | killed |

## Attacked, not broken

The 512-key refusal; masks, ties and head edges; dropout tokens, resume and divergence; E-TRAIN-001 at the step the CPU names; three-run determinism; the fast-math run within the run tolerance; eval steps under `--eval-every 3`; the `--stats` counters.

```findings
- file: crates/modelspec-runtime/src/spirv/emit/attention.rs
  line: 297
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: an overflowing Div gives NaN on the device where the CPU gives inf, because the residual step computes inf - inf; no reachable input found
- file: DESIGN.md
  line: 598
  category: property
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: the stepwise twin leaves the run tolerance in its key biases by 80 steps (ratio 2.306 at l1_bk[31]), so the rule holds only for short runs
- file: crates/modelspec-cli/src/train.rs
  line: 270
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the --profile help and the generated CLI page do not mention that a stepwise run refuses --profile, as DESIGN section 8 does
- file: crates/modelspec-runtime/src/vulkan/stepwise.rs
  line: 547
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: held-out evaluation runs inside the step's command buffer with zero extra host waits, departing from D73(d)'s one host wait per evaluated step
- file: crates/modelspec-core/src/schedule.rs
  line: 1309
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: cross-entropy schedules with no 512-class cap (V=8192 runs), unlike D73(b)
- file: crates/modelspec-runtime/tests/vulkan_bert_attention_loss_adamw.rs
  line: 386
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the unit's suite survives the descending-key-order and uncorrected-Div mutants, which only the adversary cases kill
```
