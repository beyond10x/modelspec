---
format: aep.planning-md/3
id: review-result:adversary-m0-reverse-mode-autodiff-pass-1
kind: review-result
status: active
title: Adversary pass 1 on m0-reverse-mode-autodiff
relations:
- reviews: story:m0-reverse-mode-autodiff
revision: 1
---
unit: story:m0-reverse-mode-autodiff, uncommitted working tree of ~/.local/state/worktree/trees/b10x/modelspec/modelspec-impl-m0-reverse-mode-autodiff (base 49e2dd4)
verdict: CONFIRMED (2 test-suite gaps a mutant gets through); 5 INFEASIBLE refusals on builder-made programs; no wrong gradient found
cases: executed 344→352, red 5
origin: introduced 8 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 directory tree, ~/.cache/modelspec-wave-m0/m0-reverse-mode-autodiff/scratch/adversary/ (180M in total, of which scratch target/ is 179M)
needs-coordinator: whether a parameter the loss does not use gets a gradient output (finding 8)

**1. Diff.** `git diff --stat` shows only the implementor's 4 implementation files: autodiff.rs, ir/training.rs, ir/verify.rs, ops.rs. `cmp` against copies taken before my work shows all of them unchanged. My only changes are 2 new untracked test files:
- `crates/modelspec-core/tests/autodiff_adversary.rs` (204 lines)
- `crates/modelspec-runtime/tests/gradients_adversary.rs` (193 lines)

No non-test path was touched.

**2. Cases, each run alone before the suite.** Line numbers are from before I ran `rustfmt` on my two files.

| Case | Exit | Output (verbatim) |
|---|---|---|
| `verifier_refuses_a_gradient_that_does_not_depend_on_the_loss` | 101 | `verify_training accepted the parameter read %0 as the gradient of #0: Ok(())` |
| `verifier_refuses_a_gradient_node_whose_only_loss_dependency_is_a_mistagged_forward_node` | 101 | `verify_training accepted forward matmul %3 tagged Gradient because its operand %0 was tagged Loss: Ok(())` |
| `verifier_refuses_a_forward_tag_on_a_node_that_depends_on_the_seed` | 101 | `verify_training accepted gradient %11, computed from the backward seed, tagged Forward: Ok(())` |
| `verifier_refuses_a_missing_gradient_for_a_loss_reachable_parameter` | 101 | `verify_training accepted a training program with no gradient for loss-reachable #0: Ok(())` |
| `differentiate_does_not_panic_on_the_largest_value_ids` | 101 | `panicked at crates/modelspec-core/src/autodiff.rs:169:49: attempt to add with overflow` |
| `no_adjoint_is_built_for_an_input_operand` | 0 | Passes on the tree. Red on mutants (see below). |
| `shared_value_used_three_times` | 0 | Passes on the tree. Red on a mutant (see below). |
| `parameter_read_three_times` | 0 | Passes on the tree. Red on a mutant (see below). |

Mutation runs, on the scratch copy only, script `mutate.sh`:

| Mutant | Implementor's `gradients` (42 cases) | My catching case |
|---|---|---|
| `Sink::sum` keeps only 2 contributions (`rest.iter().take(1)`) | exit 0, survives | gradients_adversary exit 101: `f64 error 3.333333136905869e-1` |
| `Add` ignores `wanted` | exit 0, survives | `no_adjoint…` exit 101 |
| `Sub` ignores `wanted[0]` | exit 0, survives | `no_adjoint…` exit 101 |
| `Sub` ignores `wanted[1]` | exit 0, survives | `no_adjoint…` exit 101 |
| `Mul` ignores `wanted[0]` | exit 0, survives | `no_adjoint…` exit 101 |
| `Mul` ignores `wanted[1]` | exit 0, survives | `no_adjoint…` exit 101: `%11 = Mul[ValueId(9), ValueId(0)]` |
| `MatMul` ignores `wanted[0]` | exit 101 | (control: already caught) |
| `Mse` ignores `wanted[1]` | exit 101 | (control: already caught) |
| `unbroadcast` without the rank check on `c` | exit 0 | exit 0. Equivalent: only adds a rank-0 ReduceSum, so not a finding |
| verifier: seed not counted as on the loss | exit 101 | (control: already caught) |

**3. Suite run, after the cases existed.**
- `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot task ci` exited 201. Build, `fmt --check` and clippy passed. It stopped at `test result: FAILED. 1 passed; 5 failed` in `autodiff_adversary`.
- The same with `cargo test --workspace --no-fail-fast` exited 101: 347 passed, 5 failed, 352 run. `gradients` and `gradients_adversary` were green.

**4. Findings**

| # | file:line | Verdict | Origin | Measured | What reaches it |
|---|---|---|---|---|---|
| 1 | crates/modelspec-core/src/ir/verify.rs:321 | INFEASIBLE | introduced | The depends-on-loss check accepts any node tagged Loss as "the loss". A forward ParamRead tagged Loss lets the forward matmul pass as a Gradient node. | Only one caller, `differentiate` (autodiff.rs:125), and it tags only the loss and the seed as Loss. I built this program by hand. |
| 2 | crates/modelspec-core/src/ir/verify.rs:301 | INFEASIBLE | introduced | A gradient value is never checked to be tagged Gradient or to depend on the loss. `gradients[w] = ParamRead(w)` passes. | Built by hand. Every rule in `differentiate` uses `g`. |
| 3 | crates/modelspec-core/src/ir/verify.rs:323 | INFEASIBLE | introduced | A node computed from the seed may be tagged Forward. | Built by hand. Sink tagging is correct, and the implementor's `assert_structure` checks it. |
| 4 | crates/modelspec-core/src/ir/verify.rs:272 | INFEASIBLE | introduced | A parameter the loss uses can have no gradient at all: the map and the outputs agree with each other, not with the graph. | Built by hand. |
| 5 | crates/modelspec-core/src/autodiff.rs:169 | INFEASIBLE | introduced | `n.id.0 + 1` overflows on a verified program whose largest id is u32::MAX: a panic, not an `AutodiffError`. | The builder allocates ids in sequence (tensor.rs `define`), so only a hand-made program gets here. |
| 6 | crates/modelspec-runtime/tests/gradients.rs:551 | CONFIRMED | introduced | No case has a value used 3 or more times. The 2-contribution mutant passes all 42 cases. | DESIGN.md § 4 "accumulate every incoming adjoint exactly once"; the brief asked for this case. |
| 7 | crates/modelspec-runtime/tests/gradients.rs:516 | CONFIRMED | introduced | "Nothing for inputs" is pinned for matmul and mse only. 5 Add/Sub/Mul mutants that build dead adjoints for inputs pass. | The story says "none for inputs or targets". Any input operand of add, sub or mul reaches it. |
| 8 | crates/modelspec-core/src/autodiff.rs:116 | CONFIRMED | introduced | The outputs carry gradients only for parameters the loss uses. The story's "Lands in" says "a gradient output per ParamId". | story:m0-cpu-sgd-training adds "`p_next` … one output per parameter" after DCE, and will meet parameters with no gradient. |

**5. Attacked and could not break** (probes in scratch `tree/crates/modelspec-runtime/tests/probes.rs`, 20 cases, exit 0)
- matmul with the same value as both operands ([3,3]); f64 error 7.4e-9.
- Shapes with a dimension of 1: [3,1]@[1,2], [1,4]@[4,1], a two-layer graph of [1,1] tensors, add of [1] and a rank-0 value.
- mean and sum of a rank-0 value; the loss being the ParamRead itself; mse with a trainable target and an input prediction; a transpose chain; a mixed rank-0 chain.
- A parameter the loss does not use, between two it does: gradient keys are [0, 2], outputs `[loss, g0, g2]`, both checked against finite differences.
- relu at ±1e-30, ±1e-7 and ±the smallest f32 subnormal: gradients exactly [1,0,1,0,1,0] in f32 and f64.
- f32 constants for 1/n and 2/n: in f64 the error is 2.98e-8 at n=3, 4.47e-8 at n=7, 2.51e-8 at n=91 and 4.75e-8 at n=1000. That is about 17 times under 1e-6. It does set the floor of the f64 error (most cases measure ≈3e-8, against 1e-12 for a rule with no constant).
- Determinism: two `differentiate` runs give equal `TrainingProgram`s in all 20 probes. The code uses only `BTreeMap`.
- `AutodiffError::Rule` and `::Training`: I found no verified forward program that reaches either; every rule emits operands of the right types.
- Each wrong rule the brief names is caught, by the implementor's own 27 mutants or by my 3 controls: a dropped transpose, a missing factor 2, a missing 1/n, a sign flip.

**6. Paths written outside the worktree**, all under the assigned ~/.cache/modelspec-wave-m0/m0-reverse-mode-autodiff/scratch/adversary/:
- `tree/`: a copy of the worktree without `.git`, `.engineering` or its `target/`; its own `target/` is 179M.
- `pristine/`, `logs/`, `mutate.sh`, `mutate.out`, `probe_cases.rs.txt`.

The worktree's own `target/` went from 374M to 418M with my builds.

```findings
- file: crates/modelspec-core/src/ir/verify.rs
  line: 321
  category: judgement
  severity: warning
  verdict: INFEASIBLE
  origin: introduced
  message: the depends-on-loss check treats every Loss-tagged node as the loss, so a mistagged forward node makes forward nodes pass as Gradient nodes
- file: crates/modelspec-core/src/ir/verify.rs
  line: 301
  category: judgement
  severity: warning
  verdict: INFEASIBLE
  origin: introduced
  message: a gradient value is not required to be tagged Gradient or to depend on the loss; the parameter's own ParamRead is accepted as its gradient
- file: crates/modelspec-core/src/ir/verify.rs
  line: 323
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: a node computed from the backward seed may be tagged Forward without refusal
- file: crates/modelspec-core/src/ir/verify.rs
  line: 272
  category: judgement
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: a training program with no gradient for a loss-reachable parameter verifies
- file: crates/modelspec-core/src/autodiff.rs
  line: 169
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: id allocation overflows and panics on a verified program whose largest value id is u32::MAX instead of returning an error
- file: crates/modelspec-runtime/tests/gradients.rs
  line: 551
  category: mutant
  severity: warning
  verdict: CONFIRMED
  origin: introduced
  message: no case uses a value three times, so a sum that keeps only two adjoint contributions passes all 42 cases
- file: crates/modelspec-runtime/tests/gradients.rs
  line: 516
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: none-for-inputs is pinned only for matmul and mse; five Add/Sub/Mul mutants that build dead adjoints for input operands pass
- file: crates/modelspec-core/src/autodiff.rs
  line: 116
  category: contract-drift
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: outputs carry gradients only for loss-reachable parameters while the story says a gradient output per ParamId and the SGD story adds p_next per parameter
```
