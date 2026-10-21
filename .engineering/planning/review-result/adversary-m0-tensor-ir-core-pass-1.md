---
format: aep.planning-md/3
id: review-result:adversary-m0-tensor-ir-core-pass-1
kind: review-result
status: active
title: Adversary pass 1 on m0-tensor-ir-core
relations:
- reviews: story:m0-tensor-ir-core
revision: 1
---
I made the suite red with 4 cases. All 4 are warnings and none is a blocker: no code in the tree builds the bad states yet.

```
unit: story:m0-tensor-ir-core, uncommitted working tree on base 1c33fe0 (modelspec-impl-m0-tensor-ir-core)
verdict: CONFIRMED (3 test-coverage gaps); 4 red verifier cases are INFEASIBLE (no producer yet)
cases: executed 29→36 (modelspec-core; 29 is the implementor's reported count), red 4
origin: introduced 7 / pre-existing 0 / undecided 0
wrote-outside-worktree: 1 directory (part 6)
needs-coordinator: yes. Should the verifier enforce the builder's uniqueness and input-list rules? Also, the 4 red cases keep `task ci` red until someone decides.
```

**1. Diff against 1c33fe0**

`git --no-pager diff --stat 1c33fe0` is unchanged from the implementor's (`print.rs`, `tensor.rs`, `verify.rs`, `ops.rs`). That diff is the implementor's change, not mine. My only addition is the new untracked test file `<worktree>/crates/modelspec-core/tests/tensor_ir_adversary.rs`. I changed no implementation file.

**2. Each case run alone** (`b10x-build-slot cargo test -p modelspec-core --test tensor_ir_adversary -- --exact <case>`)

| Case | Exit | What it asserts |
|---|---|---|
| `adv_verifier_rejects_two_parameters_with_one_canonical_path` | 101 | Two `ParamId`s with the path `M.a.w` are rejected. Got: "yet verify accepted" (panic at :54) |
| `adv_verifier_rejects_two_input_nodes_with_one_name` | 101 | Two `Input "x"` nodes are rejected. Got: "yet verify accepted" (panic at :68) |
| `adv_verifier_rejects_input_node_missing_from_inputs_list` | 101 | An `Input` node left out of `inputs` is rejected. Got: "inputs = [ValueId(0)]" (panic at :80) |
| `adv_verifier_rejects_inputs_list_naming_one_node_twice` | 101 | `inputs = [%0, %0, %1]` is rejected. Got: accepted (panic at :92) |
| `adv_relu_grad_rejects_operands_of_different_shapes` | 0 | Passes now. It fails if the shape check at `ops.rs:117` is dropped |
| `adv_matmul_rejects_rank3_rhs` | 0 | Passes now. It fails if the rank check at `ops.rs:126` is dropped |
| `adv_printer_pins_scalar_and_broadcast_scalar_lines` | 0 | Passes now. It fails if the `BroadcastScalar` brackets change at `print.rs:43` |

I tested each of the three mutants on a copy of the tree in scratch, never on the tree itself. In every run all 20 `tensor_ir` cases stayed green (exit 0) and my matching case went red (exit 101). Log: `scratch/adversary/mutate.log`.

**3. Suite run** (`CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only b10x-build-slot task ci`): exit 201.
- build, fmt and clippy passed.
- `tensor_ir`: 20 passed, `no_vulkan`: 7 passed, lib: 2 passed.
- `tensor_ir_adversary`: "3 passed; 4 failed", then `task: Failed to run task "test": exit status 101`.

**4. Findings** (all in the working tree on 1c33fe0)

| # | file:line | Verdict | Origin | Measured | What reaches it |
|---|---|---|---|---|---|
| F1 | `verify.rs:67` | INFEASIBLE | introduced | duplicate canonical path accepted | Nothing found. `lower.rs` and `autodiff.rs` are 1-line stubs, and the builder refuses it (`DuplicateParamPath`). It contradicts the verifier's own doc ("checks a program made or changed by any other means") and DESIGN § 2.3 / § 3.2 (parameter-ID ownership) |
| F2 | `verify.rs:127` | INFEASIBLE | introduced | duplicate `Input` name accepted | Nothing found. The builder refuses it (`DuplicateInput`) |
| F3 | `verify.rs:127` | INFEASIBLE | introduced | an `Input` node missing from `inputs` is accepted, and so is a node listed twice | Nothing found. It contradicts the doc on `tensor.rs:213`. The interpreter story will feed inputs from that list |
| F4 | `ops.rs:117` | CONFIRMED | introduced | the unit's suite misses the mutant with the `ReluGrad` shape check removed | Every caller of `infer` |
| F5 | `ops.rs:126` | CONFIRMED | introduced | the unit's suite misses the mutant with the `MatMul` rhs rank check removed. A rank-3 rhs would then be typed silently, and a rank-1 rhs would panic on an index | Every caller of `infer` |
| F6 | `print.rs:43` | CONFIRMED | introduced | no test pins the printed form of `Scalar` or `BroadcastScalar` (the golden has neither) | `modelspec ir` (DESIGN § 8) |

Suggested fix for F1 to F3, which I did not apply: `verify` checks unique parameter paths and unique `Input` names, and that `inputs` lists exactly the `Input` nodes, each once, in node order.

Not covered: my scalar case uses 0.5, which prints the same under `{:?}` and `{}`. It therefore does not catch a change from Debug to Display formatting, which matters for 1.0.

**5. What I attacked and could not break**
- Elementwise ops: a scalar operand on either side, both scalars, and a shape mismatch.
- `Transpose` and `MatMul` lhs rank; the `MatMul` result shape and the contracting-dimension error text, checked against DESIGN § 8.1.
- `Mse`, `ReduceSum` and `ReduceMean` result types.
- `BroadcastScalar` with a symbolic or zero dimension: the node-type check catches both.
- Verifier: an operand referring to its own node, use before definition, an undefined operand, an undeclared `ParamId`, `Dim::Symbol`, a zero dimension in a parameter or a node, element-count and byte-count overflow, arity, a type that disagrees with the registry, and an undefined output.
- Printer: parameter lines in `ParamId` order, reads printed by canonical path, the golden.

**6. Paths written outside the worktree**
- `~/.cache/modelspec-wave-m0/m0-tensor-ir-core/scratch/adversary/`, which holds:
  - one log per case (`adv_*.log`) and `ci.log`
  - `mutate.sh`, `edit.py` and `mutate.log`
  - `mutant/`: a copy of the tree with its own `target/`. It is a large build directory and the coordinator should clean it up.

```findings
- file: crates/modelspec-core/src/ir/verify.rs
  line: 67
  category: acceptance
  severity: warning
  verdict: INFEASIBLE
  origin: introduced
  message: verify accepts two ParamIds sharing one canonical path, which the builder refuses as DuplicateParamPath; no producer of such a program exists yet
- file: crates/modelspec-core/src/ir/verify.rs
  line: 127
  category: acceptance
  severity: warning
  verdict: INFEASIBLE
  origin: introduced
  message: verify accepts two Input nodes with the same name, which the builder refuses as DuplicateInput; no producer of such a program exists yet
- file: crates/modelspec-core/src/ir/verify.rs
  line: 127
  category: contract-drift
  severity: warning
  verdict: INFEASIBLE
  origin: introduced
  message: verify accepts an inputs list that omits an Input node or names one twice, contrary to the TensorProgram.inputs doc; no producer of such a program exists yet
- file: crates/modelspec-core/src/ops.rs
  line: 117
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: removing the ReluGrad shape check leaves all 20 tensor_ir cases green; adv_relu_grad_rejects_operands_of_different_shapes catches it
- file: crates/modelspec-core/src/ops.rs
  line: 126
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: removing the MatMul rhs rank check leaves all 20 tensor_ir cases green; adv_matmul_rejects_rank3_rhs catches it
- file: crates/modelspec-core/src/ir/print.rs
  line: 43
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: the printed forms of Scalar and BroadcastScalar are unpinned by the unit's golden; adv_printer_pins_scalar_and_broadcast_scalar_lines catches a bracket mutation
```
