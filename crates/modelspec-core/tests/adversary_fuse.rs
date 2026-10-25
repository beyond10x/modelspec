//! Adversary cases for story m1-fuse-elementwise: the ReLU epilogue, the scaled SGD update and
//! the folded scalars of `schedule::schedule`, on models other than MNIST, and the documents the
//! story changed read as contracts.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{self, CompiledTraining};
use modelspec_core::ir::kernel::{Epilogue, ExecutionPlan, Operand, PlanOp, PlanStep};
use modelspec_core::ir::print::print_kernel;
use modelspec_core::ir::tensor::{TensorNode, TensorOp};
use modelspec_core::ir::verify::verify_plan;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn artifact(source: &str) -> CompiledArtifact {
    match compile::compile_artifact("adv.mspec", source, Backend::Vulkan) {
        Ok(artifact) => artifact,
        Err(diagnostics) => panic!("compile_artifact refused the source: {diagnostics:#?}"),
    }
}

fn training(source: &str) -> CompiledTraining {
    compile::compile_training("adv.mspec", source).expect("the source compiles for training")
}

fn nodes(training: &CompiledTraining) -> &[TensorNode] {
    &training.training.training.program.nodes
}

fn model(name: &str, params: &str, body: &str, out: &str, loss: &str) -> String {
    format!(
        "model {name} {{
{params}
    fn forward(x: Tensor<f32, [10, 12]>)
        -> Tensor<f32, {out}> {{
{body}
    }}
}}

train Fit {{
    model = {name};
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, {out}>;
    loss = {loss};
    optimizer = sgd(lr = 0.07);
    steps = 30;
    seed = 3;
    device = cpu;
}}
"
    )
}

const W1W2: &str = "    param w1: Tensor<f32, [12, 9]> = xavier_uniform();
    param w2: Tensor<f32, [9, 5]> = xavier_uniform();
";
const W1: &str = "    param w1: Tensor<f32, [12, 9]> = xavier_uniform();
";

fn two_relus() -> String {
    model(
        "TwoRelu",
        W1W2,
        "        let z = matmul(x, w1);
        let h = add(relu(z), mul(relu(z), z));
        return matmul(h, w2);",
        "[10, 5]",
        "mse(model.forward(x), y)",
    )
}

fn relu_is_prediction() -> String {
    model(
        "PredRelu",
        W1,
        "        return relu(matmul(x, w1));",
        "[10, 9]",
        "mse(model.forward(x), y)",
    )
}

fn relu_of_prediction() -> String {
    model(
        "ReluOfPred",
        W1,
        "        return matmul(x, w1);",
        "[10, 9]",
        "mul(0.3, mse(relu(model.forward(x)), y))",
    )
}

fn transposed() -> String {
    "model Transposed {
    param w1: Tensor<f32, [10, 7]> = xavier_uniform();
    param w2: Tensor<f32, [7, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [10, 12]>)
        -> Tensor<f32, [12, 3]> {
        let h = relu(matmul(transpose(x), w1));
        return relu(matmul(h, w2));
    }
}

train Fit {
    model = Transposed;
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [12, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.3);
    steps = 30;
    seed = 13;
    device = cpu;
}
"
    .to_owned()
}

fn scaled_chain() -> String {
    model(
        "ScaledChain",
        W1W2,
        "        return matmul(relu(matmul(x, w1)), w2);",
        "[10, 5]",
        "mul(0.7, mul(0.3, mse(model.forward(x), y)))",
    )
}

/// `h = relu(matmul(x, w1))` read by `n` matmuls whose products are added: the adjoint of `h`
/// is a chain of `n - 1` `Add`s into the `ReluGrad` (story:fuse-elementwise-chains-and-gradient-sums).
fn adjoint_sum(n: usize) -> String {
    let mut params = "    param w1: Tensor<f32, [12, 9]> = xavier_uniform();\n".to_owned();
    let mut sum = "matmul(h, w2)".to_owned();
    for i in 2..=n + 1 {
        params.push_str(&format!(
            "    param w{i}: Tensor<f32, [9, 5]> = xavier_uniform();\n"
        ));
        if i > 2 {
            sum = format!("add({sum}, matmul(h, w{i}))");
        }
    }
    model(
        "AdjointSum",
        &params,
        &format!("        let h = relu(matmul(x, w1));\n        return {sum};"),
        "[10, 5]",
        "mse(model.forward(x), y)",
    )
}

/// `a` read by two ops, so no chain intermediate; `relu(a)` read once by the `add`.
fn fan_out() -> String {
    model(
        "FanOut",
        W1W2,
        "        let a = mul(matmul(x, w1), 2.0);
        return matmul(add(a, relu(a)), w2);",
        "[10, 5]",
        "mse(model.forward(x), y)",
    )
}

/// Every model of this file, with its name.
fn battery() -> Vec<(&'static str, String)> {
    vec![
        ("two_relus", two_relus()),
        ("relu_is_prediction", relu_is_prediction()),
        ("relu_of_prediction", relu_of_prediction()),
        ("transposed", transposed()),
        ("scaled_chain", scaled_chain()),
        ("adjoint_sum_2", adjoint_sum(2)),
        ("adjoint_sum_3", adjoint_sum(3)),
        ("adjoint_sum_6", adjoint_sum(6)),
        ("fan_out", fan_out()),
        (
            "mnist_mlp",
            std::fs::read_to_string(repo_root().join("examples/mnist_mlp.mspec"))
                .expect("the example reads"),
        ),
    ]
}

fn epilogue_steps<'a>(plan: &'a ExecutionPlan, steps: &'a [PlanStep]) -> Vec<&'a PlanStep> {
    steps
        .iter()
        .filter(|s| {
            s.op == PlanOp::Dispatch
                && s.kernel_module_id
                    .and_then(|id| plan.kernel(id))
                    .is_some_and(|k| k.epilogue.is_some())
        })
        .collect()
}

/// Two ReLUs read one matmul output: the first is the epilogue, which keeps the pre-activation
/// because the second ReLU (and the ReluGrad, and the `mul`) read it; the second ReLU is its own
/// dispatch reading that pre-activation view.
#[test]
fn two_relus_on_one_matmul_one_epilogue_and_one_relu_dispatch_reading_the_pre_activation() {
    let source = two_relus();
    let artifact = artifact(&source);
    let plan = &artifact.execution;
    verify_plan(plan).expect("the plan verifies");
    let training = training(&source);
    let relus: Vec<&TensorNode> = nodes(&training)
        .iter()
        .filter(|n| n.op == TensorOp::Relu)
        .collect();
    assert_eq!(relus.len(), 2, "two relu nodes survive lowering");

    let fused = epilogue_steps(plan, &plan.step);
    let [fused] = fused[..] else {
        panic!("one epilogue dispatch\n{}", print_kernel(plan));
    };
    let kernel = plan.kernel(fused.kernel_module_id.unwrap()).unwrap();
    assert_eq!(
        kernel.epilogue,
        Some(Epilogue::Relu {
            writes_pre_activation: true
        })
    );
    let [pre, _activation] = fused.writes[..] else {
        panic!("the epilogue writes two views: {fused:?}");
    };
    let own: Vec<&PlanStep> = plan
        .step
        .iter()
        .filter(|s| {
            s.op == PlanOp::Dispatch
                && s.kernel_module_id
                    .and_then(|id| plan.kernel(id))
                    .is_some_and(|k| k.op == "Relu" && k.epilogue.is_none())
        })
        .collect();
    let [second] = own[..] else {
        panic!("one relu dispatch of its own\n{}", print_kernel(plan));
    };
    assert_eq!(
        second.reads,
        vec![pre],
        "the second relu reads the pre-activation"
    );
}

/// The prediction is `relu(matmul(x, w1))`: the activation the epilogue writes is the
/// prediction view, written by both plans.
#[test]
fn a_relu_that_is_the_prediction_is_written_by_the_epilogue_in_both_plans() {
    let artifact = artifact(&relu_is_prediction());
    let plan = &artifact.execution;
    verify_plan(plan).expect("the plan verifies");
    for (name, steps) in [("step", &plan.step), ("forward", &plan.forward)] {
        let fused = epilogue_steps(plan, steps);
        let [fused] = fused[..] else {
            panic!("{name}: one epilogue dispatch\n{}", print_kernel(plan));
        };
        assert_eq!(
            fused.writes.last(),
            Some(&plan.prediction_view),
            "{name}: the activation is the prediction\n{}",
            print_kernel(plan)
        );
    }
}

/// The prediction is the matmul and the loss applies the ReLU: the epilogue keeps the
/// pre-activation, and that view is the prediction in both plans.
#[test]
fn a_relu_of_the_prediction_keeps_the_pre_activation_as_the_prediction_view() {
    let artifact = artifact(&relu_of_prediction());
    let plan = &artifact.execution;
    verify_plan(plan).expect("the plan verifies");
    for (name, steps) in [("step", &plan.step), ("forward", &plan.forward)] {
        let fused = epilogue_steps(plan, steps);
        let [fused] = fused[..] else {
            panic!("{name}: one epilogue dispatch\n{}", print_kernel(plan));
        };
        assert_eq!(
            fused.writes.first(),
            Some(&plan.prediction_view),
            "{name}: the pre-activation is the prediction\n{}",
            print_kernel(plan)
        );
        assert_eq!(fused.writes.len(), 2, "{name}");
    }
}

/// Within one barrier group (the dispatches and copies between two barriers), no step reads or
/// writes a buffer another step of the group writes. Checked independently of `verify_plan`, on
/// every model of the battery and on both plans.
#[test]
fn no_barrier_group_holds_a_hazard_on_any_model() {
    for (name, source) in battery() {
        let artifact = artifact(&source);
        let plan = &artifact.execution;
        verify_plan(plan).unwrap_or_else(|e| panic!("{name}: {e}"));
        for (which, steps) in [("step", &plan.step), ("forward", &plan.forward)] {
            let mut group: Vec<(usize, BTreeSet<u32>, BTreeSet<u32>)> = Vec::new();
            for (i, step) in steps.iter().enumerate() {
                match step.op {
                    PlanOp::Barrier => group.clear(),
                    PlanOp::Dispatch | PlanOp::Copy => {
                        let reads: BTreeSet<u32> = step.reads.iter().map(|v| v.buffer).collect();
                        let writes: BTreeSet<u32> = step.writes.iter().map(|v| v.buffer).collect();
                        for (j, r, w) in &group {
                            let raw = reads.intersection(w).next();
                            let war = writes.intersection(r).next();
                            let waw = writes.intersection(w).next();
                            assert!(
                                raw.is_none() && war.is_none() && waw.is_none(),
                                "{name} {which}: steps {j} and {i} share a barrier group with a \
                                 hazard on {raw:?} {war:?} {waw:?}\n{}",
                                print_kernel(plan)
                            );
                        }
                        group.push((i, reads, writes));
                    }
                    PlanOp::Allocate | PlanOp::Complete => {}
                }
            }
        }
    }
}

/// `spec/domains/artifact.yaml`, `ExecutionPlan.buffers`: "In M1 each stored value has its own
/// buffer, with six exceptions": an SGD update writes its parameter's buffer in place, a folded
/// scalar has no buffer, the `lr * g` of a fused update has none, a matmul pre-activation that
/// no op but its ReLU epilogue reads, and that is not the loss or the prediction, has none, and a
/// transpose that no op but matmuls reads, and that is not the loss or the prediction, has none
/// (no MNIST transpose reads a parameter after its update), and each value of an elementwise
/// chain but its last has none (story:fuse-elementwise-chains-and-gradient-sums). So the
/// plan has at least one buffer per data declaration, per parameter, and per stored (non-`Scalar`)
/// computing value of the step outside those exceptions, plus the loss.
#[test]
fn every_value_has_its_own_buffer_as_the_artifact_spec_states() {
    let spec = std::fs::read_to_string(repo_root().join("spec/domains/artifact.yaml"))
        .expect("the spec reads");
    assert!(
        spec.contains("In M1 each stored value has its own buffer, with six exceptions."),
        "the sentence this case drives from is in the spec"
    );
    let source = std::fs::read_to_string(repo_root().join("examples/mnist_mlp.mspec"))
        .expect("the example reads");
    let artifact = artifact(&source);
    let training = training(&source);
    let updates: BTreeSet<_> = training
        .training
        .optimizer
        .updates
        .values()
        .copied()
        .collect();
    let bound = [training.training.training.loss, training.forward.prediction];
    let all = nodes(&training);
    let readers = |value| {
        all.iter()
            .filter(move |r| r.operands.contains(&value))
            .collect::<Vec<_>>()
    };
    let op_of = |value| all.iter().find(|n| n.id == value).map(|n| &n.op);
    let scaled_product = |n: &TensorNode| {
        n.op == TensorOp::Mul
            && n.operands
                .iter()
                .filter(|&&o| matches!(op_of(o), Some(TensorOp::Scalar(_))))
                .count()
                == 1
            && matches!(readers(n.id)[..], [r] if updates.contains(&r.id) && r.op == TensorOp::Sub)
    };
    let epilogue_only = |n: &TensorNode| {
        n.op == TensorOp::MatMul
            && !bound.contains(&n.id)
            && matches!(readers(n.id)[..], [r] if r.op == TensorOp::Relu)
    };
    let folded_transpose = |n: &TensorNode| {
        let readers = readers(n.id);
        n.op == TensorOp::Transpose
            && !bound.contains(&n.id)
            && !readers.is_empty()
            && readers.iter().all(|r| r.op == TensorOp::MatMul)
    };
    let elementwise = |op: &TensorOp| {
        matches!(
            op,
            TensorOp::Add | TensorOp::Sub | TensorOp::Mul | TensorOp::Relu | TensorOp::ReluGrad
        )
    };
    // At most every elementwise value read once, by one elementwise op of its type alone: a
    // chain intermediate of the scheduler is one of these, so leaving them all out bounds the
    // count from below.
    let chain_intermediate = |n: &TensorNode| {
        elementwise(&n.op)
            && !bound.contains(&n.id)
            && matches!(readers(n.id)[..], [r] if elementwise(&r.op)
                && r.ty == n.ty
                && r.operands.iter().filter(|&&o| o == n.id).count() == 1)
    };
    let stored = all
        .iter()
        .filter(|n| {
            !matches!(
                n.op,
                TensorOp::Input { .. } | TensorOp::ParamRead { .. } | TensorOp::Scalar(_)
            ) && n.ty.rank() > 0
                && !updates.contains(&n.id)
                && !scaled_product(n)
                && !epilogue_only(n)
                && !folded_transpose(n)
                && !chain_intermediate(n)
        })
        .count();
    let plan = &artifact.execution;
    let needed = plan.data.len() + plan.parameters.len() + stored + 1;
    assert!(
        plan.buffers.len() >= needed,
        "{} buffers for {} data, {} parameters, {stored} stored values and the loss\n{}",
        plan.buffers.len(),
        plan.data.len(),
        plan.parameters.len(),
        print_kernel(plan)
    );
}

/// The module documentation of `schedule`: "An `Add`, `Sub` or `Mul` of rank 0 whose operands are
/// all compile-time scalars is folded ... a `Relu` or `ReluGrad` of compile-time scalars is never
/// folded and stays a kernel whose operands are immediates." A `relu` of a folded product, read by
/// an `add`: the product folds to the immediate 6.0, and the `relu` is the one kernel computing
/// from compile-time scalars alone.
#[test]
fn a_relu_of_compile_time_scalars_stays_a_kernel_as_the_schedule_documentation_states() {
    let doc = std::fs::read_to_string(repo_root().join("crates/modelspec-core/src/schedule.rs"))
        .expect("the schedule reads");
    assert!(
        doc.contains("a `Relu` or `ReluGrad` of compile-time scalars is never\n//! folded")
            || doc.contains("a `Relu` or `ReluGrad` of compile-time scalars is never folded"),
        "the sentence this case drives from is in the schedule documentation"
    );
    let source = model(
        "ReluScalar",
        W1,
        "        return add(matmul(x, w1), relu(mul(2.0, 3.0)));",
        "[10, 9]",
        "mse(model.forward(x), y)",
    );
    let artifact = artifact(&source);
    let plan = &artifact.execution;
    let scalar_only: Vec<String> = plan
        .kernels
        .iter()
        .filter(|k| {
            k.operands
                .iter()
                .all(|o| matches!(o, Operand::Immediate(_)))
        })
        .map(|k| k.signature())
        .collect();
    let relu_of_six = plan.kernels.iter().filter(|k| {
        k.op == "Relu" && k.operands.len() == 1 && k.operands[0] == Operand::Immediate(6.0)
    });
    assert!(
        scalar_only.len() == 1 && relu_of_six.count() == 1,
        "one kernel computes from compile-time scalars alone, the relu of the folded 6.0: \
         {scalar_only:#?}"
    );
}

/// DESIGN.md § 3.1 and the `buffers` summary of `spec/domains/artifact.yaml` state the chain
/// rule of story:fuse-elementwise-chains-and-gradient-sums, and every model of the battery obeys
/// it: each kernel whose op names a chain (`First>Op@at>…`) is an `elementwise` kernel of at
/// least two of the five elementwise ops, binds at most 3 buffers besides its result, and no
/// step reads a view that only a chain's intermediate would have had.
#[test]
fn the_elementwise_chain_rule_is_stated_and_every_chain_obeys_it() {
    let design = std::fs::read_to_string(repo_root().join("DESIGN.md")).expect("DESIGN.md reads");
    assert!(
        design.contains("**Revision:** 0.42 (2026-10-25)"),
        "DESIGN.md is at revision 0.42"
    );
    assert!(
        design.contains("each op but the last read once and by the next op alone"),
        "DESIGN.md § 3.1 states the chain rule"
    );
    assert!(
        design.contains("left to right, as the IR adds them"),
        "DESIGN.md § 3.1 states the order of an adjoint's sum"
    );
    let spec = std::fs::read_to_string(repo_root().join("spec/domains/artifact.yaml"))
        .expect("the spec reads");
    assert!(
        spec.contains("each value of an elementwise chain but its last"),
        "the artifact spec states the chain exception"
    );
    let mut seen = BTreeSet::new();
    for (name, source) in battery() {
        let artifact = artifact(&source);
        let plan = &artifact.execution;
        for kernel in plan.kernels.iter().filter(|k| k.op.contains('>')) {
            assert_eq!(
                kernel.kind,
                modelspec_core::ir::kernel::KernelKind::Elementwise,
                "{name}: {}",
                kernel.signature()
            );
            let ops: Vec<&str> = kernel
                .op
                .split('>')
                .map(|l| l.split('@').next().unwrap_or(l))
                .collect();
            assert!(ops.len() >= 2, "{name}: {}", kernel.op);
            for op in &ops {
                assert!(
                    ["Add", "Sub", "Mul", "Relu", "ReluGrad"].contains(op),
                    "{name}: {}",
                    kernel.op
                );
            }
            let buffers = kernel
                .operands
                .iter()
                .filter(|o| matches!(o, Operand::Buffer(_)))
                .count();
            assert!(buffers <= 3, "{name}: {}", kernel.signature());
            seen.insert(kernel.op);
        }
    }
    for chain in ["Sub>Mul@0", "Add>ReluGrad@1", "Add>Add@0", "Relu>Add@1"] {
        assert!(seen.contains(chain), "the battery fuses {chain}: {seen:?}");
    }
}
