//! story:fuse-elementwise-chains-and-gradient-sums: a chain of elementwise ops, each but the last
//! read once and only by the next, is one `elementwise` dispatch (DESIGN.md § 3.1). Covers the
//! loss gradient `(y - t) * c` of the MNIST example, an adjoint's `Add` chain into its
//! `ReluGrad`, adjoint sums of three and five contributions, a value read by two ops, and a
//! prediction read by one elementwise op.

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::artifact::{Backend, CompiledArtifact, DataRole};
use modelspec_core::compile::{self, CompiledTraining};
use modelspec_core::ir::kernel::{
    BufferView, DispatchCount, ExecutionPlan, Kernel, KernelKind, Operand, PlanOp, PlanStep,
};
use modelspec_core::ir::print::print_kernel;
use modelspec_core::ir::tensor::{TensorNode, TensorOp, ValueId};
use modelspec_core::ir::verify::verify_plan;

fn artifact(source: &str) -> CompiledArtifact {
    match compile::compile_artifact("chain.mspec", source, Backend::Vulkan) {
        Ok(artifact) => artifact,
        Err(diagnostics) => panic!("compile_artifact refused the source: {diagnostics:#?}"),
    }
}

fn training(source: &str) -> CompiledTraining {
    compile::compile_training("chain.mspec", source).expect("the source compiles for training")
}

fn nodes(training: &CompiledTraining) -> &[TensorNode] {
    &training.training.training.program.nodes
}

fn node(training: &CompiledTraining, value: ValueId) -> &TensorNode {
    nodes(training)
        .iter()
        .find(|n| n.id == value)
        .unwrap_or_else(|| panic!("no node {value}"))
}

/// A model of input `x: [10, 12]` with parameters `params`, the forward `body` and output type
/// `out`, trained with `loss`.
fn model(params: &str, body: &str, out: &str, loss: &str) -> String {
    format!(
        "model Chain {{
{params}
    fn forward(x: Tensor<f32, [10, 12]>)
        -> Tensor<f32, {out}> {{
{body}
    }}
}}

train Fit {{
    model = Chain;
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, {out}>;
    loss = {loss};
    optimizer = sgd(lr = 0.05);
    steps = 30;
    seed = 7;
    device = cpu;
}}
"
    )
}

/// `w1: [12, 9]` and `w2` to `w<n>`, each `[9, 5]`.
fn params(n: usize) -> String {
    let mut text = "    param w1: Tensor<f32, [12, 9]> = xavier_uniform();\n".to_owned();
    for i in 2..=n {
        text.push_str(&format!(
            "    param w{i}: Tensor<f32, [9, 5]> = xavier_uniform();\n"
        ));
    }
    text
}

/// `h = relu(matmul(x, w1))` read by `n` matmuls, their products added left to right: the
/// adjoint of `h` sums `n` contributions.
fn h_read_by(n: usize) -> String {
    let mut sum = "matmul(h, w2)".to_owned();
    for i in 3..=n + 1 {
        sum = format!("add({sum}, matmul(h, w{i}))");
    }
    model(
        &params(n + 1),
        &format!("        let h = relu(matmul(x, w1));\n        return {sum};"),
        "[10, 5]",
        "mse(model.forward(x), y)",
    )
}

fn dispatches(steps: &[PlanStep]) -> impl Iterator<Item = &PlanStep> {
    steps.iter().filter(|s| s.op == PlanOp::Dispatch)
}

fn kernel_of<'a>(plan: &'a ExecutionPlan, step: &PlanStep) -> &'a Kernel {
    step.kernel_module_id
        .and_then(|id| plan.kernel(id))
        .expect("a dispatch names a kernel of the plan")
}

/// The step plan's dispatch whose value is `value`.
fn dispatch_of(plan: &ExecutionPlan, value: ValueId) -> &PlanStep {
    dispatches(&plan.step)
        .find(|s| s.value == Some(value))
        .unwrap_or_else(|| panic!("no dispatch of {value}\n{}", print_kernel(plan)))
}

/// The view the step plan computes `value` into: the last write of its dispatch.
fn view_of(plan: &ExecutionPlan, value: ValueId) -> BufferView {
    *dispatch_of(plan, value)
        .writes
        .last()
        .expect("a dispatch writes")
}

/// The view an epilogue matmul writes its pre-activation `value` into.
fn pre_activation_view(plan: &ExecutionPlan, matmul: ValueId) -> BufferView {
    let step = dispatch_of(plan, matmul);
    assert_eq!(
        step.writes.len(),
        2,
        "the epilogue keeps the pre-activation"
    );
    step.writes[0]
}

/// Every elementwise kernel of `plan` named `op`.
fn kernels_named<'a>(plan: &'a ExecutionPlan, op: &str) -> Vec<&'a Kernel> {
    plan.kernels
        .iter()
        .filter(|k| k.kind == KernelKind::Elementwise && k.op == op)
        .collect()
}

/// The adjoint sum the `ReluGrad` of `training` reads, as the chain of `Add`s autodiff built:
/// the contributions in the order they are added.
fn adjoint_sum(training: &CompiledTraining) -> (ValueId, ValueId, Vec<ValueId>) {
    let grads: Vec<&TensorNode> = nodes(training)
        .iter()
        .filter(|n| n.op == TensorOp::ReluGrad)
        .collect();
    let [grad] = grads[..] else {
        panic!("one ReluGrad");
    };
    let root = grad.operands[1];
    let mut contributions = Vec::new();
    let mut at = root;
    loop {
        let add = node(training, at);
        if add.op != TensorOp::Add {
            contributions.push(at);
            break;
        }
        contributions.push(add.operands[1]);
        at = add.operands[0];
    }
    contributions.reverse();
    (grad.id, root, contributions)
}

/// The matmul whose `relu` epilogue is the forward `h`.
fn epilogue_matmul(training: &CompiledTraining) -> ValueId {
    let relus: Vec<&TensorNode> = nodes(training)
        .iter()
        .filter(|n| n.op == TensorOp::Relu)
        .collect();
    let [relu] = relus[..] else {
        panic!("one Relu");
    };
    relu.operands[0]
}

#[test]
fn the_mnist_loss_gradient_is_one_sub_mul_dispatch_with_no_buffer_for_the_difference() {
    let source = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/mnist_mlp.mspec"),
    )
    .expect("the example reads");
    let artifact = artifact(&source);
    let plan = &artifact.execution;
    verify_plan(plan).expect("the plan verifies");
    let chains = kernels_named(plan, "Sub>Mul@0");
    let [chain] = chains[..] else {
        panic!("one Sub>Mul@0 kernel\n{}", print_kernel(plan));
    };
    assert_eq!(
        chain.operands,
        vec![
            Operand::Buffer(chain.result.clone()),
            Operand::Buffer(chain.result.clone()),
            Operand::Immediate(0.002),
        ]
    );
    assert_eq!(chain.flop(), Some(2_000), "one sub and one mul per element");
    assert!(
        kernels_named(plan, "Sub").is_empty(),
        "{}",
        print_kernel(plan)
    );
    assert!(
        kernels_named(plan, "Mul").is_empty(),
        "{}",
        print_kernel(plan)
    );

    let training = training(&source);
    let mul = nodes(&training)
        .iter()
        .find(|n| {
            n.op == TensorOp::Mul
                && n.ty.rank() == 2
                && node(&training, n.operands[0]).op == TensorOp::Sub
        })
        .expect("the gradient `(y - t) * c`");
    let sub = node(&training, mul.operands[0]);
    let step = dispatch_of(plan, mul.id);
    assert_eq!(kernel_of(plan, step).op, "Sub>Mul@0");
    let target = plan
        .data
        .iter()
        .find(|d| d.role == DataRole::Target)
        .expect("a target")
        .view;
    assert_eq!(
        step.reads,
        vec![view_of(plan, sub.operands[0]), target],
        "the chain reads the prediction and the target"
    );
    assert!(
        dispatches(&plan.step).all(|s| s.value != Some(sub.id)),
        "the difference is no dispatch of its own"
    );
    assert_eq!(
        plan.dispatch_count(step),
        Some(DispatchCount::Counted {
            flop: 2_000,
            bytes: 12_000
        })
    );
    let count = |steps: &[PlanStep], op: PlanOp| steps.iter().filter(|s| s.op == op).count();
    assert_eq!(
        (
            count(&plan.step, PlanOp::Dispatch),
            count(&plan.step, PlanOp::Barrier)
        ),
        (10, 7),
        "{}",
        print_kernel(plan)
    );
}

/// The adjoint of `h`, read by two matmuls, is `c0 + c1`, and its only reader is the `ReluGrad`
/// of `h`'s `relu`, which reads it as operand 1: one `Add>ReluGrad@1` dispatch over
/// `(c0, c1, z)`, `z` the pre-activation the epilogue keeps.
#[test]
fn an_adjoint_sum_into_its_relu_grad_is_one_add_relu_grad_dispatch() {
    let source = h_read_by(2);
    let artifact = artifact(&source);
    let plan = &artifact.execution;
    verify_plan(plan).expect("the plan verifies");
    let training = training(&source);
    let (grad, root, contributions) = adjoint_sum(&training);
    assert_eq!(contributions.len(), 2, "two contributions");
    let step = dispatch_of(plan, grad);
    let kernel = kernel_of(plan, step);
    assert_eq!(kernel.op, "Add>ReluGrad@1", "{}", print_kernel(plan));
    assert_eq!(
        kernel.flop(),
        Some(2 * 90),
        "two ops over the 10 x 9 elements of `h`"
    );
    let z = epilogue_matmul(&training);
    assert_eq!(
        step.reads,
        vec![
            view_of(plan, contributions[0]),
            view_of(plan, contributions[1]),
            pre_activation_view(plan, z),
        ],
        "{}",
        print_kernel(plan)
    );
    assert!(
        dispatches(&plan.step).all(|s| s.value != Some(root)),
        "the sum is no dispatch of its own"
    );
    assert!(kernels_named(plan, "ReluGrad").is_empty());
}

/// An adjoint of three contributions is one `((c0 + c1) + c2)` dispatch in that order. With the
/// pre-activation the `ReluGrad` would read a fourth buffer, past the chain's three, so the
/// `ReluGrad` stays its own dispatch, reading the sum.
#[test]
fn an_adjoint_sum_of_three_contributions_is_one_dispatch_in_the_order_of_the_ir() {
    let source = h_read_by(3);
    let artifact = artifact(&source);
    let plan = &artifact.execution;
    verify_plan(plan).expect("the plan verifies");
    let training = training(&source);
    let (grad, root, contributions) = adjoint_sum(&training);
    assert_eq!(contributions.len(), 3, "three contributions");
    let step = dispatch_of(plan, root);
    assert_eq!(
        kernel_of(plan, step).op,
        "Add>Add@0",
        "{}",
        print_kernel(plan)
    );
    let expected: Vec<BufferView> = contributions.iter().map(|&c| view_of(plan, c)).collect();
    assert_eq!(step.reads, expected, "{}", print_kernel(plan));
    let relu_grad = dispatch_of(plan, grad);
    assert_eq!(kernel_of(plan, relu_grad).op, "ReluGrad");
    assert_eq!(relu_grad.reads[1], view_of(plan, root));
}

/// Five contributions: `((c0 + c1) + c2)` is one dispatch and `((s + c3) + c4)` the next, `s`
/// the first's result, so the order of the sum is the IR's.
#[test]
fn an_adjoint_sum_of_five_contributions_is_two_dispatches_keeping_the_order() {
    let source = h_read_by(5);
    let artifact = artifact(&source);
    let plan = &artifact.execution;
    verify_plan(plan).expect("the plan verifies");
    let training = training(&source);
    let (_, root, contributions) = adjoint_sum(&training);
    assert_eq!(contributions.len(), 5, "five contributions");
    let last = dispatch_of(plan, root);
    assert_eq!(
        kernel_of(plan, last).op,
        "Add>Add@0",
        "{}",
        print_kernel(plan)
    );
    let s = node(&training, node(&training, root).operands[0]).operands[0];
    let first = dispatch_of(plan, s);
    assert_eq!(kernel_of(plan, first).op, "Add>Add@0");
    let view = |c: ValueId| view_of(plan, c);
    assert_eq!(
        first.reads,
        vec![
            view(contributions[0]),
            view(contributions[1]),
            view(contributions[2])
        ]
    );
    assert_eq!(
        last.reads,
        vec![view(s), view(contributions[3]), view(contributions[4])]
    );
    let sums = nodes(&training)
        .iter()
        .filter(|n| n.op == TensorOp::Add)
        .count();
    let summed = dispatches(&plan.step)
        .filter(|s| kernel_of(plan, s).op.starts_with("Add"))
        .count();
    assert!(summed < sums, "{summed} add dispatches for {sums} adds");
}

/// `a` is read by `relu(a)` and by `add`: it stays its own dispatch, and the `relu`, read only
/// by the `add`, is fused into it as `Relu>Add@1` over `(a, a)`.
#[test]
fn a_value_read_by_two_ops_is_no_chain_intermediate() {
    let source = model(
        &params(2),
        "        let a = mul(matmul(x, w1), 2.0);
        return matmul(add(a, relu(a)), w2);",
        "[10, 5]",
        "mse(model.forward(x), y)",
    );
    let artifact = artifact(&source);
    let plan = &artifact.execution;
    verify_plan(plan).expect("the plan verifies");
    let training = training(&source);
    let a = nodes(&training)
        .iter()
        .find(|n| n.op == TensorOp::Mul && n.ty.rank() == 2)
        .expect("a");
    assert_eq!(kernel_of(plan, dispatch_of(plan, a.id)).op, "Mul");
    let add = nodes(&training)
        .iter()
        .find(|n| n.op == TensorOp::Add && n.operands[0] == a.id)
        .expect("add(a, relu(a))");
    let step = dispatch_of(plan, add.id);
    assert_eq!(
        kernel_of(plan, step).op,
        "Relu>Add@1",
        "{}",
        print_kernel(plan)
    );
    assert_eq!(step.reads, vec![view_of(plan, a.id), view_of(plan, a.id)]);
}

/// The prediction `mul(matmul(x, w1), 2.0)` has one reader, `add(prediction, 1.0)` of the loss,
/// and is never a chain intermediate: the prediction keeps its own view in both plans.
#[test]
fn a_prediction_read_by_one_elementwise_op_keeps_its_view() {
    let source = model(
        &params(1),
        "        return mul(matmul(x, w1), 2.0);",
        "[10, 9]",
        "mse(add(model.forward(x), 1.0), y)",
    );
    let artifact = artifact(&source);
    let plan = &artifact.execution;
    verify_plan(plan).expect("the plan verifies");
    for (name, steps) in [("step", &plan.step), ("forward", &plan.forward)] {
        let writers: Vec<&PlanStep> = dispatches(steps)
            .filter(|s| s.writes.contains(&plan.prediction_view))
            .collect();
        let [writer] = writers[..] else {
            panic!(
                "{name}: one dispatch writes the prediction\n{}",
                print_kernel(plan)
            );
        };
        assert_eq!(kernel_of(plan, writer).op, "Mul", "{name}");
    }
}

/// Every model of this file and the examples: the plan verifies, no barrier group holds a
/// hazard, every chain kernel reads at most 3 buffers, so it binds at most 4 storage buffers
/// with its result, and every kernel op names a chain of the five elementwise ops or no chain.
#[test]
fn every_chain_binds_at_most_four_buffers_and_every_plan_is_hazard_free() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut sources = vec![
        h_read_by(2),
        h_read_by(3),
        h_read_by(5),
        h_read_by(8),
        model(
            &params(2),
            "        let a = mul(matmul(x, w1), 2.0);
        return matmul(add(a, relu(a)), w2);",
            "[10, 5]",
            "mse(model.forward(x), y)",
        ),
    ];
    for example in ["mnist_mlp", "linear_regression"] {
        sources.push(
            std::fs::read_to_string(root.join(format!("examples/{example}.mspec")))
                .expect("the example reads"),
        );
    }
    let mut chains = BTreeSet::new();
    for source in &sources {
        let artifact = artifact(source);
        let plan = &artifact.execution;
        verify_plan(plan).unwrap_or_else(|e| panic!("{e}\n{}", print_kernel(plan)));
        for kernel in plan
            .kernels
            .iter()
            .filter(|k| k.kind == KernelKind::Elementwise && k.op.contains('>'))
        {
            let buffers = kernel
                .operands
                .iter()
                .filter(|o| matches!(o, Operand::Buffer(_)))
                .count();
            assert!(buffers <= 3, "{}", kernel.signature());
            for (i, link) in kernel.op.split('>').enumerate() {
                let op = if i == 0 {
                    link
                } else {
                    let (op, at) = link.split_once('@').expect("a later op names its operand");
                    assert!(matches!(at, "0" | "1"), "{}", kernel.op);
                    op
                };
                assert!(
                    ["Add", "Sub", "Mul", "Relu", "ReluGrad"].contains(&op),
                    "{}",
                    kernel.op
                );
            }
            chains.insert(kernel.op);
        }
        for steps in [&plan.step, &plan.forward] {
            let mut group: Vec<(BTreeSet<u32>, BTreeSet<u32>)> = Vec::new();
            for step in steps {
                match step.op {
                    PlanOp::Barrier => group.clear(),
                    PlanOp::Dispatch | PlanOp::Copy => {
                        let reads: BTreeSet<u32> = step.reads.iter().map(|v| v.buffer).collect();
                        let writes: BTreeSet<u32> = step.writes.iter().map(|v| v.buffer).collect();
                        for (r, w) in &group {
                            assert!(
                                reads.is_disjoint(w)
                                    && writes.is_disjoint(r)
                                    && writes.is_disjoint(w),
                                "a hazard in one barrier group\n{}",
                                print_kernel(plan)
                            );
                        }
                        group.push((reads, writes));
                    }
                    PlanOp::Allocate | PlanOp::Complete => {}
                }
            }
        }
    }
    let expected: BTreeSet<&str> = ["Sub>Mul@0", "Add>ReluGrad@1", "Add>Add@0", "Relu>Add@1"]
        .into_iter()
        .collect();
    assert!(
        expected.is_subset(&chains),
        "chains seen: {chains:?}, expected at least {expected:?}"
    );
}

/// The intermediates of every chain have no buffer: the plan has one buffer fewer per absorbed
/// op than the same plan with every elementwise op its own dispatch would need. Checked on the
/// MNIST example, whose one chain absorbs the `Sub`.
#[test]
fn the_mnist_plan_has_one_buffer_fewer_than_before_the_fusion() {
    let source = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/mnist_mlp.mspec"),
    )
    .expect("the example reads");
    let plan = artifact(&source).execution;
    assert_eq!(plan.buffers.len(), 13, "{}", print_kernel(&plan));
    let written: BTreeMap<u32, usize> =
        dispatches(&plan.step)
            .flat_map(|s| &s.writes)
            .fold(BTreeMap::new(), |mut m, v| {
                *m.entry(v.buffer).or_default() += 1;
                m
            });
    assert!(written.len() <= plan.buffers.len());
}
