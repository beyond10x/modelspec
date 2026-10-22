//! Kernel IR and the execution plan of a training step (DESIGN.md § 3.1, § 5.3, § 7.1;
//! story m1-kernel-plan).
//!
//! Every case compiles a real `.mspec` file through `compile::compile_artifact` and reads the
//! artifact's [`ExecutionPlan`]: one Dispatch per computing op of the training step but the fused
//! and folded ones of DESIGN.md § 3.1 (story m1-fuse-elementwise), the `mse` loss
//! sum as one Dispatch of one workgroup, every other reduce as `Allocate`, stage 1, `Barrier`,
//! stage 2, no `Complete` step, one view per parameter, and a forward plan that writes no
//! parameter.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{self, CompiledTraining};
use modelspec_core::ir::kernel::{
    Epilogue, ExecutionPlan, Kernel, KernelKind, Operand, PlanOp, PlanStep,
};
use modelspec_core::ir::print::print_kernel;
use modelspec_core::ir::tensor::{TensorNode, TensorOp, ValueId};
use modelspec_core::ir::verify::verify_plan;
use modelspec_core::schedule;

const LINEAR: &str = "examples/linear_regression.mspec";
const TWO_LAYER: &str = "crates/modelspec-docs/models/two_layer_relu.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(path: &str) -> String {
    std::fs::read_to_string(repo_root().join(path)).expect("the model file exists")
}

fn artifact(source: &str, backend: Backend) -> CompiledArtifact {
    match compile::compile_artifact("test.mspec", source, backend) {
        Ok(artifact) => artifact,
        Err(diagnostics) => panic!("compile_artifact refused the source: {diagnostics:#?}"),
    }
}

fn training(source: &str) -> CompiledTraining {
    compile::compile_training("test.mspec", source).expect("the source compiles for training")
}

/// The ops of the training step that compute: every op but `Input`, `ParamRead` and `Scalar`.
fn computing(training: &CompiledTraining) -> Vec<&TensorNode> {
    training
        .training
        .training
        .program
        .nodes
        .iter()
        .filter(|n| {
            !matches!(
                n.op,
                TensorOp::Input { .. } | TensorOp::ParamRead { .. } | TensorOp::Scalar(_)
            )
        })
        .collect()
}

fn dispatches_of(steps: &[PlanStep], value: ValueId) -> Vec<usize> {
    steps
        .iter()
        .enumerate()
        .filter(|(_, s)| s.op == PlanOp::Dispatch && s.value == Some(value))
        .map(|(i, _)| i)
        .collect()
}

fn kind(plan: &ExecutionPlan, step: &PlanStep) -> KernelKind {
    let id = step
        .kernel_module_id
        .expect("a Dispatch names its kernel module");
    plan.kernel(id)
        .expect("the module is one of the plan's kernels")
        .kind
}

/// The buffers that hold a parameter.
fn parameter_buffers(plan: &ExecutionPlan) -> BTreeSet<u32> {
    plan.parameters
        .iter()
        .flat_map(|p| p.views.iter().map(|v| v.buffer))
        .collect()
}

/// How the step plan computes one op (DESIGN.md § 3.1).
#[derive(Debug, PartialEq)]
enum Cover {
    /// One Dispatch names the op.
    Own,
    /// A `relu` the Dispatch of its matmul computes as the kernel's epilogue.
    Epilogue,
    /// An `lr * g` the Dispatch of its only reader, an SGD update, computes.
    Scaled,
    /// A rank-0 op of compile-time scalars, an immediate equal to its `f32` value in every
    /// kernel that reads it.
    Folded,
}

fn node_of(training: &CompiledTraining, value: ValueId) -> &TensorNode {
    training
        .training
        .training
        .program
        .node(value)
        .expect("the value is a node of the step")
}

fn readers(training: &CompiledTraining, value: ValueId) -> Vec<&TensorNode> {
    training
        .training
        .training
        .program
        .nodes
        .iter()
        .filter(|n| n.operands.contains(&value))
        .collect()
}

fn kernel_of<'a>(plan: &'a ExecutionPlan, step: &PlanStep) -> &'a Kernel {
    plan.kernel(step.kernel_module_id.expect("a Dispatch names its module"))
        .expect("the module is one of the plan's kernels")
}

/// The `f32` value of a compile-time scalar: a `Scalar`, or an `Add`, `Sub` or `Mul` of them,
/// each one correctly rounded operation.
fn constant(training: &CompiledTraining, value: ValueId) -> Option<f32> {
    let node = node_of(training, value);
    let operands: Option<Vec<f32>> = node
        .operands
        .iter()
        .map(|&o| constant(training, o))
        .collect();
    match (&node.op, operands?.as_slice()) {
        (TensorOp::Scalar(v), []) => Some(*v),
        (TensorOp::Add, [a, b]) => Some(a + b),
        (TensorOp::Sub, [a, b]) => Some(a - b),
        (TensorOp::Mul, [a, b]) => Some(a * b),
        _ => None,
    }
}

/// Every way the step plan of `plan` computes `node`; exactly one is right.
fn coverage(plan: &ExecutionPlan, training: &CompiledTraining, node: &TensorNode) -> Vec<Cover> {
    let mut covers = Vec::new();
    let own = dispatches_of(&plan.step, node.id);
    if own.len() == 1 {
        covers.push(Cover::Own);
    }
    if !own.is_empty() {
        return covers;
    }
    if let (TensorOp::Relu, [matmul]) = (&node.op, &node.operands[..])
        && let [at] = dispatches_of(&plan.step, *matmul)[..]
    {
        let kernel = kernel_of(plan, &plan.step[at]);
        if kernel.kind == KernelKind::MatMul && kernel.epilogue.is_some() {
            covers.push(Cover::Epilogue);
        }
    }
    let updates: BTreeSet<ValueId> = training
        .training
        .optimizer
        .updates
        .values()
        .copied()
        .collect();
    if let [reader] = readers(training, node.id)[..]
        && node.op == TensorOp::Mul
        && updates.contains(&reader.id)
        && let [at] = dispatches_of(&plan.step, reader.id)[..]
    {
        let kernel = kernel_of(plan, &plan.step[at]);
        if kernel.kind == KernelKind::SgdUpdate && kernel.operands.len() == 3 {
            covers.push(Cover::Scaled);
        }
    }
    if let Some(value) = constant(training, node.id).filter(|_| node.ty.rank() == 0) {
        let immediate_everywhere = readers(training, node.id).iter().all(|reader| {
            let mut sites = dispatches_of(&plan.step, reader.id);
            sites.extend(
                readers(training, reader.id)
                    .iter()
                    .filter(|r| updates.contains(&r.id))
                    .flat_map(|r| dispatches_of(&plan.step, r.id)),
            );
            sites.iter().any(|&at| {
                kernel_of(plan, &plan.step[at])
                    .operands
                    .iter()
                    .any(|o| matches!(o, Operand::Immediate(v) if v.to_bits() == value.to_bits()))
            })
        });
        if immediate_everywhere {
            covers.push(Cover::Folded);
        }
    }
    covers
}

/// Acceptance 2 for one model file.
fn the_step_plan_of_a_mse_model_is_one_dispatch_per_computing_op(path: &str) {
    let text = source(path);
    let artifact = artifact(&text, Backend::Vulkan);
    let plan = &artifact.execution;
    let training = training(&text);
    verify_plan(plan).expect("the plan verifies");

    let computing = computing(&training);
    assert!(!computing.is_empty());
    let mut own = 0;
    for node in &computing {
        let covers = coverage(plan, &training, node);
        assert_eq!(
            covers.len(),
            1,
            "{} {} is covered by {covers:?}\n{}",
            node.id,
            node.op.name(),
            print_kernel(plan)
        );
        if covers == [Cover::Own] {
            own += 1;
        }
    }
    let all: usize = plan
        .step
        .iter()
        .filter(|s| s.op == PlanOp::Dispatch)
        .count();
    assert_eq!(all, own, "no Dispatch outside the computing ops");

    let mse = computing
        .iter()
        .find(|n| n.op == TensorOp::Mse)
        .expect("the loss is mse");
    let [at] = dispatches_of(&plan.step, mse.id)[..] else {
        panic!("one Mse dispatch");
    };
    let step = &plan.step[at];
    assert_eq!(step.workgroups, Some([1, 1, 1]));
    assert_eq!(kind(plan, step), KernelKind::LossSum);
    let kernel = plan.kernel(step.kernel_module_id.unwrap()).unwrap();
    assert_eq!(kernel.workgroup_size, 64);
    assert_eq!(kernel.workgroups, [1, 1, 1]);

    assert_eq!(
        plan.parameters.len(),
        training.training.training.program.params.len()
    );
    for parameter in &plan.parameters {
        assert_eq!(parameter.views.len(), 1, "{}", parameter.path);
    }
    assert!(
        plan.step.iter().all(|s| s.op != PlanOp::Complete),
        "the step plan has no completion point"
    );
    assert!(plan.forward.iter().all(|s| s.op != PlanOp::Complete));

    let params = parameter_buffers(plan);
    for step in &plan.forward {
        for view in &step.writes {
            assert!(
                !params.contains(&view.buffer),
                "the forward plan writes parameter buffer b{}",
                view.buffer
            );
        }
        if step.op == PlanOp::Dispatch {
            assert_ne!(kind(plan, step), KernelKind::SgdUpdate);
        }
    }
    assert!(
        plan.forward
            .iter()
            .any(|s| s.writes.contains(&plan.prediction_view)),
        "the forward plan writes the prediction"
    );
    assert!(
        plan.step.iter().any(|s| s.writes.contains(&plan.loss_view)),
        "the step plan writes the loss"
    );
    assert!(
        plan.forward
            .iter()
            .all(|s| !s.writes.contains(&plan.loss_view)),
        "the forward plan computes no loss"
    );
}

#[test]
fn linear_regression_step_plan_has_one_dispatch_per_computing_op() {
    the_step_plan_of_a_mse_model_is_one_dispatch_per_computing_op(LINEAR);
}

#[test]
fn two_layer_relu_step_plan_has_one_dispatch_per_computing_op() {
    the_step_plan_of_a_mse_model_is_one_dispatch_per_computing_op(TWO_LAYER);
}

#[test]
fn every_sgd_update_writes_its_parameter_view_in_place() {
    let text = source(TWO_LAYER);
    let artifact = artifact(&text, Backend::Vulkan);
    let plan = &artifact.execution;
    let training = training(&text);
    for (param, &update) in &training.training.optimizer.updates {
        let [at] = dispatches_of(&plan.step, update)[..] else {
            panic!("one dispatch for the update of {param}");
        };
        let step = &plan.step[at];
        assert_eq!(kind(plan, step), KernelKind::SgdUpdate);
        let path = &training.training.training.program.params[param].path;
        let views = &plan
            .parameters
            .iter()
            .find(|p| &p.path == path)
            .expect("the parameter has views")
            .views;
        assert_eq!(step.writes, vec![views[0]]);
        assert!(step.reads.contains(&views[0]));
    }
}

#[test]
fn data_views_follow_the_train_block_declarations() {
    let artifact = artifact(&source(LINEAR), Backend::Vulkan);
    let data: Vec<(&str, String)> = artifact
        .execution
        .data
        .iter()
        .map(|d| (d.name.as_str(), d.role.to_string()))
        .collect();
    assert_eq!(data, [("x", "input".into()), ("y", "target".into())]);
    for spec in &artifact.execution.data {
        let elements: u64 = spec
            .ty
            .shape
            .iter()
            .map(|d| match d {
                modelspec_core::ir::tensor::Dim::Static(n) => *n,
                other => panic!("unresolved {other}"),
            })
            .product();
        assert_eq!(spec.view.element_count, elements, "{}", spec.name);
    }
}

// Acceptance 3: a reduce other than the mse loss sum is two stages around a partials buffer.

/// The linear-regression example with the loss `mean(mul(model.forward(x), y))`.
fn mean_loss_source() -> String {
    source(LINEAR).replace(
        "loss = mse(model.forward(x), y);",
        "loss = mean(mul(model.forward(x), y));",
    )
}

#[test]
fn a_reduce_mean_is_allocate_stage_1_barrier_stage_2() {
    let text = mean_loss_source();
    assert_ne!(text, source(LINEAR), "the loss was replaced");
    let artifact = artifact(&text, Backend::Vulkan);
    let plan = &artifact.execution;
    let training = training(&text);
    verify_plan(plan).expect("the plan verifies");

    let computing = computing(&training);
    let mean = computing
        .iter()
        .find(|n| n.op == TensorOp::ReduceMean)
        .expect("the loss is a ReduceMean");
    let [stage1, stage2] = dispatches_of(&plan.step, mean.id)[..] else {
        panic!("the ReduceMean has two dispatches\n{}", print_kernel(plan));
    };
    assert_eq!(stage2, stage1 + 2, "{}", print_kernel(plan));
    let allocate = &plan.step[stage1 - 1];
    let barrier = &plan.step[stage1 + 1];
    let (first, second) = (&plan.step[stage1], &plan.step[stage2]);
    assert_eq!(allocate.op, PlanOp::Allocate);
    assert_eq!(first.op, PlanOp::Dispatch);
    assert_eq!(barrier.op, PlanOp::Barrier);
    assert_eq!(second.op, PlanOp::Dispatch);
    assert_eq!(kind(plan, first), KernelKind::ReduceStage1);
    assert_eq!(kind(plan, second), KernelKind::ReduceStage2);

    let [partials] = allocate.writes[..] else {
        panic!("the Allocate makes one partials view");
    };
    assert_eq!(first.writes, vec![partials]);
    assert!(first.depends_on.contains(&(stage1 - 1)));
    assert_eq!(second.reads, vec![partials]);
    assert!(second.depends_on.contains(&(stage1 + 1)));
    assert!(barrier.depends_on.contains(&stage1));
    assert_eq!(second.workgroups, Some([1, 1, 1]));

    for node in computing.iter().filter(|n| n.id != mean.id) {
        let covers = coverage(plan, &training, node);
        assert_eq!(covers.len(), 1, "{} {} {covers:?}", node.id, node.op.name());
    }
    assert!(plan.step.iter().all(|s| s.op != PlanOp::Complete));
}

#[test]
fn an_allocate_comes_before_every_use_of_its_buffer() {
    let artifact = artifact(&mean_loss_source(), Backend::Vulkan);
    for steps in [&artifact.execution.step, &artifact.execution.forward] {
        for (i, step) in steps.iter().enumerate() {
            if step.op != PlanOp::Allocate {
                continue;
            }
            for earlier in &steps[..i] {
                for view in earlier.reads.iter().chain(&earlier.writes) {
                    assert!(
                        step.writes.iter().all(|w| w.buffer != view.buffer),
                        "buffer b{} is used before its Allocate",
                        view.buffer
                    );
                }
            }
        }
    }
}

// Acceptance 5: stable output, a backend- and rate-independent schema hash, fixed workgroup sizes.

#[test]
fn two_compiles_print_byte_equal_kernel_ir_and_equal_module_ids() {
    for path in [LINEAR, TWO_LAYER] {
        let text = source(path);
        let first = artifact(&text, Backend::Vulkan);
        let second = artifact(&text, Backend::Vulkan);
        assert_eq!(
            print_kernel(&first.execution),
            print_kernel(&second.execution)
        );
        let ids =
            |a: &CompiledArtifact| -> Vec<_> { a.execution.kernels.iter().map(|k| k.id).collect() };
        assert_eq!(ids(&first), ids(&second));
        assert!(!ids(&first).is_empty());
    }
}

#[test]
fn kernel_module_ids_are_distinct_within_an_artifact() {
    let artifact = artifact(&source(TWO_LAYER), Backend::Vulkan);
    let ids: BTreeSet<_> = artifact.execution.kernels.iter().map(|k| k.id).collect();
    assert_eq!(ids.len(), artifact.execution.kernels.len());
}

#[test]
fn schema_hash_ignores_backend_and_learning_rate_and_follows_shapes() {
    let text = source(LINEAR);
    let vulkan = artifact(&text, Backend::Vulkan);
    let cpu = artifact(&text, Backend::Cpu);
    assert_eq!(vulkan.schema_hash, cpu.schema_hash);

    let other_rate = text.replace("sgd(lr = 0.1)", "sgd(lr = 0.05)");
    assert_ne!(other_rate, text);
    let other_rate = artifact(&other_rate, Backend::Vulkan);
    assert_eq!(other_rate.schema_hash, vulkan.schema_hash);
    assert_ne!(other_rate.ir_hash, vulkan.ir_hash, "the rate is in the IR");

    let wider = text
        .replace("Tensor<f32, [4, 1]>", "Tensor<f32, [4, 2]>")
        .replace("Tensor<f32, [8, 1]>", "Tensor<f32, [8, 2]>");
    assert_ne!(wider, text);
    let wider = artifact(&wider, Backend::Vulkan);
    assert_ne!(wider.schema_hash, vulkan.schema_hash);
}

#[test]
fn the_source_hash_follows_the_source() {
    let text = source(LINEAR);
    let a = artifact(&text, Backend::Vulkan);
    let b = artifact(&format!("{text}\n"), Backend::Vulkan);
    assert_eq!(a.source_hash, artifact(&text, Backend::Cpu).source_hash);
    assert_ne!(a.source_hash, b.source_hash);
    assert_eq!(a.source_hash.0.len(), 64);
}

#[test]
fn a_cpu_artifact_has_no_kernels_and_empty_plans() {
    let artifact = artifact(&source(LINEAR), Backend::Cpu);
    assert_eq!(artifact.backend, Backend::Cpu);
    assert!(artifact.execution.kernels.is_empty());
    assert!(artifact.execution.step.is_empty());
    assert!(artifact.execution.forward.is_empty());
    assert_eq!(artifact.execution.parameters.len(), 1);
    verify_plan(&artifact.execution).expect("a Cpu plan verifies");
}

#[test]
fn the_schedule_entry_point_takes_no_device_profile() {
    let text = source(LINEAR);
    let compiled = training(&text);
    let plan: ExecutionPlan = schedule::schedule(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
    )
    .expect("the step schedules");
    assert_eq!(plan, artifact(&text, Backend::Vulkan).execution);
    let code = include_str!("../src/schedule.rs");
    assert!(!code.contains("DeviceProfile"));
}

#[test]
fn every_printed_kernel_shows_the_workgroup_size_of_its_kind() {
    for path in [LINEAR, TWO_LAYER] {
        let artifact = artifact(&source(path), Backend::Vulkan);
        let printed = print_kernel(&artifact.execution);
        for kernel in &artifact.execution.kernels {
            assert_eq!(kernel.workgroup_size, kernel.kind.workgroup_size());
            let line = printed
                .lines()
                .find(|l| l.contains(&kernel.id.to_string()))
                .unwrap_or_else(|| panic!("{} is printed\n{printed}", kernel.id));
            assert!(
                line.contains(&format!("workgroup_size {}", kernel.kind.workgroup_size())),
                "{line}"
            );
        }
    }
    for kind in [
        KernelKind::LossSum,
        KernelKind::Elementwise,
        KernelKind::Broadcast,
        KernelKind::ReduceStage1,
        KernelKind::ReduceStage2,
        KernelKind::Transpose,
        KernelKind::MatMul,
        KernelKind::SgdUpdate,
    ] {
        assert_eq!(kind.workgroup_size(), 64, "{kind}");
    }
}

#[test]
fn both_plans_end_with_a_fence() {
    for path in [LINEAR, TWO_LAYER] {
        let plan = artifact(&source(path), Backend::Vulkan).execution;
        for steps in [&plan.step, &plan.forward] {
            let last = steps.last().expect("the plan has steps");
            assert_eq!(last.op, PlanOp::Barrier, "{path}");
            assert!(last.depends_on.is_empty(), "{path}");
            let fences = steps
                .iter()
                .filter(|s| s.op == PlanOp::Barrier && s.depends_on.is_empty())
                .count();
            assert_eq!(fences, 1, "{path}");
        }
    }
}

#[test]
fn a_forward_returning_its_parameter_copies_it_into_the_prediction_view() {
    let text = source(LINEAR)
        .replace("-> Tensor<f32, [8, 1]>", "-> Tensor<f32, [4, 1]>")
        .replace("return matmul(x, w);", "return w;")
        .replace(
            "target y: Tensor<f32, [8, 1]>",
            "target y: Tensor<f32, [4, 1]>",
        );
    let plan = artifact(&text, Backend::Vulkan).execution;
    let w = plan.parameters[0].views[0];
    assert_ne!(plan.prediction_view.buffer, w.buffer);
    for steps in [&plan.step, &plan.forward] {
        let copy = steps
            .iter()
            .position(|s| s.op == PlanOp::Copy)
            .expect("the plan copies the parameter");
        assert_eq!(steps[copy].reads, vec![w]);
        assert_eq!(steps[copy].writes, vec![plan.prediction_view]);
        if let Some(update) = steps
            .iter()
            .position(|s| s.writes.contains(&w) && s.op == PlanOp::Dispatch)
        {
            assert!(copy < update, "the copy reads w before its update");
        }
    }
    assert!(plan.forward.iter().all(|s| s.op != PlanOp::Dispatch));
}

#[test]
fn a_forward_returning_a_literal_fills_the_prediction_view() {
    let text = source(LINEAR)
        .replace("-> Tensor<f32, [8, 1]>", "-> Tensor<f32, []>")
        .replace("return matmul(x, w);", "return 1.0;")
        .replace("target y: Tensor<f32, [8, 1]>", "target y: Tensor<f32, []>")
        .replace(
            "loss = mse(model.forward(x), y);",
            "loss = mean(mul(model.forward(x), mul(y, sum(x))));",
        );
    let plan = artifact(&text, Backend::Vulkan).execution;
    let fill = plan
        .forward
        .iter()
        .find(|s| s.op == PlanOp::Dispatch)
        .expect("the forward plan fills the prediction");
    assert_eq!(fill.writes, vec![plan.prediction_view]);
    assert!(fill.reads.is_empty());
    assert_eq!(kind(&plan, fill), KernelKind::Broadcast);
}

#[test]
fn a_shape_too_large_to_dispatch_is_refused_with_its_target_code() {
    let text = source(LINEAR)
        .replace("[8, 4]", "[18014398509481984, 4]")
        .replace("[8, 1]", "[18014398509481984, 1]");
    let diagnostics = compile::compile_artifact("test.mspec", &text, Backend::Vulkan)
        .expect_err("2^50 workgroups do not fit in three dimensions of 65535");
    assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
    assert_eq!(diagnostics[0].code, compile::codes::TOO_LARGE_FOR_TARGET);
    assert_eq!(diagnostics[0].code.as_str(), "E-TARGET-010");
}

// Story m1-fuse-elementwise: the fusions and folded constants of DESIGN.md § 3.1.

const MNIST: &str = "examples/mnist_mlp.mspec";

#[test]
fn mnist_mlp_step_plan_covers_every_computing_op_once() {
    the_step_plan_of_a_mse_model_is_one_dispatch_per_computing_op(MNIST);
}

/// The dispatch that names `value`, which must be the only one.
fn only_dispatch(plan: &ExecutionPlan, value: ValueId) -> &PlanStep {
    let [at] = dispatches_of(&plan.step, value)[..] else {
        panic!("{value} has one dispatch\n{}", print_kernel(plan));
    };
    &plan.step[at]
}

/// The first node of `op` in the step.
fn first<'a>(training: &'a CompiledTraining, op: &TensorOp) -> &'a TensorNode {
    training
        .training
        .training
        .program
        .nodes
        .iter()
        .find(|n| &n.op == op)
        .expect("the step has the op")
}

#[test]
fn the_mnist_step_plan_is_14_dispatches_in_8_barrier_groups() {
    let plan = artifact(&source(MNIST), Backend::Vulkan).execution;
    verify_plan(&plan).expect("the plan verifies");
    let count = |op: PlanOp| plan.step.iter().filter(|s| s.op == op).count();
    assert_eq!(count(PlanOp::Dispatch), 14, "{}", print_kernel(&plan));
    assert_eq!(count(PlanOp::Barrier), 8, "{}", print_kernel(&plan));
}

#[test]
fn no_kernel_computes_from_compile_time_scalars_alone() {
    for path in [MNIST, LINEAR, TWO_LAYER] {
        let plan = artifact(&source(path), Backend::Vulkan).execution;
        for kernel in &plan.kernels {
            assert!(
                kernel
                    .operands
                    .iter()
                    .any(|o| matches!(o, Operand::Buffer(_))),
                "{path}: {}",
                kernel.signature()
            );
        }
    }
}

/// On MNIST, `relu(%4)` has a second reader of `%4`: the `ReluGrad` of the backward pass.
#[test]
fn a_relu_whose_matmul_has_a_second_reader_is_an_epilogue_writing_both() {
    let text = source(MNIST);
    let plan = artifact(&text, Backend::Vulkan).execution;
    let training = training(&text);
    let relu = first(&training, &TensorOp::Relu);
    let matmul = relu.operands[0];
    assert_eq!(node_of(&training, matmul).op, TensorOp::MatMul);

    let step = only_dispatch(&plan, matmul);
    let kernel = kernel_of(&plan, step);
    assert_eq!(kernel.kind, KernelKind::MatMul);
    assert_eq!(
        kernel.epilogue,
        Some(Epilogue::Relu {
            writes_pre_activation: true
        })
    );
    assert_eq!(
        plan.kernels[0].id, kernel.id,
        "the ReLU is the epilogue of k0"
    );
    let [pre_activation, activation] = step.writes[..] else {
        panic!("the kernel writes two views: {:?}", step.writes);
    };
    assert_ne!(pre_activation.buffer, activation.buffer);
    assert!(dispatches_of(&plan.step, relu.id).is_empty());

    let relu_grad = first(&training, &TensorOp::ReluGrad);
    assert_eq!(relu_grad.operands[0], matmul);
    assert_eq!(
        only_dispatch(&plan, relu_grad.id).reads[0],
        pre_activation,
        "ReluGrad reads the pre-activation"
    );
    for reader in readers(&training, relu.id) {
        assert!(
            only_dispatch(&plan, reader.id).reads.contains(&activation),
            "{} reads the activation",
            reader.id
        );
    }
}

/// `relu(matmul(x, transpose(x)))` with `x` an input: no gradient reads the pre-activation.
fn no_second_reader_source() -> String {
    "model Gram {
    param w: Tensor<f32, [16, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [16, 4]>)
        -> Tensor<f32, [16, 1]> {
        let h = relu(matmul(x, transpose(x)));
        return matmul(h, w);
    }
}

train FitGram {
    model = Gram;
    input x: Tensor<f32, [16, 4]>;
    target y: Tensor<f32, [16, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 10;
    seed = 7;
    device = cpu;
}
"
    .to_string()
}

#[test]
fn a_relu_whose_matmul_has_no_other_reader_writes_only_the_activation() {
    let text = no_second_reader_source();
    let plan = artifact(&text, Backend::Vulkan).execution;
    let training = training(&text);
    verify_plan(&plan).expect("the plan verifies");
    let relu = first(&training, &TensorOp::Relu);
    let matmul = relu.operands[0];
    assert_eq!(
        readers(&training, matmul).len(),
        1,
        "the relu is the matmul's only reader"
    );

    let step = only_dispatch(&plan, matmul);
    let kernel = kernel_of(&plan, step);
    assert_eq!(
        kernel.epilogue,
        Some(Epilogue::Relu {
            writes_pre_activation: false
        })
    );
    let [activation] = step.writes[..] else {
        panic!("the kernel writes the activation alone: {:?}", step.writes);
    };
    for reader in readers(&training, relu.id) {
        assert!(only_dispatch(&plan, reader.id).reads.contains(&activation));
    }

    let named: BTreeSet<u32> = plan
        .step
        .iter()
        .chain(&plan.forward)
        .flat_map(|s| s.reads.iter().chain(&s.writes))
        .map(|v| v.buffer)
        .chain(plan.data.iter().map(|d| d.view.buffer))
        .chain(parameter_buffers(&plan))
        .collect();
    assert_eq!(
        named.len(),
        plan.buffers.len(),
        "every buffer is used; none is left for a pre-activation\n{}",
        print_kernel(&plan)
    );
}

#[test]
fn each_sgd_update_is_one_kernel_reading_its_gradient() {
    for path in [MNIST, TWO_LAYER, LINEAR] {
        let text = source(path);
        let plan = artifact(&text, Backend::Vulkan).execution;
        let training = training(&text);
        let lr = training.training.optimizer.lr;
        for (param, &update) in &training.training.optimizer.updates {
            let node = node_of(&training, update);
            assert_eq!(node.op, TensorOp::Sub);
            let product = node_of(&training, node.operands[1]);
            assert_eq!(product.op, TensorOp::Mul);
            let gradient = product.operands[1];
            assert!(dispatches_of(&plan.step, product.id).is_empty(), "{path}");

            let step = only_dispatch(&plan, update);
            let kernel = kernel_of(&plan, step);
            assert_eq!(kernel.kind, KernelKind::SgdUpdate, "{path}");
            let path_of = &training.training.training.program.params[param].path;
            let view = plan
                .parameters
                .iter()
                .find(|p| &p.path == path_of)
                .expect("the parameter has a view")
                .views[0];
            assert_eq!(
                kernel.operands,
                vec![
                    Operand::Buffer(node.ty.clone()),
                    Operand::Immediate(lr),
                    Operand::Buffer(node_of(&training, gradient).ty.clone()),
                ],
                "{path}"
            );
            let gradient_view = only_dispatch(&plan, gradient).writes[0];
            assert_eq!(step.reads, vec![view, gradient_view], "{path}");
            assert_eq!(step.writes, vec![view], "{path}");
        }
    }
}

/// `%11 = Mul %8, %10` of MNIST, `1.0 * 0.002`, is the immediate of `%12`, with the bits the
/// device computed for it before: one round-to-nearest-even `f32` product.
#[test]
fn a_mul_of_two_scalars_is_folded_into_its_readers_immediate() {
    let text = source(MNIST);
    let plan = artifact(&text, Backend::Vulkan).execution;
    let training = training(&text);
    let folded = training
        .training
        .training
        .program
        .nodes
        .iter()
        .find(|n| {
            n.op == TensorOp::Mul
                && n.operands
                    .iter()
                    .all(|&o| matches!(node_of(&training, o).op, TensorOp::Scalar(_)))
        })
        .expect("the step scales the loss gradient by two scalars");
    let [a, b] = folded.operands[..] else {
        panic!("a Mul has two operands");
    };
    let (TensorOp::Scalar(a), TensorOp::Scalar(b)) =
        (&node_of(&training, a).op, &node_of(&training, b).op)
    else {
        unreachable!()
    };
    assert_eq!((a.to_bits(), b.to_bits()), (0x3F80_0000, 0x3B03_126F));
    assert_eq!((a * b).to_bits(), 0x3B03_126F);
    assert!(dispatches_of(&plan.step, folded.id).is_empty());

    let [reader] = readers(&training, folded.id)[..] else {
        panic!("one reader");
    };
    let kernel = kernel_of(&plan, only_dispatch(&plan, reader.id));
    let immediates: Vec<u32> = kernel
        .operands
        .iter()
        .filter_map(|o| match o {
            Operand::Immediate(v) => Some(v.to_bits()),
            Operand::Buffer(_) => None,
        })
        .collect();
    assert_eq!(immediates, vec![0x3B03_126F], "{}", kernel.signature());
}
