//! Kernel IR and the execution plan of a training step (DESIGN.md § 3.1, § 5.3, § 7.1;
//! story m1-kernel-plan).
//!
//! Every case compiles a real `.mspec` file through `compile::compile_artifact` and reads the
//! artifact's [`ExecutionPlan`]: one Dispatch per computing op of the training step, the `mse` loss
//! sum as one Dispatch of one invocation, every other reduce as `Allocate`, stage 1, `Barrier`,
//! stage 2, no `Complete` step, one view per parameter, and a forward plan that writes no
//! parameter.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{self, CompiledTraining};
use modelspec_core::ir::kernel::{ExecutionPlan, KernelKind, PlanOp, PlanStep};
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

/// Acceptance 2 for one model file.
fn the_step_plan_of_a_mse_model_is_one_dispatch_per_computing_op(path: &str) {
    let text = source(path);
    let artifact = artifact(&text, Backend::Vulkan);
    let plan = &artifact.execution;
    let training = training(&text);
    verify_plan(plan).expect("the plan verifies");

    let computing = computing(&training);
    assert!(!computing.is_empty());
    for node in &computing {
        let dispatches = dispatches_of(&plan.step, node.id);
        assert_eq!(
            dispatches.len(),
            1,
            "{} {} has {} dispatches\n{}",
            node.id,
            node.op.name(),
            dispatches.len(),
            print_kernel(plan)
        );
    }
    let all: usize = plan
        .step
        .iter()
        .filter(|s| s.op == PlanOp::Dispatch)
        .count();
    assert_eq!(
        all,
        computing.len(),
        "no Dispatch outside the computing ops"
    );

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
    assert_eq!(kernel.workgroup_size, 1);
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
        assert_eq!(
            dispatches_of(&plan.step, node.id).len(),
            1,
            "{} {}",
            node.id,
            node.op.name()
        );
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
    assert_eq!(KernelKind::LossSum.workgroup_size(), 1);
    for kind in [
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
