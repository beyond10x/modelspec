//! Adversary cases for `compile_artifact` and `schedule` (story m1-kernel-plan) on sources the
//! checker accepts: every checked source schedules without a compiler bug, and the prediction and
//! loss views are what `spec/domains/artifact.yaml` says they are.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use modelspec_core::artifact::Backend;
use modelspec_core::compile::{self, codes::COMPILER_BUG};
use modelspec_core::ir::kernel::{ExecutionPlan, PlanOp};
use modelspec_core::ir::verify::verify_plan;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(path: &str) -> String {
    std::fs::read_to_string(repo_root().join(path)).expect("the source exists")
}

fn linear_with(forward: &str, target: &str, loss: &str) -> String {
    format!(
        "model LinearReg {{
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> {forward}
}}

train Fit {{
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: {target};
    loss = {loss};
    optimizer = sgd(lr = 0.1);
    steps = 5;
    seed = 42;
    device = cpu;
}}
"
    )
}

fn parameter_buffers(plan: &ExecutionPlan) -> BTreeSet<u32> {
    plan.parameters
        .iter()
        .flat_map(|p| p.views.iter().map(|v| v.buffer))
        .collect()
}

/// `compile_training` accepts `text`; `compile_artifact` must not report a compiler bug for it.
fn artifact_of_checked(text: &str) -> ExecutionPlan {
    if let Err(diagnostics) = compile::compile_training("test.mspec", text) {
        panic!("precondition: the source compiles for training: {diagnostics:?}");
    }
    match compile::compile_artifact("test.mspec", text, Backend::Vulkan) {
        Ok(artifact) => artifact.execution,
        Err(diagnostics) => {
            let bugs: Vec<_> = diagnostics
                .iter()
                .filter(|d| d.code == COMPILER_BUG)
                .map(|d| d.message.clone())
                .collect();
            panic!("a source the checker accepts reaches {COMPILER_BUG}: {bugs:?}");
        }
    }
}

#[test]
fn a_forward_returning_a_literal_schedules() {
    let text = linear_with(
        "Tensor<f32, []> {\n        return 1.0;\n    }",
        "Tensor<f32, []>",
        "mean(mul(model.forward(x), y))",
    );
    let plan = artifact_of_checked(&text);
    verify_plan(&plan).expect("the plan verifies");
}

#[test]
fn a_forward_returning_its_parameter_has_a_prediction_view_apart_from_the_parameter() {
    let text = linear_with(
        "Tensor<f32, [4, 1]> {\n        return w;\n    }",
        "Tensor<f32, [4, 1]>",
        "mse(model.forward(x), y)",
    );
    let plan = artifact_of_checked(&text);
    assert!(
        !parameter_buffers(&plan).contains(&plan.prediction_view.buffer),
        "the prediction view is the parameter view the step plan updates in place: {}",
        plan.prediction_view
    );
}

#[test]
fn a_shape_too_large_to_dispatch_is_not_a_compiler_bug() {
    let text = source("examples/linear_regression.mspec")
        .replace("[8, 4]", "[18014398509481984, 4]")
        .replace("[8, 1]", "[18014398509481984, 1]");
    let Ok(_) = compile::compile_training("test.mspec", &text) else {
        return;
    };
    if let Err(diagnostics) = compile::compile_artifact("test.mspec", &text, Backend::Vulkan) {
        assert!(
            diagnostics.iter().all(|d| d.code != COMPILER_BUG),
            "a source the checker accepts reaches {COMPILER_BUG}: {diagnostics:?}"
        );
    }
}

#[test]
fn a_reduce_inside_the_forward_is_scheduled_in_both_plans() {
    let text = linear_with(
        "Tensor<f32, []> {\n        return sum(matmul(x, w));\n    }",
        "Tensor<f32, []>",
        "mean(mul(model.forward(x), y))",
    );
    let plan = artifact_of_checked(&text);
    verify_plan(&plan).expect("the plan verifies");
    let allocates = |steps: &[_]| {
        steps
            .iter()
            .filter(|s: &&modelspec_core::ir::kernel::PlanStep| s.op == PlanOp::Allocate)
            .count()
    };
    assert_eq!(allocates(&plan.forward), 1);
    assert_eq!(allocates(&plan.step), 2);
    let writes_prediction = plan
        .forward
        .iter()
        .any(|s| s.writes.contains(&plan.prediction_view));
    assert!(writes_prediction);
}

#[test]
fn the_prediction_and_loss_views_are_written_by_the_plans_that_name_them() {
    for path in [
        "examples/linear_regression.mspec",
        "crates/modelspec-docs/models/two_layer_relu.mspec",
    ] {
        let text = source(path);
        let compiled = compile::compile_training("test.mspec", &text).expect("compiles");
        let plan = artifact_of_checked(&text);
        let last_forward = plan
            .forward
            .iter()
            .rev()
            .find(|s| s.op == PlanOp::Dispatch)
            .expect("the forward plan dispatches");
        assert_eq!(last_forward.writes, vec![plan.prediction_view], "{path}");
        assert_eq!(
            last_forward.value,
            Some(compiled.forward.prediction),
            "{path}"
        );
        let forward_node = compiled
            .forward
            .program
            .node(compiled.forward.prediction)
            .expect("the prediction is a forward value");
        let step_node = compiled
            .training
            .training
            .program
            .node(compiled.forward.prediction)
            .expect("the prediction is a step value");
        assert_eq!(forward_node.op.name(), step_node.op.name(), "{path}");
        assert_eq!(forward_node.ty, step_node.ty, "{path}");
        let loss_writers = plan
            .step
            .iter()
            .filter(|s| s.writes.contains(&plan.loss_view))
            .count();
        assert_eq!(loss_writers, 1, "{path}");
    }
}
