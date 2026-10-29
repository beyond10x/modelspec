//! `ExecutionPlan::prediction_steps`, the steps a held-out evaluation records
//! (story:bert-tiny-vs-pytorch-cuda, diagnosis of the step bottleneck): the loss steps without
//! the loss and without any step only the loss needs, so a held-out batch runs no loss sum. On
//! `examples/bert_tiny.mspec` the loss sum is the `CrossEntropy` kernel, one workgroup that took
//! 36 ms of each held-out batch on the RTX 3090.

use std::collections::BTreeSet;

use modelspec_core::artifact::Backend;
use modelspec_core::compile;
use modelspec_core::ir::kernel::{ExecutionPlan, PlanOp};
use modelspec_core::ir::verify::verify_plan;

fn plan(name: &str, source: &str) -> ExecutionPlan {
    let artifact = compile::compile_artifact(name, source, Backend::Vulkan)
        .unwrap_or_else(|d| panic!("{name}: compile_artifact refused: {d:#?}"));
    verify_plan(&artifact.execution).unwrap_or_else(|e| panic!("{name}: {e}"));
    artifact.execution
}

fn examples() -> [(&'static str, &'static str); 3] {
    [
        (
            "bert_tiny.mspec",
            include_str!("../../../examples/bert_tiny.mspec"),
        ),
        (
            "mnist_mlp.mspec",
            include_str!("../../../examples/mnist_mlp.mspec"),
        ),
        (
            "transformer_block.mspec",
            include_str!("../../../examples/transformer_block.mspec"),
        ),
    ]
}

fn writes_loss(plan: &ExecutionPlan, at: usize) -> bool {
    plan.step[at]
        .writes
        .iter()
        .any(|view| view.buffer == plan.loss_view.buffer)
}

#[test]
fn the_prediction_steps_are_the_loss_steps_without_the_loss() {
    for (name, source) in examples() {
        let plan = plan(name, source);
        let loss_steps = plan.loss_steps();
        let chosen = plan.prediction_steps();
        let loss_set: BTreeSet<usize> = loss_steps.iter().copied().collect();
        assert!(
            chosen.iter().all(|at| loss_set.contains(at)),
            "{name}: a prediction step that is no loss step"
        );
        assert!(
            chosen.windows(2).all(|pair| pair[0] < pair[1]),
            "{name}: the prediction steps are not in plan order"
        );
        let dispatches = |steps: &[usize]| -> Vec<usize> {
            steps
                .iter()
                .copied()
                .filter(|&at| plan.step[at].op == PlanOp::Dispatch)
                .collect()
        };
        assert!(
            dispatches(&chosen)
                .iter()
                .all(|&at| !writes_loss(&plan, at)),
            "{name}: a prediction step writes the loss"
        );
        let writer = dispatches(&chosen)
            .into_iter()
            .chain(
                chosen
                    .iter()
                    .copied()
                    .filter(|&at| plan.step[at].op == PlanOp::Copy),
            )
            .any(|at| {
                plan.step[at]
                    .writes
                    .iter()
                    .any(|view| view.buffer == plan.prediction_view.buffer)
            });
        assert!(writer, "{name}: no prediction step writes the prediction");
        let dropped: Vec<usize> = dispatches(&loss_steps)
            .into_iter()
            .filter(|at| !chosen.contains(at))
            .collect();
        assert!(
            dropped.iter().all(|&at| !plan.step[at]
                .writes
                .iter()
                .any(|view| view.buffer == plan.prediction_view.buffer)),
            "{name}: a dropped loss step writes the prediction"
        );
        assert!(
            dropped.iter().any(|&at| writes_loss(&plan, at)),
            "{name}: the loss sum is still among the prediction steps"
        );
    }
}

#[test]
fn bert_tiny_drops_exactly_the_cross_entropy_loss_sum() {
    let (name, source) = examples()[0];
    let plan = plan(name, source);
    let chosen: BTreeSet<usize> = plan.prediction_steps().into_iter().collect();
    let dropped: Vec<&str> = plan
        .loss_steps()
        .into_iter()
        .filter(|at| !chosen.contains(at) && plan.step[*at].op == PlanOp::Dispatch)
        .map(|at| {
            let id = plan.step[at]
                .kernel_module_id
                .expect("a dispatch names its kernel");
            plan.kernel(id).expect("the plan has the kernel").op
        })
        .collect();
    assert_eq!(dropped, ["CrossEntropy"]);
}
