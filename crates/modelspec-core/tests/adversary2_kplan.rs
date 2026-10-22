//! Adversary pass 2 on story m1-kernel-plan: the fence semantics of `verify_plan`, the claim that
//! pass-1 case 7 (a read of a parameter after its update inside one training step) is refused,
//! the parameter-after-update check, and the prediction and loss views the artifact summary says
//! the plans write.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use modelspec_core::artifact::Backend;
use modelspec_core::compile;
use modelspec_core::ir::kernel::{BufferView, ExecutionPlan, KernelKind, PlanOp, PlanStep};
use modelspec_core::ir::verify::{verify_plan, verify_recorded_back_to_back};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn plan_of_file(path: &str) -> ExecutionPlan {
    let source = std::fs::read_to_string(repo_root().join(path)).expect("the source exists");
    plan_of(&source)
}

fn plan_of(source: &str) -> ExecutionPlan {
    let plan = compile::compile_artifact("test.mspec", source, Backend::Vulkan)
        .expect("the source compiles")
        .execution;
    verify_plan(&plan).expect("the unaltered plan verifies");
    plan
}

fn linear() -> ExecutionPlan {
    plan_of_file("examples/linear_regression.mspec")
}

fn two_layer() -> ExecutionPlan {
    plan_of_file("crates/modelspec-docs/models/two_layer_relu.mspec")
}

fn dispatches_of(plan: &ExecutionPlan, kind: KernelKind) -> Vec<usize> {
    plan.step
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            s.op == PlanOp::Dispatch
                && s.kernel_module_id
                    .and_then(|id| plan.kernel(id))
                    .is_some_and(|k| k.kind == kind)
        })
        .map(|(i, _)| i)
        .collect()
}

fn parameter_buffers(plan: &ExecutionPlan) -> BTreeSet<u32> {
    plan.parameters
        .iter()
        .flat_map(|p| p.views.iter().map(|v| v.buffer))
        .collect()
}

fn barrier(depends_on: Vec<usize>) -> PlanStep {
    PlanStep {
        op: PlanOp::Barrier,
        kernel_module_id: None,
        workgroups: None,
        reads: Vec::new(),
        writes: Vec::new(),
        depends_on,
        value: None,
    }
}

/// A fresh buffer of `count` f32 elements and its whole view.
fn fresh_view(plan: &mut ExecutionPlan, count: u64) -> BufferView {
    let buffer = u32::try_from(plan.buffers.len()).expect("few buffers");
    plan.buffers.push(count * 4);
    BufferView {
        buffer,
        element_count: count,
        offset: 0,
        size_bytes: count * 4,
        alignment: 4,
        ..plan.loss_view
    }
}

/// Pass-1 case 7 exactly as it was written (a barrier after the update, then the forward matmul
/// again), with the one change the new `PlanNotFenced` rule asks for: a fence after it. The
/// artifact's `plan` is "the execution DAG of one training step" that "ends with its only
/// fence", and "within one recording no step reads a parameter after its update"; a runner
/// records this whole plan once per training step, so the appended matmul reads version n + 1
/// inside step n.
#[test]
fn pass_1_case_7_with_a_fence_appended_is_still_refused() {
    let mut plan = linear();
    let update = dispatches_of(&plan, KernelKind::SgdUpdate)[0];
    let first = plan.step[0].clone();
    assert!(
        first
            .reads
            .iter()
            .any(|v| parameter_buffers(&plan).contains(&v.buffer)),
        "the first dispatch reads the parameter"
    );
    let after_update = plan.step.len();
    plan.step.push(barrier(vec![update]));
    plan.step.push(PlanStep {
        depends_on: vec![after_update],
        ..first
    });
    plan.step.push(barrier(Vec::new()));
    let fences = plan
        .step
        .iter()
        .filter(|s| s.op == PlanOp::Barrier && s.depends_on.is_empty())
        .count();
    assert_eq!(fences, 2, "the altered plan has a second fence");
    let result = verify_plan(&plan);
    assert!(
        result.is_err(),
        "a step plan with two fences, whose second segment reads `w` after the update of the \
         same training step, verifies: {result:?}"
    );
}

/// `ReadAfterUpdate` looks only at the first parameter view a step reads. A step between the
/// two SGD updates of the two-layer model that reads `w2` (not yet updated) and then `w1`
/// (already updated) reads version n + 1 of `w1` inside step n, and is ordered correctly
/// otherwise (barriers on both sides), so only that check could refuse it.
#[test]
fn a_step_reading_an_updated_parameter_as_its_second_operand_is_refused() {
    let mut plan = two_layer();
    let updates = dispatches_of(&plan, KernelKind::SgdUpdate);
    let [first_update, second_update] = updates[..] else {
        panic!("two SGD updates, got {updates:?}");
    };
    let updated = plan.step[first_update].writes[0];
    let pending = plan.step[second_update].writes[0];
    assert_ne!(updated.buffer, pending.buffer);
    let guard = plan.step[second_update].depends_on.clone();
    assert_eq!(guard.len(), 1, "the second update waits on one barrier");
    let guard = guard[0];
    assert!(guard > first_update && plan.step[guard].op == PlanOp::Barrier);

    let out = fresh_view(&mut plan, pending.element_count);
    let at = second_update;
    let reader = PlanStep {
        op: PlanOp::Copy,
        kernel_module_id: None,
        workgroups: None,
        reads: vec![pending, updated],
        writes: vec![out],
        depends_on: vec![guard],
        value: None,
    };
    let mut steps: Vec<PlanStep> = plan.step[..at].to_vec();
    steps.push(reader);
    steps.push(barrier(vec![guard, at]));
    for step in &plan.step[at..] {
        let mut moved = step.clone();
        for on in &mut moved.depends_on {
            if *on >= at {
                *on += 2;
            }
        }
        steps.push(moved);
    }
    steps[at + 2].depends_on = vec![at + 1];
    plan.step = steps;

    let result = verify_plan(&plan);
    assert!(
        result.is_err(),
        "step {at} reads parameter buffer b{} after update step {first_update} wrote it in the \
         same recording: {result:?}",
        updated.buffer
    );
}

/// The artifact summary: `prediction_view` is "the plan buffer view the forward plan and the step
/// plan write the model's prediction into". A prediction view in a buffer no step writes leaves
/// the data loader reading garbage.
#[test]
fn a_prediction_view_no_plan_writes_is_refused() {
    let mut plan = linear();
    let count = plan.prediction_view.element_count;
    plan.prediction_view = fresh_view(&mut plan, count);
    let result = verify_plan(&plan);
    assert!(
        result.is_err(),
        "the prediction view names buffer b{}, which neither plan writes: {result:?}",
        plan.prediction_view.buffer
    );
}

/// The artifact summary: `loss_view` is "the plan buffer view the step plan writes the rank-0
/// loss into". Here it is eight elements in a buffer no step writes.
#[test]
fn a_loss_view_the_step_plan_never_writes_is_refused() {
    let mut plan = linear();
    plan.loss_view = fresh_view(&mut plan, 8);
    let result = verify_plan(&plan);
    assert!(
        result.is_err(),
        "the loss view is eight elements in buffer b{}, which no step writes: {result:?}",
        plan.loss_view.buffer
    );
}

/// The fence claim itself, across every source shape this story schedules: each plan recorded
/// three times back to back verifies as one plan.
#[test]
fn every_plan_recorded_three_times_verifies() {
    let linear_source =
        std::fs::read_to_string(repo_root().join("examples/linear_regression.mspec")).unwrap();
    let mean = linear_source.replace(
        "loss = mse(model.forward(x), y);",
        "loss = mean(mul(model.forward(x), y));",
    );
    let returns_w = linear_source
        .replace("-> Tensor<f32, [8, 1]> {", "-> Tensor<f32, [4, 1]> {")
        .replace("return matmul(x, w);", "return w;")
        .replace(
            "target y: Tensor<f32, [8, 1]>;",
            "target y: Tensor<f32, [4, 1]>;",
        )
        .replace(
            "loss = mse(model.forward(x), y);",
            "loss = mean(mul(model.forward(x), mul(y, sum(x))));",
        );
    let plans = [linear(), two_layer(), plan_of(&mean), plan_of(&returns_w)];
    for (n, plan) in plans.iter().enumerate() {
        let result = verify_recorded_back_to_back(plan, 3);
        assert!(
            result.is_ok(),
            "plan {n} (step and forward) recorded three times: {result:?}"
        );
    }
}
