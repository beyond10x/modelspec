//! Adversary cases for `verify_plan` (story m1-kernel-plan): hand-altered plans of real compiles
//! that a runner could not execute safely, each expected to be refused, and the step plan recorded
//! back to back, as the artifact's `plan` summary says a runner may.

use std::collections::BTreeSet;
use std::path::Path;

use modelspec_core::artifact::Backend;
use modelspec_core::compile;
use modelspec_core::ir::kernel::{BufferView, ExecutionPlan, KernelKind, PlanOp, PlanStep};
use modelspec_core::ir::verify::{verify_plan, verify_recorded_back_to_back};

fn linear() -> ExecutionPlan {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = std::fs::read_to_string(root.join("examples/linear_regression.mspec"))
        .expect("the example exists");
    let plan = compile::compile_artifact("test.mspec", &source, Backend::Vulkan)
        .expect("the example compiles")
        .execution;
    verify_plan(&plan).expect("the unaltered plan verifies");
    plan
}

fn dispatch_of(plan: &ExecutionPlan, kind: KernelKind) -> usize {
    plan.step
        .iter()
        .position(|s| {
            s.op == PlanOp::Dispatch
                && s.kernel_module_id
                    .and_then(|id| plan.kernel(id))
                    .is_some_and(|k| k.kind == kind)
        })
        .unwrap_or_else(|| panic!("the plan has a {kind} dispatch"))
}

fn parameter_buffers(plan: &ExecutionPlan) -> BTreeSet<u32> {
    plan.parameters
        .iter()
        .flat_map(|p| p.views.iter().map(|v| v.buffer))
        .collect()
}

#[test]
fn a_dispatch_reading_a_write_with_no_barrier_between_is_refused() {
    let mut plan = linear();
    let loss = dispatch_of(&plan, KernelKind::LossSum);
    plan.step[loss].depends_on.clear();
    assert!(
        verify_plan(&plan).is_err(),
        "a read of the prediction races its write"
    );
}

#[test]
fn the_step_plan_recorded_twice_back_to_back_verifies() {
    let plan = linear();
    let result = verify_recorded_back_to_back(&plan, 2);
    assert!(
        result.is_ok(),
        "the artifact says a runner may record the step plan many times into one command \
         buffer, yet recording it twice is refused: {result:?}"
    );
}

#[test]
fn a_data_view_aliasing_a_parameter_buffer_is_refused() {
    let mut plan = linear();
    plan.data[0].view = plan.parameters[0].views[0];
    assert!(
        verify_plan(&plan).is_err(),
        "a runner writing a batch of `x` would overwrite parameter `w`"
    );
}

#[test]
fn a_loss_view_inside_a_parameter_buffer_is_refused() {
    let mut plan = linear();
    let parameter = plan.parameters[0].views[0];
    plan.loss_view = BufferView {
        element_count: 1,
        size_bytes: 4,
        ..parameter
    };
    assert!(parameter_buffers(&plan).contains(&plan.loss_view.buffer));
    assert!(
        verify_plan(&plan).is_err(),
        "the loss view names the buffer the update writes"
    );
}

#[test]
fn a_parameter_view_smaller_than_its_parameter_is_refused() {
    let mut plan = linear();
    let view = &mut plan.parameters[0].views[0];
    view.element_count = 1;
    view.size_bytes = 4;
    assert!(
        verify_plan(&plan).is_err(),
        "a [4, 1] parameter bound to a view of one element"
    );
}

#[test]
fn a_loss_sum_dispatched_over_two_workgroups_is_refused() {
    let mut plan = linear();
    let loss = dispatch_of(&plan, KernelKind::LossSum);
    plan.step[loss].workgroups = Some([2, 1, 1]);
    assert!(
        verify_plan(&plan).is_err(),
        "the single-invocation loss sum dispatched twice races on the loss"
    );
}

#[test]
fn a_copy_writing_a_buffer_a_dispatch_then_reads_unsynced_is_refused() {
    let mut plan = linear();
    let x = plan.data[0].view;
    let copy = PlanStep {
        op: PlanOp::Copy,
        kernel_module_id: None,
        workgroups: None,
        reads: vec![x],
        writes: vec![x],
        depends_on: Vec::new(),
        value: None,
    };
    for step in &mut plan.step {
        for on in &mut step.depends_on {
            *on += 1;
        }
    }
    plan.step.insert(0, copy);
    assert_eq!(plan.step[1].op, PlanOp::Dispatch);
    assert!(plan.step[1].reads.iter().any(|v| v.buffer == x.buffer));
    assert!(
        verify_plan(&plan).is_err(),
        "the first dispatch reads `x` while the copy writes it, with no barrier"
    );
}

#[test]
fn a_dispatch_reading_its_parameter_after_the_update_is_refused() {
    let mut plan = linear();
    let update = dispatch_of(&plan, KernelKind::SgdUpdate);
    let first = plan.step[0].clone();
    assert!(
        first
            .reads
            .iter()
            .any(|v| parameter_buffers(&plan).contains(&v.buffer))
    );
    let barrier = plan.step.len();
    plan.step.push(PlanStep {
        op: PlanOp::Barrier,
        kernel_module_id: None,
        workgroups: None,
        reads: Vec::new(),
        writes: Vec::new(),
        depends_on: vec![update],
        value: None,
    });
    plan.step.push(PlanStep {
        depends_on: vec![barrier],
        ..first
    });
    assert!(
        verify_plan(&plan).is_err(),
        "the forward matmul after the in-place update reads version n + 1 inside step n"
    );
}
