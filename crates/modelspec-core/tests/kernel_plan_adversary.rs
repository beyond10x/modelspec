//! Adversary cases for the plan verifier (`verify_plan`, story m1-kernel-plan acceptance 4): ten
//! altered plans of real compiles, each refused with its own error, and the two-view parameter
//! the type admits, accepted.

use std::path::{Path, PathBuf};

use modelspec_core::artifact::Backend;
use modelspec_core::compile;
use modelspec_core::ir::kernel::{BufferView, ExecutionPlan, KernelKind, PlanOp, PlanStep};
use modelspec_core::ir::verify::{
    PlanName, PlanVerifyError, verify_plan, verify_recorded_back_to_back,
};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn linear_source() -> String {
    std::fs::read_to_string(repo_root().join("examples/linear_regression.mspec"))
        .expect("the example exists")
}

fn plan_of(source: &str) -> ExecutionPlan {
    let plan = compile::compile_artifact("test.mspec", source, Backend::Vulkan)
        .expect("the source compiles")
        .execution;
    verify_plan(&plan).expect("the unaltered plan verifies");
    plan
}

/// The verified step plan of the linear-regression example.
fn linear() -> ExecutionPlan {
    plan_of(&linear_source())
}

/// The verified step plan of the example with the loss `mean(mul(model.forward(x), y))`.
fn mean_loss() -> ExecutionPlan {
    let source = linear_source().replace(
        "loss = mse(model.forward(x), y);",
        "loss = mean(mul(model.forward(x), y));",
    );
    plan_of(&source)
}

fn refused(plan: &ExecutionPlan) -> PlanVerifyError {
    verify_plan(plan).expect_err("the verifier refuses the altered plan")
}

/// The position of the first step-plan Dispatch whose kernel is of `kind`.
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

#[test]
fn a_dependency_on_a_later_step_is_refused() {
    let mut plan = linear();
    let last = plan.step.len() - 1;
    plan.step[0].depends_on.push(last);
    let error = refused(&plan);
    assert!(
        matches!(error, PlanVerifyError::DependencyNotEarlier { step: 0, on, .. } if on == last),
        "{error:?}"
    );
}

#[test]
fn an_update_that_does_not_depend_on_a_reader_of_its_parameter_is_refused() {
    let mut plan = linear();
    let update = dispatch_of(&plan, KernelKind::SgdUpdate);
    plan.step[update].depends_on.clear();
    let error = refused(&plan);
    assert!(
        matches!(error, PlanVerifyError::UpdateMissesReader { update: u, .. } if u == update),
        "{error:?}"
    );
}

#[test]
fn a_view_past_the_end_of_its_buffer_is_refused() {
    let mut plan = linear();
    let matmul = dispatch_of(&plan, KernelKind::MatMul);
    plan.step[matmul].reads[0].offset += 4;
    let error = refused(&plan);
    assert!(
        matches!(error, PlanVerifyError::ViewOutsideBuffer { .. }),
        "{error:?}"
    );
}

#[test]
fn a_workgroup_of_129_invocations_is_refused() {
    let mut plan = linear();
    let kernel = plan
        .kernels
        .iter_mut()
        .find(|k| k.kind == KernelKind::MatMul)
        .expect("a matmul kernel");
    kernel.workgroup_size = 129;
    let error = refused(&plan);
    assert!(
        matches!(error, PlanVerifyError::WorkgroupTooLarge { size: 129, .. }),
        "{error:?}"
    );
}

#[test]
fn a_workgroup_of_128_invocations_is_accepted() {
    let mut plan = linear();
    for kernel in plan.kernels.iter_mut() {
        if kernel.kind == KernelKind::MatMul {
            kernel.workgroup_size = 128;
        }
    }
    verify_plan(&plan).expect("128 invocations is within every device's limit");
}

#[test]
fn a_dimension_of_65536_workgroups_is_refused() {
    let mut plan = linear();
    let matmul = dispatch_of(&plan, KernelKind::MatMul);
    plan.step[matmul].workgroups = Some([1, 65_536, 1]);
    let error = refused(&plan);
    assert!(
        matches!(
            error,
            PlanVerifyError::TooManyWorkgroups { step, dimension: 1, count: 65_536, .. }
                if step == matmul
        ),
        "{error:?}"
    );
}

#[test]
fn a_dispatch_without_a_kernel_module_is_refused() {
    let mut plan = linear();
    let matmul = dispatch_of(&plan, KernelKind::MatMul);
    plan.step[matmul].kernel_module_id = None;
    let error = refused(&plan);
    assert!(
        matches!(error, PlanVerifyError::MissingKernelModule { step, .. } if step == matmul),
        "{error:?}"
    );
}

#[test]
fn a_dispatch_naming_a_module_the_plan_does_not_have_is_refused() {
    let mut plan = linear();
    let matmul = dispatch_of(&plan, KernelKind::MatMul);
    let module = plan.step[matmul].kernel_module_id.unwrap();
    plan.kernels.retain(|k| k.id != module);
    let error = refused(&plan);
    assert!(
        matches!(error, PlanVerifyError::MissingKernelModule { step, .. } if step == matmul),
        "{error:?}"
    );
}

#[test]
fn a_stage_2_whose_partials_no_earlier_allocate_made_is_refused() {
    let mut plan = mean_loss();
    let stage1 = dispatch_of(&plan, KernelKind::ReduceStage1);
    assert_eq!(plan.step[stage1 - 1].op, PlanOp::Allocate);
    plan.step[stage1 - 1].writes.clear();
    let error = refused(&plan);
    assert!(
        matches!(error, PlanVerifyError::PartialsNotAllocated { step, .. } if step == stage1 + 2),
        "{error:?}"
    );
}

#[test]
fn a_stage_2_without_a_barrier_after_its_stage_1_is_refused() {
    let mut plan = mean_loss();
    let stage1 = dispatch_of(&plan, KernelKind::ReduceStage1);
    let stage2 = dispatch_of(&plan, KernelKind::ReduceStage2);
    plan.step[stage2].depends_on = vec![stage1];
    let error = refused(&plan);
    assert!(
        matches!(error, PlanVerifyError::Stage2WithoutBarrier { step, .. } if step == stage2),
        "{error:?}"
    );
}

#[test]
fn a_parameter_with_no_view_is_refused() {
    let mut plan = linear();
    plan.parameters[0].views.clear();
    let error = refused(&plan);
    assert!(
        matches!(error, PlanVerifyError::ParameterWithoutView { .. }),
        "{error:?}"
    );
}

#[test]
fn a_parameter_with_three_views_is_refused() {
    let mut plan = linear();
    let view = plan.parameters[0].views[0];
    plan.parameters[0].views = vec![view, view, view];
    let error = refused(&plan);
    assert!(
        matches!(
            error,
            PlanVerifyError::TooManyParameterViews { count: 3, .. }
        ),
        "{error:?}"
    );
}

#[test]
fn a_parameter_with_two_views_is_accepted() {
    let mut plan = linear();
    let mut second = plan.parameters[0].views[0];
    second.buffer = plan.buffers.len() as u32;
    plan.buffers.push(second.size_bytes);
    plan.parameters[0].views.push(second);
    verify_plan(&plan).expect("ping-pong views are admitted (DESIGN.md § 6.3 item 3)");
}

#[test]
fn a_loss_view_in_a_parameter_buffer_is_refused() {
    let mut plan = linear();
    let parameter = plan.parameters[0].views[0];
    plan.loss_view = BufferView {
        element_count: 1,
        size_bytes: 4,
        ..parameter
    };
    let error = refused(&plan);
    assert!(
        matches!(&error, PlanVerifyError::BoundViewAliasesParameter { name, .. } if name == "loss"),
        "{error:?}"
    );
}

#[test]
fn a_one_element_loss_view_no_step_writes_is_refused() {
    let mut plan = linear();
    let buffer = u32::try_from(plan.buffers.len()).expect("few buffers");
    plan.buffers.push(4);
    plan.loss_view = BufferView {
        buffer,
        ..plan.loss_view
    };
    let error = refused(&plan);
    assert_eq!(
        error,
        PlanVerifyError::BoundViewNotWritten {
            name: "loss".to_owned(),
            plan: PlanName::Step
        }
    );
}

#[test]
fn a_loss_view_of_more_than_one_element_is_refused() {
    let mut plan = linear();
    let prediction = plan.prediction_view;
    plan.loss_view = prediction;
    let error = refused(&plan);
    assert_eq!(error, PlanVerifyError::LossViewNotScalar { count: 8 });
}

#[test]
fn a_parameter_view_of_another_element_count_is_refused() {
    let mut plan = linear();
    let view = &mut plan.parameters[0].views[0];
    view.element_count = 1;
    view.size_bytes = 4;
    let error = refused(&plan);
    assert!(
        matches!(
            error,
            PlanVerifyError::ViewDisagreesWithType { count: 1, .. }
        ),
        "{error:?}"
    );
}

#[test]
fn a_dispatch_with_other_workgroups_than_its_kernel_is_refused() {
    let mut plan = linear();
    let loss = dispatch_of(&plan, KernelKind::LossSum);
    plan.step[loss].workgroups = Some([2, 1, 1]);
    let error = refused(&plan);
    assert!(
        matches!(error, PlanVerifyError::WorkgroupsDisagree { step, .. } if step == loss),
        "{error:?}"
    );
}

#[test]
fn a_copy_racing_a_dispatch_is_refused() {
    let mut plan = linear();
    let x = plan.data[0].view;
    for step in &mut plan.step {
        for on in &mut step.depends_on {
            *on += 1;
        }
    }
    plan.step.insert(
        0,
        PlanStep {
            op: PlanOp::Copy,
            kernel_module_id: None,
            workgroups: None,
            reads: vec![x],
            writes: vec![x],
            depends_on: Vec::new(),
            value: None,
        },
    );
    let error = refused(&plan);
    assert!(
        matches!(
            error,
            PlanVerifyError::UnsyncedAccess {
                step: 1,
                earlier: 0,
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn a_read_of_a_parameter_after_its_update_in_the_same_recording_is_refused() {
    let mut plan = linear();
    let update = dispatch_of(&plan, KernelKind::SgdUpdate);
    let fence = plan.step.pop().expect("the step plan ends with its fence");
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
    let first = plan.step[0].clone();
    plan.step.push(PlanStep {
        depends_on: vec![barrier],
        ..first
    });
    plan.step.push(fence);
    let error = refused(&plan);
    assert!(
        matches!(error, PlanVerifyError::ReadAfterUpdate { update: u, .. } if u == update),
        "{error:?}"
    );
}

#[test]
fn a_step_plan_without_its_closing_fence_is_refused() {
    let mut plan = linear();
    let fence = plan.step.pop().expect("the step plan ends with a step");
    assert_eq!(fence.op, PlanOp::Barrier);
    assert!(fence.depends_on.is_empty());
    let error = refused(&plan);
    assert!(
        matches!(
            error,
            PlanVerifyError::PlanNotFenced {
                plan: PlanName::Step
            }
        ),
        "{error:?}"
    );
}

#[test]
fn the_mean_loss_step_plan_recorded_twice_verifies() {
    let plan = mean_loss();
    let mut twice = plan.clone();
    let shift = plan.step.len();
    for step in &plan.step {
        let mut again = step.clone();
        again.depends_on = step.depends_on.iter().map(|on| on + shift).collect();
        twice.step.push(again);
    }
    verify_recorded_back_to_back(&plan, 2)
        .expect("each recording allocates and reduces after the fence before it");
    let error = verify_plan(&twice).expect_err("one plan has one fence, its last step");
    assert_eq!(
        error,
        PlanVerifyError::FenceNotLast {
            plan: PlanName::Step,
            step: shift - 1
        }
    );
}

#[test]
fn the_ten_refusals_are_ten_different_errors() {
    let mut names = std::collections::BTreeSet::new();
    let mut plans = Vec::new();
    let mut add = |alter: &dyn Fn(&mut ExecutionPlan), base: fn() -> ExecutionPlan| {
        let mut plan = base();
        alter(&mut plan);
        plans.push(plan);
    };
    add(
        &|p| {
            let last = p.step.len() - 1;
            p.step[0].depends_on.push(last)
        },
        linear,
    );
    add(
        &|p| {
            let u = dispatch_of(p, KernelKind::SgdUpdate);
            p.step[u].depends_on.clear()
        },
        linear,
    );
    add(
        &|p| {
            let m = dispatch_of(p, KernelKind::MatMul);
            p.step[m].reads[0].offset += 4
        },
        linear,
    );
    add(
        &|p| {
            p.kernels[0].workgroup_size = 129;
        },
        linear,
    );
    add(
        &|p| {
            let m = dispatch_of(p, KernelKind::MatMul);
            p.step[m].workgroups = Some([65_536, 1, 1])
        },
        linear,
    );
    add(
        &|p| {
            let m = dispatch_of(p, KernelKind::MatMul);
            p.step[m].kernel_module_id = None
        },
        linear,
    );
    add(
        &|p| {
            let s = dispatch_of(p, KernelKind::ReduceStage1);
            p.step[s - 1].writes.clear()
        },
        mean_loss,
    );
    add(
        &|p| {
            let s1 = dispatch_of(p, KernelKind::ReduceStage1);
            let s2 = dispatch_of(p, KernelKind::ReduceStage2);
            p.step[s2].depends_on = vec![s1]
        },
        mean_loss,
    );
    add(&|p| p.parameters[0].views.clear(), linear);
    add(
        &|p| {
            let v = p.parameters[0].views[0];
            p.parameters[0].views = vec![v, v, v]
        },
        linear,
    );
    for plan in &plans {
        let error = refused(plan);
        let debug = format!("{error:?}");
        let name = debug
            .split(|c: char| !c.is_alphanumeric())
            .next()
            .unwrap()
            .to_owned();
        names.insert(name);
    }
    assert_eq!(names.len(), 10, "{names:?}");
}
