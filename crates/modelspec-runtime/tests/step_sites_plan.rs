//! story:step-sites-beat-pytorch, without a device: the plans of the `mlp_sweep` b8192 h4096
//! point and of `examples/mnist_mlp.mspec`, scheduled for the RTX 3090 profile
//! (`tests/fixtures/rtx3090.device.json`).
//!
//! - The `mse` loss sum, which no op of the step reads, is recorded right after the matmul with
//!   the most FLOP among the dispatches after it that touch none of its buffers, the gradient of
//!   `w1`, with no barrier between the two, so the one-workgroup sum runs beside that matmul
//!   instead of in a barrier interval of its own (`benchmarks/raw/2026-10-25-step-sites/`).
//! - The loss evaluation after a run and `infer` record only the steps the loss is computed from
//!   ([`ExecutionPlan::loss_steps`]), not the gradients recorded before the moved loss sum.
//! - The `relu` gradient is the epilogue of the matmul that produces its adjoint, `[B, 10] x
//!   [10, H]ᵀ`: it reads the pre-activation as a third operand and writes `x > 0 ? sum : 0.0`, so
//!   the step has one dispatch and one barrier fewer.
//! - Every other site of the sweep's step keeps its kernel module, in its order.

#[allow(dead_code)]
#[path = "../benches/mlp_sweep/sweep.rs"]
mod sweep;

use modelspec_core::artifact::Backend;
use modelspec_core::compile::compile_artifact_for;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{ExecutionPlan, Kernel, KernelKind, PlanOp, PlanStep};
use modelspec_core::ir::tensor::{Dim, TensorType};

const RTX3090: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

const MNIST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/mnist_mlp.mspec"
));

fn plan_of(name: &str, source: &str) -> ExecutionPlan {
    let profile = DeviceProfile::from_json(RTX3090).expect("the fixture is a valid profile");
    compile_artifact_for(name, source, Backend::Vulkan, Some(&profile))
        .expect("compiles for Vulkan")
        .execution
}

fn sweep_plan() -> ExecutionPlan {
    let point = sweep::Point {
        batch: 8192,
        hidden: 4096,
    };
    plan_of(&point.file_name(), &point.source(sweep::WARMUP_STEPS))
}

fn kernel_of<'p>(plan: &'p ExecutionPlan, step: &PlanStep) -> &'p Kernel {
    let id = step.kernel_module_id.expect("a dispatch names its module");
    plan.kernels
        .iter()
        .find(|k| k.id == id)
        .expect("the plan holds the kernel")
}

fn shape(ty: &TensorType) -> Vec<u64> {
    ty.shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
            Dim::Symbol(_) => panic!("a static shape"),
        })
        .collect()
}

/// The position of the step plan's one loss-sum dispatch.
fn loss_at(plan: &ExecutionPlan) -> usize {
    let at: Vec<usize> = plan
        .step
        .iter()
        .enumerate()
        .filter(|(_, s)| s.op == PlanOp::Dispatch && kernel_of(plan, s).kind == KernelKind::LossSum)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(at.len(), 1, "one loss sum per step");
    at[0]
}

#[test]
fn the_loss_sum_is_recorded_right_after_the_w1_gradient_matmul_with_no_barrier_between() {
    for (name, plan, w1) in [
        ("sweep b8192 h4096", sweep_plan(), [784, 4096]),
        ("mnist_mlp", plan_of("mnist_mlp.mspec", MNIST), [784, 64]),
    ] {
        let at = loss_at(&plan);
        let before = &plan.step[at - 1];
        assert_eq!(before.op, PlanOp::Dispatch, "{name}: {before:?}");
        let matmul = kernel_of(&plan, before);
        assert_eq!(matmul.kind, KernelKind::MatMul, "{name}");
        assert_eq!(shape(&matmul.result), w1, "{name}: the gradient of w1");
        let flop = matmul.flop().expect("counted");
        for later in &plan.step[at + 1..] {
            if later.op == PlanOp::Dispatch {
                let kernel = kernel_of(&plan, later);
                assert!(
                    kernel.flop().unwrap_or(0) < flop || kernel.kind != KernelKind::MatMul,
                    "{name}: no larger matmul after the loss sum"
                );
            }
        }
        assert_eq!(plan.step[at + 1].op, PlanOp::Barrier, "{name}");
        let loss = &plan.step[at];
        assert_eq!(loss.writes, vec![plan.loss_view], "{name}");
        assert!(
            loss.depends_on.iter().all(|&d| d < at - 1),
            "{name}: the loss sum waits on no step of its own interval: {:?}",
            loss.depends_on
        );
    }
}

#[test]
fn the_loss_evaluation_records_only_the_steps_the_loss_is_computed_from() {
    for (name, plan) in [
        ("sweep b8192 h4096", sweep_plan()),
        ("mnist_mlp", plan_of("mnist_mlp.mspec", MNIST)),
    ] {
        let steps = plan.loss_steps();
        let kinds: Vec<(KernelKind, Vec<u64>)> = steps
            .iter()
            .map(|&i| &plan.step[i])
            .filter(|s| s.op == PlanOp::Dispatch)
            .map(|s| {
                let k = kernel_of(&plan, s);
                (k.kind, shape(&k.result))
            })
            .collect();
        let (batch, hidden) = if name == "mnist_mlp" {
            (100, 64)
        } else {
            (8192, 4096)
        };
        assert_eq!(
            kinds,
            vec![
                (KernelKind::MatMul, vec![batch, hidden]),
                (KernelKind::MatMul, vec![batch, 10]),
                (KernelKind::LossSum, vec![]),
            ],
            "{name}"
        );
        let ops: Vec<PlanOp> = steps.iter().map(|&i| plan.step[i].op).collect();
        assert_eq!(
            ops,
            vec![
                PlanOp::Dispatch,
                PlanOp::Barrier,
                PlanOp::Dispatch,
                PlanOp::Barrier,
                PlanOp::Dispatch,
            ],
            "{name}"
        );
        assert!(steps.windows(2).all(|w| w[0] < w[1]), "{name}: plan order");
        assert!(steps.iter().all(|&i| plan.step[i].writes.iter().all(|v| {
            plan.parameters
                .iter()
                .all(|p| p.views[0].buffer != v.buffer)
        })));
    }
}

#[test]
fn the_relu_gradient_is_the_epilogue_of_the_matmul_that_produces_its_adjoint() {
    for (name, plan, [batch, hidden]) in [
        ("sweep b8192 h4096", sweep_plan(), [8192, 4096]),
        ("mnist_mlp", plan_of("mnist_mlp.mspec", MNIST), [100, 64]),
    ] {
        let dispatches: Vec<&PlanStep> = plan
            .step
            .iter()
            .filter(|s| s.op == PlanOp::Dispatch)
            .collect();
        assert!(
            dispatches
                .iter()
                .all(|s| !kernel_of(&plan, s).op.contains("ReluGrad")),
            "{name}: no dispatch of its own computes the relu gradient"
        );
        let fused: Vec<&&PlanStep> = dispatches
            .iter()
            .filter(|s| {
                kernel_of(&plan, s).epilogue.map(|e| e.to_string())
                    == Some("epilogue ReluGrad".to_string())
            })
            .collect();
        assert_eq!(fused.len(), 1, "{name}");
        let step = fused[0];
        let kernel = kernel_of(&plan, step);
        assert_eq!(kernel.kind, KernelKind::MatMul, "{name}");
        assert_eq!(shape(&kernel.result), [batch, hidden], "{name}");
        assert_eq!(
            kernel.operands.len(),
            3,
            "{name}: a, b and the pre-activation"
        );
        assert_eq!(
            kernel.operands[2].to_string(),
            kernel.result.to_string(),
            "{name}"
        );
        assert_eq!(step.reads.len(), 3, "{name}");
        assert_eq!(step.writes.len(), 1, "{name}");
        let forward = dispatches
            .iter()
            .find(|s| {
                kernel_of(&plan, s).epilogue.map(|e| e.to_string())
                    == Some("epilogue Relu writes pre-activation".to_string())
            })
            .expect("the forward matmul keeps its pre-activation");
        assert_eq!(
            step.reads[2], forward.writes[0],
            "{name}: the epilogue reads the pre-activation the forward matmul wrote"
        );
        assert_eq!(dispatches.len(), 9, "{name}: one dispatch fewer");
    }
}

#[test]
fn every_other_site_of_the_sweep_step_keeps_its_kernel_module_and_order() {
    let plan = sweep_plan();
    let ids: Vec<String> = plan
        .step
        .iter()
        .filter(|s| s.op == PlanOp::Dispatch)
        .map(|s| s.kernel_module_id.expect("a module").to_string())
        .collect();
    // Position 3, the gradient of the hidden activation, was module
    // 545b0394-3c57-8ee0-b3d0-5d8e483e033a before the change; it now carries the relu gradient
    // and reads the pre-activation as a third operand, so its module is a new one, the module
    // the profiled runs of `benchmarks/raw/2026-10-25-step-sites/sites/` time at that site.
    assert_eq!(
        ids,
        [
            "55ee7765-96e3-8fe0-a2b3-bb064c893270",
            "f39a0278-629b-8a45-91a6-c3bd874b89d8",
            "d58c985d-59cd-89c3-b67b-ab96193fe102",
            "3db8a6bb-05d0-826f-9121-0de6c559b61c",
            "66d0d33c-e218-8098-a57a-4454996f06df",
            "ac5b3fba-1cd9-8dd9-a98d-4d2cf349c408",
            "d661a871-498f-8049-ba2d-d5fffdd46a69",
            "5e255200-6dd1-8e6a-b722-15b0efb04e7c",
            "22c4c89c-aec7-8d43-be1c-a59edb97236d",
        ]
    );
    let barriers = plan.step.iter().filter(|s| s.op == PlanOp::Barrier).count();
    assert_eq!(barriers, 6, "the step plan's barriers, its fence included");
}
