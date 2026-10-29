//! Adversary cases for story:profile-achieved-throughput, the counting rule of DESIGN.md § 3.1
//! (approval-record:benchmark-wave-scope D1).
//!
//! - `Add` and an unfused `Relu` are two of the five elementwise ops the rule names, and no
//!   other case compiles a plan that has either: the MNIST and linear-regression plans have
//!   neither, and the unit tests build `Sub` and `Exp` only. A compiled `mean` loss is the only
//!   place its reduce stages and the gradient's broadcast meet the rule through the compiler.
//! - A count past `u64` is `uncounted` (DESIGN.md § 3.1); no case reaches it, so a wrapping
//!   product that prints 0 FLOP would pass every other case.

use modelspec_core::artifact::Backend;
use modelspec_core::compile;
use modelspec_core::ir::kernel::{
    BufferView, DispatchCount, ExecutionPlan, Kernel, KernelKind, KernelModuleId, Operand, PlanOp,
    PlanStep,
};
use modelspec_core::ir::print::print_kernel;
use modelspec_core::ir::tensor::{DType, Dim, TensorType};

/// A residual block whose weight is read twice, so its gradient is an `Add` of two terms, with a
/// `relu` of an `add` that no matmul epilogue absorbs, under a `mean` loss.
const RESIDUAL: &str = "model Residual {
    param w: Tensor<f32, [4, 4]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 4]> {
        return matmul(relu(add(matmul(x, w), x)), w);
    }
}

train Fit {
    model = Residual;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 4]>;
    loss = mean(mul(sub(model.forward(x), y), y));
    optimizer = sgd(lr = 0.1);
    steps = 10;
    seed = 42;
    device = vulkan(0);
}
";

fn static_dims(ty: &TensorType) -> Vec<u64> {
    ty.shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
            Dim::Symbol(_) => panic!("a compiled plan has static shapes"),
        })
        .collect()
}

/// Every Dispatch of the step plan as `(kind, op, result shape, count)`.
fn dispatches(plan: &ExecutionPlan) -> Vec<(KernelKind, &'static str, Vec<u64>, DispatchCount)> {
    plan.step
        .iter()
        .filter(|s| s.op == PlanOp::Dispatch)
        .map(|s| {
            let kernel = plan
                .kernel(s.kernel_module_id.expect("a dispatch names its module"))
                .expect("the plan has the module");
            (
                kernel.kind,
                kernel.op,
                static_dims(&kernel.result),
                plan.dispatch_count(s).expect("a dispatch has a count"),
            )
        })
        .collect()
}

fn counted(flop: u64, bytes: u64) -> DispatchCount {
    DispatchCount::Counted { flop, bytes }
}

/// D1: per output element one FLOP for `add` and `relu`; one per input element for each reduce
/// stage; none for a broadcast. Bytes: the read views plus the write views, 4 bytes an `f32`.
#[test]
fn a_compiled_add_unfused_relu_and_mean_count_by_the_rule() {
    let artifact = compile::compile_artifact("residual.mspec", RESIDUAL, Backend::Vulkan)
        .unwrap_or_else(|d| panic!("the residual block compiles: {d:#?}"));
    let plan = &artifact.execution;
    let got = dispatches(plan);
    let printed = print_kernel(plan);
    let find = |kind: KernelKind, op: &str, shape: &[u64]| -> Vec<DispatchCount> {
        got.iter()
            .filter(|(k, o, s, _)| *k == kind && *o == op && s[..] == *shape)
            .map(|(_, _, _, c)| *c)
            .collect()
    };
    // add(matmul(x, w), x): two [8, 4] reads, one [8, 4] write.
    assert_eq!(
        find(KernelKind::Elementwise, "Add", &[8, 4]),
        vec![counted(32, 3 * 128)],
        "{printed}"
    );
    // relu of the add: one [8, 4] read, one write.
    assert_eq!(
        find(KernelKind::Elementwise, "Relu", &[8, 4]),
        vec![counted(32, 2 * 128)],
        "{printed}"
    );
    // The gradient of w, read twice: the sum of two [4, 4] terms.
    assert_eq!(
        find(KernelKind::Elementwise, "Add", &[4, 4]),
        vec![counted(16, 3 * 64)],
        "{printed}"
    );
    // mean over 32 values: stage 1 reads them into one partial, stage 2 reads the partial.
    assert_eq!(
        find(KernelKind::ReduceStage1, "ReduceMean", &[1]),
        vec![counted(32, 128 + 4)],
        "{printed}"
    );
    assert_eq!(
        find(KernelKind::ReduceStage2, "ReduceMean", &[]),
        vec![counted(1, 4 + 4)],
        "{printed}"
    );
    // The mean's gradient, 1/32 broadcast to [8, 4]: no FLOP, one [8, 4] write.
    assert_eq!(
        find(KernelKind::Broadcast, "BroadcastScalar", &[8, 4]),
        vec![counted(0, 128)],
        "{printed}"
    );
    assert!(
        got.iter()
            .all(|(_, _, _, c)| *c != DispatchCount::Uncounted),
        "{printed}"
    );
}

fn kernel(
    kind: KernelKind,
    op: &'static str,
    operands: Vec<Operand>,
    result: TensorType,
) -> Kernel {
    let mut kernel = Kernel {
        id: KernelModuleId([0; 16]),
        kind,
        op,
        dtype: DType::F32,
        operands,
        result,
        reduces: None,
        epilogue: None,
        transposed: [false, false],
        softmax_stats: None,
        workgroup_size: kind.workgroup_size(),
        workgroups: [1, 1, 1],
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

fn view(buffer: u32, size_bytes: u64) -> BufferView {
    BufferView {
        buffer,
        dtype: DType::F32,
        element_count: size_bytes / 4,
        offset: 0,
        size_bytes,
        alignment: 4,
    }
}

fn dispatch(kernel: &Kernel, reads: Vec<BufferView>, writes: Vec<BufferView>) -> PlanStep {
    PlanStep {
        op: PlanOp::Dispatch,
        kernel_module_id: Some(kernel.id),
        workgroups: Some(kernel.workgroups),
        reads,
        writes,
        depends_on: Vec::new(),
        value: None,
    }
}

/// DESIGN.md § 3.1: "a count passes `u64`" is `uncounted`, never 0 and never a wrapped or
/// saturated number: `2·m·k·n` of `[2^32, 2^32] x [2^32, 2^32]` is 2^97, the output elements
/// of a `[2^32, 2^32]` `add` 2^64, and three views of 2^63 bytes 3·2^63.
#[test]
fn a_count_past_u64_is_uncounted_never_wrapped() {
    let huge = TensorType::f32(&[1 << 32, 1 << 32]);
    let small = TensorType::f32(&[4]);
    let matmul = kernel(
        KernelKind::MatMul,
        "MatMul",
        vec![Operand::Buffer(huge.clone()), Operand::Buffer(huge.clone())],
        huge.clone(),
    );
    let add = kernel(
        KernelKind::Elementwise,
        "Add",
        vec![Operand::Buffer(huge.clone()), Operand::Buffer(huge.clone())],
        huge,
    );
    let sub = kernel(
        KernelKind::Elementwise,
        "Sub",
        vec![
            Operand::Buffer(small.clone()),
            Operand::Buffer(small.clone()),
        ],
        small,
    );
    assert_eq!(matmul.flop(), None, "2^97 FLOP");
    assert_eq!(add.flop(), None, "2^64 output elements");
    assert_eq!(sub.flop(), Some(4));

    let half = 1u64 << 63;
    let steps = vec![
        dispatch(&matmul, vec![view(0, 16), view(1, 16)], vec![view(2, 16)]),
        dispatch(&add, vec![view(0, 16), view(1, 16)], vec![view(2, 16)]),
        dispatch(
            &sub,
            vec![view(0, half), view(1, half)],
            vec![view(2, half)],
        ),
    ];
    let plan = ExecutionPlan {
        buffers: vec![u64::MAX; 3],
        parameters: Vec::new(),
        data: Vec::new(),
        kernels: vec![matmul, add, sub],
        step: steps.clone(),
        forward: Vec::new(),
        loss_view: view(0, 4),
        prediction_view: view(0, 4),
    };
    for step in &steps {
        assert_eq!(
            plan.dispatch_count(step),
            Some(DispatchCount::Uncounted),
            "{step:?}"
        );
    }
}
