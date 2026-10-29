//! DESIGN.md § 5.2, no Vulkan instance: `CrossEntropyGrad` is two dispatches. A `per_row`
//! kernel over the logits, the labels and the weights writes the maximum and the exponential sum
//! of each row and, from one more invocation, the weight sum into a `[N + 1, 2]` buffer an
//! `Allocate` makes; after a barrier a `per_element` kernel, one invocation per gradient
//! element, reads them, the logits, the labels, the weights and the seed. Each stage passes
//! `spirv-val --target-env vulkan1.2` on the hidden-32 twin's shapes and on shapes whose edges
//! do not fill a workgroup. Needs `spirv-val` on the path and fails, not skips, without it.
#![cfg(feature = "vulkan")]

use modelspec_core::compile::compile_training;
use modelspec_core::ir::kernel::{ExecutionPlan, Kernel, KernelKind, Operand, PlanOp};
use modelspec_core::ir::tensor::{Dim, TensorOp, TensorType};
use modelspec_core::ir::verify::verify_plan;
use modelspec_core::schedule::{attention_loss_kernel, schedule};
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::spirv::validate::validate;
use modelspec_runtime::vulkan::dispatch::kernel_bindings;

/// The hidden-32 twin of BERT-tiny: attention, cross-entropy, dropout and AdamW.
const TWIN: &str = include_str!("../../modelspec-cli/tests/fixtures/bert_cpu.mspec");

/// `[n, v]`: one class; unaligned edges; 512 classes, the longest row of a device test.
const SHAPES: [(u64, u64); 5] = [(1, 1), (3, 5), (63, 65), (65, 512), (130, 129)];

fn f32s(shape: &[u64]) -> Operand {
    Operand::Buffer(TensorType::f32(shape))
}

fn i32s(shape: &[u64]) -> Operand {
    Operand::Buffer(TensorType::i32(shape))
}

fn twin_plan() -> ExecutionPlan {
    let compiled = compile_training("bert_cpu.mspec", TWIN)
        .unwrap_or_else(|errors| panic!("the twin compiles: {errors:#?}"));
    schedule(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
    )
    .unwrap_or_else(|error| panic!("the twin's step schedules for Vulkan: {error}"))
}

/// The one dispatch of `kernel` in `plan`.
fn dispatch_of(plan: &ExecutionPlan, kernel: &Kernel) -> usize {
    let found: Vec<usize> = (0..plan.step.len())
        .filter(|&i| {
            plan.step[i].op == PlanOp::Dispatch && plan.step[i].kernel_module_id == Some(kernel.id)
        })
        .collect();
    assert_eq!(found.len(), 1, "{} is dispatched once", kernel.signature());
    found[0]
}

/// The twin's `CrossEntropyGrad` is a `per_row` kernel of `n + 1` invocations into the row
/// statistics, `n·(4v + 1)` FLOP, then a `per_element` kernel of `n·v` invocations, `7·n·v`
/// FLOP, that reads the statistics first and then the views the first read.
#[test]
fn cross_entropy_grad_is_row_statistics_then_one_invocation_per_element() {
    let plan = twin_plan();
    let grad: Vec<&Kernel> = plan
        .kernels
        .iter()
        .filter(|k| k.op == "CrossEntropyGrad")
        .collect();
    let mut kinds: Vec<KernelKind> = grad.iter().map(|k| k.kind).collect();
    kinds.sort();
    assert_eq!(
        kinds,
        [KernelKind::PerElement, KernelKind::PerRow],
        "{grad:#?}"
    );
    let of = |kind: KernelKind| *grad.iter().find(|k| k.kind == kind).unwrap();
    let (stats, elements) = (of(KernelKind::PerRow), of(KernelKind::PerElement));
    let (n, v) = match &stats.operands[0] {
        Operand::Buffer(ty) => match ty.shape[..] {
            [Dim::Static(n), Dim::Static(v)] => (n, v),
            _ => panic!("static logits: {ty}"),
        },
        Operand::Immediate(_) => panic!("buffered logits"),
    };
    assert_eq!(stats.operands, [f32s(&[n, v]), i32s(&[n]), f32s(&[n])]);
    assert_eq!(stats.result, TensorType::f32(&[n + 1, 2]));
    assert_eq!(stats.workgroups, [(n + 1).div_ceil(64) as u32, 1, 1]);
    assert_eq!(stats.flop(), Some(n * (4 * v + 1)));
    assert_eq!(elements.operands.len(), 5, "{}", elements.signature());
    assert_eq!(
        elements.operands[..4],
        [f32s(&[n + 1, 2]), f32s(&[n, v]), i32s(&[n]), f32s(&[n])]
    );
    assert_eq!(elements.result, TensorType::f32(&[n, v]));
    assert_eq!(elements.flop(), Some(7 * n * v));
    let bindings = kernel_bindings(elements).expect("the gradient binds");
    assert_eq!(bindings.invocations, n * v);

    let (s, e) = (dispatch_of(&plan, stats), dispatch_of(&plan, elements));
    assert!(s < e, "the row statistics before the gradient");
    let [partials] = plan.step[s].writes[..] else {
        panic!("the statistics kernel writes one view");
    };
    assert_eq!(plan.step[e].reads[0], partials);
    assert_eq!(plan.step[e].reads[1..4], plan.step[s].reads[..]);
    assert!(
        plan.step[..s]
            .iter()
            .any(|step| step.op == PlanOp::Allocate && step.writes == [partials]),
        "an Allocate makes the statistics' buffer"
    );
    assert!(
        plan.step[e]
            .depends_on
            .iter()
            .any(|&b| b > s && plan.step[b].op == PlanOp::Barrier),
        "the gradient waits on a barrier after the statistics"
    );
    verify_plan(&plan).unwrap_or_else(|error| panic!("the twin's plan verifies: {error}"));
}

/// On shapes whose edges do not fill a workgroup, with the seed as an immediate and as a rank-0
/// buffer: the statistics kernel runs `n + 1` invocations and the gradient `n·v`, each in
/// workgroups of 64 that cover them, and each module passes `spirv-val`.
#[test]
fn cross_entropy_grad_stages_on_unaligned_shapes_pass_spirv_val() {
    let op = TensorOp::CrossEntropyGrad;
    let mut modules = 0;
    for (n, v) in SHAPES {
        let inputs = vec![f32s(&[n, v]), i32s(&[n]), f32s(&[n])];
        let stats = attention_loss_kernel(&op, inputs.clone(), TensorType::f32(&[n + 1, 2]))
            .unwrap_or_else(|| panic!("[{n}, {v}] has a statistics kernel"));
        assert_eq!(stats.kind, KernelKind::PerRow);
        let mut kernels = vec![(stats, n + 1)];
        for seed in [Operand::Immediate(1.0), f32s(&[])] {
            let mut operands = vec![f32s(&[n + 1, 2])];
            operands.extend(inputs.iter().cloned());
            operands.push(seed);
            let elements = attention_loss_kernel(&op, operands, TensorType::f32(&[n, v]))
                .unwrap_or_else(|| panic!("[{n}, {v}] has a gradient kernel"));
            assert_eq!(elements.kind, KernelKind::PerElement);
            kernels.push((elements, n * v));
        }
        for (kernel, invocations) in kernels {
            let signature = kernel.signature();
            assert_eq!(kernel.workgroup_size, 64, "{signature}");
            assert_eq!(
                kernel.workgroups,
                [invocations.div_ceil(64) as u32, 1, 1],
                "{signature}"
            );
            let bindings = kernel_bindings(&kernel).expect("the kernel binds");
            assert_eq!(bindings.invocations, invocations, "{signature}");
            let words = emit(&kernel).unwrap_or_else(|error| panic!("{signature}: {error}"));
            if let Err(error) = validate(&words) {
                panic!("{signature}: {error}");
            }
            modules += 1;
        }
    }
    assert_eq!(modules, 3 * SHAPES.len());
}
