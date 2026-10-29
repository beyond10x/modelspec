//! story:bert-vulkan-kernels-attention-loss-adamw, DESIGN.md § 9 gate 6, no Vulkan instance: the
//! eight kernels `Attention`, `AttentionGradQ`, `AttentionGradK`, `AttentionGradV`,
//! `CrossEntropy`, `CrossEntropyGrad`, `Sqrt` and `Div` are scheduled as one dispatch each, but
//! `CrossEntropy` (story:vulkan-cross-entropy-sum-per-row) and `CrossEntropyGrad` as two, with
//! a kernel record, a workgroup size and a module id, printed by `ir --stage kernel` with a FLOP
//! and byte count; each module passes `spirv-val --target-env vulkan1.2`, on the shapes of the
//! hidden-32 twin's AdamW step and on shapes whose edges do not fill a workgroup, and decorates
//! every float add, subtract, multiply and `Fma` it computes `NoContraction` in the
//! deterministic mode and none under `fast_math`. The step plan writes each AdamW moment back
//! into the buffer its input is read from, and every host input of the step (the dropout RNG
//! tokens, the scheduled rate, the bias corrections and the moments) has a view of the plan
//! (coordinator decision D73, 2026-10-25). Needs `spirv-val` on the path and fails, not skips,
//! without it.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeSet;

use common::words::{
    DECORATION_NO_CONTRACTION, GLSL_FMA, OP_DECORATE, OP_EXT_INST, OP_F_ADD, OP_F_MUL, OP_F_SUB,
    instructions,
};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    DispatchCount, ExecutionPlan, Kernel, KernelKind, Operand, PlanOp,
};
use modelspec_core::ir::print::print_kernel;
use modelspec_core::ir::tensor::{Dim, TensorOp, TensorType};
use modelspec_core::ir::training::{C1_INPUT, C2_INPUT, LR_T_INPUT, dropout_sites, moment_input};
use modelspec_core::ir::verify::verify_plan;
use modelspec_core::schedule::{
    self, ATTENTION_WORKGROUP, attention_loss_kernel, host_inputs, schedule,
};
use modelspec_runtime::spirv::emit::{emit, emit_kernels};
use modelspec_runtime::spirv::validate::validate;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

/// The hidden-32 twin of BERT-tiny: attention, cross-entropy, dropout and AdamW.
const TWIN: &str = include_str!("../../modelspec-cli/tests/fixtures/bert_cpu.mspec");

/// The eight ops by their Tensor IR names.
const OPS: [&str; 8] = [
    "Attention",
    "AttentionGradQ",
    "AttentionGradK",
    "AttentionGradV",
    "CrossEntropy",
    "CrossEntropyGrad",
    "Sqrt",
    "Div",
];

fn twin() -> (CompiledTraining, ExecutionPlan) {
    let compiled = compile_training("bert_cpu.mspec", TWIN)
        .unwrap_or_else(|errors| panic!("the twin compiles: {errors:#?}"));
    let plan = schedule(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
    )
    .unwrap_or_else(|error| panic!("the twin's step schedules for Vulkan: {error}"));
    (compiled, plan)
}

fn twin_artifact() -> CompiledArtifact {
    let (compiled, plan) = twin();
    CompiledArtifact::new(Backend::Vulkan, TWIN, &compiled.training, plan)
}

fn is_new(kernel: &Kernel) -> bool {
    OPS.contains(&kernel.op)
}

fn assert_valid(kernel: &Kernel) {
    let words = emit(kernel).unwrap_or_else(|error| panic!("{}: {error}", kernel.signature()));
    if let Err(error) = validate(&words) {
        panic!("{}: {error}", kernel.signature());
    }
}

/// Scenario 2: the twin's AdamW step schedules for Vulkan, its plan verifies, every module of
/// it passes `spirv-val` as `emit_kernels` builds it for the RTX 3090 profile, and the eight ops
/// are all there.
#[test]
fn every_module_of_the_twin_step_passes_spirv_val() {
    let (_, plan) = twin();
    verify_plan(&plan).unwrap_or_else(|error| panic!("the twin's plan verifies: {error}"));
    let artifact = twin_artifact();
    let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a profile");
    let modules = emit_kernels(&artifact, &profile).expect("every kernel emits");
    let mut ops = BTreeSet::new();
    for (kernel, module) in artifact.execution.kernels.iter().zip(&modules) {
        if let Err(error) = validate(&module.spirv) {
            panic!("{}: {error}", kernel.signature());
        }
        if is_new(kernel) {
            ops.insert(kernel.op);
        }
    }
    let expected: BTreeSet<&str> = OPS.into_iter().collect();
    assert_eq!(ops, expected);
}

/// Each of the eight ops is one dispatch with a kernel record and a module id derived from it,
/// of workgroup size `ATTENTION_WORKGROUP` for attention and its gradients and 64 for the
/// others; `ir --stage kernel` prints it with its kind and a FLOP and byte count, never
/// `uncounted`. Attention and its gradients run per row, cross-entropy per row and then as the
/// loss sum, its gradient per row and then per element, `Sqrt` and `Div` per element of the
/// elementwise family.
#[test]
fn each_new_op_is_one_counted_dispatch_printed_with_its_kernel() {
    let artifact = twin_artifact();
    let plan = &artifact.execution;
    let printed = print_kernel(plan);
    for op in OPS {
        let kernels: Vec<&Kernel> = plan.kernels.iter().filter(|k| k.op == op).collect();
        assert!(!kernels.is_empty(), "no kernel of {op}:\n{printed}");
        let kinds: &[KernelKind] = match op {
            "CrossEntropy" => &[KernelKind::PerRow, KernelKind::LossSum],
            "CrossEntropyGrad" => &[KernelKind::PerRow, KernelKind::PerElement],
            "Sqrt" | "Div" => &[KernelKind::Elementwise],
            _ => &[KernelKind::PerRow],
        };
        for kernel in kernels {
            assert!(kinds.contains(&kernel.kind), "{}", kernel.signature());
            let size = if op.starts_with("Attention") {
                ATTENTION_WORKGROUP
            } else {
                64
            };
            assert_eq!(kernel.workgroup_size, size, "{}", kernel.signature());
            assert_eq!(kernel.id, Kernel::module_id(&kernel.signature()));
            let dispatches: Vec<_> = plan
                .step
                .iter()
                .filter(|s| s.op == PlanOp::Dispatch && s.kernel_module_id == Some(kernel.id))
                .collect();
            assert!(!dispatches.is_empty(), "{op} is never dispatched");
            for step in dispatches {
                assert!(
                    matches!(
                        plan.dispatch_count(step),
                        Some(DispatchCount::Counted { flop, bytes }) if flop > 0 && bytes > 0
                    ),
                    "{op}: {:?}",
                    plan.dispatch_count(step)
                );
            }
            let id = kernel.id.to_string();
            let line = printed
                .lines()
                .find(|l| l.starts_with("kernel ") && l.contains(&id))
                .unwrap_or_else(|| panic!("no kernel line of {op}:\n{printed}"));
            assert!(
                line.contains(&format!(" {id} {} {op} ", kernel.kind)),
                "{line}"
            );
            assert!(line.contains(&format!("workgroup_size {size}")), "{line}");
        }
    }
    for line in printed.lines().filter(|l| l.contains(": dispatch ")) {
        assert!(!line.ends_with(" uncounted"), "{line}");
    }
}

fn f32s(shape: &[u64]) -> Operand {
    Operand::Buffer(TensorType::f32(shape))
}

fn i32s(shape: &[u64]) -> Operand {
    Operand::Buffer(TensorType::i32(shape))
}

/// The attention kernels of `b` sequences of `s` rows of width `h` in `heads` heads.
fn attention_kernels(b: u64, s: u64, h: u64, heads: u64) -> Vec<Kernel> {
    let n = b * s;
    let rows = TensorType::f32(&[n, h]);
    let forward = vec![f32s(&[n, h]), f32s(&[n, h]), f32s(&[n, h]), f32s(&[b, s])];
    let mut backward = forward.clone();
    backward.push(f32s(&[n, h]));
    [
        (TensorOp::Attention { heads }, forward),
        (TensorOp::AttentionGradQ { heads }, backward.clone()),
        (TensorOp::AttentionGradK { heads }, backward.clone()),
        (TensorOp::AttentionGradV { heads }, backward),
    ]
    .into_iter()
    .map(|(op, operands)| {
        attention_loss_kernel(&op, operands, rows.clone())
            .unwrap_or_else(|| panic!("{} [{b}, {s}] x {h} / {heads} has a kernel", op.name()))
    })
    .collect()
}

/// The cross-entropy kernels of `n` rows of `v` classes: the loss's row losses and their sum,
/// and its gradient's row statistics and elements with the backward seed as an immediate and as
/// a rank-0 buffer.
fn cross_entropy_kernels(n: u64, v: u64) -> Vec<Kernel> {
    let operands = vec![f32s(&[n, v]), i32s(&[n]), f32s(&[n])];
    let mut seeded = operands.clone();
    seeded.push(Operand::Immediate(1.0));
    let mut buffered = operands.clone();
    buffered.push(f32s(&[]));
    let loss = schedule::cross_entropy_kernels(operands, TensorType::f32_scalar())
        .unwrap_or_else(|| panic!("CrossEntropy [{n}, {v}] has kernels"));
    let gradients = [seeded, buffered].into_iter().flat_map(|operands| {
        schedule::cross_entropy_grad_kernels(operands, TensorType::f32(&[n, v]))
            .unwrap_or_else(|| panic!("CrossEntropyGrad [{n}, {v}] has kernels"))
    });
    loss.into_iter().chain(gradients).collect()
}

/// The `Sqrt` and `Div` kernels of `shape`: `Div` of two tensors, of a tensor by a rank-0
/// buffer, of a rank-0 buffer by a tensor and of a tensor by an immediate.
fn optimizer_kernels(shape: &[u64]) -> Vec<Kernel> {
    let ty = TensorType::f32(shape);
    [
        (TensorOp::Sqrt, vec![f32s(shape)]),
        (TensorOp::Div, vec![f32s(shape), f32s(shape)]),
        (TensorOp::Div, vec![f32s(shape), f32s(&[])]),
        (TensorOp::Div, vec![f32s(&[]), f32s(shape)]),
        (TensorOp::Div, vec![f32s(shape), Operand::Immediate(3.0)]),
    ]
    .into_iter()
    .map(|(op, operands)| {
        attention_loss_kernel(&op, operands, ty.clone())
            .unwrap_or_else(|| panic!("{} {shape:?} has a kernel", op.name()))
    })
    .collect()
}

/// `[b, s, h, heads]`: one row; edges one short of, at and one past a workgroup of 64; and 512
/// keys, the longest softmax the device tests run.
const ATTENTION_SHAPES: [(u64, u64, u64, u64); 6] = [
    (1, 1, 1, 1),
    (2, 3, 4, 2),
    (1, 63, 6, 3),
    (2, 64, 8, 2),
    (3, 65, 8, 4),
    (1, 512, 16, 2),
];

/// `[n, v]`: one class; unaligned edges; 512 classes, the longest reduction of a device test.
const CROSS_ENTROPY_SHAPES: [(u64, u64); 5] = [(1, 1), (3, 5), (63, 65), (65, 512), (130, 129)];

/// The optimizer shapes: one element, unaligned vectors and a matrix.
const OPTIMIZER_SHAPES: [&[u64]; 5] = [&[1], &[63], &[65], &[7, 9], &[300, 32]];

#[test]
fn every_new_kernel_on_unaligned_shapes_passes_spirv_val() {
    let mut modules = 0;
    for (b, s, h, heads) in ATTENTION_SHAPES {
        let kernels = attention_kernels(b, s, h, heads);
        for kernel in &kernels {
            assert_valid(kernel);
            modules += 1;
        }
    }
    for (n, v) in CROSS_ENTROPY_SHAPES {
        for kernel in &cross_entropy_kernels(n, v) {
            assert_valid(kernel);
            modules += 1;
        }
    }
    for shape in OPTIMIZER_SHAPES {
        for kernel in &optimizer_kernels(shape) {
            assert_valid(kernel);
            modules += 1;
        }
    }
    assert_eq!(
        modules,
        4 * ATTENTION_SHAPES.len() + 6 * CROSS_ENTROPY_SHAPES.len() + 5 * OPTIMIZER_SHAPES.len()
    );
}

/// What the schedule refuses: `heads` not dividing the width, a mask whose rows do not cover
/// the sequence rows, labels that are not `i32`, and a gradient seed that is not rank 0.
#[test]
fn ill_shaped_operands_have_no_kernel() {
    let rows = TensorType::f32(&[6, 4]);
    let refused = [
        (
            TensorOp::Attention { heads: 3 },
            vec![f32s(&[6, 4]), f32s(&[6, 4]), f32s(&[6, 4]), f32s(&[2, 3])],
            rows.clone(),
        ),
        (
            TensorOp::Attention { heads: 2 },
            vec![f32s(&[6, 4]), f32s(&[6, 4]), f32s(&[6, 4]), f32s(&[2, 2])],
            rows.clone(),
        ),
        (
            TensorOp::CrossEntropy,
            vec![f32s(&[6, 4]), f32s(&[6])],
            TensorType::f32(&[6]),
        ),
    ];
    for (op, operands, result) in refused {
        assert!(
            attention_loss_kernel(&op, operands.clone(), result).is_none(),
            "{} {operands:?}",
            op.name()
        );
    }
    let float_labels = vec![f32s(&[6, 4]), f32s(&[6]), f32s(&[6])];
    assert!(
        schedule::cross_entropy_kernels(float_labels, TensorType::f32_scalar()).is_none(),
        "CrossEntropy of f32 labels"
    );
    let vector_seed = vec![f32s(&[6, 4]), i32s(&[6]), f32s(&[6]), f32s(&[6])];
    assert!(
        schedule::cross_entropy_grad_kernels(vector_seed, rows).is_none(),
        "CrossEntropyGrad of a seed that is not rank 0"
    );
}

/// The float adds, subtracts, multiplies and `Fma`s of a module, and how many of them are
/// decorated `NoContraction`.
fn float_arithmetic(words: &[u32]) -> (usize, usize) {
    let all = instructions(words);
    let decorated: BTreeSet<u32> = all
        .iter()
        .filter(|(op, w)| *op == OP_DECORATE && w[2] == DECORATION_NO_CONTRACTION)
        .map(|(_, w)| w[1])
        .collect();
    let mut total = 0;
    let mut exact = 0;
    for (op, w) in &all {
        let result = match *op {
            OP_F_ADD | OP_F_SUB | OP_F_MUL => w[2],
            OP_EXT_INST if w[4] == GLSL_FMA => w[2],
            _ => continue,
        };
        total += 1;
        if decorated.contains(&result) {
            exact += 1;
        }
    }
    (total, exact)
}

/// ADR-009 for the eight kernels: in the deterministic mode every float add, subtract, multiply
/// and `Fma` is decorated `NoContraction`; the `fast_math` modules carry none. Every kernel but
/// `Sqrt` and `Div` sums products, each after the first fused by an `Fma`, and those two correct
/// their quotient or root by one `Fma` residual step, so each has an `Fma`.
#[test]
fn every_new_float_op_is_uncontracted_except_under_fast_math() {
    let artifact = twin_artifact();
    let mut fast = artifact.clone();
    fast.fast_math = true;
    let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a profile");
    let off = emit_kernels(&artifact, &profile).expect("emits");
    let on = emit_kernels(&fast, &profile).expect("emits");
    let mut fused = BTreeSet::new();
    for ((kernel, off), on) in artifact.execution.kernels.iter().zip(&off).zip(&on) {
        if !is_new(kernel) {
            continue;
        }
        let (total, exact) = float_arithmetic(&off.spirv);
        assert!(total > 0, "{}", kernel.signature());
        assert_eq!(exact, total, "{}", kernel.signature());
        let (fast_total, fast_exact) = float_arithmetic(&on.spirv);
        assert_eq!(fast_total, total, "{}", kernel.signature());
        assert_eq!(fast_exact, 0, "{}", kernel.signature());
        let fmas = instructions(&off.spirv)
            .into_iter()
            .filter(|(op, w)| *op == OP_EXT_INST && w[4] == GLSL_FMA)
            .count();
        if fmas > 0 {
            fused.insert(kernel.op);
        }
    }
    let expected: BTreeSet<&str> = OPS.into_iter().collect();
    assert_eq!(fused, expected);
}

/// Every host input of the twin's step has a view of the plan: one `i32[2]` RNG token per
/// dropout site, the rank-0 scheduled rate and bias corrections, and the two moments of every
/// parameter at its type, each in a buffer of its own that holds it. The step plan writes each
/// moment's next value into the view the moment is read from, and writes no other host input.
#[test]
fn every_host_input_has_a_view_and_each_moment_is_written_in_place() {
    let (compiled, plan) = twin();
    let inputs = host_inputs(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
    )
    .expect("the host inputs schedule");
    let names: BTreeSet<&str> = inputs.iter().map(|i| i.name.as_str()).collect();
    let sites = dropout_sites(&compiled.training.training.program);
    assert_eq!(sites.len(), 5, "{sites:?}");
    let mut expected: BTreeSet<String> = sites.iter().cloned().collect();
    expected.extend([LR_T_INPUT, C1_INPUT, C2_INPUT].map(str::to_owned));
    for spec in &plan.parameters {
        for moment in ["m", "v"] {
            let name = moment_input(
                if moment == "m" {
                    modelspec_core::ir::training::Moment::First
                } else {
                    modelspec_core::ir::training::Moment::Second
                },
                &spec.path,
            );
            expected.insert(name);
        }
    }
    let expected: BTreeSet<&str> = expected.iter().map(String::as_str).collect();
    assert_eq!(names, expected);
    let mut buffers = BTreeSet::new();
    for input in &inputs {
        let view = input.view;
        assert!(
            (view.buffer as usize) < plan.buffers.len(),
            "{}: {view}",
            input.name
        );
        assert_eq!(view.size_bytes, plan.buffers[view.buffer as usize]);
        assert!(
            buffers.insert(view.buffer),
            "{} shares b{}",
            input.name,
            view.buffer
        );
        let written = plan
            .step
            .iter()
            .any(|s| s.writes.iter().any(|w| w.buffer == view.buffer));
        let moment =
            input.name.starts_with("optimizer.m@") || input.name.starts_with("optimizer.v@");
        assert_eq!(written, moment, "{}", input.name);
        assert!(
            plan.forward
                .iter()
                .all(|s| s.writes.iter().all(|w| w.buffer != view.buffer)),
            "{}",
            input.name
        );
    }
    for input in inputs.iter().filter(|i| i.name.starts_with("optimizer.")) {
        let spec = plan
            .parameters
            .iter()
            .find(|p| input.name.ends_with(&format!("@{}", p.path)));
        if let Some(spec) = spec {
            assert_eq!(input.ty, spec.ty, "{}", input.name);
        } else {
            assert_eq!(input.ty, TensorType::f32_scalar(), "{}", input.name);
        }
    }
}

/// The `[n, v]` logits of the twin's `CrossEntropy`, from its row-loss kernel.
fn logits_of(plan: &ExecutionPlan) -> (u64, u64) {
    let rows = plan
        .kernels
        .iter()
        .find(|k| k.op == "CrossEntropy" && k.kind == KernelKind::PerRow)
        .expect("the row-loss kernel");
    match &rows.operands[0] {
        Operand::Buffer(ty) => match ty.shape[..] {
            [Dim::Static(n), Dim::Static(v)] => (n, v),
            _ => panic!("static logits: {ty}"),
        },
        Operand::Immediate(_) => panic!("buffered logits"),
    }
}

/// story:vulkan-cross-entropy-sum-per-row: the twin's `CrossEntropy` is two dispatches. A
/// `per_row` kernel, one invocation per row, writes the `n` row losses into a buffer an
/// `Allocate` makes; after a barrier a `loss_sum` kernel of one workgroup reads them and the
/// weights and writes the loss. The two count the FLOP the one-workgroup kernel counted,
/// `n·(4v + 3)` and `3n + 1` against `n·(4v + 6) + 1`.
#[test]
fn cross_entropy_is_a_per_row_kernel_then_one_ordered_sum() {
    let (_, plan) = twin();
    let (n, v) = logits_of(&plan);
    let entropy: Vec<&Kernel> = plan
        .kernels
        .iter()
        .filter(|k| k.op == "CrossEntropy")
        .collect();
    let mut kinds: Vec<KernelKind> = entropy.iter().map(|k| k.kind).collect();
    kinds.sort();
    assert_eq!(
        kinds,
        [KernelKind::LossSum, KernelKind::PerRow],
        "{entropy:#?}"
    );
    let of = |kind: KernelKind| *entropy.iter().find(|k| k.kind == kind).unwrap();
    let (rows, sum) = (of(KernelKind::PerRow), of(KernelKind::LossSum));
    assert_eq!(
        rows.operands,
        [
            Operand::Buffer(TensorType::f32(&[n, v])),
            Operand::Buffer(TensorType::i32(&[n]))
        ]
    );
    assert_eq!(rows.result, TensorType::f32(&[n]));
    assert_eq!(rows.workgroups, [n.div_ceil(64) as u32, 1, 1]);
    assert_eq!(
        sum.operands,
        [
            Operand::Buffer(TensorType::f32(&[n])),
            Operand::Buffer(TensorType::f32(&[n]))
        ]
    );
    assert_eq!(sum.result, TensorType::f32_scalar());
    assert_eq!(sum.workgroups, [1, 1, 1]);
    assert_eq!(rows.flop(), Some(n * (4 * v + 3)));
    assert_eq!(sum.flop(), Some(3 * n + 1));

    let at = |kernel: &Kernel| {
        let found: Vec<usize> = (0..plan.step.len())
            .filter(|&i| {
                plan.step[i].op == PlanOp::Dispatch
                    && plan.step[i].kernel_module_id == Some(kernel.id)
            })
            .collect();
        assert_eq!(found.len(), 1, "{} is dispatched once", kernel.signature());
        found[0]
    };
    let (r, s) = (at(rows), at(sum));
    assert!(r < s, "the row losses before their sum");
    let [partials] = plan.step[r].writes[..] else {
        panic!("the per-row kernel writes one view");
    };
    assert_eq!(plan.step[s].reads[0], partials);
    assert_eq!(plan.step[s].writes, [plan.loss_view]);
    assert!(
        plan.step[..r]
            .iter()
            .any(|step| step.op == PlanOp::Allocate && step.writes == [partials]),
        "an Allocate makes the row losses' buffer"
    );
    assert!(
        plan.step[s]
            .depends_on
            .iter()
            .any(|&b| b > r && plan.step[b].op == PlanOp::Barrier),
        "the sum waits on a barrier after the row losses"
    );
    verify_plan(&plan).unwrap_or_else(|error| panic!("the twin's plan verifies: {error}"));
}

/// The optimizer shapes with their element counts, multiplied out by hand: 1, 63, 65, 7·9 = 63
/// and 300·32 = 9,600.
const OPTIMIZER_ELEMENTS: [(&[u64], u64); 5] = [
    (&[1], 1),
    (&[63], 63),
    (&[65], 65),
    (&[7, 9], 63),
    (&[300, 32], 9_600),
];

/// story:bert-tiny-vs-pytorch-cuda, acceptance of the counts: AdamW's `Sqrt` against values
/// computed by hand from DESIGN.md § 3.1 "Counting rule" ("`elementwise` `Sqrt` and `Div` one
/// per output element"): one FLOP per element of each optimizer shape.
#[test]
fn sqrt_counts_one_flop_per_element_by_hand() {
    for (shape, flop) in OPTIMIZER_ELEMENTS {
        let kernels: Vec<Kernel> = optimizer_kernels(shape)
            .into_iter()
            .filter(|k| k.op == "Sqrt")
            .collect();
        assert_eq!(kernels.len(), 1, "{shape:?}");
        assert_eq!(kernels[0].kind, KernelKind::Elementwise);
        assert_eq!(kernels[0].flop(), Some(flop), "{}", kernels[0].signature());
    }
}

/// story:bert-tiny-vs-pytorch-cuda, acceptance of the counts: AdamW's `Div` against values
/// computed by hand from DESIGN.md § 3.1 "Counting rule" ("`elementwise` `Sqrt` and `Div` one
/// per output element"): one FLOP per element of each optimizer shape, whether it divides two
/// tensors, a tensor by a rank-0 buffer, a rank-0 buffer by a tensor or a tensor by an
/// immediate.
#[test]
fn div_counts_one_flop_per_element_by_hand() {
    for (shape, flop) in OPTIMIZER_ELEMENTS {
        let kernels: Vec<Kernel> = optimizer_kernels(shape)
            .into_iter()
            .filter(|k| k.op == "Div")
            .collect();
        assert_eq!(kernels.len(), 4, "{shape:?}");
        for kernel in kernels {
            assert_eq!(kernel.kind, KernelKind::Elementwise);
            assert_eq!(kernel.flop(), Some(flop), "{}", kernel.signature());
        }
    }
}

/// The `(flop, bytes)` of every dispatch of `op` in the twin's step plan, sorted.
fn twin_counts(op: &str) -> Vec<(u64, u64)> {
    let (_, plan) = twin();
    let mut counts: Vec<(u64, u64)> = plan
        .step
        .iter()
        .filter(|s| {
            s.op == PlanOp::Dispatch
                && plan
                    .kernels
                    .iter()
                    .any(|k| k.op == op && s.kernel_module_id == Some(k.id))
        })
        .map(|s| match plan.dispatch_count(s) {
            Some(DispatchCount::Counted { flop, bytes }) => (flop, bytes),
            other => panic!("{op}: {other:?}"),
        })
        .collect();
    counts.sort_unstable();
    counts
}

/// story:bert-tiny-vs-pytorch-cuda, acceptance of the counts, on the hidden-32 twin of
/// BERT-tiny (H = 32, batches of 2 sequences of 16, so N = 32 rows, a 300 by 32 token table and
/// a 16 by 32 position table), each figure computed by hand from DESIGN.md § 3.1 "Counting
/// rule" (FLOP per kind as cited on the tests above; bytes "the `size_bytes` of the step's read
/// views plus those of its write views", 4 B per f32 or i32 element):
///
/// - `Embedding`, two dispatches of 0 FLOP: the token table 38,400 B + ids 128 B read and
///   f32[32, 32] 4,096 B written, 42,624 B; the position table 2,048 B + 128 B + 4,096 B, 6,272 B.
/// - `Dropout` and `DropoutGrad`, five sites each on f32[32, 32]: 1,024 FLOP and 4,096 B +
///   the i32[2] token 8 B + 4,096 B = 8,200 B per dispatch.
/// - `Sqrt` and `Div`, one each per parameter (41), of the parameter's shape: 28,684 FLOP per
///   step in all (embeddings 9,600 + 512 + 64; per layer 4·(1,024 + 32) + 64 + 2,048 + 64 +
///   2,048 + 32 + 64 = 8,544, twice; MLM head 1,024 + 32 + 64 + 300 = 1,420); `Sqrt` reads one
///   and writes one parameter-shaped view, 2·4·28,684 = 229,472 B, `Div` reads two and writes
///   one, 3·4·28,684 = 344,208 B.
#[test]
fn the_twin_step_counts_its_new_kinds_by_hand() {
    assert_eq!(twin_counts("Embedding"), [(0, 6_272), (0, 42_624)]);
    assert_eq!(twin_counts("Dropout"), [(1_024, 8_200); 5]);
    assert_eq!(twin_counts("DropoutGrad"), [(1_024, 8_200); 5]);
    for (op, bytes) in [("Sqrt", 229_472), ("Div", 344_208)] {
        let counts = twin_counts(op);
        assert_eq!(counts.len(), 41, "{op}: {counts:?}");
        let flop: u64 = counts.iter().map(|(flop, _)| flop).sum();
        let total: u64 = counts.iter().map(|(_, bytes)| bytes).sum();
        assert_eq!((flop, total), (28_684, bytes), "{op}");
    }
}
