//! story:bert-vulkan-kernels-rowwise, DESIGN.md § 9 gate 6, no Vulkan instance: the thirteen
//! row-wise kernels (`Embedding`, `EmbeddingGrad`, `AddRows`, `SumRows`, `Gelu`, `GeluGrad`,
//! `Dropout`, `DropoutGrad`, `LayerNorm`, `LayerNormGradX`, `LayerNormGradGamma`, `Softmax`,
//! `SoftmaxGrad`) are scheduled as one dispatch each, `EmbeddingGrad` as two (a sort and a
//! gradient) and `LayerNormGradGamma` as two (row moments and a column sum), with a kernel
//! record, a workgroup size and a module id, printed by `ir --stage kernel` with a FLOP and
//! byte count; each module passes `spirv-val --target-env vulkan1.2`, on the shapes of a compiled
//! program and on shapes whose edges do not fill a workgroup, and decorates every float add,
//! subtract, multiply and `Fma` it computes `NoContraction` in the deterministic mode and none
//! under `fast_math`.
//! Needs `spirv-val` on the path and fails, not skips, without it.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::words::{
    DECORATION_NO_CONTRACTION, GLSL_FMA, OP_DECORATE, OP_EXT_INST, OP_F_ADD, OP_F_MUL, OP_F_SUB,
    instructions,
};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    DispatchCount, ExecutionPlan, Kernel, KernelKind, Operand, PlanOp,
};
use modelspec_core::ir::print::print_kernel;
use modelspec_core::ir::tensor::{Dim, TensorOp, TensorType};
use modelspec_core::schedule::{
    embedding_grad_kernels, layer_norm_grad_gamma_kernels, rowwise_kernel,
};
use modelspec_runtime::spirv::emit::{emit, emit_kernels};
use modelspec_runtime::spirv::validate::validate;
use modelspec_runtime::vulkan::dispatch::kernel_bindings;
use modelspec_runtime::vulkan::runner::{DeviceRunError, infer_on_device};

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

/// The thirteen row-wise ops by their Tensor IR names.
const ROWWISE: [&str; 13] = [
    "Embedding",
    "EmbeddingGrad",
    "AddRows",
    "SumRows",
    "Gelu",
    "GeluGrad",
    "Dropout",
    "DropoutGrad",
    "LayerNorm",
    "LayerNormGradX",
    "LayerNormGradGamma",
    "Softmax",
    "SoftmaxGrad",
];

/// A program whose training step holds every row-wise op: an embedding of 5 ids into an 11 by
/// 13 table, a bias, `gelu`, `layer_norm`, `dropout` and `softmax`, and their gradients.
const BLOCK: &str = "model Block {
    param table: Tensor<f32, [11, 13]> = xavier_uniform();
    param b: Tensor<f32, [13]> = zeros();
    param gamma: Tensor<f32, [13]> = ones();
    param beta: Tensor<f32, [13]> = zeros();

    fn forward(ids: Tensor<i32, [5]>) -> Tensor<f32, [5, 13]> {
        let h = gelu(add_rows(embedding(table, ids), b));
        let n = layer_norm(h, gamma, beta);
        return softmax(dropout(n, p = 0.25));
    }
}

train Fit {
    model = Block;
    input ids: Tensor<i32, [5]>;
    target y: Tensor<f32, [5, 13]>;
    loss = mse(model.forward(ids), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
";

fn block() -> CompiledArtifact {
    compile_artifact("block.mspec", BLOCK, Backend::Vulkan)
        .unwrap_or_else(|errors| panic!("the block compiles for Vulkan: {errors:#?}"))
}

fn is_rowwise(kernel: &Kernel) -> bool {
    matches!(
        kernel.kind,
        KernelKind::PerElement | KernelKind::PerRow | KernelKind::PerColumn
    )
}

fn assert_valid(kernel: &Kernel) {
    let words = emit(kernel).unwrap_or_else(|error| panic!("{}: {error}", kernel.signature()));
    if let Err(error) = validate(&words) {
        panic!("{}: {error}", kernel.signature());
    }
}

/// Scenario 2: every row-wise kernel of the block's Vulkan artifact passes `spirv-val`, as
/// `emit_kernels` builds it for the RTX 3090 profile, and the thirteen ops are all there.
#[test]
fn every_rowwise_module_of_the_block_passes_spirv_val() {
    let artifact = block();
    let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a profile");
    let modules = emit_kernels(&artifact, &profile).expect("every kernel emits");
    let mut ops = BTreeSet::new();
    for (kernel, module) in artifact.execution.kernels.iter().zip(&modules) {
        if let Err(error) = validate(&module.spirv) {
            panic!("{}: {error}", kernel.signature());
        }
        if is_rowwise(kernel) {
            ops.insert(kernel.op);
        }
    }
    let expected: BTreeSet<&str> = ROWWISE.into_iter().collect();
    assert_eq!(ops, expected);
}

/// Each row-wise op is one dispatch with a kernel record of workgroup size 64 and a module id
/// derived from it; `ir --stage kernel` prints it with its kind and a FLOP and byte count, never
/// `uncounted`.
#[test]
fn each_rowwise_op_is_one_counted_dispatch_printed_with_its_kernel() {
    let artifact = block();
    let plan = &artifact.execution;
    let printed = print_kernel(plan);
    for op in ROWWISE {
        let kernels: Vec<&Kernel> = plan.kernels.iter().filter(|k| k.op == op).collect();
        assert!(!kernels.is_empty(), "no kernel of {op}:\n{printed}");
        for kernel in kernels {
            assert!(is_rowwise(kernel), "{}", kernel.signature());
            assert_eq!(kernel.workgroup_size, 64, "{}", kernel.signature());
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
                        Some(DispatchCount::Counted { flop: _, bytes }) if bytes > 0
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
            assert!(line.contains("workgroup_size 64"), "{line}");
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

/// The row-wise kernels of one `[n, h]` shape and a `v`-row table, in the order of [`ROWWISE`]:
/// one per op, and the sort and the gradient of `EmbeddingGrad`.
fn kernels_of_shape(n: u64, h: u64, v: u64) -> Vec<Kernel> {
    let rows = TensorType::f32(&[n, h]);
    let row = TensorType::f32(&[h]);
    let cases: Vec<(TensorOp, Vec<Operand>, TensorType)> = vec![
        (
            TensorOp::Embedding,
            vec![f32s(&[v, h]), i32s(&[n])],
            rows.clone(),
        ),
        (
            TensorOp::EmbeddingGrad {
                rows: Dim::Static(v),
            },
            vec![i32s(&[n]), f32s(&[n, h])],
            TensorType::f32(&[v, h]),
        ),
        (
            TensorOp::AddRows,
            vec![f32s(&[n, h]), f32s(&[h])],
            rows.clone(),
        ),
        (TensorOp::SumRows, vec![f32s(&[n, h])], row.clone()),
        (TensorOp::Gelu, vec![f32s(&[n, h])], rows.clone()),
        (
            TensorOp::GeluGrad,
            vec![f32s(&[n, h]), f32s(&[n, h])],
            rows.clone(),
        ),
        (
            TensorOp::Dropout { p: 0.1 },
            vec![f32s(&[n, h]), i32s(&[2])],
            rows.clone(),
        ),
        (
            TensorOp::DropoutGrad { p: 0.1 },
            vec![i32s(&[2]), f32s(&[n, h])],
            rows.clone(),
        ),
        (
            TensorOp::LayerNorm,
            vec![f32s(&[n, h]), f32s(&[h]), f32s(&[h])],
            rows.clone(),
        ),
        (
            TensorOp::LayerNormGradX,
            vec![f32s(&[n, h]), f32s(&[h]), f32s(&[n, h])],
            rows.clone(),
        ),
        (
            TensorOp::LayerNormGradGamma,
            vec![f32s(&[n, h]), f32s(&[n, h])],
            row,
        ),
        (TensorOp::Softmax, vec![f32s(&[n, h])], rows.clone()),
        (
            TensorOp::SoftmaxGrad,
            vec![f32s(&[n, h]), f32s(&[n, h])],
            rows,
        ),
    ];
    cases
        .into_iter()
        .flat_map(|(op, operands, result)| {
            let missing = format!("{} has a row-wise kernel", op.name());
            match op {
                TensorOp::EmbeddingGrad { .. } => {
                    Vec::from(embedding_grad_kernels(operands, result).expect(&missing))
                }
                TensorOp::LayerNormGradGamma => {
                    Vec::from(layer_norm_grad_gamma_kernels(operands, result).expect(&missing))
                }
                _ => vec![rowwise_kernel(&op, operands, result).expect(&missing)],
            }
        })
        .collect()
}

/// The shapes of the edge cases: one element, edges one short of, at and one past a workgroup
/// of 64, and rows of 512 terms, the longest reduction the device tests run.
const SHAPES: [(u64, u64, u64); 6] = [
    (1, 1, 1),
    (3, 5, 7),
    (63, 65, 64),
    (64, 64, 65),
    (65, 63, 3),
    (130, 512, 129),
];

#[test]
fn every_rowwise_kernel_on_unaligned_shapes_passes_spirv_val() {
    let mut modules = 0;
    for (n, h, v) in SHAPES {
        let kernels = kernels_of_shape(n, h, v);
        let ops: Vec<&str> = kernels.iter().map(|k| k.op).collect();
        let mut expected = ROWWISE.to_vec();
        expected.insert(1, "EmbeddingGrad");
        let gamma = expected
            .iter()
            .position(|op| *op == "LayerNormGradGamma")
            .expect("LayerNormGradGamma is row-wise");
        expected.insert(gamma, "LayerNormGradGamma");
        assert_eq!(
            ops, expected,
            "one kernel per op, two for EmbeddingGrad and LayerNormGradGamma"
        );
        for kernel in &kernels {
            assert_valid(kernel);
            modules += 1;
        }
    }
    assert_eq!(modules, 15 * SHAPES.len());
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

/// ADR-009 for the row-wise kernels: in the deterministic mode every float add, subtract,
/// multiply and `Fma` is decorated `NoContraction`; the `fast_math` modules carry none. A
/// kernel that computes a sum of products fuses each product after the first with an `Fma`.
#[test]
fn every_rowwise_float_op_is_uncontracted_except_under_fast_math() {
    let artifact = block();
    let mut fast = artifact.clone();
    fast.fast_math = true;
    let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a profile");
    let off = emit_kernels(&artifact, &profile).expect("emits");
    let on = emit_kernels(&fast, &profile).expect("emits");
    let mut fused = BTreeSet::new();
    for ((kernel, off), on) in artifact.execution.kernels.iter().zip(&off).zip(&on) {
        if !is_rowwise(kernel) {
            continue;
        }
        let (total, exact) = float_arithmetic(&off.spirv);
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
    let expected: BTreeSet<&str> = [
        "LayerNorm",
        "LayerNormGradX",
        "LayerNormGradGamma",
        "SoftmaxGrad",
    ]
    .into_iter()
    .collect();
    assert_eq!(fused, expected);
}

/// The dropout kernels compile and a training run uploads their RNG tokens
/// (story:bert-vulkan-kernels-attention-loss-adamw), but inference runs the step plan with its
/// dropouts: inference on the device of a program with dropout is `E-TARGET-011` at its
/// `dropout` call, refused before Vulkan is loaded, so this needs no device.
#[test]
fn device_inference_with_dropout_is_refused_before_vulkan_is_loaded() {
    let training = compile_training("block.mspec", BLOCK).expect("the block compiles");
    let artifact = block();
    let refused = match infer_on_device(None, &training, &artifact, &BTreeMap::new()) {
        Err(DeviceRunError::Refused(diagnostic)) => diagnostic,
        Err(other) => panic!("refused before the run, not {other}"),
        Ok(_) => panic!("inference with dropout is refused"),
    };
    assert_eq!(refused.code.to_string(), "E-TARGET-011", "{refused}");
    assert!(
        refused
            .to_string()
            .starts_with("E-TARGET-011: operator `dropout` has no Vulkan run yet"),
        "{refused}"
    );
    let span = refused.span.as_ref().expect("a span");
    assert_eq!((span.line, span.column), (10, 24), "{refused}");
}

/// The sort and the gradient of each `EmbeddingGrad` of `plan`, then of tables whose elements do
/// not fill a workgroup, each with its `(n, h, v)`.
fn embedding_grad_pairs(plan: &ExecutionPlan) -> Vec<([Kernel; 2], u64, u64, u64)> {
    let of_kind = |kind: KernelKind| -> Vec<Kernel> {
        plan.kernels
            .iter()
            .filter(|k| k.op == "EmbeddingGrad" && k.kind == kind)
            .cloned()
            .collect()
    };
    let (sorts, grads) = (of_kind(KernelKind::PerRow), of_kind(KernelKind::PerElement));
    assert_eq!((sorts.len(), grads.len()), (1, 1), "{sorts:#?} {grads:#?}");
    let mut pairs = vec![([sorts[0].clone(), grads[0].clone()], 5, 13, 11)];
    for (n, h, v) in [(1, 1, 1), (63, 65, 3), (65, 7, 129), (130, 33, 2)] {
        let built: Vec<Kernel> = kernels_of_shape(n, h, v)
            .into_iter()
            .filter(|k| k.op == "EmbeddingGrad")
            .collect();
        let pair = <[Kernel; 2]>::try_from(built).expect("a sort and a gradient");
        pairs.push((pair, n, h, v));
    }
    pairs
}

/// story:vulkan-embedding-grad-per-element and story:vulkan-embedding-grad-sorted-ids:
/// `EmbeddingGrad` is two kernels. The sort is one invocation per id, `n` of them in workgroups
/// of 64, writing `i32[n, 2]` and counting no FLOP; the gradient is one invocation per element of
/// the `[v, h]` table gradient, `v·h` of them in workgroups of 64, over the sorted ids and the
/// gradient rows, each adding the gradient column of every position of its row in ascending
/// order, so it still counts one FLOP per element of the gradient it reads: on the block's 11 by
/// 13 table and on tables whose elements do not fill a workgroup. The step plan dispatches the
/// sort, then after a barrier the gradient.
#[test]
fn embedding_grad_is_one_invocation_per_table_element() {
    let artifact = block();
    let plan = &artifact.execution;
    let pairs = embedding_grad_pairs(plan);
    for ([sort, grad], n, h, v) in &pairs {
        let signature = sort.signature();
        assert_eq!(sort.kind, KernelKind::PerRow, "{signature}");
        assert_eq!(sort.workgroup_size, 64, "{signature}");
        assert_eq!(
            sort.workgroups,
            [n.div_ceil(64) as u32, 1, 1],
            "{signature}"
        );
        assert_eq!(sort.result, TensorType::i32(&[*n, 2]), "{signature}");
        assert_eq!(sort.flop(), Some(0), "{signature}");
        let bindings = kernel_bindings(sort).expect("the sort binds");
        assert_eq!(u64::from(bindings.count), *n, "{signature}");
        assert_eq!(bindings.invocations, *n, "{signature}");
        assert_valid(sort);
        let signature = grad.signature();
        assert_eq!(grad.kind, KernelKind::PerElement, "{signature}");
        assert_eq!(grad.workgroup_size, 64, "{signature}");
        assert_eq!(
            grad.workgroups,
            [(v * h).div_ceil(64) as u32, 1, 1],
            "{signature}"
        );
        assert_eq!(
            grad.operands,
            [
                Operand::Buffer(TensorType::i32(&[*n, 2])),
                Operand::Buffer(TensorType::f32(&[*n, *h]))
            ],
            "{signature}"
        );
        assert_eq!(grad.flop(), Some(n * h), "{signature}");
        let bindings = kernel_bindings(grad).expect("the gradient binds");
        assert_eq!(u64::from(bindings.count), v * h, "{signature}");
        assert_eq!(bindings.invocations, v * h, "{signature}");
        assert_valid(grad);
    }
    let [sort, grad] = &pairs[0].0;
    let printed = print_kernel(plan);
    for (id, kind) in [(sort.id, "per_row"), (grad.id, "per_element")] {
        assert!(
            printed
                .lines()
                .any(|l| l.starts_with("kernel ")
                    && l.contains(&format!(" {id} {kind} EmbeddingGrad "))),
            "{printed}"
        );
    }
    let at = |id| {
        plan.step
            .iter()
            .position(|s| s.op == PlanOp::Dispatch && s.kernel_module_id == Some(id))
            .expect("dispatched")
    };
    let (first, second) = (at(sort.id), at(grad.id));
    assert!(first < second, "the sort before the gradient");
    assert!(
        plan.step[first..second]
            .iter()
            .any(|s| s.op == PlanOp::Barrier),
        "a barrier between the sort and the gradient"
    );
}

/// `OpVariable`, its storage class `Workgroup`, and `OpControlBarrier`.
const OP_VARIABLE: u32 = 59;
const STORAGE_WORKGROUP: u32 = 4;
const OP_CONTROL_BARRIER: u32 = 224;

/// The sort of `EmbeddingGrad` stages the ids in workgroup memory, 64 at a time, between
/// barriers, so each invocation compares its id with ids it reads from there: one `Workgroup`
/// variable and a barrier; the gradient reads the sorted ids and needs neither. On the block's
/// table and on tables whose elements do not fill a workgroup.
#[test]
fn embedding_grad_stages_the_ids_in_workgroup_memory() {
    let artifact = block();
    let pairs = embedding_grad_pairs(&artifact.execution);
    assert_eq!(pairs.len(), 5);
    for ([sort, grad], ..) in &pairs {
        for (kernel, stages) in [(sort, true), (grad, false)] {
            let words = emit(kernel).expect("the kernel emits");
            let module = instructions(&words);
            let workgroup = module
                .iter()
                .filter(|(op, w)| *op == OP_VARIABLE && w[3] == STORAGE_WORKGROUP)
                .count();
            let barriers = module
                .iter()
                .filter(|(op, _)| *op == OP_CONTROL_BARRIER)
                .count();
            assert_eq!(workgroup, usize::from(stages), "{}", kernel.signature());
            assert_eq!(barriers > 0, stages, "{}", kernel.signature());
            assert_valid(kernel);
        }
    }
}

/// The one kernel of `op` among the row-wise kernels of `[n, h]` and a `v`-row table.
fn kernel_of_shape(op: &str, n: u64, h: u64, v: u64) -> Kernel {
    let mut found: Vec<Kernel> = kernels_of_shape(n, h, v)
        .into_iter()
        .filter(|k| k.op == op)
        .collect();
    assert_eq!(found.len(), 1, "one {op} kernel of [{n}, {h}] and {v} rows");
    found.remove(0)
}

/// The `(flop, bytes)` of every dispatch of `op` in the block's step plan, in plan order.
fn block_counts(op: &str) -> Vec<(u64, u64)> {
    let artifact = block();
    let plan = &artifact.execution;
    plan.step
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
        .collect()
}

/// story:bert-tiny-vs-pytorch-cuda, acceptance of the counts: `Embedding` against values
/// computed by hand from DESIGN.md § 3.1 "Counting rule" ("`per_element` `Embedding` none";
/// bytes "the `size_bytes` of the step's read views plus those of its write views"). A row
/// copy counts 0 FLOP, a count, not `uncounted`, on every shape. The block's one embedding
/// reads its f32[11, 13] table (572 B) and i32[5] ids (20 B) and writes f32[5, 13] (260 B):
/// 852 B.
#[test]
fn embedding_counts_no_flop_by_hand() {
    for (n, h, v) in [(1, 1, 1), (3, 5, 7), (65, 63, 3), (130, 512, 129)] {
        let kernel = kernel_of_shape("Embedding", n, h, v);
        assert_eq!(
            kernel.kind,
            KernelKind::PerElement,
            "{}",
            kernel.signature()
        );
        assert_eq!(kernel.flop(), Some(0), "{}", kernel.signature());
    }
    assert_eq!(block_counts("Embedding"), [(0, 852)]);
}

/// story:bert-tiny-vs-pytorch-cuda, acceptance of the counts: `Dropout` against values computed
/// by hand from DESIGN.md § 3.1 "Counting rule" ("`per_element` ... `Dropout` and `DropoutGrad`
/// one per output element"): `[1, 1]` 1, `[3, 5]` 15, `[65, 63]` 4,095 and `[130, 512]` 66,560
/// FLOP. The block's one dropout of f32[5, 13] counts 65 FLOP and reads 260 B and its i32[2] RNG
/// tokens (8 B) and writes 260 B: 528 B.
#[test]
fn dropout_counts_one_flop_per_element_by_hand() {
    for (n, h, flop) in [(1, 1, 1), (3, 5, 15), (65, 63, 4_095), (130, 512, 66_560)] {
        let kernel = kernel_of_shape("Dropout", n, h, 3);
        assert_eq!(
            kernel.kind,
            KernelKind::PerElement,
            "{}",
            kernel.signature()
        );
        assert_eq!(kernel.flop(), Some(flop), "{}", kernel.signature());
    }
    assert_eq!(block_counts("Dropout"), [(65, 528)]);
}

/// story:bert-tiny-vs-pytorch-cuda, acceptance of the counts: `DropoutGrad` against values
/// computed by hand from DESIGN.md § 3.1 "Counting rule" ("`per_element` ... `Dropout` and
/// `DropoutGrad` one per output element"): `[1, 1]` 1, `[3, 5]` 15, `[65, 63]` 4,095 and
/// `[130, 512]` 66,560 FLOP. The block's one dropout gradient of f32[5, 13] counts 65 FLOP and
/// reads the i32[2] RNG tokens (8 B) and the 260 B gradient and writes 260 B: 528 B.
#[test]
fn dropout_grad_counts_one_flop_per_element_by_hand() {
    for (n, h, flop) in [(1, 1, 1), (3, 5, 15), (65, 63, 4_095), (130, 512, 66_560)] {
        let kernel = kernel_of_shape("DropoutGrad", n, h, 3);
        assert_eq!(
            kernel.kind,
            KernelKind::PerElement,
            "{}",
            kernel.signature()
        );
        assert_eq!(kernel.flop(), Some(flop), "{}", kernel.signature());
    }
    assert_eq!(block_counts("DropoutGrad"), [(65, 528)]);
}
