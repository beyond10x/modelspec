//! Adversary pass 1 on story:bert-vulkan-kernels-rowwise, no Vulkan instance: `spirv-val` over
//! shapes and drop probabilities the unit's suite does not emit, the `NoContraction` claim for
//! every `OpFDiv` (DESIGN.md § 5.2, revision 0.46), the counting rule of DESIGN.md § 3.1
//! recomputed by hand for one plan, the emitter's refusals of a kernel record the schedule
//! never builds, and the E-TARGET-011 sites that remain (`attention`, `cross_entropy`).
//! Needs `spirv-val` on the path.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeSet;

use common::words::{DECORATION_NO_CONTRACTION, OP_DECORATE, OP_EXT_INST, instructions};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::compile_artifact;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{DispatchCount, Kernel, KernelKind, Operand, PlanOp};
use modelspec_core::ir::tensor::{Dim, TensorOp, TensorType};
use modelspec_core::schedule::rowwise_kernel;
use modelspec_runtime::spirv::emit::{emit, emit_kernels};
use modelspec_runtime::spirv::validate::validate;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

/// `OpFDiv`.
const OP_F_DIV: u32 = 136;
/// `Exp` and `Sqrt` of `GLSL.std.450`.
const GLSL_EXP: u32 = 27;
const GLSL_SQRT: u32 = 31;

fn f32s(shape: &[u64]) -> Operand {
    Operand::Buffer(TensorType::f32(shape))
}

fn i32s(shape: &[u64]) -> Operand {
    Operand::Buffer(TensorType::i32(shape))
}

/// The thirteen row-wise kernels of one `[n, h]` shape, a `v`-row table and drop
/// probability `p`.
fn kernels_of_shape(n: u64, h: u64, v: u64, p: f32) -> Vec<Kernel> {
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
            TensorOp::Dropout { p },
            vec![f32s(&[n, h]), i32s(&[2])],
            rows.clone(),
        ),
        (
            TensorOp::DropoutGrad { p },
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
        .map(|(op, operands, result)| {
            rowwise_kernel(&op, operands, result)
                .unwrap_or_else(|| panic!("{} [{n}, {h}] v {v} has a kernel", op.name()))
        })
        .collect()
}

/// Hypothesis 6: every module of N in {1, 64, 65, 4096} rows, D in {1, 2, 64, 65, 1024}
/// columns and tables of 1 and 50,000 rows passes `spirv-val`, the dropout modules at drop
/// probabilities 0, 0.1, 0.999 and the largest `f32` below 1.
#[test]
fn adv_rowwise_every_module_of_the_extended_grid_passes_spirv_val() {
    let below_one = f32::from_bits(1.0f32.to_bits() - 1);
    let ps = [0.0_f32, 0.1, 0.999, below_one];
    let mut modules = 0;
    let mut k = 0;
    for n in [1_u64, 64, 65, 4096] {
        for h in [1_u64, 2, 64, 65, 1024] {
            for v in [1_u64, 50_000] {
                let p = ps[k % ps.len()];
                k += 1;
                for kernel in kernels_of_shape(n, h, v, p) {
                    let words =
                        emit(&kernel).unwrap_or_else(|e| panic!("{}: {e}", kernel.signature()));
                    if let Err(error) = validate(&words) {
                        panic!("{}: {error}", kernel.signature());
                    }
                    modules += 1;
                }
            }
        }
    }
    eprintln!("{modules} row-wise modules pass spirv-val");
    assert_eq!(modules, 4 * 5 * 2 * 13);
}

const BLOCK: &str = "model Block {
    param w: Tensor<f32, [8, 8]> = xavier_uniform();
    param b: Tensor<f32, [8]> = zeros();
    param gamma: Tensor<f32, [8]> = ones();
    param beta: Tensor<f32, [8]> = zeros();

    fn forward(x: Tensor<f32, [4, 8]>) -> Tensor<f32, [4, 8]> {
        let h = relu(add(mul(matmul(x, w), x), x));
        let g = sub(gelu(add_rows(h, b)), x);
        let n = layer_norm(mul(relu(g), x), gamma, beta);
        return mul(softmax(add(n, x)), x);
    }
}

train Fit {
    model = Block;
    input x: Tensor<f32, [4, 8]>;
    target y: Tensor<f32, [4, 8]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 3;
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

/// The `OpFDiv`s of a module, and how many of them are decorated `NoContraction`.
fn divisions(words: &[u32]) -> (usize, usize) {
    let all = instructions(words);
    let decorated: BTreeSet<u32> = all
        .iter()
        .filter(|(op, w)| *op == OP_DECORATE && w[2] == DECORATION_NO_CONTRACTION)
        .map(|(_, w)| w[1])
        .collect();
    let divs: Vec<u32> = all
        .iter()
        .filter(|(op, _)| *op == OP_F_DIV)
        .map(|(_, w)| w[2])
        .collect();
    let exact = divs.iter().filter(|id| decorated.contains(id)).count();
    (divs.len(), exact)
}

fn uses_glsl(words: &[u32], op: u32) -> bool {
    instructions(words)
        .iter()
        .any(|(code, w)| *code == OP_EXT_INST && w[4] == op)
}

/// DESIGN.md § 5.2 and the module comment of `spirv/emit.rs`: every `OpFDiv` of a row-wise
/// module is decorated `NoContraction` in the deterministic mode and none is under
/// `fast_math`; the ops whose CPU definition divides (`Gelu`, `GeluGrad`, `LayerNorm`,
/// `LayerNormGradX`, `LayerNormGradGamma`, `Softmax`) have divisions, and those that take an
/// `exp` or a square root import them. The unit's own check counts adds, subtracts, multiplies
/// and `Fma`s only.
#[test]
fn adv_rowwise_every_division_is_uncontracted_outside_fast_math() {
    let artifact = block();
    let mut fast = artifact.clone();
    fast.fast_math = true;
    let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a profile");
    let off = emit_kernels(&artifact, &profile).expect("emits");
    let on = emit_kernels(&fast, &profile).expect("emits");
    let mut dividing = BTreeSet::new();
    for ((kernel, off), on) in artifact.execution.kernels.iter().zip(&off).zip(&on) {
        if !is_rowwise(kernel) {
            continue;
        }
        let (total, exact) = divisions(&off.spirv);
        assert_eq!(exact, total, "{}", kernel.signature());
        let (fast_total, fast_exact) = divisions(&on.spirv);
        assert_eq!(fast_total, total, "{}", kernel.signature());
        assert_eq!(fast_exact, 0, "{}", kernel.signature());
        if total > 0 {
            dividing.insert(kernel.op);
        }
        let exp = matches!(kernel.op, "Gelu" | "GeluGrad" | "Softmax");
        assert_eq!(
            uses_glsl(&off.spirv, GLSL_EXP),
            exp,
            "{}",
            kernel.signature()
        );
        let sqrt = kernel.op.starts_with("LayerNorm");
        assert_eq!(
            uses_glsl(&off.spirv, GLSL_SQRT),
            sqrt,
            "{}",
            kernel.signature()
        );
    }
    let expected: BTreeSet<&str> = [
        "Gelu",
        "GeluGrad",
        "LayerNorm",
        "LayerNormGradX",
        "LayerNormGradGamma",
        "Softmax",
    ]
    .into_iter()
    .collect();
    assert_eq!(dividing, expected);
}

/// Hypothesis 5: the counting rule of DESIGN.md § 3.1 recomputed by hand for the `[4, 8]`
/// block. `LayerNorm` `n·(8h + 3)` = 4·67 = 268 FLOP, bytes (32 + 8 + 8 + 32)·4 = 320;
/// `LayerNormGradX` `n·(14h + 3)` = 4·115 = 460, (32 + 8 + 32 + 32)·4 = 416;
/// `LayerNormGradGamma` 268, (32 + 32 + 8)·4 = 288; `Softmax` 5·32 = 160, (32 + 32)·4 = 256;
/// `SoftmaxGrad` 4·32 = 128, 384; `Gelu` 10·32 = 320, 256; `GeluGrad` 24·32 = 768, 384;
/// `AddRows` 32, (32 + 8 + 32)·4 = 288; `SumRows` 32, (32 + 8)·4 = 160. Every row-wise op is
/// one dispatch per use, and the elementwise ops around it are not fused into it.
#[test]
fn adv_rowwise_counting_rule_recomputed_by_hand() {
    let artifact = block();
    let plan = &artifact.execution;
    let expected: [(&str, u64, u64); 9] = [
        ("LayerNorm", 268, 320),
        ("LayerNormGradX", 460, 416),
        ("LayerNormGradGamma", 268, 288),
        ("Softmax", 160, 256),
        ("SoftmaxGrad", 128, 384),
        ("Gelu", 320, 256),
        ("GeluGrad", 768, 384),
        ("AddRows", 32, 288),
        ("SumRows", 32, 160),
    ];
    for (op, flop, bytes) in expected {
        let kernel = plan
            .kernels
            .iter()
            .find(|k| k.op == op)
            .unwrap_or_else(|| panic!("no {op} kernel"));
        let steps: Vec<_> = plan
            .step
            .iter()
            .filter(|s| s.op == PlanOp::Dispatch && s.kernel_module_id == Some(kernel.id))
            .collect();
        assert!(!steps.is_empty(), "{op} never dispatched");
        for step in steps {
            assert_eq!(
                plan.dispatch_count(step),
                Some(DispatchCount::Counted { flop, bytes }),
                "{op}"
            );
        }
    }
    for kernel in &plan.kernels {
        if kernel.kind == KernelKind::Elementwise {
            assert!(
                !kernel.op.contains("Gelu")
                    && !kernel.op.contains("AddRows")
                    && !kernel.op.contains("Softmax")
                    && !kernel.op.contains("LayerNorm"),
                "a row-wise op in a chain: {}",
                kernel.signature()
            );
        }
    }
}

/// The emitter refuses a row-wise kernel record the schedule never builds: the wrong kind, a
/// workgroup size other than 64, workgroups that cover too few invocations, a drop
/// probability outside `[0, 1)`.
#[test]
fn adv_rowwise_emit_refuses_records_the_schedule_never_builds() {
    let gelu = rowwise_kernel(
        &TensorOp::Gelu,
        vec![f32s(&[65, 3])],
        TensorType::f32(&[65, 3]),
    )
    .expect("a kernel");
    assert!(emit(&gelu).is_ok());
    let mut wrong_kind = gelu.clone();
    wrong_kind.kind = KernelKind::PerRow;
    assert!(emit(&wrong_kind).is_err(), "a per-row Gelu");
    let mut wide = gelu.clone();
    wide.workgroup_size = 128;
    assert!(emit(&wide).is_err(), "workgroups of 128");
    let mut short = gelu.clone();
    short.workgroups = [3, 1, 1];
    assert!(emit(&short).is_err(), "192 invocations for 195 elements");
    let mut dropout = rowwise_kernel(
        &TensorOp::Dropout { p: 0.5 },
        vec![f32s(&[4]), i32s(&[2])],
        TensorType::f32(&[4]),
    )
    .expect("a kernel");
    dropout.operands[2] = Operand::Immediate(1.0);
    assert!(emit(&dropout).is_err(), "p = 1");
    assert!(
        rowwise_kernel(
            &TensorOp::Gelu,
            vec![Operand::Buffer(TensorType {
                dtype: modelspec_core::ir::tensor::DType::F32,
                shape: vec![Dim::Symbol(0), Dim::Static(3)],
            })],
            TensorType::f32(&[2, 3]),
        )
        .is_none(),
        "a symbolic dimension has no kernel"
    );
}

const CROSS_ENTROPY: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../modelspec-cli/tests/fixtures/cross_entropy_train.mspec"
));

/// Hypothesis 7: `cross_entropy` is still E-TARGET-011 at its call for Vulkan now that the
/// `embedding` before it has kernels; no test of the unit or its base asserted this site.
#[test]
fn adv_rowwise_cross_entropy_is_still_e_target_011_at_its_call() {
    let errors = compile_artifact("cross_entropy_train.mspec", CROSS_ENTROPY, Backend::Vulkan)
        .expect_err("cross_entropy has no Vulkan kernel");
    assert_eq!(errors.len(), 1, "{errors:#?}");
    let text = errors[0].to_string();
    assert!(
        text.starts_with("E-TARGET-011: operator `cross_entropy` has no Vulkan kernel yet"),
        "{text}"
    );
    let span = errors[0].span.as_ref().expect("a span");
    assert_eq!((span.line, span.column), (14, 12), "{text}");
}
