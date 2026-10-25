//! Adversary pass on `story:fast-math-mode` (DESIGN.md § 5.4, § 12 ADR-008), no Vulkan instance.
//!
//! Attacks 1 and 2 of the pass, over every kernel kind the plan emits and not only the four the
//! story names: the MNIST and linear-regression examples, the `mlp_sweep` sources of the three
//! story points (b8192 h4096, b8192 h1024, b1000 h4096, whose large matmuls are register-blocked),
//! two-stage `sum` and `mean` reductions and a transpose read elementwise, each against the
//! RTX 3090 fixture and the same profile reporting the three float controls. For every kernel:
//!
//! - the `fast_math` module has no `OpDecorate ... NoContraction`;
//! - its execution modes, capabilities and the count of every opcode equal the deterministic
//!   module's, `OpDecorate` less exactly the deterministic module's `NoContraction` count;
//! - in the deterministic module every `NoContraction` names an `OpFAdd`, `OpFSub`, `OpFMul` or
//!   `GLSL.std.450 Fma` (DESIGN.md § 12 ADR-009), every float add, subtract, multiply and fused
//!   multiply-add of a non-elementwise module, of a three-operand SGD update and of a fused
//!   elementwise chain carries one, and any other elementwise module carries none (the default
//!   the base emitted);
//! - the two cache keys are equal but for `fast_math`, and the module ids equal.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use common::words::{
    DECORATION_NO_CONTRACTION, OP_CAPABILITY, OP_DECORATE, OP_EXECUTION_MODE, OP_F_ADD, OP_F_MUL,
    OP_F_SUB, instructions,
};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::compile_artifact;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Kernel, KernelKind, MatMulTile};
use modelspec_runtime::spirv::emit::emit_kernels;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

fn profile(float_controls: bool) -> DeviceProfile {
    let mut p = DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile");
    if float_controls {
        p.shader_rounding_mode_rte_f32 = Some(true);
        p.shader_denorm_preserve_f32 = Some(true);
        p.shader_signed_zero_inf_nan_preserve_f32 = Some(true);
    }
    p
}

fn example(path: &str) -> CompiledArtifact {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = std::fs::read_to_string(root.join(path)).expect("the example reads");
    compile_artifact(path, &source, Backend::Vulkan).expect("the example compiles")
}

/// The source `benches/mlp_sweep/sweep.rs` writes for batch `b`, hidden `h`.
fn sweep(b: u64, h: u64) -> CompiledArtifact {
    let source = format!(
        "model SweepMlp {{\n    param w1: Tensor<f32, [784, {h}]> = xavier_uniform();\n    \
         param w2: Tensor<f32, [{h}, 10]> = xavier_uniform();\n\n    \
         fn forward(x: Tensor<f32, [{b}, 784]>)\n        -> Tensor<f32, [{b}, 10]> {{\n        \
         return matmul(relu(matmul(x, w1)), w2);\n    }}\n}}\n\ntrain Fit {{\n    \
         model = SweepMlp;\n    input x: Tensor<f32, [{b}, 784]>;\n    \
         target y: Tensor<f32, [{b}, 10]>;\n    loss = mse(model.forward(x), y);\n    \
         optimizer = sgd(lr = 0.001);\n    steps = 220;\n    seed = 42;\n    \
         device = vulkan(0);\n}}\n"
    );
    compile_artifact("sweep.mspec", &source, Backend::Vulkan).expect("the sweep source compiles")
}

/// A program whose loss is the `op` of `n` products: two reduce stages and a broadcast.
fn reduction(n: u64, op: &str) -> CompiledArtifact {
    let source = format!(
        "model R {{\n    param w: Tensor<f32, [{n}, 1]> = xavier_uniform();\n\n    \
         fn forward(x: Tensor<f32, [{n}, 1]>)\n        -> Tensor<f32, [{n}, 1]> {{\n        \
         return mul(x, w);\n    }}\n}}\n\ntrain Fit {{\n    model = R;\n    \
         input x: Tensor<f32, [{n}, 1]>;\n    target y: Tensor<f32, [{n}, 1]>;\n    \
         loss = {op}(mul(model.forward(x), y));\n    optimizer = sgd(lr = 0.1);\n    \
         steps = 1;\n    seed = 42;\n    device = vulkan(0);\n}}\n"
    );
    compile_artifact("reduction.mspec", &source, Backend::Vulkan).expect("the source compiles")
}

/// A program whose forward multiplies the transpose of its input by a parameter, elementwise: a
/// transpose that a kernel other than a matmul reads stays a `Transpose` dispatch (DESIGN.md § 3.1).
fn elementwise_transpose(n: u64) -> CompiledArtifact {
    let source = format!(
        "model T {{\n    param w: Tensor<f32, [{n}, {n}]> = xavier_uniform();\n\n    \
         fn forward(x: Tensor<f32, [{n}, {n}]>)\n        -> Tensor<f32, [{n}, {n}]> {{\n        \
         return mul(transpose(x), w);\n    }}\n}}\n\ntrain Fit {{\n    model = T;\n    \
         input x: Tensor<f32, [{n}, {n}]>;\n    target y: Tensor<f32, [{n}, {n}]>;\n    \
         loss = mse(model.forward(x), y);\n    optimizer = sgd(lr = 0.1);\n    \
         steps = 1;\n    seed = 42;\n    device = vulkan(0);\n}}\n"
    );
    compile_artifact("transpose.mspec", &source, Backend::Vulkan).expect("the source compiles")
}

fn opcode_counts(module: &[u32]) -> BTreeMap<u32, usize> {
    let mut counts = BTreeMap::new();
    for (op, _) in instructions(module) {
        *counts.entry(op).or_default() += 1;
    }
    counts
}

fn with_op(module: &[u32], wanted: u32) -> Vec<Vec<u32>> {
    instructions(module)
        .into_iter()
        .filter(|(op, _)| *op == wanted)
        .map(|(_, w)| w.to_vec())
        .collect()
}

/// The ids decorated `NoContraction`.
fn no_contraction_targets(module: &[u32]) -> Vec<u32> {
    with_op(module, OP_DECORATE)
        .into_iter()
        .filter(|w| w[2] == DECORATION_NO_CONTRACTION)
        .map(|w| w[1])
        .collect()
}

/// The result ids of every `OpFAdd`, `OpFSub` and `OpFMul`.
/// `OpExtInst`, and `Fma` in `GLSL.std.450`, the only instruction set a module imports.
const OP_EXT_INST: u32 = 12;
const GLSL_FMA: u32 = 50;

fn float_arithmetic(module: &[u32]) -> BTreeSet<u32> {
    instructions(module)
        .into_iter()
        .filter(|(op, w)| {
            [OP_F_ADD, OP_F_SUB, OP_F_MUL].contains(op) || (*op == OP_EXT_INST && w[4] == GLSL_FMA)
        })
        .map(|(_, w)| w[2])
        .collect()
}

/// A fused elementwise chain (story:fuse-elementwise-chains-and-gradient-sums), whose every
/// add, subtract and multiply carries `NoContraction`.
fn is_chain(kernel: &Kernel) -> bool {
    kernel.kind == KernelKind::Elementwise && kernel.op.contains('>')
}

fn is_scaled_sub(kernel: &Kernel) -> bool {
    kernel.kind == KernelKind::SgdUpdate && kernel.operands.len() == 3
}

fn is_register_blocked(kernel: &Kernel) -> bool {
    kernel.kind == KernelKind::MatMul
        && Kernel::matmul_shape(&kernel.operands)
            .map(|(m, k, n)| MatMulTile::of(m, k, n))
            .is_some_and(|tile| tile.rows > 2 || tile.cols_per_invocation > 1)
}

#[test]
fn every_kernel_kind_of_a_fast_math_artifact_is_its_deterministic_module_less_no_contraction() {
    let artifacts = [
        ("mnist_mlp", example("examples/mnist_mlp.mspec")),
        (
            "linear_regression",
            example("examples/linear_regression.mspec"),
        ),
        ("sweep b8192 h4096", sweep(8192, 4096)),
        ("sweep b8192 h1024", sweep(8192, 1024)),
        ("sweep b1000 h4096", sweep(1000, 4096)),
        ("sum over 65", reduction(65, "sum")),
        ("mean over 78400", reduction(78_400, "mean")),
        ("transpose read elementwise", elementwise_transpose(24)),
    ];
    let mut seen = BTreeMap::<String, usize>::new();
    let mut removed_total = 0usize;
    for (name, deterministic) in &artifacts {
        let mut fast = deterministic.clone();
        fast.fast_math = true;
        for float_controls in [false, true] {
            let p = profile(float_controls);
            let off = emit_kernels(deterministic, &p).expect("emits");
            let on = emit_kernels(&fast, &p).expect("emits");
            assert_eq!(off.len(), on.len(), "{name}");
            for ((kernel, off), on) in deterministic.execution.kernels.iter().zip(&off).zip(&on) {
                let what = format!("{name} fc={float_controls}: {}", kernel.signature());
                assert_eq!(on.kernel_module_id, off.kernel_module_id, "{what}");
                let mut key = on.cache_key.clone();
                assert!(key.fast_math && !off.cache_key.fast_math, "{what}");
                key.fast_math = false;
                assert_eq!(key, off.cache_key, "{what}: keys differ only in fast_math");

                assert!(
                    no_contraction_targets(&on.spirv).is_empty(),
                    "{what}: a fast_math module keeps NoContraction"
                );
                assert_eq!(
                    with_op(&on.spirv, OP_EXECUTION_MODE),
                    with_op(&off.spirv, OP_EXECUTION_MODE),
                    "{what}: execution modes"
                );
                assert_eq!(
                    with_op(&on.spirv, OP_CAPABILITY),
                    with_op(&off.spirv, OP_CAPABILITY),
                    "{what}: capabilities"
                );
                let removed = no_contraction_targets(&off.spirv).len();
                let mut expected = opcode_counts(&off.spirv);
                if removed > 0 {
                    let decorations = expected.get_mut(&OP_DECORATE).expect("decorations");
                    *decorations -= removed;
                    if *decorations == 0 {
                        expected.remove(&OP_DECORATE);
                    }
                }
                assert_eq!(
                    opcode_counts(&on.spirv),
                    expected,
                    "{what}: opcode counts other than the dropped decorations"
                );
                assert_eq!(on.spirv[..5], off.spirv[..5], "{what}: header");

                let targets: BTreeSet<u32> =
                    no_contraction_targets(&off.spirv).into_iter().collect();
                let arithmetic = float_arithmetic(&off.spirv);
                assert!(
                    targets.is_subset(&arithmetic),
                    "{what}: NoContraction on a non-arithmetic id"
                );
                assert_eq!(
                    float_arithmetic(&on.spirv).len(),
                    arithmetic.len(),
                    "{what}: same float arithmetic"
                );
                match kernel.kind {
                    KernelKind::Elementwise | KernelKind::SgdUpdate
                        if !is_scaled_sub(kernel) && !is_chain(kernel) =>
                    {
                        assert!(targets.is_empty(), "{what}: elementwise decorated")
                    }
                    _ => assert_eq!(targets, arithmetic, "{what}: a float op left undecorated"),
                }
                if is_scaled_sub(kernel) {
                    assert_eq!(targets.len(), 2, "{what}: product and difference");
                }
                removed_total += removed;
                *seen.entry(format!("{:?}", kernel.kind)).or_default() += 1;
                if kernel.epilogue.is_some() {
                    *seen.entry("MatMul+epilogue".into()).or_default() += 1;
                }
                if is_register_blocked(kernel) {
                    *seen.entry("MatMul register-blocked".into()).or_default() += 1;
                }
                if is_scaled_sub(kernel) {
                    *seen.entry("SgdUpdate scaled".into()).or_default() += 1;
                }
                if is_chain(kernel) {
                    *seen.entry("Elementwise chain".into()).or_default() += 1;
                }
                if kernel.transposed.contains(&true) {
                    *seen.entry("MatMul transposed".into()).or_default() += 1;
                }
            }
        }
    }
    eprintln!("adversary_fastmath_emission: kinds {seen:?}, decorations dropped {removed_total}");
    for kind in [
        "Elementwise",
        "Broadcast",
        "ReduceStage1",
        "ReduceStage2",
        "LossSum",
        "Transpose",
        "MatMul",
        "SgdUpdate",
        "MatMul+epilogue",
        "MatMul register-blocked",
        "SgdUpdate scaled",
        "Elementwise chain",
        "MatMul transposed",
    ] {
        assert!(
            seen.contains_key(kind),
            "no {kind} kernel was checked: {seen:?}"
        );
    }
    assert!(removed_total > 0);
}
