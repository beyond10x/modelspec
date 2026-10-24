//! `story:fast-math-mode`, acceptance `emission-checked` (DESIGN.md § 5.4, § 12 ADR-008), no
//! Vulkan instance (runs in `task test`): the modules `emit_kernels` builds for an artifact whose
//! `fast_math` is set carry no `NoContraction` on any `OpFMul`, `OpFAdd` or `OpFSub` of a matmul,
//! reduction, loss-sum or SGD module, and those of the same artifact with `fast_math` off carry it
//! on every one. Nothing else in a module changes: each `fast_math` module is the deterministic
//! module with its `NoContraction` decorations removed, word for word, and passes `spirv-val`.
//! Every cache key names the artifact's `fast_math`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::path::PathBuf;

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::compile_artifact;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Kernel, KernelKind};
use modelspec_runtime::spirv::emit::emit_kernels;
use modelspec_runtime::spirv::validate::validate;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

/// The RTX 3090 fixture, or the same profile reporting all three float controls.
fn profile(float_controls: bool) -> DeviceProfile {
    let mut p = DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile");
    if float_controls {
        p.shader_rounding_mode_rte_f32 = Some(true);
        p.shader_denorm_preserve_f32 = Some(true);
        p.shader_signed_zero_inf_nan_preserve_f32 = Some(true);
    }
    p
}

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn example(path: &str) -> CompiledArtifact {
    let source = std::fs::read_to_string(repository().join(path)).expect("the example reads");
    compile_artifact(path, &source, Backend::Vulkan).expect("the example compiles")
}

/// A program whose loss is the `sum` or `mean` of `n` products: a two-stage reduction.
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

/// The four module classes ADR-008 lets contract, or `None` for a module with no such arithmetic.
fn class(kernel: &Kernel) -> Option<&'static str> {
    match kernel.kind {
        KernelKind::MatMul => Some("matmul"),
        KernelKind::ReduceStage1 | KernelKind::ReduceStage2 => Some("reduction"),
        KernelKind::LossSum => Some("loss-sum"),
        KernelKind::SgdUpdate if kernel.operands.len() == 3 => Some("sgd"),
        _ => None,
    }
}

/// `module` without its `OpDecorate <id> NoContraction` instructions; the header is kept.
fn without_no_contraction(module: &[u32]) -> Vec<u32> {
    let mut out = module[..5].to_vec();
    for (op, words) in common::words::instructions(module) {
        let dropped = op == common::words::OP_DECORATE
            && words[2] == common::words::DECORATION_NO_CONTRACTION;
        if !dropped {
            out.extend_from_slice(words);
        }
    }
    out
}

#[test]
fn a_fast_math_artifact_emits_no_no_contraction_and_a_deterministic_one_emits_it_on_every_float_op()
{
    let artifacts = [
        ("mnist_mlp", example("examples/mnist_mlp.mspec")),
        (
            "linear_regression",
            example("examples/linear_regression.mspec"),
        ),
        ("sum over 4097", reduction(4_097, "sum")),
        ("mean over 4097", reduction(4_097, "mean")),
    ];
    let mut seen = std::collections::BTreeMap::<&str, usize>::new();
    for (name, deterministic) in &artifacts {
        assert!(
            !deterministic.fast_math,
            "{name}: compiled with fast_math off"
        );
        let mut fast = deterministic.clone();
        fast.fast_math = true;
        for float_controls in [false, true] {
            let profile = profile(float_controls);
            let exact = emit_kernels(deterministic, &profile).expect("emits");
            let contracted = emit_kernels(&fast, &profile).expect("emits");
            assert_eq!(exact.len(), deterministic.execution.kernels.len(), "{name}");
            assert_eq!(contracted.len(), exact.len(), "{name}");
            for ((kernel, off), on) in deterministic
                .execution
                .kernels
                .iter()
                .zip(&exact)
                .zip(&contracted)
            {
                let what = format!(
                    "{name} float_controls={float_controls}: {}",
                    kernel.signature()
                );
                assert!(
                    !off.cache_key.fast_math,
                    "{what}: the key names fast_math off"
                );
                assert!(on.cache_key.fast_math, "{what}: the key names fast_math on");
                assert_eq!(
                    on.kernel_module_id, off.kernel_module_id,
                    "{what}: one kernel"
                );
                validate(&on.spirv).unwrap_or_else(|e| panic!("{what}: fast_math: {e}"));
                assert_eq!(
                    on.spirv,
                    without_no_contraction(&off.spirv),
                    "{what}: the fast_math module is the deterministic one without NoContraction"
                );
                let Some(class) = class(kernel) else {
                    continue;
                };
                let (missing, total) = common::words::uncontracted_arithmetic(&off.spirv);
                assert!(total > 0, "{what}: has float arithmetic");
                assert!(
                    missing.is_empty(),
                    "{what}: fast_math off leaves {missing:?} of {total} without NoContraction"
                );
                let (missing, total) = common::words::uncontracted_arithmetic(&on.spirv);
                assert_eq!(
                    missing.len(),
                    total,
                    "{what}: fast_math on leaves {} of {total} decorated NoContraction",
                    total - missing.len()
                );
                *seen.entry(class).or_default() += 1;
            }
        }
    }
    eprintln!("fast_math_emission: modules checked per class: {seen:?}");
    for class in ["matmul", "reduction", "loss-sum", "sgd"] {
        assert!(
            seen.get(class).copied().unwrap_or(0) > 0,
            "no {class} module was checked: {seen:?}"
        );
    }
}
