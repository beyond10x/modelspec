//! Adversary pass 1 on `story:m1-vulkan-reduce-matmul`, no Vulkan instance (runs in `task ci`):
//! the modules `emit_kernels` builds for the planned artifacts against the module's own doc and
//! DESIGN.md § 5.4 (`NoContraction` on every reduction, loss-sum and matmul `OpFAdd`/`OpFSub`/
//! `OpFMul`; the cache key names exactly the capabilities the module declares, each once), and
//! the binding sizes of a planned reduction against the plan's partials view, past 64 * 64
//! partials and on a two-dimensional stage-1 grid.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::path::PathBuf;

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::compile_artifact;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{KernelKind, PlanOp};
use modelspec_runtime::spirv::emit::{emit_kernels, emit_loss_sum_undivided};
use modelspec_runtime::spirv::validate::validate;
use modelspec_runtime::vulkan::dispatch::kernel_bindings;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

fn profile(all_true: bool) -> DeviceProfile {
    let mut p = DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile");
    if all_true {
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

fn reduction_source(n: u64, op: &str) -> CompiledArtifact {
    let source = format!(
        "model R {{\n    param w: Tensor<f32, [{n}, 1]> = xavier_uniform();\n\n    \
         fn forward(x: Tensor<f32, [{n}, 1]>)\n        -> Tensor<f32, [{n}, 1]> {{\n        \
         return mul(x, w);\n    }}\n}}\n\ntrain Fit {{\n    model = R;\n    \
         input x: Tensor<f32, [{n}, 1]>;\n    target y: Tensor<f32, [{n}, 1]>;\n    \
         loss = {op}(mul(model.forward(x), y));\n    optimizer = sgd(lr = 0.1);\n    \
         steps = 1;\n    seed = 42;\n    device = vulkan(0);\n}}\n"
    );
    compile_artifact("adv.mspec", &source, Backend::Vulkan).expect("the source compiles")
}

/// The capability enumerants of every `OpCapability`, by the names the cache key uses.
fn declared_capabilities(module: &[u32]) -> Vec<String> {
    common::words::instructions(module)
        .into_iter()
        .filter(|(op, _)| *op == common::words::OP_CAPABILITY)
        .map(|(_, w)| match w[1] {
            1 => "Shader".to_owned(),
            4464 => "DenormPreserve".to_owned(),
            4466 => "SignedZeroInfNanPreserve".to_owned(),
            4467 => "RoundingModeRTE".to_owned(),
            other => format!("capability {other}"),
        })
        .collect()
}

#[test]
fn every_reduction_loss_sum_and_matmul_module_of_the_examples_has_no_contractable_arithmetic_and_a_key_naming_its_capabilities()
 {
    let mut checked = 0;
    for path in [
        "examples/linear_regression.mspec",
        "crates/modelspec-docs/models/two_layer_relu.mspec",
        "examples/mnist_mlp.mspec",
    ] {
        let mut artifacts = vec![(path.to_owned(), example(path))];
        if path == "examples/linear_regression.mspec" {
            artifacts.push(("sum over 4097".to_owned(), reduction_source(4_097, "sum")));
            artifacts.push(("mean over 4097".to_owned(), reduction_source(4_097, "mean")));
        }
        for (name, artifact) in artifacts {
            for all_true in [false, true] {
                let profile = profile(all_true);
                let modules = emit_kernels(&artifact, &profile).expect("emits");
                assert_eq!(modules.len(), artifact.execution.kernels.len(), "{name}");
                for (module, kernel) in modules.iter().zip(&artifact.execution.kernels) {
                    let what = format!("{name} all_true={all_true}: {}", kernel.signature());
                    validate(&module.spirv).unwrap_or_else(|e| panic!("{what}: {e}"));
                    let mut declared = declared_capabilities(&module.spirv);
                    let mut keyed = module.cache_key.capabilities.clone();
                    let mut distinct = keyed.clone();
                    distinct.sort();
                    distinct.dedup();
                    assert_eq!(distinct.len(), keyed.len(), "{what}: distinct by name");
                    declared.sort();
                    keyed.sort();
                    assert_eq!(
                        declared, keyed,
                        "{what}: the key names what the module declares"
                    );
                    if kernel.kind != KernelKind::Elementwise
                        && kernel.kind != KernelKind::Broadcast
                        && kernel.kind != KernelKind::SgdUpdate
                    {
                        let (missing, _) = common::words::uncontracted_arithmetic(&module.spirv);
                        assert!(
                            missing.is_empty(),
                            "{what}: no NoContraction on {missing:?}"
                        );
                        checked += 1;
                    }
                    if kernel.kind == KernelKind::LossSum {
                        let undivided = emit_loss_sum_undivided(kernel, &profile).expect("emits");
                        let mut declared = declared_capabilities(&undivided.spirv);
                        let mut keyed = undivided.cache_key.capabilities.clone();
                        declared.sort();
                        keyed.sort();
                        assert_eq!(declared, keyed, "{what}: undivided");
                    }
                }
            }
        }
    }
    assert!(checked > 0);
}

#[test]
fn a_planned_reduction_binds_the_plan_partials_view_past_4096_partials_and_on_a_2d_grid() {
    for n in [
        1u64,
        64,
        65,
        4_096 * 64,
        4_096 * 64 + 1,
        65_535 * 64,
        65_535 * 64 + 1,
    ] {
        for op in ["sum", "mean"] {
            let artifact = reduction_source(n, op);
            let plan = &artifact.execution;
            let groups = n.div_ceil(64);
            let reduced = modelspec_core::ir::tensor::TensorType::f32(&[n, 1]);
            let stage1 = plan
                .kernels
                .iter()
                .find(|k| {
                    k.kind == KernelKind::ReduceStage1 && k.reduces.as_ref() == Some(&reduced)
                })
                .expect("stage 1");
            let stage2 = plan
                .kernels
                .iter()
                .find(|k| {
                    k.kind == KernelKind::ReduceStage2 && k.reduces.as_ref() == Some(&reduced)
                })
                .expect("stage 2");
            let b1 = kernel_bindings(stage1).expect("binds");
            let b2 = kernel_bindings(stage2).expect("binds");
            assert_eq!((b1.count, b1.invocations), (n as u32, n), "{op} n={n}");
            assert_eq!(b1.elements, vec![n, groups], "{op} n={n}");
            assert_eq!((b2.count, b2.invocations), (groups as u32, 1), "{op} n={n}");
            assert_eq!(b2.elements, vec![groups, 1], "{op} n={n}");
            let covered: u64 = stage1
                .workgroups
                .iter()
                .map(|&w| u64::from(w))
                .product::<u64>()
                * u64::from(stage1.workgroup_size);
            assert!(covered >= n, "{op} n={n}: {:?}", stage1.workgroups);
            if groups > 65_535 {
                assert!(stage1.workgroups[1] > 1, "{op} n={n}: a 2D grid");
            }
            assert_eq!(stage2.workgroups, [1, 1, 1]);
            let dispatch = plan
                .step
                .iter()
                .find(|s| s.op == PlanOp::Dispatch && s.kernel_module_id == Some(stage1.id))
                .expect("the stage-1 dispatch");
            let view = dispatch.writes[0];
            assert_eq!(view.element_count, groups, "{op} n={n}");
            assert!(view.size_bytes >= groups * 4, "{op} n={n}");
            let allocates = plan
                .step
                .iter()
                .any(|s| s.op == PlanOp::Allocate && s.writes.contains(&view));
            assert!(
                allocates,
                "{op} n={n}: an Allocate step makes the partials view"
            );
            let reader = plan
                .step
                .iter()
                .find(|s| s.op == PlanOp::Dispatch && s.kernel_module_id == Some(stage2.id))
                .expect("the stage-2 dispatch");
            assert_eq!(
                reader.reads,
                vec![view],
                "{op} n={n}: stage 2 reads the partials"
            );
            let modules = emit_kernels(&artifact, &profile(false)).expect("emits");
            for module in &modules {
                validate(&module.spirv).expect("valid");
            }
        }
    }
}
