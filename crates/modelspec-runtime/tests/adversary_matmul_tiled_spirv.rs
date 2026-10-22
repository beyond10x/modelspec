//! Adversary pass 1 on `story:m1-f-forward-matmul-tiled`, no Vulkan instance (runs in `task ci`).
//!
//! The tiled matmul unrolls every tile of `k` into straight-line SPIR-V, so its module grows
//! linearly with `k`. Acceptance 4 says `spirv-val` passes on every emitted matmul module. A
//! full-batch linear regression over `N` samples has a weight-gradient matmul
//! `[4, N] x [N, 1]`, `k = N`; the schedule accepts it, so `emit_kernels` must either refuse it
//! or emit modules `spirv-val` accepts. The SPIR-V universal limit on the `<id>` bound is
//! 4,194,303 (SPIR-V 1.6 § 2.17).
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use modelspec_core::artifact::Backend;
use modelspec_core::compile::compile_artifact;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::KernelKind;
use modelspec_runtime::spirv::emit::emit_kernels;
use modelspec_runtime::spirv::validate::validate;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

/// The SPIR-V universal limit on the `<id>` bound.
const MAX_ID_BOUND: u32 = 4_194_303;

fn full_batch_linear_regression(samples: u64) -> String {
    format!(
        "model LinearReg {{\n    param w: Tensor<f32, [4, 1]> = xavier_uniform();\n\n    \
         fn forward(x: Tensor<f32, [{samples}, 4]>)\n        -> Tensor<f32, [{samples}, 1]> {{\n        \
         return matmul(x, w);\n    }}\n}}\n\ntrain Fit {{\n    model = LinearReg;\n    \
         input x: Tensor<f32, [{samples}, 4]>;\n    target y: Tensor<f32, [{samples}, 1]>;\n    \
         loss = mse(model.forward(x), y);\n    optimizer = sgd(lr = 0.1);\n    \
         steps = 1;\n    seed = 42;\n    device = vulkan(0);\n}}\n"
    )
}

#[test]
fn every_matmul_module_of_a_full_batch_weight_gradient_passes_spirv_val() {
    let samples = 600_000;
    let source = full_batch_linear_regression(samples);
    let artifact = compile_artifact("adv.mspec", &source, Backend::Vulkan)
        .expect("the schedule accepts a full-batch linear regression of 600,000 samples");
    let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile");
    let modules = emit_kernels(&artifact, &profile)
        .expect("emit_kernels emits the artifact or refuses it with a target error");
    let mut matmuls = 0;
    let mut failures = Vec::new();
    for module in &modules {
        let kernel = artifact
            .execution
            .kernels
            .iter()
            .find(|kernel| kernel.id == module.kernel_module_id)
            .expect("each module has its kernel");
        if kernel.kind != KernelKind::MatMul {
            continue;
        }
        matmuls += 1;
        let bound = module.spirv[3];
        eprintln!(
            "{}: {} words, id bound {bound}",
            kernel.signature(),
            module.spirv.len()
        );
        if bound > MAX_ID_BOUND {
            failures.push(format!(
                "{}: id bound {bound} > {MAX_ID_BOUND}",
                kernel.signature()
            ));
        }
        if let Err(error) = validate(&module.spirv) {
            let text = error.to_string();
            failures.push(format!(
                "{}: spirv-val: {}",
                kernel.signature(),
                &text[..text.len().min(400)]
            ));
        }
    }
    assert!(matmuls >= 2, "the forward and the weight-gradient matmul");
    assert!(failures.is_empty(), "{failures:#?}");
}
