//! `story:matmul-tile-from-device-invocation-limit`, without a device: the modules of the plans
//! scheduled for the RTX 3090 profile (`tests/fixtures/rtx3090.device.json`, 1024 invocations).
//!
//! - Every module of those plans passes `spirv-val --target-env vulkan1.2`; a 128 by 128 block
//!   declares `LocalSize` 256 and decorates every `OpFAdd` and `OpFMul` `NoContraction`.
//! - `default-unchanged`: in this build, the artifact of a plan scheduled for a profile of 128
//!   invocations is the artifact `compile_artifact` builds without a profile, and `emit_kernels`
//!   gives both the same modules and cache keys. Both sides are compiled by the same build; the
//!   case says nothing about the artifact a build before this story wrote.
//! - `workgroup-memory-bounded`: a block whose staged bytes pass the profile's workgroup memory is
//!   refused by the emitter with `EmitError::SharedMemory`; the plan never holds one, because the
//!   schedule gives such a profile the tile of 128 invocations.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::path::{Path, PathBuf};

use common::words::{execution_modes, uncontracted_arithmetic};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Kernel, KernelKind, MatMulLimits};
use modelspec_core::schedule::schedule_for;
use modelspec_runtime::spirv::emit::{EmitError, emit_kernels, emit_module};
use modelspec_runtime::spirv::validate::validate;

/// The execution mode `LocalSize`.
const MODE_LOCAL_SIZE: u32 = 17;

const RTX3090: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

fn rtx3090() -> DeviceProfile {
    DeviceProfile::from_json(RTX3090).expect("the fixture is a valid profile")
}

fn limited(invocations: u32, shared_bytes: u32) -> DeviceProfile {
    let mut profile = rtx3090();
    profile.max_workgroup_invocations = invocations;
    profile.max_shared_memory_bytes = shared_bytes;
    profile
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sweep(batch: u64, hidden: u64) -> String {
    format!(
        "model SweepMlp {{
    param w1: Tensor<f32, [784, {hidden}]> = xavier_uniform();
    param w2: Tensor<f32, [{hidden}, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [{batch}, 784]>)
        -> Tensor<f32, [{batch}, 10]> {{
        return matmul(relu(matmul(x, w1)), w2);
    }}
}}

train Fit {{
    model = SweepMlp;
    input x: Tensor<f32, [{batch}, 784]>;
    target y: Tensor<f32, [{batch}, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 220;
    seed = 42;
    device = vulkan(0);
}}
"
    )
}

fn sources() -> Vec<(String, String)> {
    let mut sources = Vec::new();
    for example in ["mnist_mlp", "linear_regression"] {
        let path = repo_root().join(format!("examples/{example}.mspec"));
        sources.push((
            example.to_owned(),
            std::fs::read_to_string(path).expect("the example reads"),
        ));
    }
    for (batch, hidden) in [(1000, 1024), (8192, 1024), (8192, 4096)] {
        sources.push((format!("sweep_b{batch}_h{hidden}"), sweep(batch, hidden)));
    }
    sources
}

/// The Vulkan artifact of `source` with the plan scheduled for `profile`.
fn artifact_for(name: &str, source: &str, profile: &DeviceProfile) -> CompiledArtifact {
    let compiled = compile_training(name, source).unwrap_or_else(|d| panic!("{name}: {d:?}"));
    let plan = schedule_for(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
        Some(MatMulLimits::of(profile)),
    )
    .expect("the profile's schedule");
    CompiledArtifact::new(Backend::Vulkan, source, &compiled.training, plan)
}

fn local_size(words: &[u32]) -> Vec<u32> {
    execution_modes(words)
        .into_iter()
        .find(|(mode, _)| *mode == MODE_LOCAL_SIZE)
        .map(|(_, sizes)| sizes)
        .expect("a LocalSize")
}

#[test]
fn every_module_of_the_rtx3090_plans_passes_spirv_val_and_a_block_runs_256_invocations() {
    let profile = rtx3090();
    let mut blocks = 0;
    for (name, source) in sources() {
        let artifact = artifact_for(&name, &source, &profile);
        let modules = emit_kernels(&artifact, &profile).expect("every kernel emits");
        for (kernel, module) in artifact.execution.kernels.iter().zip(&modules) {
            if let Err(error) = validate(&module.spirv) {
                panic!("{name}: {}: {error}", kernel.signature());
            }
            assert_eq!(
                local_size(&module.spirv),
                vec![kernel.workgroup_size, 1, 1],
                "{name}: {}",
                kernel.signature()
            );
            if kernel.kind == KernelKind::MatMul && kernel.workgroup_size == 256 {
                blocks += 1;
                let (missing, all) = uncontracted_arithmetic(&module.spirv);
                assert!(
                    all > 0 && missing.is_empty(),
                    "{name}: {}",
                    kernel.signature()
                );
            }
        }
    }
    // The forward matmul at b8192 h1024 (512 blocks), and at b8192 h4096 the forward matmul
    // (2,048 blocks) and the weight gradient of w1, `[784, 8192] x [8192, 4096]` (224 blocks).
    assert_eq!(blocks, 3);
}

/// Compile without a profile (`compile_artifact`) against compile with a profile of 128
/// invocations, at 48 and 16 KiB of workgroup memory, both in this build: the same artifact, and
/// for the RTX 3090 the same modules, cache keys and module ids.
#[test]
fn a_profile_of_128_invocations_gives_the_artifact_and_the_modules_of_compile_artifact() {
    for (name, source) in sources() {
        let before = compile_artifact(&name, &source, Backend::Vulkan).expect("compiles");
        for profile in [limited(128, 49_152), limited(128, 16_384)] {
            let after = artifact_for(&name, &source, &profile);
            assert_eq!(after, before, "{name}");
            let rtx = rtx3090();
            let (now, then) = (
                emit_kernels(&after, &rtx).expect("emits"),
                emit_kernels(&before, &rtx).expect("emits"),
            );
            assert_eq!(now.len(), then.len());
            for (now, then) in now.iter().zip(&then) {
                assert_eq!(now.spirv, then.spirv, "{name}");
                assert_eq!(now.cache_key, then.cache_key, "{name}");
                assert_eq!(now.kernel_module_id, then.kernel_module_id, "{name}");
            }
        }
    }
}

#[test]
fn a_block_past_the_profile_workgroup_memory_is_refused_by_the_emitter() {
    let profile = rtx3090();
    let artifact = artifact_for("sweep_b8192_h4096", &sweep(8192, 4096), &profile);
    let blocks: Vec<&Kernel> = artifact
        .execution
        .kernels
        .iter()
        .filter(|k| k.kind == KernelKind::MatMul && k.workgroup_size == 256)
        .collect();
    assert_eq!(blocks.len(), 2);
    for kernel in blocks {
        let tile = kernel.matmul_tile().expect("a tile");
        let padded = tile.layout(kernel.transposed).shared_bytes();
        let needs = tile.layout([false, false]).shared_bytes();
        emit_module(kernel, &limited(1024, padded)).expect("the padded layout fits exactly");
        emit_module(kernel, &limited(1024, needs)).expect("the unpadded layout fits exactly");
        if padded > needs {
            assert_ne!(
                emit_module(kernel, &limited(1024, padded))
                    .expect("padded")
                    .spirv,
                emit_module(kernel, &limited(1024, padded - 1))
                    .expect("unpadded")
                    .spirv,
                "{}: one byte short of the padded layout stages unpadded",
                kernel.signature()
            );
        }
        match emit_module(kernel, &limited(1024, needs - 1)) {
            Err(EmitError::SharedMemory {
                needs: found,
                provides,
                ..
            }) => {
                assert_eq!((found, provides), (needs, needs - 1));
            }
            other => panic!("{}: not refused: {other:?}", kernel.signature()),
        }
    }
}
