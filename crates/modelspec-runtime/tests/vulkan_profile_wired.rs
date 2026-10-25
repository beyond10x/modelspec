//! `story:matmul-block-kernel-throughput`, acceptance `profile-wired` on a Vulkan device: the
//! plan `train --backend vulkan` runs is scheduled for the profile of the device it selects
//! ([`device_profile`], [`compile_artifact_for`]), so on a device that allows 256 invocations the
//! `mlp_sweep` b8192 h1024 forward matmul takes the block, and the run gives the losses and
//! parameters of the plan for every device bit for bit; an artifact scheduled for limits past
//! the device's is refused before the device is opened (E-ARTIFACT-005), naming both values.
//!
//! Ignored by default: they create a Vulkan instance. Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_profile_wired -- --ignored`.
#![cfg(feature = "vulkan")]

use modelspec_core::artifact::{Backend, CompiledArtifact, E_ARTIFACT_005};
use modelspec_core::compile::{CompiledTraining, compile_artifact_for, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_runtime::train::runner::Run;
use modelspec_runtime::vulkan::runner::{DeviceRunError, device_profile, train_on_device};

/// The `mlp_sweep` source at batch 8192 and hidden width 1024, `steps` steps.
fn sweep(steps: u64) -> String {
    format!(
        "model SweepMlp {{
    param w1: Tensor<f32, [784, 1024]> = xavier_uniform();
    param w2: Tensor<f32, [1024, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8192, 784]>)
        -> Tensor<f32, [8192, 10]> {{
        return matmul(relu(matmul(x, w1)), w2);
    }}
}}

train Fit {{
    model = SweepMlp;
    input x: Tensor<f32, [8192, 784]>;
    target y: Tensor<f32, [8192, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = {steps};
    seed = 42;
    device = vulkan(0);
}}
"
    )
}

fn compiled(profile: Option<&DeviceProfile>) -> (CompiledTraining, CompiledArtifact) {
    let source = sweep(3);
    let training = compile_training("mlp_sweep.mspec", &source).expect("compiles");
    let artifact = compile_artifact_for("mlp_sweep.mspec", &source, Backend::Vulkan, profile)
        .unwrap_or_else(|d| panic!("{d:?}"));
    (training, artifact)
}

fn bits(run: &Run) -> (Vec<u32>, Vec<Vec<u32>>) {
    (
        run.losses.iter().map(|loss| loss.to_bits()).collect(),
        run.state
            .values
            .values()
            .map(|tensor| tensor.data().iter().map(|v| v.to_bits()).collect())
            .collect(),
    )
}

#[test]
#[ignore = "needs a Vulkan device"]
fn the_plan_for_the_device_takes_the_block_and_trains_to_the_bits_of_the_plan_for_every_device() {
    let profile = device_profile(None).expect("device 0");
    let limits = modelspec_core::ir::kernel::MatMulLimits::of(&profile);
    let (training, for_device) = compiled(Some(&profile));
    assert_eq!(for_device.matmul_limits, Some(limits));
    let blocks = for_device
        .execution
        .kernels
        .iter()
        .filter(|kernel| kernel.workgroup_size == 256)
        .count();
    if limits.invocations >= 256 && limits.shared_bytes >= 16_896 {
        assert!(blocks > 0, "a device of {limits:?} takes the block");
    }
    let (_, for_every) = compiled(None);
    assert_eq!(for_every.matmul_limits, None);
    let (scheduled, _) = train_on_device(None, &training, &for_device).expect("runs");
    let (plain, _) = train_on_device(None, &training, &for_every).expect("runs");
    assert_eq!(scheduled.losses.len(), 4);
    assert_eq!(bits(&scheduled), bits(&plain), "{blocks} block kernels");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn an_artifact_scheduled_past_the_device_limits_is_refused_before_the_device_is_opened() {
    let mut beyond = device_profile(None).expect("device 0");
    beyond.max_workgroup_invocations = beyond.max_workgroup_invocations.max(256) * 2;
    beyond.max_workgroup_size.x = beyond.max_workgroup_invocations;
    beyond.max_shared_memory_bytes += 4;
    let (training, artifact) = compiled(Some(&beyond));
    let refused = match train_on_device(None, &training, &artifact) {
        Err(DeviceRunError::Refused(diagnostic)) => diagnostic,
        Err(other) => panic!("refused as the device opens, not {other}"),
        Ok(_) => panic!("a device below the limits is refused"),
    };
    assert_eq!(refused.code, E_ARTIFACT_005, "{refused}");
    let shown = refused.to_string();
    for value in [
        beyond.max_workgroup_invocations,
        beyond.max_shared_memory_bytes,
        beyond.max_shared_memory_bytes - 4,
    ] {
        assert!(shown.contains(&value.to_string()), "{shown} names {value}");
    }
}
