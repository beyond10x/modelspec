//! Adversary pass 1 of story:sweep-run-starts-busy, without a device.
//!
//! The bench compiles a point once, at `sweep::WARMUP_STEPS` steps, and runs its calibration and
//! timed runs on that plan with step counts the calibration picks (`benches/mlp_sweep/runs.rs`).
//! The unit's own test compares 20 steps with 5000 only. This one compares the step counts at
//! the command-buffer boundary and the extremes the calibration can return.

#[allow(dead_code)]
#[path = "../benches/mlp_sweep/sweep.rs"]
mod sweep;

use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact_for, compile_training};
use modelspec_core::device::DeviceProfile;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

#[test]
fn adv_a_point_s_plan_is_the_same_at_boundary_step_counts() {
    let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile");
    for point in sweep::points() {
        let name = point.file_name();
        let warm_source = point.source(sweep::WARMUP_STEPS);
        let warm = compile_training(&name, &warm_source).expect("compiles");
        let warm_artifact =
            compile_artifact_for(&name, &warm_source, Backend::Vulkan, Some(&profile))
                .expect("compiles for Vulkan");
        for steps in [1, 255, 256, 257, 1021, 10_000_000] {
            let source = point.source(steps);
            let other = compile_training(&name, &source).expect("compiles");
            assert_eq!(warm.training, other.training, "{name} at {steps} steps");
            assert_eq!(warm.forward, other.forward, "{name} at {steps} steps");
            let mut config = other.config.clone();
            config.steps = warm.config.steps;
            assert_eq!(config, warm.config, "{name} at {steps} steps");
            let artifact = compile_artifact_for(&name, &source, Backend::Vulkan, Some(&profile))
                .expect("compiles for Vulkan");
            assert_eq!(
                warm_artifact.execution, artifact.execution,
                "{name} at {steps} steps"
            );
            assert_eq!(
                warm_artifact.matmul_limits, artifact.matmul_limits,
                "{name} at {steps} steps"
            );
        }
    }
}
