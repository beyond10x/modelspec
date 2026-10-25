//! The device part of the MLP size sweep (`benches/mlp_sweep.rs`): a point's warm-up,
//! calibration and timed runs on an opened device, shared with the device test
//! `tests/vulkan_mlp_sweep.rs`.

use std::time::{SystemTime, UNIX_EPOCH};

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{CompiledTraining, compile_artifact_for, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::vulkan::Device;
use modelspec_runtime::vulkan::negotiate::Negotiated;
use modelspec_runtime::vulkan::runner::train_repeatedly;

use crate::sweep::{self, Point, TimedRun};

/// The calibration runs a point gets before its step count is taken as it stands.
const CALIBRATIONS: usize = 4;

/// The opened device a point runs on.
pub struct On<'d> {
    pub device: &'d Device<'d>,
    pub profile: &'d DeviceProfile,
    pub negotiated: &'d Negotiated,
    /// The `fast_math` every point's artifact is compiled with.
    pub fast_math: bool,
}

/// The point's source for `steps` steps, compiled for training and for Vulkan with `fast_math`,
/// its plan scheduled within the limits of the device `profile` as `train --backend vulkan`
/// schedules it.
fn compile(
    point: Point,
    steps: u64,
    fast_math: bool,
    profile: &DeviceProfile,
) -> Result<(CompiledTraining, CompiledArtifact), String> {
    let name = point.file_name();
    let source = point.source(steps);
    let refusal = |diagnostics: Vec<modelspec_core::diag::Diagnostic>| {
        diagnostics
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    };
    let compiled = compile_training(&name, &source).map_err(refusal)?;
    let mut artifact =
        compile_artifact_for(&name, &source, Backend::Vulkan, Some(profile)).map_err(refusal)?;
    artifact.fast_math = fast_math;
    Ok((compiled, artifact))
}

/// Where a point's series of runs is.
#[derive(Clone, Copy)]
enum Phase {
    /// The [`sweep::WARMUP_STEPS`]-step warm-up run.
    Warmup,
    /// The calibration run with this number, from 1.
    Calibration(usize),
    /// The timed runs.
    Timed,
}

/// The warm-up, the calibration and the timed runs of `point`: the steps of each timed run and
/// the runs.
///
/// All of them are one series on one setup ([`train_repeatedly`]): the point is compiled once,
/// at [`sweep::WARMUP_STEPS`] steps, its batch made once, and each run after the warm-up is given
/// its step count; a point's plan does not depend on its step count. Between one run's host wait
/// and the next run's first submit the device idles only while the host reads the run back,
/// checks it and records the next run's first command buffer.
pub fn measure(
    on: &On<'_>,
    point: Point,
    min_seconds: f64,
) -> Result<(u64, Vec<TimedRun>), String> {
    let target = min_seconds * 1.2;
    let (compiled, artifact) = compile(point, sweep::WARMUP_STEPS, on.fast_math, on.profile)?;
    let data = synthetic(&compiled.config, &compiled.forward).map_err(|error| error.to_string())?;
    let mut phase = Phase::Warmup;
    let mut steps = sweep::WARMUP_STEPS;
    let mut runs = Vec::with_capacity(sweep::RUNS);
    train_repeatedly(
        on.device,
        on.profile,
        on.negotiated,
        &compiled,
        &artifact,
        Some(data),
        &mut |_, stats| {
            let end = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |since| since.as_millis() as u64);
            let seconds = stats.elapsed.as_secs_f64();
            match phase {
                Phase::Warmup => {
                    steps = sweep::steps_for(seconds, stats.steps, target);
                    phase = Phase::Calibration(1);
                }
                Phase::Calibration(made) => {
                    eprintln!("mlp_sweep:   calibration: {steps} steps in {seconds:.3} s");
                    let long_enough = seconds >= min_seconds;
                    if !long_enough {
                        steps = sweep::steps_for(seconds, steps, target).max(steps + 1);
                    }
                    phase = if long_enough || made == CALIBRATIONS {
                        Phase::Timed
                    } else {
                        Phase::Calibration(made + 1)
                    };
                }
                Phase::Timed => {
                    eprintln!(
                        "mlp_sweep:   timed: {steps} steps in {seconds:.3} s, {:.3} us per step",
                        stats.us_per_step()
                    );
                    if seconds < min_seconds {
                        eprintln!("mlp_sweep:   warning: the run was shorter than {min_seconds} s");
                    }
                    let elapsed_ms = stats.elapsed.as_millis() as u64;
                    runs.push(TimedRun {
                        us_per_step: stats.us_per_step(),
                        seconds,
                        span_unix_ms: [end.saturating_sub(elapsed_ms), end],
                    });
                    if runs.len() == sweep::RUNS {
                        return None;
                    }
                }
            }
            Some(steps)
        },
    )
    .map_err(|error| error.to_string())?;
    Ok((steps, runs))
}
