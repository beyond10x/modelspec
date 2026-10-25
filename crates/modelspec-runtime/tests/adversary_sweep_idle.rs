//! Adversary pass 1 of story:sweep-run-starts-busy, on the device.
//!
//! The unit's own series test (`tests/vulkan_mlp_sweep.rs`) runs at batch 100, hidden 256 and
//! compares a series with `train_with`, which goes through the same `execute_runs` body. These
//! cases run the series at the point the bench times (batch 8192, hidden 4096, deterministic and
//! fast-math), compare each run's counters with values written from the source's shapes rather
//! than with another run of the same code, and drive a series whose second run diverges.
//!
//! Ignored by default: they open a Vulkan device. Run them with
//! `cargo test --release -p modelspec-runtime --features vulkan --test adversary_sweep_idle -- --ignored --test-threads=1`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
#[path = "../benches/mlp_sweep/sweep.rs"]
mod sweep;

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{CompiledTraining, compile_artifact_for, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::train::runner::Run;
use modelspec_runtime::vulkan::negotiate::{Negotiated, negotiate};
use modelspec_runtime::vulkan::runner::{
    CHUNK_STEPS, RunOptions, RunStats, VulkanRunError, train_repeatedly, train_with,
};
use modelspec_runtime::vulkan::{Device, Vulkan};

use sweep::Point;

fn compile_source(
    name: &str,
    source: &str,
    profile: &DeviceProfile,
    fast_math: bool,
) -> (CompiledTraining, CompiledArtifact) {
    let compiled = compile_training(name, source).expect("compiles");
    let mut artifact = compile_artifact_for(name, source, Backend::Vulkan, Some(profile))
        .expect("compiles for Vulkan");
    artifact.fast_math = fast_math;
    (compiled, artifact)
}

/// Every counter of a run but its time and the series-wide memory counts.
fn counted(stats: &RunStats) -> (u64, u64, u64, u64, u64, u64, u64, u64, String) {
    (
        stats.steps,
        stats.submits,
        stats.host_waits,
        stats.dispatches,
        stats.dispatches_per_step,
        stats.barriers_per_step,
        stats.bytes_uploaded,
        stats.bytes_downloaded,
        stats.class.to_string(),
    )
}

/// The bytes a run of `point` copies up (both parameters and the batch) and down (both
/// parameters and `steps + 1` losses), from the source's shapes.
fn expected_bytes(point: Point, steps: u64) -> (u64, u64) {
    let (b, h) = (point.batch, point.hidden);
    let params = 784 * h + h * 10;
    let batch = b * 784 + b * 10;
    (4 * (params + batch), 4 * (params + steps + 1))
}

fn series_is_train_with(
    device: &Device<'_>,
    profile: &DeviceProfile,
    negotiated: &Negotiated,
    point: Point,
    fast_math: bool,
    series: &[u64],
) {
    let name = point.file_name();
    let (compiled, artifact) = compile_source(&name, &point.source(series[0]), profile, fast_math);
    let data = synthetic(&compiled.config, &compiled.forward).expect("a batch");
    let mut finished: Vec<(Run, RunStats)> = Vec::new();
    train_repeatedly(
        device,
        profile,
        negotiated,
        &compiled,
        &artifact,
        Some(data.clone()),
        &mut |run, stats| {
            finished.push((run, stats));
            series.get(finished.len()).copied()
        },
    )
    .expect("the series runs");
    assert_eq!(finished.len(), series.len());
    for (&steps, (run, stats)) in series.iter().zip(&finished) {
        let (compiled, artifact) = compile_source(&name, &point.source(steps), profile, fast_math);
        let (alone, alone_stats) = train_with(
            device,
            profile,
            negotiated,
            &compiled,
            &artifact,
            RunOptions {
                data: Some(data.clone()),
                ..RunOptions::default()
            },
        )
        .expect("the run alone finishes");
        assert!(
            run == &alone,
            "{name} fast_math {fast_math}, {steps} steps: the series run differs from train_with"
        );
        assert_eq!(
            counted(stats),
            counted(&alone_stats),
            "{name} fast_math {fast_math}, {steps} steps"
        );
        let (up, down) = expected_bytes(point, steps);
        assert_eq!(
            (
                stats.host_waits,
                stats.submits,
                stats.bytes_uploaded,
                stats.bytes_downloaded
            ),
            (1, steps.div_ceil(CHUNK_STEPS).max(1), up, down),
            "{name} fast_math {fast_math}, {steps} steps: one wait, one submit per chunk, the \
             whole upload and read-back inside the run"
        );
    }
}

fn open() -> (Vulkan, DeviceProfile, Negotiated) {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    (vulkan, profile, negotiated)
}

/// The point the bench times and the story measures: each run of a series (down, up past one
/// command buffer, repeated) is `train_with` of its source, bit for bit, with every counter, and
/// the counters are the ones the source's shapes give.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_series_at_the_widest_point_is_bit_for_bit_train_with() {
    let (vulkan, profile, negotiated) = open();
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let point = Point {
        batch: 8192,
        hidden: 4096,
    };
    series_is_train_with(
        &device,
        &profile,
        &negotiated,
        point,
        false,
        &[20, 3, CHUNK_STEPS + 5, 20],
    );
}

/// The same with `fast_math`, as `mlp_sweep --fast-math` runs the point.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_series_at_the_widest_point_with_fast_math_is_bit_for_bit_train_with() {
    let (vulkan, profile, negotiated) = open();
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let point = Point {
        batch: 8192,
        hidden: 4096,
    };
    series_is_train_with(&device, &profile, &negotiated, point, true, &[20, 3, 20]);
}

/// The unit's own point, with the counters checked against the source's shapes.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_series_counters_are_the_source_s_at_the_unit_s_point() {
    let (vulkan, profile, negotiated) = open();
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let point = Point {
        batch: 100,
        hidden: 256,
    };
    series_is_train_with(
        &device,
        &profile,
        &negotiated,
        point,
        false,
        &[20, 7, CHUNK_STEPS + 44, CHUNK_STEPS + 44, 1],
    );
}

/// A series whose second run diverges ends with the refusal `train_with` of that run's source
/// makes: the same diagnostic, the same run through the failing step, the same counters. The
/// first run, shorter than the divergence, is `train_with` of its source.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_a_series_run_that_diverges_is_the_refusal_train_with_makes() {
    let (vulkan, profile, negotiated) = open();
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let point = Point {
        batch: 100,
        hidden: 256,
    };
    let name = point.file_name();
    let source = |lr: &str, steps: u64| {
        point
            .source(steps)
            .replace(&format!("lr = {}", sweep::LR), &format!("lr = {lr}"))
    };
    let alone = |lr: &str, steps: u64, data: &modelspec_runtime::train::data::SyntheticData| {
        let (compiled, artifact) = compile_source(&name, &source(lr, steps), &profile, false);
        train_with(
            &device,
            &profile,
            &negotiated,
            &compiled,
            &artifact,
            RunOptions {
                data: Some(data.clone()),
                ..RunOptions::default()
            },
        )
    };
    // The first learning rate whose run diverges after its first step and within 200.
    let mut found = None;
    'search: for lr in ["1.0", "3.0", "10.0", "30.0", "100.0", "1000.0"] {
        let (compiled, _) = compile_source(&name, &source(lr, 1), &profile, false);
        let data = synthetic(&compiled.config, &compiled.forward).expect("a batch");
        for steps in 1..=200u64 {
            match alone(lr, steps, &data) {
                Ok(_) => {}
                Err(VulkanRunError::NonFinite(_)) if steps > 1 => {
                    found = Some((lr, steps, data));
                    break 'search;
                }
                Err(VulkanRunError::NonFinite(_)) => continue 'search,
                Err(other) => panic!("lr {lr}, {steps} steps: {other}"),
            }
        }
    }
    let (lr, first_bad, data) = found.expect("a learning rate that diverges after step 1");
    let series = [first_bad - 1, first_bad + 10];
    let (compiled, artifact) = compile_source(&name, &source(lr, series[0]), &profile, false);
    let mut finished: Vec<(Run, RunStats)> = Vec::new();
    let ended = train_repeatedly(
        &device,
        &profile,
        &negotiated,
        &compiled,
        &artifact,
        Some(data.clone()),
        &mut |run, stats| {
            finished.push((run, stats));
            series.get(finished.len()).copied()
        },
    );
    assert_eq!(finished.len(), 1, "lr {lr}: the first run finishes");
    let (first, first_stats) = alone(lr, series[0], &data).expect("the first run alone finishes");
    assert!(finished[0].0 == first, "lr {lr}: the first run differs");
    assert_eq!(counted(&finished[0].1), counted(&first_stats));
    let Err(VulkanRunError::NonFinite(series_refusal)) = ended else {
        panic!("lr {lr}: the series ends with {ended:?}, not E-TRAIN-001");
    };
    let Err(VulkanRunError::NonFinite(alone_refusal)) = alone(lr, series[1], &data) else {
        panic!("lr {lr}: {} steps alone do not diverge", series[1]);
    };
    assert_eq!(
        series_refusal.diverged.diagnostic.to_string(),
        alone_refusal.diverged.diagnostic.to_string(),
        "lr {lr}"
    );
    assert_eq!(
        format!("{:?}", series_refusal.diverged.run),
        format!("{:?}", alone_refusal.diverged.run),
        "lr {lr}: the refused run differs"
    );
    assert_eq!(
        counted(&series_refusal.stats),
        counted(&alone_refusal.stats),
        "lr {lr}"
    );
}
