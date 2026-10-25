//! story:sweep-run-starts-busy: the timed runs of the MLP size sweep (`benches/mlp_sweep.rs`)
//! follow one another without leaving the device idle long enough for the driver to lower its
//! clock.
//!
//! The device idles from the end of one timed run's host wait to the next run's first submit.
//! A run's recorded span is `[return − elapsed, return]`, so the gap between two consecutive
//! spans is that idle, give or take the difference of the two runs' host work after their wait.
//! On the RTX 3090 an idle of 40 ms before each run left the SM clock at or above 1,845 MHz
//! through the run, 66 ms let it fall to 1,590 MHz and 141 ms to 915 MHz, where it climbs back
//! over two to three seconds (`benchmarks/raw/2026-10-25-sweep-idle/diagnosis.md`). The widest
//! point, batch 8192 and hidden 4096, has the most host work per run.
//!
//! Ignored by default: it opens a Vulkan device, which `task ci` never does. Run it with
//! `cargo test --release -p modelspec-runtime --features vulkan --test vulkan_mlp_sweep -- --ignored --test-threads=1`,
//! one test at a time so that the other does not share the device while the gaps are timed.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
#[path = "../benches/mlp_sweep/sweep.rs"]
mod sweep;

#[allow(dead_code)]
#[path = "../benches/mlp_sweep/runs.rs"]
mod runs;

use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact_for, compile_training};
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::vulkan::Vulkan;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::runner::{
    CHUNK_STEPS, RunOptions, RunStats, train_repeatedly, train_with,
};

use runs::{On, measure};
use sweep::Point;

/// The longest idle of the device between two timed runs, in milliseconds.
const IDLE_MS: u64 = 40;

#[test]
#[ignore = "needs a Vulkan device"]
fn the_timed_runs_of_the_widest_point_leave_the_device_idle_under_40_ms_between_them() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let on = On {
        device: &device,
        profile: &profile,
        negotiated: &negotiated,
        fast_math: false,
    };
    let point = Point {
        batch: 8192,
        hidden: 4096,
    };
    let (steps, runs) = measure(&on, point, 0.5).expect("the point runs");
    let gaps: Vec<u64> = runs
        .windows(2)
        .map(|pair| pair[1].span_unix_ms[0].saturating_sub(pair[0].span_unix_ms[1]))
        .collect();
    println!("{steps} steps per timed run; ms between consecutive timed spans: {gaps:?}");
    assert_eq!(gaps.len(), sweep::RUNS - 1);
    assert!(
        gaps.iter().all(|&gap| gap < IDLE_MS),
        "the device idled {gaps:?} ms between timed runs, at least {IDLE_MS} ms"
    );
}

/// Each run of a series on one setup ([`train_repeatedly`]) is the run [`train_with`] makes of the
/// point's source at that step count on the same batch, bit for bit: the initial values, every
/// loss, the final parameters and their versions and the teacher, and the same submits, host
/// waits, dispatches and bytes copied each way. The step counts go down, up past one command
/// buffer of [`CHUNK_STEPS`] steps and repeat.
#[test]
#[ignore = "needs a Vulkan device"]
fn every_run_of_a_series_is_bit_for_bit_the_run_of_the_source_at_its_step_count() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let point = Point {
        batch: 100,
        hidden: 256,
    };
    let name = point.file_name();
    let compile = |steps: u64| {
        let source = point.source(steps);
        let compiled = compile_training(&name, &source).expect("compiles");
        let artifact = compile_artifact_for(&name, &source, Backend::Vulkan, Some(&profile))
            .expect("compiles for Vulkan");
        (compiled, artifact)
    };
    let series = [20, 7, CHUNK_STEPS + 44, CHUNK_STEPS + 44, 1];
    let (compiled, artifact) = compile(series[0]);
    let data = synthetic(&compiled.config, &compiled.forward).expect("a batch");
    let mut finished = Vec::new();
    train_repeatedly(
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
    )
    .expect("the series runs");
    assert_eq!(finished.len(), series.len());
    for (&steps, (run, stats)) in series.iter().zip(&finished) {
        let (compiled, artifact) = compile(steps);
        let options = RunOptions {
            data: Some(data.clone()),
            ..RunOptions::default()
        };
        let (alone, alone_stats) = train_with(
            &device,
            &profile,
            &negotiated,
            &compiled,
            &artifact,
            options,
        )
        .expect("the run alone finishes");
        assert_eq!(run.losses.len() as u64, steps + 1, "{steps} steps");
        assert!(run == &alone, "{steps} steps: the series run differs");
        let counted = |stats: &RunStats| {
            (
                stats.steps,
                stats.submits,
                stats.host_waits,
                stats.dispatches,
                stats.bytes_uploaded,
                stats.bytes_downloaded,
            )
        };
        assert_eq!(counted(stats), counted(&alone_stats), "{steps} steps");
    }
}
