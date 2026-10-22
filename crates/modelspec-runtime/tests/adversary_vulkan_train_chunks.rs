//! Adversary cases for `story:m1-vulkan-train`: the chunk boundaries of the one-wait device run
//! and whether the device ran exactly `steps` updates.
//!
//! `tests/vulkan_train.rs` runs 500 and 1,800 steps only: neither 0, 1, nor a run that ends on or
//! just past a 256-step chunk boundary. Here, for `examples/linear_regression.mspec` at 0, 1, 255,
//! 256, 257 and 512 steps:
//!
//! - every loss and every final parameter passes the bounds of `tests/vulkan_train.rs` against
//!   the CPU run of the same step count;
//! - there are `steps + 1` losses, one host wait, `max(1, ceil(steps / 256))` submits, the
//!   parameters and the batch up once, the parameters and the `steps + 1` losses down once;
//! - the parameter versions equal the CPU's.
//!
//! And for `examples/mnist_mlp.mspec` at 257 steps (one step in the last chunk), the device
//! parameters are nearer to the CPU after 257 updates than after 256: a device run that skips
//! the update of its last step, or of its last chunk, fails here.
//!
//! Ignored by default (they create a Vulkan instance). Run with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test adversary_vulkan_train_chunks -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::{KernelOp, run_within, term_tolerance};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{CompiledTraining, compile_artifact, compile_training};
use modelspec_runtime::train::runner::{self, Run};
use modelspec_runtime::vulkan::Vulkan;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::runner::{CHUNK_STEPS, RunStats, train};

const LINEAR_REGRESSION: &str = "examples/linear_regression.mspec";
const MNIST_MLP: &str = "examples/mnist_mlp.mspec";

fn compiled(file: &str) -> (CompiledTraining, CompiledArtifact) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(file);
    let source = std::fs::read_to_string(&path).expect("the example reads");
    let compiled = compile_training(file, &source).expect("the example compiles");
    let artifact =
        compile_artifact(file, &source, Backend::Vulkan).expect("the example compiles for Vulkan");
    (compiled, artifact)
}

/// Runs `compiled` with each of `step_counts` on device 0, all on one device, and returns the
/// device runs with their stats; every validation message fails.
fn device_runs(
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    step_counts: &[u64],
) -> Vec<(u64, Run, RunStats)> {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut out = Vec::new();
    for &steps in step_counts {
        let mut compiled = compiled.clone();
        compiled.config.steps = steps;
        let (run, stats) = train(&device, &profile, &negotiated, &compiled, artifact)
            .unwrap_or_else(|error| panic!("{steps} steps: the device run fails: {error}"));
        out.push((steps, run, stats));
    }
    drop(device);
    let messages = vulkan.take_validation_messages();
    assert!(messages.is_empty(), "validation messages: {messages:#?}");
    out
}

fn cpu_run(compiled: &CompiledTraining, steps: u64) -> Run {
    let mut compiled = compiled.clone();
    compiled.config.steps = steps;
    runner::run(&compiled).expect("the CPU run finishes")
}

fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_runs_on_and_around_the_chunk_boundary_match_the_cpu() {
    let (compiled, artifact) = compiled(LINEAR_REGRESSION);
    let plan = &artifact.execution;
    let param_bytes: u64 = plan.parameters.iter().map(|p| p.views[0].size_bytes).sum();
    let data_bytes: u64 = plan.data.iter().map(|d| d.view.size_bytes).sum();
    let counts = [
        0,
        1,
        CHUNK_STEPS - 1,
        CHUNK_STEPS,
        CHUNK_STEPS + 1,
        2 * CHUNK_STEPS,
    ];
    let tolerance = term_tolerance(KernelOp::Mse);
    let mut failures = Vec::new();
    for (steps, gpu, stats) in device_runs(&compiled, &artifact, &counts) {
        let cpu = cpu_run(&compiled, steps);
        eprintln!("{steps} steps:\n{stats}");
        let mut check = |ok: bool, what: String| {
            if !ok {
                failures.push(format!("{steps} steps: {what}"));
            }
        };
        check(
            gpu.losses.len() as u64 == steps + 1,
            format!("{} losses, not {}", gpu.losses.len(), steps + 1),
        );
        for (k, (&g, &c)) in gpu.losses.iter().zip(&cpu.losses).enumerate() {
            let r = tolerance.r(c, f64::from(c), 1);
            check(
                tolerance.accepts(g, c, r),
                format!("loss {k}: gpu {g}, cpu {c}"),
            );
        }
        for (param, c) in &cpu.state.values {
            let g = &gpu.state.values[param];
            check(
                run_within(g.data(), c.data()),
                format!("{param}: gpu {:?}, cpu {:?}", g.data(), c.data()),
            );
        }
        check(
            gpu.state.versions == cpu.state.versions,
            format!(
                "versions gpu {:?}, cpu {:?}",
                gpu.state.versions, cpu.state.versions
            ),
        );
        check(
            stats.host_waits == 1,
            format!("{} host waits", stats.host_waits),
        );
        let submits = steps.div_ceil(CHUNK_STEPS).max(1);
        check(
            stats.submits == submits,
            format!("{} submits, not {submits}", stats.submits),
        );
        check(
            stats.bytes_uploaded == param_bytes + data_bytes,
            format!("{} bytes up", stats.bytes_uploaded),
        );
        check(
            stats.bytes_downloaded == param_bytes + 4 * (steps + 1),
            format!("{} bytes down", stats.bytes_downloaded),
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_the_last_chunk_runs_its_update() {
    let (compiled, artifact) = compiled(MNIST_MLP);
    let steps = CHUNK_STEPS + 1;
    let runs = device_runs(&compiled, &artifact, &[steps]);
    let (_, gpu, stats) = &runs[0];
    eprintln!("{steps} steps:\n{stats}");
    let after = cpu_run(&compiled, steps);
    let before = cpu_run(&compiled, steps - 1);
    for (param, c) in &after.state.values {
        let g = gpu.state.values[param].data();
        let to_after = l2_distance(g, c.data());
        let to_before = l2_distance(g, before.state.values[param].data());
        assert!(
            run_within(g, c.data()),
            "{param}: outside the whole-run bound of the CPU after {steps} updates"
        );
        assert!(
            to_after < to_before,
            "{param}: the device parameters are {to_after:e} from the CPU after {steps} updates \
             and {to_before:e} from the CPU after {} updates",
            steps - 1
        );
    }
}
