//! DESIGN.md § 6 and § 9 gate 5 for whole training runs on a Vulkan device
//! (`story:m1-vulkan-train`): both examples train on the device from the CPU run's seed, and
//!
//! - every parameter after the last step passes the whole-run bound [`common::run_within`]
//!   against the CPU run of the same file and seed;
//! - each of the `steps + 1` losses passes the `mse` tolerance of `tests/common/mod.rs`
//!   ([`common::term_tolerance`] of [`KernelOp::Mse`]) against the CPU loss;
//! - the run reports the `tolerance` numerics class;
//! - the host waits once, submits at most one command buffer per [`CHUNK_STEPS`] steps, and
//!   copies the parameters and the batch up once and the parameters and the losses down once.
//!
//! Each test prints its [`RunStats`]: the steady-state time per step and the dispatches per step
//! are reported, not checked. The tests are ignored by default: they create a Vulkan instance,
//! which `task ci` never does. Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_train -- --ignored`;
//! with `MODELSPEC_VK_VALIDATION=1` every message of the Khronos validation layer fails them.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::{GPU_ATOL, GPU_RTOL, KernelOp, RUN_L2_RTOL, run_within, term_tolerance};
use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_runtime::train::runner::{self, Run};
use modelspec_runtime::vulkan::negotiate::{NumericsClass, negotiate};
use modelspec_runtime::vulkan::runner::{CHUNK_STEPS, RunStats, train};
use modelspec_runtime::vulkan::{VALIDATION_ENV, Vulkan};

const LINEAR_REGRESSION: &str = "examples/linear_regression.mspec";
const MNIST_MLP: &str = "examples/mnist_mlp.mspec";

fn validation_requested() -> bool {
    std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
}

/// The CPU run and the device run of the example at `file`, the device run's stats, and the
/// bytes of its parameters and of its data declarations.
struct Runs {
    cpu: Run,
    gpu: Run,
    stats: RunStats,
    param_bytes: u64,
    data_bytes: u64,
}

fn runs(file: &str) -> Runs {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(file);
    let source = std::fs::read_to_string(&path).expect("the example reads");
    runs_of(file, &source)
}

/// [`runs`] of the program `source`, named `file` in its diagnostics.
fn runs_of(file: &str, source: &str) -> Runs {
    let compiled = compile_training(file, source).expect("the example compiles");
    let artifact =
        compile_artifact(file, source, Backend::Vulkan).expect("the example compiles for Vulkan");
    let cpu = runner::run(&compiled).expect("the CPU run finishes");

    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        validation_requested(),
        "the validation layer is on exactly when {VALIDATION_ENV}=1"
    );
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let setup = vulkan.take_validation_messages();
    assert!(setup.is_empty(), "validation messages at setup: {setup:#?}");
    let (gpu, stats) = train(&device, &profile, &negotiated, &compiled, &artifact)
        .expect("the device run finishes");
    drop(device);
    let messages = vulkan.take_validation_messages();
    assert!(
        messages.is_empty(),
        "{file}: {} validation messages: {messages:#?}",
        messages.len()
    );
    eprintln!("{file}: validation {}", vulkan.validation_enabled());
    eprintln!("{file}:\n{stats}");

    let plan = &artifact.execution;
    let param_bytes = plan.parameters.iter().map(|p| p.views[0].size_bytes).sum();
    let data_bytes = plan.data.iter().map(|d| d.view.size_bytes).sum();
    Runs {
        cpu,
        gpu,
        stats,
        param_bytes,
        data_bytes,
    }
}

/// Acceptance 1: the device run starts where the CPU run starts, every final parameter passes
/// the whole-run bound, every loss passes the `mse` tolerance, and the class is `tolerance`.
fn assert_matches_the_cpu(file: &str, runs: &Runs) {
    let (cpu, gpu) = (&runs.cpu, &runs.gpu);
    assert_eq!(gpu.start_step, 0, "{file}");
    assert_eq!(
        gpu.initial, cpu.initial,
        "{file}: the same initial parameters"
    );
    assert_eq!(gpu.teacher, cpu.teacher, "{file}: the same teacher");
    assert_eq!(
        gpu.state.versions, cpu.state.versions,
        "{file}: the same parameter versions"
    );
    assert_eq!(
        gpu.state.values.keys().collect::<Vec<_>>(),
        cpu.state.values.keys().collect::<Vec<_>>(),
        "{file}: the same parameters"
    );
    for (param, c) in &cpu.state.values {
        let g = &gpu.state.values[param];
        assert_eq!(g.shape(), c.shape(), "{file}: {param} shape");
        assert!(
            run_within(g.data(), c.data()),
            "{file}: {param} is outside ||g - c||_2 <= {RUN_L2_RTOL:e} * ||c||_2"
        );
    }

    assert_eq!(
        gpu.losses.len(),
        cpu.losses.len(),
        "{file}: steps + 1 losses"
    );
    // Every term of an `mse` is a square, so the sum of their absolute values over their count
    // is the CPU loss itself: the reference `r` of the `mse` tolerance.
    let tolerance = term_tolerance(KernelOp::Mse);
    let mut outside = Vec::new();
    for (k, (&g, &c)) in gpu.losses.iter().zip(&cpu.losses).enumerate() {
        let r = tolerance.r(c, f64::from(c), 1);
        if !tolerance.accepts(g, c, r) {
            outside.push((k, g, c));
        }
    }
    assert!(
        outside.is_empty(),
        "{file}: {} of {} losses outside {tolerance} (GPU_ATOL {GPU_ATOL:e}, GPU_RTOL \
         {GPU_RTOL:e}); first (step, gpu, cpu): {:?}",
        outside.len(),
        gpu.losses.len(),
        &outside[..outside.len().min(5)]
    );
    assert_eq!(
        runs.stats.class,
        NumericsClass::Tolerance,
        "{file}: a Vulkan training run is in the tolerance class in M1"
    );
}

/// Acceptance 2 for any file: one host wait, one submit per chunk of [`CHUNK_STEPS`] steps,
/// the parameters and the batch up once, the parameters and the `steps + 1` losses down once.
fn assert_one_wait_and_the_counted_bytes(file: &str, runs: &Runs, steps: u64) {
    let stats = &runs.stats;
    assert_eq!(stats.steps, steps, "{file}");
    assert_eq!(
        stats.host_waits, 1,
        "{file}: one host wait for the whole run"
    );
    assert_eq!(
        stats.submits,
        steps.div_ceil(CHUNK_STEPS).max(1),
        "{file}: one submit per chunk of {CHUNK_STEPS} steps for {steps} steps"
    );
    assert_eq!(
        stats.bytes_uploaded,
        runs.param_bytes + runs.data_bytes,
        "{file}: bytes up"
    );
    assert_eq!(
        stats.bytes_downloaded,
        runs.param_bytes + 4 * (steps + 1),
        "{file}: bytes down"
    );
}

#[test]
#[ignore = "needs a Vulkan device"]
fn linear_regression_trains_on_the_device_within_tolerance_of_the_cpu() {
    let runs = runs(LINEAR_REGRESSION);
    assert_matches_the_cpu(LINEAR_REGRESSION, &runs);
    assert_one_wait_and_the_counted_bytes(LINEAR_REGRESSION, &runs, 500);
}

#[test]
#[ignore = "needs a Vulkan device"]
fn mnist_mlp_trains_on_the_device_with_one_host_wait() {
    let runs = runs(MNIST_MLP);
    assert_matches_the_cpu(MNIST_MLP, &runs);
    assert_one_wait_and_the_counted_bytes(MNIST_MLP, &runs, 1800);
    let stats = &runs.stats;
    assert!(stats.submits <= 8, "{} submits", stats.submits);
    assert_eq!(
        stats.bytes_uploaded,
        203_264 + 313_600 + 4_000,
        "params, x and y up"
    );
    assert_eq!(stats.bytes_uploaded, 520_864);
    assert_eq!(
        stats.bytes_downloaded,
        203_264 + 7_204,
        "params and 1,801 losses down"
    );
    assert_eq!(stats.bytes_downloaded, 210_468);
    assert_eq!(
        stats.dispatches_per_step, 10,
        "the fused step plan of story m1-fuse-elementwise, its transposes read in place and \
         its loss gradient one elementwise chain (story:fuse-elementwise-chains-and-gradient-sums)"
    );
    eprintln!(
        "{MNIST_MLP}: steady state {:.3} us/step, {} dispatches/step",
        stats.us_per_step(),
        stats.dispatches_per_step
    );
}

/// `relu(matmul(x, transpose(x)))` with `x` an input: no gradient reads the pre-activation, so
/// the matmul kernel's ReLU epilogue writes only the activation (story m1-fuse-elementwise).
const GRAM: &str = "model Gram {
    param w: Tensor<f32, [16, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [16, 4]>)
        -> Tensor<f32, [16, 1]> {
        let h = relu(matmul(x, transpose(x)));
        return matmul(h, w);
    }
}

train FitGram {
    model = Gram;
    input x: Tensor<f32, [16, 4]>;
    target y: Tensor<f32, [16, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 200;
    seed = 7;
    device = cpu;
}
";

#[test]
#[ignore = "needs a Vulkan device"]
fn a_relu_epilogue_without_its_pre_activation_trains_on_the_device_within_tolerance() {
    let file = "gram.mspec";
    let artifact = compile_artifact(file, GRAM, Backend::Vulkan).expect("the model compiles");
    let epilogues: Vec<String> = artifact
        .execution
        .kernels
        .iter()
        .filter_map(|k| k.epilogue.map(|e| e.to_string()))
        .collect();
    assert_eq!(
        epilogues,
        vec!["epilogue Relu"],
        "one ReLU epilogue, without the pre-activation"
    );
    let runs = runs_of(file, GRAM);
    assert_matches_the_cpu(file, &runs);
    assert_one_wait_and_the_counted_bytes(file, &runs, 200);
}
