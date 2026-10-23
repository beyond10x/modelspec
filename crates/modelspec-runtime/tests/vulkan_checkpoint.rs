//! Checkpoints of a training run on a Vulkan device, resume on the device, and inference on the
//! device (DESIGN.md § 6.2, § 9 gate 7; `story:m1-vulkan-checkpoint-infer`):
//!
//! - a run of `examples/mnist_mlp.mspec` with a checkpoint every 600 of its 1,800 steps waits
//!   for the device 3 times (at 600, at 1,200 and at the end) and writes 2 checkpoints; without
//!   them it waits once;
//! - 900 steps, a checkpoint, then 900 steps resumed from it, pass against 1,800 steps that
//!   never stopped: every final parameter the whole-run bound [`common::run_within`], every loss
//!   from step 900 to 1,800 the `mse` tolerance of `tests/common/mod.rs`. Whether the two runs
//!   are bit-equal is printed, not checked;
//! - the forward program on the CPU with the final checkpoint of a device run passes the `mse`
//!   tolerance against the run's last loss, and inference on the device passes it against the
//!   CPU, with one host wait.
//!
//! Every test fails on a message of the Khronos validation layer. They are ignored by default:
//! they create a Vulkan instance, which `task ci` never does. Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_checkpoint -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use common::{GPU_ATOL, GPU_RTOL, KernelOp, RUN_L2_RTOL, run_within, term_tolerance};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{CompiledTraining, compile_artifact, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::tensor::ParamId;
use modelspec_runtime::cpu::{HostTensor, evaluate};
use modelspec_runtime::train::checkpoint::{self, Checkpoint};
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::train::runner::Run;
use modelspec_runtime::vulkan::negotiate::{Negotiated, negotiate};
use modelspec_runtime::vulkan::runner::{
    Checkpoints, Inference, RunOptions, RunStats, infer, train_with,
};
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

const MNIST_MLP: &str = "examples/mnist_mlp.mspec";

type Values = BTreeMap<ParamId, HostTensor<f32>>;

fn validation_requested() -> bool {
    std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
}

fn compiled(steps: u64) -> (CompiledTraining, CompiledArtifact) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(MNIST_MLP);
    let source = std::fs::read_to_string(&path).expect("the example reads");
    let mut compiled = compile_training(MNIST_MLP, &source).expect("the example compiles");
    compiled.config.steps = steps;
    let artifact = compile_artifact(MNIST_MLP, &source, Backend::Vulkan)
        .expect("the example compiles for Vulkan");
    (compiled, artifact)
}

/// A fresh directory under the test temp dir.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("vulkan_checkpoint")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    dir
}

/// Opens device 0 and calls `body` with it; fails on a validation message at setup or after
/// `body`, the device dropped.
fn on_device<T>(
    label: &str,
    body: impl FnOnce(&Device<'_>, &DeviceProfile, &Negotiated) -> T,
) -> T {
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
    let result = body(&device, &profile, &negotiated);
    drop(device);
    let messages = vulkan.take_validation_messages();
    assert!(
        messages.is_empty(),
        "{label}: {} validation messages: {messages:#?}",
        messages.len()
    );
    eprintln!(
        "{label}: validation {}, 0 validation messages",
        vulkan.validation_enabled()
    );
    result
}

/// Trains on `device` with `options`.
fn train(
    device: &Device<'_>,
    profile: &DeviceProfile,
    negotiated: &Negotiated,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    options: RunOptions<'_>,
) -> (Run, RunStats) {
    train_with(device, profile, negotiated, compiled, artifact, options)
        .expect("the device run finishes")
}

/// Whether the loss `g` passes the `mse` tolerance against `c`, with the reference `r` of
/// `tests/vulkan_train.rs`: every term of an `mse` is a square, so `c` itself.
fn loss_within(g: f32, c: f32) -> bool {
    let tolerance = term_tolerance(KernelOp::Mse);
    tolerance.accepts(g, c, tolerance.r(c, f64::from(c), 1))
}

/// The largest `||g - c||_2 / ||c||_2` over the parameters of `c`.
fn largest_relative_l2(g: &Values, c: &Values) -> f64 {
    c.iter()
        .map(|(param, c)| {
            let g = &g[param];
            let difference: f64 = g
                .data()
                .iter()
                .zip(c.data())
                .map(|(&g, &c)| (f64::from(g) - f64::from(c)).powi(2))
                .sum();
            let norm: f64 = c.data().iter().map(|&c| f64::from(c).powi(2)).sum();
            difference.sqrt() / norm.sqrt()
        })
        .fold(0.0, f64::max)
}

/// The largest `|g - c|` of two loss lists.
fn largest_loss_gap(g: &[f32], c: &[f32]) -> f64 {
    g.iter()
        .zip(c)
        .map(|(&g, &c)| (f64::from(g) - f64::from(c)).abs())
        .fold(0.0, f64::max)
}

/// Acceptance 2: a checkpoint every 600 of 1,800 steps costs one host wait each and writes two
/// files; the run without them waits once.
#[test]
#[ignore = "needs a Vulkan device"]
fn mnist_mlp_waits_once_per_requested_checkpoint_and_once_at_the_end() {
    let (compiled, artifact) = compiled(1800);
    let dir = scratch("every-600");
    let mut written = Vec::new();
    let mut write = |step: u64, values: &Values| {
        let path = dir.join(format!("run-{step}.msckpt"));
        written.push(step);
        checkpoint::write(&path, &Checkpoint::of_state(&compiled, step, values))
    };
    let (every, plain) = on_device("every 600", |device, profile, negotiated| {
        let every = train(
            device,
            profile,
            negotiated,
            &compiled,
            &artifact,
            RunOptions {
                checkpoints: Some(Checkpoints {
                    every: 600,
                    write: &mut write,
                }),
                ..RunOptions::default()
            },
        );
        let plain = train(
            device,
            profile,
            negotiated,
            &compiled,
            &artifact,
            RunOptions::default(),
        );
        (every, plain)
    });
    eprintln!("with --checkpoint-every 600:\n{}", every.1);
    eprintln!("without:\n{}", plain.1);
    assert_eq!(written, vec![600, 1200], "checkpoints at 600 and 1,200");
    assert_eq!(every.1.host_waits, 3, "at 600, at 1,200 and at the end");
    assert_eq!(plain.1.host_waits, 1, "at the end");
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .expect("the dir reads")
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    assert_eq!(files, ["run-1200.msckpt", "run-600.msckpt"]);
    for step in [600, 1200] {
        let read = checkpoint::read(&dir.join(format!("run-{step}.msckpt")))
            .expect("the checkpoint reads back");
        assert_eq!(read.step, step);
        read.resume_into(&compiled)
            .expect("the checkpoint matches the file");
    }
    let (every, plain) = (every.0, plain.0);
    assert_eq!(every.losses.len(), 1801);
    assert_eq!(every.state.versions, plain.state.versions);
    for (param, c) in &plain.state.values {
        assert!(
            run_within(every.state.values[param].data(), c.data()),
            "{param}: the run with checkpoints is outside ||g - c||_2 <= {RUN_L2_RTOL:e} * ||c||_2 \
             of the run without"
        );
    }
    eprintln!(
        "with and without checkpoints: parameters bit-equal {}, losses bit-equal {}",
        every.state.values == plain.state.values,
        every.losses == plain.losses
    );
}

/// Acceptance 3: 900 steps, a checkpoint, 900 steps resumed from it on the device, against
/// 1,800 steps that never stopped.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_resumed_device_run_passes_against_the_run_that_never_stopped() {
    let (whole, artifact) = compiled(1800);
    let (half, _) = compiled(900);
    let dir = scratch("resume");
    let mid = dir.join("mid.msckpt");
    let (uninterrupted, resumed, stats) = on_device("resume", |device, profile, negotiated| {
        let (uninterrupted, _) = train(
            device,
            profile,
            negotiated,
            &whole,
            &artifact,
            RunOptions::default(),
        );
        let (first, _) = train(
            device,
            profile,
            negotiated,
            &half,
            &artifact,
            RunOptions::default(),
        );
        checkpoint::write(&mid, &Checkpoint::of_state(&half, 900, &first.state.values))
            .expect("the checkpoint is written");
        let read = checkpoint::read(&mid).expect("the checkpoint reads back");
        let init = read
            .resume_into(&half)
            .expect("the checkpoint matches the file");
        assert_eq!(read.step, 900);
        let (resumed, stats) = train(
            device,
            profile,
            negotiated,
            &half,
            &artifact,
            RunOptions {
                start: read.step,
                init,
                ..RunOptions::default()
            },
        );
        (uninterrupted, resumed, stats)
    });
    eprintln!("resumed:\n{stats}");
    assert_eq!(resumed.start_step, 900);
    assert_eq!(stats.host_waits, 1);
    assert_eq!(resumed.state.versions, uninterrupted.state.versions);
    assert_eq!(resumed.losses.len(), 901, "steps 900 to 1,800");
    for (param, c) in &uninterrupted.state.values {
        assert!(
            run_within(resumed.state.values[param].data(), c.data()),
            "{param} is outside ||g - c||_2 <= {RUN_L2_RTOL:e} * ||c||_2"
        );
    }
    let tail = &uninterrupted.losses[900..];
    let outside: Vec<(usize, f32, f32)> = resumed
        .losses
        .iter()
        .zip(tail)
        .enumerate()
        .filter(|(_, (g, c))| !loss_within(**g, **c))
        .map(|(i, (&g, &c))| (900 + i, g, c))
        .collect();
    assert!(
        outside.is_empty(),
        "{} of 901 losses outside the mse tolerance (GPU_ATOL {GPU_ATOL:e}, GPU_RTOL \
         {GPU_RTOL:e}); first (step, resumed, uninterrupted): {:?}",
        outside.len(),
        &outside[..outside.len().min(5)]
    );
    eprintln!(
        "resumed against uninterrupted: largest ||g - c||_2 / ||c||_2 {:e}, largest loss gap {:e}, \
         parameters bit-equal {}, losses bit-equal {}",
        largest_relative_l2(&resumed.state.values, &uninterrupted.state.values),
        largest_loss_gap(&resumed.losses, tail),
        resumed.state.values == uninterrupted.state.values,
        resumed.losses == tail
    );
}

/// Acceptance 4: the final checkpoint of a device run loads into CPU inference, whose loss
/// passes against the run's last loss, and into device inference, whose loss passes against the
/// CPU's with one host wait.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_device_checkpoint_loads_into_cpu_and_device_inference() {
    let (compiled, artifact) = compiled(1800);
    let dir = scratch("infer");
    let last = dir.join("run.msckpt");
    let (run, params, inference, stats) = on_device("infer", |device, profile, negotiated| {
        let (run, _) = train(
            device,
            profile,
            negotiated,
            &compiled,
            &artifact,
            RunOptions::default(),
        );
        checkpoint::write(
            &last,
            &Checkpoint::of_state(&compiled, 1800, &run.state.values),
        )
        .expect("the checkpoint is written");
        let params = checkpoint::read(&last)
            .and_then(|c| c.parameters_for(&compiled))
            .expect("the checkpoint matches the file");
        let (inference, stats): (Inference, RunStats) =
            infer(device, profile, negotiated, &compiled, &artifact, &params)
                .expect("the device inference finishes");
        (run, params, inference, stats)
    });
    eprintln!("device inference:\n{stats}");
    let batch = synthetic(&compiled.config, &compiled.forward).expect("the batch is drawn");
    let cpu = evaluate(&compiled.forward.program, &batch.inputs, &params)
        .expect("the CPU forward program runs")[0]
        .data()[0];
    let gpu_last = *run.losses.last().unwrap();
    eprintln!(
        "last device loss {gpu_last}, CPU infer loss {cpu}, device infer loss {}",
        inference.loss
    );
    assert!(
        loss_within(cpu, gpu_last),
        "CPU infer {cpu} against the run's last loss {gpu_last}"
    );
    assert!(
        loss_within(inference.loss, cpu),
        "device infer {} against CPU infer {cpu}",
        inference.loss
    );
    assert_eq!(stats.host_waits, 1, "one host wait");
    assert_eq!(stats.steps, 0);
    let prediction = compiled
        .forward
        .program
        .node(compiled.forward.prediction)
        .expect("the forward program has a prediction");
    assert_eq!(
        inference.prediction.shape().len(),
        prediction.ty.shape.len()
    );
}
