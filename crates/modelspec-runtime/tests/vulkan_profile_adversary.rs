//! Adversarial cases for `train --profile` (`story:m1-vulkan-profile`).
//!
//! The device cases are ignored by default; run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_profile_adversary -- --ignored`.
#![cfg(feature = "vulkan")]

use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_core::ir::kernel::{KernelKind, PlanOp};
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::profile::PlanName;
use modelspec_runtime::vulkan::runner::{train, train_profiled};
use modelspec_runtime::vulkan::{VALIDATION_ENV, Vulkan};

const MNIST_MLP: &str = "examples/mnist_mlp.mspec";

fn source() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(MNIST_MLP);
    std::fs::read_to_string(&path).expect("the example reads")
}

/// Step counts on both sides of the 256-step command buffer boundary: every step site is
/// called once per step, every site has a positive time, the results are bit-equal to an
/// unprofiled run and the validation layer stays silent.
#[test]
#[ignore = "needs a Vulkan device"]
fn step_counts_around_the_command_buffer_boundary_time_every_dispatch() {
    let source = source();
    let artifact = compile_artifact(MNIST_MLP, &source, Backend::Vulkan).expect("compiles");
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
    );
    let candidates = vulkan.candidates().expect("the devices are probed");
    let device_profile = candidates[0].profile.clone();
    let negotiated = negotiate(&device_profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    for steps in [1u64, 21, 255, 256, 257, 512, 513] {
        let mut compiled = compile_training(MNIST_MLP, &source).expect("compiles");
        compiled.config.steps = steps;
        let (plain, plain_stats) =
            train(&device, &device_profile, &negotiated, &compiled, &artifact)
                .expect("the unprofiled run finishes");
        let (run, stats, profile) =
            train_profiled(&device, &device_profile, &negotiated, &compiled, &artifact)
                .unwrap_or_else(|e| panic!("{steps} steps: {e}"));
        let timed = profile.timings.as_ref().expect("timestamps");
        for site in &timed.sites {
            let calls = match site.site.plan {
                PlanName::Step => steps,
                PlanName::Final => 1,
            };
            assert_eq!(site.calls, calls, "{steps} steps: {site:?}");
            assert!(site.total_ns > 0.0, "{steps} steps: {site:?}");
        }
        assert_eq!(timed.steady_us_per_step.is_some(), steps > 20, "{steps}");
        assert_eq!(stats.host_waits, 1);
        assert_eq!(stats.submits, plain_stats.submits, "{steps} steps");
        assert_eq!(stats.dispatches, plain_stats.dispatches, "{steps} steps");
        assert_eq!(run.losses, plain.losses, "{steps} steps: bit-equal losses");
        assert_eq!(run.state.values, plain.state.values, "{steps} steps");
    }
    drop(device);
    let messages = vulkan.take_validation_messages();
    assert!(messages.is_empty(), "{messages:#?}");
}

/// The step sites partition a step: their mean times sum to no more than the steady-state time
/// per step, which spans every dispatch, barrier and loss copy of a step. A start stamp that is
/// written before the previous dispatch has finished counts that dispatch twice.
#[test]
#[ignore = "needs a Vulkan device"]
fn the_step_sites_do_not_count_one_interval_twice() {
    let source = source();
    let artifact = compile_artifact(MNIST_MLP, &source, Backend::Vulkan).expect("compiles");
    let mut compiled = compile_training(MNIST_MLP, &source).expect("compiles");
    compiled.config.steps = 600;
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let device_profile = candidates[0].profile.clone();
    let negotiated = negotiate(&device_profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let (_, _, profile) =
        train_profiled(&device, &device_profile, &negotiated, &compiled, &artifact)
            .expect("the profiled run finishes");
    let timed = profile.timings.expect("timestamps");
    let steady = timed
        .steady_us_per_step
        .expect("600 steps have a steady state");
    let sum: f64 = timed
        .sites
        .iter()
        .filter(|s| s.site.plan == PlanName::Step)
        .map(|s| s.mean_us())
        .sum();
    eprintln!("sum of step-site means {sum:.3} us, steady {steady:.3} us per step");
    assert!(sum <= steady, "{sum} > {steady}");
}

/// Acceptance 1 and 3 of the story: one `final` line per Dispatch step of the step plan the loss
/// is computed from (`ExecutionPlan::loss_steps`, since `story:step-sites-beat-pytorch` records
/// the loss sum after the gradients that do not feed it), named by its step-plan position, and
/// dispatches 1,800 times the step plan's plus that count.
#[test]
#[ignore = "needs a Vulkan device"]
fn the_final_lines_are_the_step_plan_dispatches_the_loss_is_computed_from() {
    let source = source();
    let artifact = compile_artifact(MNIST_MLP, &source, Backend::Vulkan).expect("compiles");
    let compiled = compile_training(MNIST_MLP, &source).expect("compiles");
    let plan = &artifact.execution;
    let dispatches_of = |steps: &[modelspec_core::ir::kernel::PlanStep]| -> Vec<usize> {
        steps
            .iter()
            .enumerate()
            .filter(|(_, s)| s.op == PlanOp::Dispatch)
            .map(|(at, _)| at)
            .collect()
    };
    let step_plan = dispatches_of(&plan.step);
    let loss_at = plan
        .step
        .iter()
        .position(|s| s.op == PlanOp::Dispatch && s.writes.contains(&plan.loss_view))
        .expect("a dispatch writes the loss view");
    let loss_steps: Vec<usize> = plan
        .loss_steps()
        .into_iter()
        .filter(|&at| plan.step[at].op == PlanOp::Dispatch)
        .collect();
    let kinds: Vec<KernelKind> = loss_steps
        .iter()
        .map(|&at| {
            let id = plan.step[at]
                .kernel_module_id
                .expect("a dispatch names a kernel");
            plan.kernels
                .iter()
                .find(|k| k.id == id)
                .expect("the plan holds the kernel")
                .kind
        })
        .collect();
    assert_eq!(
        kinds,
        [KernelKind::MatMul, KernelKind::MatMul, KernelKind::LossSum],
        "the two forward matmuls, then the loss sum"
    );
    assert_eq!(loss_steps.last(), Some(&loss_at));
    assert!(loss_steps.len() < step_plan.len());
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let device_profile = candidates[0].profile.clone();
    let negotiated = negotiate(&device_profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let (_, stats, profile) =
        train_profiled(&device, &device_profile, &negotiated, &compiled, &artifact)
            .expect("the profiled run finishes");
    let final_lines: Vec<usize> = profile
        .timings
        .as_ref()
        .expect("timestamps")
        .sites
        .iter()
        .filter(|s| s.site.plan == PlanName::Final)
        .map(|s| s.site.position)
        .collect();
    assert_eq!(
        final_lines, loss_steps,
        "the final lines name the step plan's Dispatch steps the loss is computed from"
    );
    assert_eq!(
        stats.dispatches,
        1800 * step_plan.len() as u64 + loss_steps.len() as u64
    );
}
