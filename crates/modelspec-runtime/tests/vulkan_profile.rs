//! `train --profile` on a Vulkan device (DESIGN.md § 8; `story:m1-vulkan-profile`): a timestamp
//! before and after every dispatch of every step, read back with the results after the run's one
//! host wait, summed per dispatch site.
//!
//! The first tests need no device: they feed [`timings`] timestamps by hand and check what it
//! sums, and check the lines [`Profile::report`] and the `--stats` lines of [`RunStats`] print.
//! The last two tests are ignored by default because they create a Vulkan instance, which
//! `task ci` never does; run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_profile -- --ignored`;
//! every message of the Khronos validation layer fails them.
//! One trains `examples/linear_regression.mspec` and reads its barriers per step, device
//! allocations and peak device bytes, each known from its plan (story:vulkan-step-counters).
//! The other trains `examples/mnist_mlp.mspec` with and without the profile and checks
//!
//! - one site per Dispatch step of the step plan, `calls` the steps, and one per Dispatch step
//!   the final loss evaluation records, `calls` 1 (acceptance 1);
//! - the sum of every site's total is at most the wall time of the profiled run (acceptance 1);
//! - the same host waits and bit-equal losses and parameters with and without (acceptance 2);
//! - the counter lines are the run's `RunStats` (acceptance 3);
//! - every site carries the FLOP and byte count of its plan step, and the report prints its
//!   rates and the step total (`story:profile-achieved-throughput`).
#![cfg(feature = "vulkan")]

use std::time::{Duration, Instant};

use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_core::ir::kernel::{DispatchCount, KernelKind, KernelModuleId, PlanOp};
use modelspec_runtime::vulkan::Device;
use modelspec_runtime::vulkan::negotiate::{NumericsClass, negotiate};
use modelspec_runtime::vulkan::profile::{
    Clock, PlanName, Profile, Site, Timings, WARMUP_STEPS, timings,
};
use modelspec_runtime::vulkan::runner::{RunStats, train, train_profiled};
use modelspec_runtime::vulkan::{VALIDATION_ENV, Vulkan};

const MNIST_MLP: &str = "examples/mnist_mlp.mspec";
const LINEAR_REGRESSION: &str = "examples/linear_regression.mspec";

/// The count of a dispatch of 48 FLOP over 120 bytes.
const COUNTED: DispatchCount = DispatchCount::Counted {
    flop: 48,
    bytes: 120,
};

fn site(plan: PlanName, position: usize, id: u8, count: DispatchCount) -> Site {
    Site {
        plan,
        position,
        kernel_module_id: KernelModuleId([id; 16]),
        entry: "main".to_owned(),
        count,
    }
}

fn stats() -> RunStats {
    RunStats {
        steps: 3,
        host_waits: 1,
        submits: 1,
        dispatches: 7,
        dispatches_per_step: 2,
        barriers_per_step: 4,
        bytes_uploaded: 40,
        bytes_downloaded: 28,
        device_allocations: 5,
        peak_device_bytes: 1024,
        class: NumericsClass::Tolerance,
        elapsed: Duration::from_micros(90),
        training: Duration::from_micros(90),
        evaluation: Duration::ZERO,
    }
}

/// story:vulkan-step-counters: `--stats` prints one `<name>: <value>` line per counter, the
/// barriers of one step after its dispatches and the device allocations and their peak bytes after
/// the bytes copied.
#[test]
fn the_stats_lines_name_the_barriers_per_step_the_device_allocations_and_the_peak_bytes() {
    assert_eq!(
        stats().to_string(),
        "steps: 3\n\
         host waits: 1\n\
         submits: 1\n\
         dispatches: 7\n\
         dispatches per step: 2\n\
         barriers per step: 4\n\
         bytes uploaded: 40\n\
         bytes downloaded: 28\n\
         device allocations: 5\n\
         peak device bytes: 1024\n\
         numerics class: tolerance\n\
         us per step: 30.000\n"
    );
}

/// Two step sites (positions 0 and 2, the second sharing the first's module and uncounted) and
/// one final site, three steps: the stamps are `(start, end)` pairs in recording order, steps
/// first.
fn three_steps() -> (Vec<Site>, Vec<Site>, Vec<u64>) {
    let steps = vec![
        site(PlanName::Step, 0, 1, COUNTED),
        site(PlanName::Step, 2, 1, DispatchCount::Uncounted),
    ];
    let last = vec![site(PlanName::Final, 0, 1, COUNTED)];
    let stamps = vec![
        100, 110, 120, 125, // step 0: 10 and 5 ticks
        200, 212, 230, 236, // step 1: 12 and 6 ticks
        300, 314, 330, 337, // step 2: 14 and 7 ticks
        400, 403, // the final loss: 3 ticks
    ];
    (steps, last, stamps)
}

#[test]
fn timings_sum_each_site_over_the_steps_and_count_the_final_loss_once() {
    let (steps, last, stamps) = three_steps();
    let clock = Clock {
        valid_bits: 64,
        period_ns: 2.0,
    };
    let Timings {
        sites,
        steady_us_per_step,
    } = timings(&steps, &last, 3, &stamps, clock).expect("the stamps fit the sites");
    assert_eq!(
        sites.len(),
        3,
        "two sites that share a module are two sites"
    );
    assert_eq!(sites[0].site, steps[0]);
    assert_eq!(sites[1].site, steps[1]);
    assert_eq!(sites[2].site, last[0]);
    assert_eq!(sites.iter().map(|s| s.calls).collect::<Vec<_>>(), [3, 3, 1]);
    // Ticks times 2 ns: 36, 18 and 3 ticks.
    let totals: Vec<f64> = sites.iter().map(|s| s.total_us()).collect();
    assert_eq!(totals, [0.072, 0.036, 0.006]);
    assert_eq!(sites[0].mean_us(), 0.024);
    assert_eq!(
        steady_us_per_step, None,
        "3 steps have no steady state after a {WARMUP_STEPS}-step warm-up"
    );
}

#[test]
fn a_timestamp_that_wraps_its_valid_bits_counts_the_ticks_past_the_wrap() {
    let steps = vec![site(PlanName::Step, 0, 1, COUNTED)];
    let last = vec![site(PlanName::Final, 0, 1, COUNTED)];
    let clock = Clock {
        valid_bits: 8,
        period_ns: 1000.0,
    };
    // Bits above the valid ones are garbage; 250 to 4 is 10 ticks past the wrap at 256.
    let stamps = [0xff00 | 250, 0xab00 | 4, 7, 8];
    let timed = timings(&steps, &last, 1, &stamps, clock).expect("the stamps fit");
    assert_eq!(timed.sites[0].total_us(), 10.0);
    assert_eq!(timed.sites[1].total_us(), 1.0);
}

#[test]
fn the_steady_state_runs_from_the_first_step_after_the_warm_up_to_the_final_loss() {
    let steps = vec![site(PlanName::Step, 0, 1, COUNTED)];
    let final_sites = vec![site(PlanName::Final, 0, 1, COUNTED)];
    let n = WARMUP_STEPS + 4;
    let mut stamps = Vec::new();
    for k in 0..n {
        // The warm-up steps take 1,000 ns each, the steady ones 50.
        let start = if k <= WARMUP_STEPS {
            1000 * k
        } else {
            1000 * WARMUP_STEPS + 50 * (k - WARMUP_STEPS)
        };
        stamps.extend([start, start + 10]);
    }
    let last = 1000 * WARMUP_STEPS + 50 * (n - WARMUP_STEPS);
    stamps.extend([last, last + 10]);
    let clock = Clock {
        valid_bits: 64,
        period_ns: 1.0,
    };
    let timed = timings(&steps, &final_sites, n, &stamps, clock).expect("the stamps fit");
    assert_eq!(timed.steady_us_per_step, Some(0.05));
}

#[test]
fn stamps_that_do_not_fit_the_sites_are_refused() {
    let (steps, last, mut stamps) = three_steps();
    let clock = Clock {
        valid_bits: 64,
        period_ns: 1.0,
    };
    stamps.pop();
    assert!(timings(&steps, &last, 3, &stamps, clock).is_err());
}

/// A counted site's line ends with its FLOP and bytes and their rates over its mean time: 48 FLOP
/// in 0.024 µs is 2 GFLOP/s, 120 bytes 5 GB/s. The uncounted site prints `uncounted`, and the
/// step total, with no steady state to divide by, names it.
#[test]
fn the_report_prints_one_line_per_site_then_the_counters() {
    let (steps, last, stamps) = three_steps();
    let clock = Clock {
        valid_bits: 64,
        period_ns: 2.0,
    };
    let profile = Profile {
        timings: Some(timings(&steps, &last, 3, &stamps, clock).expect("the stamps fit")),
        pipeline_creation: Duration::from_micros(1500),
        compile: Duration::from_nanos(2_250_500),
        pipeline_statistics: modelspec_runtime::vulkan::pipeline::PipelineStatistics::NotCaptured,
    };
    let id = KernelModuleId([1; 16]);
    assert_eq!(
        profile.report(&stats()),
        format!(
            "profile dispatch step 0 {id} main calls 3 total_us 0.072 mean_us 0.024 \
             flop 48 bytes 120 gflop_per_s 2.000 gb_per_s 5.000\n\
             profile dispatch step 2 {id} main calls 3 total_us 0.036 mean_us 0.012 uncounted\n\
             profile dispatch final 0 {id} main calls 1 total_us 0.006 mean_us 0.006 \
             flop 48 bytes 120 gflop_per_s 8.000 gb_per_s 20.000\n\
             profile pipeline statistics: not reported\n\
             profile dispatches 7\n\
             profile submits 1\n\
             profile host_waits 1\n\
             profile barriers_per_step 4\n\
             profile bytes_uploaded 40\n\
             profile bytes_downloaded 28\n\
             profile device_allocations 5\n\
             profile peak_device_bytes 1024\n\
             profile pipeline_creation_us 1500.000\n\
             profile compile_us 2250.500\n\
             profile steady_us_per_step unavailable\n\
             profile step_total flop 48 bytes 120 gflop_per_s unavailable gb_per_s unavailable \
             uncounted_sites 1\n"
        )
    );
}

#[test]
fn without_timestamp_queries_the_report_says_so_and_still_prints_the_counters() {
    let profile = Profile {
        timings: None,
        pipeline_creation: Duration::from_micros(1),
        compile: Duration::from_micros(2),
        pipeline_statistics: modelspec_runtime::vulkan::pipeline::PipelineStatistics::NotCaptured,
    };
    assert_eq!(
        profile.report(&stats()),
        "profile kernel timings unavailable: no timestamp queries\n\
         profile pipeline statistics: not reported\n\
         profile dispatches 7\n\
         profile submits 1\n\
         profile host_waits 1\n\
         profile barriers_per_step 4\n\
         profile bytes_uploaded 40\n\
         profile bytes_downloaded 28\n\
         profile device_allocations 5\n\
         profile peak_device_bytes 1024\n\
         profile pipeline_creation_us 1.000\n\
         profile compile_us 2.000\n\
         profile steady_us_per_step unavailable\n\
         profile step_total unavailable\n"
    );
}

/// The counter value of the report line `profile <name> <value>`.
fn counter<'a>(report: &'a str, name: &str) -> &'a str {
    let head = format!("profile {name} ");
    report
        .lines()
        .find_map(|line| line.strip_prefix(head.as_str()))
        .unwrap_or_else(|| panic!("no line {head}<value> in\n{report}"))
}

/// The stamps of `n` steps of `per_step` dispatches, then `last` final ones: the warm-up steps
/// start 1,000 ticks apart and the steady ones 50, each dispatch 2 ticks, back to back.
fn steady_stamps(n: u64, per_step: u64, last: u64) -> Vec<u64> {
    let start = |k: u64| {
        if k <= WARMUP_STEPS {
            1000 * k
        } else {
            1000 * WARMUP_STEPS + 50 * (k - WARMUP_STEPS)
        }
    };
    let mut stamps = Vec::new();
    for k in 0..=n {
        let dispatches = if k < n { per_step } else { last };
        for d in 0..dispatches {
            stamps.extend([start(k) + 2 * d, start(k) + 2 * d + 2]);
        }
    }
    stamps
}

/// story:profile-achieved-throughput (D3): the step total is the counted FLOP and bytes of the
/// step sites, 100 and 400, over the steady 0.05 µs per step, and it names the uncounted site;
/// the final loss's site is not a step site.
#[test]
fn the_step_total_divides_the_counted_step_sites_by_the_steady_time_per_step() {
    let steps = vec![
        site(
            PlanName::Step,
            0,
            1,
            DispatchCount::Counted {
                flop: 60,
                bytes: 300,
            },
        ),
        site(
            PlanName::Step,
            1,
            2,
            DispatchCount::Counted {
                flop: 40,
                bytes: 100,
            },
        ),
        site(PlanName::Step, 3, 3, DispatchCount::Uncounted),
    ];
    let final_sites = vec![site(PlanName::Final, 0, 1, COUNTED)];
    let n = WARMUP_STEPS + 4;
    let clock = Clock {
        valid_bits: 64,
        period_ns: 1.0,
    };
    let timed =
        timings(&steps, &final_sites, n, &steady_stamps(n, 3, 1), clock).expect("the stamps fit");
    assert_eq!(timed.steady_us_per_step, Some(0.05));
    let profile = Profile {
        timings: Some(timed),
        pipeline_creation: Duration::from_micros(1),
        compile: Duration::from_micros(2),
        pipeline_statistics: modelspec_runtime::vulkan::pipeline::PipelineStatistics::NotCaptured,
    };
    let report = profile.report(&stats());
    assert_eq!(
        counter(&report, "step_total"),
        "flop 100 bytes 400 gflop_per_s 2.000 gb_per_s 8.000 uncounted_sites 1",
        "{report}"
    );
    let uncounted = report
        .lines()
        .find(|l| l.starts_with("profile dispatch step 3 "))
        .expect("the uncounted site has a line");
    assert!(
        uncounted.ends_with(" mean_us 0.002 uncounted"),
        "{uncounted}"
    );
    assert!(!uncounted.contains(" flop "), "{uncounted}");
}

/// story:profile-achieved-throughput: a site carries the count of its plan step. Every step site
/// of the MNIST plan is counted, and its five matmuls are `2·m·k·n` FLOP, 20,454,400 in all.
#[test]
fn a_site_of_the_mnist_plan_carries_the_count_of_its_dispatch() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(MNIST_MLP);
    let source = std::fs::read_to_string(&path).expect("the example reads");
    let artifact = compile_artifact(MNIST_MLP, &source, Backend::Vulkan)
        .expect("the example compiles for Vulkan");
    let plan = &artifact.execution;
    let mut matmul_flop = 0;
    for (at, step) in plan.step.iter().enumerate() {
        let site = Site::new(plan, PlanName::Step, at, "main".to_owned());
        if step.op != PlanOp::Dispatch {
            assert_eq!(site, None, "step {at} is no dispatch");
            continue;
        }
        let site = site.expect("a dispatch is a site");
        assert_eq!(Some(site.kernel_module_id), step.kernel_module_id);
        assert_eq!(Some(site.count), plan.dispatch_count(step));
        let DispatchCount::Counted { flop, .. } = site.count else {
            panic!("step {at} is uncounted: {step:?}");
        };
        let kind = plan.kernel(site.kernel_module_id).map(|k| k.kind);
        if kind == Some(KernelKind::MatMul) {
            matmul_flop += flop;
        }
    }
    assert_eq!(matmul_flop, 20_454_400);
}

#[test]
#[ignore = "needs a Vulkan device"]
fn mnist_mlp_profile_times_every_dispatch_site_without_changing_the_run() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(MNIST_MLP);
    let source = std::fs::read_to_string(&path).expect("the example reads");
    let compiled = compile_training(MNIST_MLP, &source).expect("the example compiles");
    let artifact = compile_artifact(MNIST_MLP, &source, Backend::Vulkan)
        .expect("the example compiles for Vulkan");
    let plan = &artifact.execution;
    let steps = compiled.config.steps;
    assert_eq!(steps, 1800);

    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1"),
        "the validation layer is on exactly when {VALIDATION_ENV}=1"
    );
    let candidates = vulkan.candidates().expect("the devices are probed");
    let device_profile = candidates[0].profile.clone();
    assert!(
        device_profile.timestamp_queries,
        "device 0 has timestamp queries"
    );
    let negotiated = negotiate(&device_profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let (plain, plain_stats) = train(&device, &device_profile, &negotiated, &compiled, &artifact)
        .expect("the unprofiled run finishes");
    let wall = Instant::now();
    let (profiled, stats, profile) =
        train_profiled(&device, &device_profile, &negotiated, &compiled, &artifact)
            .expect("the profiled run finishes");
    let wall = wall.elapsed();
    drop(device);
    let messages = vulkan.take_validation_messages();
    assert!(
        messages.is_empty(),
        "{} validation messages: {messages:#?}",
        messages.len()
    );
    let report = profile.report(&stats);
    eprintln!("{MNIST_MLP}: wall {wall:?}\n{stats}{report}");

    // Acceptance 1: one site per Dispatch step of the step plan, called once per step, then one
    // per Dispatch step the final loss evaluation records (the steps the loss is computed from,
    // `ExecutionPlan::loss_steps`).
    let dispatch_positions: Vec<usize> = plan
        .step
        .iter()
        .enumerate()
        .filter(|(_, step)| step.op == PlanOp::Dispatch)
        .map(|(at, _)| at)
        .collect();
    let final_positions: Vec<usize> = plan
        .loss_steps()
        .into_iter()
        .filter(|&at| plan.step[at].op == PlanOp::Dispatch)
        .collect();
    assert!(!final_positions.is_empty(), "a step writes the loss");
    let timed = profile.timings.as_ref().expect("the device has timestamps");
    let expected: Vec<(PlanName, usize, KernelModuleId, u64)> = dispatch_positions
        .iter()
        .map(|&at| (PlanName::Step, at, steps))
        .chain(final_positions.iter().map(|&at| (PlanName::Final, at, 1)))
        .map(|(name, at, calls)| {
            let id = plan.step[at]
                .kernel_module_id
                .expect("a dispatch names a kernel");
            (name, at, id, calls)
        })
        .collect();
    let got: Vec<(PlanName, usize, KernelModuleId, u64)> = timed
        .sites
        .iter()
        .map(|s| {
            (
                s.site.plan,
                s.site.position,
                s.site.kernel_module_id,
                s.calls,
            )
        })
        .collect();
    assert_eq!(got, expected);
    assert_eq!(
        report
            .lines()
            .filter(|l| l.starts_with("profile dispatch "))
            .count(),
        dispatch_positions.len() + final_positions.len()
    );
    // story:profile-achieved-throughput: every site carries its step's count, every line its
    // rates, and the step total counts every step site.
    for time in &timed.sites {
        let step = &plan.step[time.site.position];
        assert_eq!(Some(time.site.count), plan.dispatch_count(step), "{time:?}");
    }
    assert!(
        report
            .lines()
            .filter(|l| l.starts_with("profile dispatch "))
            .all(|l| l.contains(" gflop_per_s ") && l.contains(" gb_per_s ")),
        "{report}"
    );
    let total = counter(&report, "step_total");
    assert!(
        total.starts_with("flop ") && total.ends_with(" uncounted_sites 0"),
        "{total}"
    );
    assert!(timed.sites.iter().all(|s| s.total_us() > 0.0), "{report}");
    let sum_us: f64 = timed.sites.iter().map(|s| s.total_us()).sum();
    assert!(
        sum_us <= wall.as_secs_f64() * 1e6,
        "the site totals sum to {sum_us} us, more than the run's wall time {wall:?}"
    );
    assert!(timed.steady_us_per_step.is_some_and(|us| us > 0.0));

    // Acceptance 2: the same host waits and the same results.
    assert_eq!(stats.host_waits, 1);
    assert_eq!(stats.host_waits, plain_stats.host_waits);
    assert_eq!(stats.submits, plain_stats.submits);
    assert_eq!(profiled.losses, plain.losses, "bit-equal losses");
    assert_eq!(
        profiled.state.values, plain.state.values,
        "bit-equal parameters"
    );
    assert_eq!(profiled.initial, plain.initial);

    // Acceptance 3: the counter lines are the run's `RunStats`.
    let per_step = dispatch_positions.len() as u64;
    assert_eq!(
        stats.dispatches,
        steps * per_step + final_positions.len() as u64
    );
    assert_eq!(stats.dispatches, plain_stats.dispatches);
    assert_eq!(counter(&report, "dispatches"), stats.dispatches.to_string());
    assert_eq!(counter(&report, "submits"), stats.submits.to_string());
    assert_eq!(counter(&report, "host_waits"), "1");
    assert_eq!(counter(&report, "bytes_uploaded"), "520864");
    assert_eq!(counter(&report, "bytes_downloaded"), "210468");
    assert_eq!(stats.bytes_uploaded, 520_864);
    assert_eq!(stats.bytes_downloaded, 210_468);
    // story:vulkan-step-counters: the three counter lines are the run's, the same with and
    // without the profile.
    assert_eq!(stats.barriers_per_step, plain_stats.barriers_per_step);
    assert_eq!(stats.device_allocations, plain_stats.device_allocations);
    assert_eq!(stats.peak_device_bytes, plain_stats.peak_device_bytes);
    assert_eq!(
        counter(&report, "barriers_per_step"),
        stats.barriers_per_step.to_string()
    );
    assert_eq!(
        counter(&report, "device_allocations"),
        (plan.buffers.len() as u64 + 3).to_string()
    );
    assert_eq!(
        counter(&report, "peak_device_bytes"),
        stats.peak_device_bytes.to_string()
    );
    for name in ["pipeline_creation_us", "compile_us", "steady_us_per_step"] {
        let value: f64 = counter(&report, name)
            .parse()
            .unwrap_or_else(|_| panic!("{name} is a number in\n{report}"));
        assert!(value > 0.0, "{name} {value}");
    }
}

/// The allocation sizes the device gives a device-local buffer of each size of `device_local`
/// and a staging buffer of each size of `staging`, summed; every buffer is freed on return.
fn allocation_bytes(device: &Device<'_>, device_local: &[u64], staging: &[u64]) -> u64 {
    let local = device_local.iter().map(|&bytes| {
        device
            .device_local_buffer(bytes)
            .expect("the device-local buffer is made")
            .allocation_bytes()
    });
    let mapped = staging.iter().map(|&bytes| {
        device
            .staging_buffer(bytes)
            .expect("the staging buffer is made")
            .allocation_bytes()
    });
    local.chain(mapped).sum()
}

/// story:vulkan-step-counters: the counters of a run of `linear_regression`, each known from its
/// plan (`crates/modelspec-cli/tests/golden/linear_regression.kernel.ir`). A step records the 4
/// barriers of the step plan and the 2 around the copy of its loss: 6. The run allocates one
/// device-local buffer per plan buffer, 7 (the transpose of `x` is read in place and the loss
/// gradient is one elementwise chain, DESIGN.md § 3.1), the loss buffer and two staging buffers,
/// one for the upload and one for the read-back: 10. All 10 are live from setup to the end, so the
/// peak is the sum of their allocation sizes, which the device may round up from the 4,460 bytes
/// asked for.
#[test]
#[ignore = "needs a Vulkan device"]
fn linear_regression_counts_its_barriers_allocations_and_peak_device_bytes() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(LINEAR_REGRESSION);
    let source = std::fs::read_to_string(&path).expect("the example reads");
    let compiled = compile_training(LINEAR_REGRESSION, &source).expect("the example compiles");
    let artifact = compile_artifact(LINEAR_REGRESSION, &source, Backend::Vulkan)
        .expect("the example compiles for Vulkan");
    let plan = &artifact.execution;
    let steps = compiled.config.steps;
    assert_eq!(steps, 500);
    let ops = |op: PlanOp| plan.step.iter().filter(|step| step.op == op).count();
    assert_eq!((ops(PlanOp::Barrier), ops(PlanOp::Copy)), (4, 0));
    assert_eq!(plan.buffers.len(), 7);
    let param_bytes: u64 = plan.parameters.iter().map(|p| p.views[0].size_bytes).sum();
    let data_bytes: u64 = plan.data.iter().map(|d| d.view.size_bytes).sum();
    let loss_bytes = 4 * (steps + 1);
    let device_local: Vec<u64> = plan.buffers.iter().copied().chain([loss_bytes]).collect();
    let staging = [param_bytes + data_bytes, param_bytes + loss_bytes];
    let asked: u64 = device_local.iter().chain(&staging).sum();
    assert_eq!(asked, 4460);

    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let device_profile = candidates[0].profile.clone();
    let negotiated = negotiate(&device_profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let allocated = allocation_bytes(&device, &device_local, &staging);
    let (_, stats) = train(&device, &device_profile, &negotiated, &compiled, &artifact)
        .expect("the run finishes");
    drop(device);
    let messages = vulkan.take_validation_messages();
    assert!(
        messages.is_empty(),
        "{} validation messages: {messages:#?}",
        messages.len()
    );
    eprintln!("{LINEAR_REGRESSION}: {asked} bytes asked for, {allocated} allocated\n{stats}");

    assert_eq!(stats.barriers_per_step, 6);
    assert_eq!(stats.device_allocations, 10);
    assert!(allocated >= asked, "{allocated} < {asked}");
    assert_eq!(stats.peak_device_bytes, allocated);
}
