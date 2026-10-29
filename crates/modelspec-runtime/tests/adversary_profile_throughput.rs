//! Adversary case for story:profile-achieved-throughput: a time of 0 gives no rate. DESIGN.md
//! § 8 and the inspect-a-run guide say `gflop_per_s` and `gb_per_s` are `unavailable` for a
//! time of 0; no other case reports a site of 0 µs or a steady time of 0, so a report that
//! divides by it, printing `inf` or `NaN`, would pass every other case. Host only: the report is
//! built from hand-made timings, no device is opened.
#![cfg(feature = "vulkan")]

use std::time::Duration;

use modelspec_core::ir::kernel::{DispatchCount, KernelModuleId};
use modelspec_runtime::vulkan::negotiate::NumericsClass;
use modelspec_runtime::vulkan::profile::{PlanName, Profile, Site, SiteTime, Timings};
use modelspec_runtime::vulkan::runner::RunStats;

fn site(plan: PlanName, position: usize, count: DispatchCount) -> Site {
    Site {
        plan,
        position,
        kernel_module_id: KernelModuleId([position as u8; 16]),
        entry: "main".to_owned(),
        count,
    }
}

fn stats() -> RunStats {
    RunStats {
        steps: 30,
        host_waits: 1,
        submits: 1,
        dispatches: 61,
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
        pipelines: 6,
        pipeline_creation: Duration::from_micros(1500),
    }
}

/// A counted site whose every dispatch took 0 ns, a transpose (0 FLOP) of 0 ns, a site with no
/// calls, and a steady time of 0 µs per step: every rate is `unavailable`, none `inf` or `NaN`.
#[test]
fn a_time_of_zero_prints_unavailable_never_inf_or_nan() {
    let matmul = DispatchCount::Counted {
        flop: 1000,
        bytes: 400,
    };
    let transpose = DispatchCount::Counted { flop: 0, bytes: 64 };
    let profile = Profile {
        timings: Some(Timings {
            sites: vec![
                SiteTime {
                    site: site(PlanName::Step, 0, matmul),
                    calls: 30,
                    total_ns: 0.0,
                },
                SiteTime {
                    site: site(PlanName::Step, 1, transpose),
                    calls: 30,
                    total_ns: 0.0,
                },
                SiteTime {
                    site: site(PlanName::Final, 0, matmul),
                    calls: 0,
                    total_ns: 0.0,
                },
            ],
            steady_us_per_step: Some(0.0),
        }),
        pipeline_creation: Duration::from_micros(1),
        compile: Duration::from_micros(2),
        pipeline_statistics: modelspec_runtime::vulkan::pipeline::PipelineStatistics::NotCaptured,
    };
    let report = profile.report(&stats());
    assert!(
        !report.contains("inf") && !report.contains("NaN"),
        "a rate divided by 0:\n{report}"
    );
    let line = |head: &str| {
        report
            .lines()
            .find(|l| l.starts_with(head))
            .unwrap_or_else(|| panic!("no line {head}… in\n{report}"))
    };
    assert!(
        line("profile dispatch step 0 ").ends_with(
            " mean_us 0.000 flop 1000 bytes 400 gflop_per_s unavailable gb_per_s unavailable"
        ),
        "{report}"
    );
    assert!(
        line("profile dispatch step 1 ").ends_with(
            " mean_us 0.000 flop 0 bytes 64 gflop_per_s unavailable gb_per_s unavailable"
        ),
        "{report}"
    );
    assert!(
        line("profile dispatch final 0 ").ends_with(
            " calls 0 total_us 0.000 mean_us 0.000 flop 1000 bytes 400 gflop_per_s unavailable \
             gb_per_s unavailable"
        ),
        "{report}"
    );
    assert_eq!(
        line("profile step_total "),
        "profile step_total flop 1000 bytes 464 gflop_per_s unavailable gb_per_s unavailable \
         uncounted_sites 0",
        "{report}"
    );
}
