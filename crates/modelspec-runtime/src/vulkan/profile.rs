//! The profile of a device run (`train --profile`, DESIGN.md § 8): the GPU time of every
//! dispatch site, from timestamp queries, and the run's counters.
//!
//! A profiled run ([`super::runner::train_profiled`]) writes a timestamp before each dispatch
//! (top of pipe) and after it (bottom of pipe), for every step and for the final loss
//! evaluation, into query pools it reads back after its one host wait. [`timings`] sums the
//! stamps per dispatch site: one site per Dispatch step of the step plan, called once per
//! step, then one per Dispatch step the final loss evaluation records, called once. Two sites
//! that dispatch one module are two sites. Each site carries the FLOP and bytes of its plan
//! step ([`ExecutionPlan::dispatch_count`], DESIGN.md § 3.1). [`Profile::report`] prints the
//! sites with their GFLOP/s and GB/s over their mean time, the counters of the run's
//! [`RunStats`], and the step total over the steady time per step.

use std::fmt::{self, Write as _};
use std::time::Duration;

use modelspec_core::ir::kernel::{DispatchCount, ExecutionPlan, KernelModuleId};

use super::pipeline::PipelineStatistics;
use super::runner::RunStats;

/// The steps a profiled run counts as warm-up before its steady state.
pub const WARMUP_STEPS: u64 = 20;

/// The plan a dispatch site is in, as a report line names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanName {
    /// The step plan, recorded once per training step.
    Step,
    /// The final loss evaluation after the last step: the steps of the step plan the loss is
    /// computed from ([`modelspec_core::ir::kernel::ExecutionPlan::loss_steps`]).
    Final,
}

impl fmt::Display for PlanName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            PlanName::Step => "step",
            PlanName::Final => "final",
        })
    }
}

/// One Dispatch step of a plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site {
    pub plan: PlanName,
    /// The index of the step in the step plan.
    pub position: usize,
    pub kernel_module_id: KernelModuleId,
    /// The entry point of the kernel's SPIR-V module.
    pub entry: String,
    /// The FLOP and bytes of one dispatch of the step (DESIGN.md § 3.1, the counting rule).
    pub count: DispatchCount,
}

impl Site {
    /// The site of the Dispatch step at `position` of `execution`'s step plan, in `plan`, with
    /// its module id and count from the step; `None` when that step is not a Dispatch.
    pub fn new(
        execution: &ExecutionPlan,
        plan: PlanName,
        position: usize,
        entry: String,
    ) -> Option<Self> {
        let step = execution.step.get(position)?;
        Some(Site {
            plan,
            position,
            kernel_module_id: step.kernel_module_id?,
            entry,
            count: execution.dispatch_count(step)?,
        })
    }
}

/// The GPU time of one dispatch site over a run.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteTime {
    pub site: Site,
    /// The dispatches of the site the run recorded.
    pub calls: u64,
    /// The sum of the site's dispatch times, in nanoseconds.
    pub total_ns: f64,
}

impl SiteTime {
    /// [`SiteTime::total_ns`] in microseconds.
    pub fn total_us(&self) -> f64 {
        self.total_ns / 1e3
    }

    /// The mean time of one dispatch of the site in microseconds.
    pub fn mean_us(&self) -> f64 {
        if self.calls == 0 {
            return 0.0;
        }
        self.total_ns / self.calls as f64 / 1e3
    }
}

/// The timestamp counter of the device's queue: its valid bits and the nanoseconds of a tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clock {
    /// `timestampValidBits` of the queue family; the bits above them are undefined.
    pub valid_bits: u32,
    /// `timestampPeriod`: the nanoseconds one tick takes.
    pub period_ns: f32,
}

impl Clock {
    /// The nanoseconds from `start` to `end`, across at most one wrap of the valid bits.
    fn span_ns(&self, start: u64, end: u64) -> f64 {
        let mask = match self.valid_bits {
            0 | 64.. => u64::MAX,
            bits => (1u64 << bits) - 1,
        };
        let ticks = (end & mask).wrapping_sub(start & mask) & mask;
        ticks as f64 * f64::from(self.period_ns)
    }
}

/// The per-site times of a run, and its steady-state time per step.
#[derive(Clone, Debug, PartialEq)]
pub struct Timings {
    /// The step sites in plan order, then the final loss evaluation's.
    pub sites: Vec<SiteTime>,
    /// From the first dispatch of step [`WARMUP_STEPS`] to the first of the final loss
    /// evaluation, over the steps between, in microseconds; `None` for a run of no more than
    /// [`WARMUP_STEPS`] steps.
    pub steady_us_per_step: Option<f64>,
}

/// Sums the `stamps` of a run of `steps` steps per site. The stamps are `(start, end)` pairs in
/// recording order: `step_sites` for each step, then `final_sites` once.
pub fn timings(
    step_sites: &[Site],
    final_sites: &[Site],
    steps: u64,
    stamps: &[u64],
    clock: Clock,
) -> Result<Timings, String> {
    let per_step = step_sites.len() as u64;
    let pairs = steps
        .checked_mul(per_step)
        .and_then(|n| n.checked_add(final_sites.len() as u64))
        .ok_or_else(|| format!("{steps} steps of {per_step} dispatches overflow"))?;
    if stamps.len() as u64 != 2 * pairs {
        return Err(format!(
            "{} timestamps for {pairs} dispatches, which need {}",
            stamps.len(),
            2 * pairs
        ));
    }
    let span = |pair: usize| clock.span_ns(stamps[2 * pair], stamps[2 * pair + 1]);
    let mut sites: Vec<SiteTime> = step_sites
        .iter()
        .chain(final_sites)
        .map(|site| SiteTime {
            site: site.clone(),
            calls: 0,
            total_ns: 0.0,
        })
        .collect();
    let stepped = (steps * per_step) as usize;
    for pair in 0..stepped {
        let site = &mut sites[pair % step_sites.len()];
        site.calls += 1;
        site.total_ns += span(pair);
    }
    for (index, site) in sites[step_sites.len()..].iter_mut().enumerate() {
        site.calls += 1;
        site.total_ns += span(stepped + index);
    }
    let steady_us_per_step = (steps > WARMUP_STEPS && per_step > 0).then(|| {
        let from = stamps[2 * (WARMUP_STEPS * per_step) as usize];
        let to = if final_sites.is_empty() {
            stamps[2 * stepped - 1]
        } else {
            stamps[2 * stepped]
        };
        clock.span_ns(from, to) / (steps - WARMUP_STEPS) as f64 / 1e3
    });
    Ok(Timings {
        sites,
        steady_us_per_step,
    })
}

/// What `train --profile` prints for a device run.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    /// The per-site times; `None` on a device without timestamp queries.
    pub timings: Option<Timings>,
    /// Creating the compute pipeline of every kernel.
    pub pipeline_creation: Duration,
    /// Compiling the kernels: the runner's SPIR-V emission, plus what the caller adds for the
    /// compile before it (the CLI adds the compile of the file and of its artifact).
    pub compile: Duration,
    /// What the run read of the statistics of its pipelines' executables, per kernel of the plan
    /// in plan order ([`PipelineStatistics`]): not captured on a device opened without
    /// `VK_KHR_pipeline_executable_properties`, and the reason when a query failed.
    pub pipeline_statistics: PipelineStatistics,
}

/// `flop <f> bytes <b> gflop_per_s <g> gb_per_s <h>` for the `(flop, bytes)` of `count`, the
/// rates being the counts over `us` microseconds, or `unavailable` without a positive time;
/// `uncounted` without a count.
fn write_throughput(out: &mut String, count: Option<(u128, u128)>, us: Option<f64>) {
    let Some((flop, bytes)) = count else {
        out.push_str("uncounted");
        return;
    };
    let _ = write!(out, "flop {flop} bytes {bytes}");
    for (label, units) in [("gflop_per_s", flop), ("gb_per_s", bytes)] {
        match us.filter(|us| *us > 0.0) {
            Some(us) => {
                let _ = write!(out, " {label} {:.3}", units as f64 / us / 1e3);
            }
            None => {
                let _ = write!(out, " {label} unavailable");
            }
        }
    }
}

impl Profile {
    /// The report lines: one `profile dispatch <plan> <position> <kernel_module_id> <entry>
    /// calls <n> total_us <t> mean_us <m> <throughput>` line per site, `<throughput>` being
    /// `flop <f> bytes <b> gflop_per_s <g> gb_per_s <h>` over the mean time, or `uncounted`; or
    /// one line saying the device has no timestamp queries. Then one `profile <counter> <value>`
    /// line per counter of `stats` and of the profile, and last `profile step_total flop <f>
    /// bytes <b> gflop_per_s <g> gb_per_s <h> uncounted_sites <n>`: the counts of the counted
    /// step sites over the steady time per step, and the step sites left out.
    pub fn report(&self, stats: &RunStats) -> String {
        let mut out = String::new();
        match &self.timings {
            Some(timings) => {
                for time in &timings.sites {
                    let site = &time.site;
                    let _ = write!(
                        out,
                        "profile dispatch {} {} {} {} calls {} total_us {:.3} mean_us {:.3} ",
                        site.plan,
                        site.position,
                        site.kernel_module_id,
                        site.entry,
                        time.calls,
                        time.total_us(),
                        time.mean_us()
                    );
                    let count = match site.count {
                        DispatchCount::Counted { flop, bytes } => Some((flop.into(), bytes.into())),
                        DispatchCount::Uncounted => None,
                    };
                    write_throughput(&mut out, count, Some(time.mean_us()));
                    out.push('\n');
                }
            }
            None => out.push_str("profile kernel timings unavailable: no timestamp queries\n"),
        }
        match &self.pipeline_statistics {
            PipelineStatistics::Reported(kernels) => {
                for kernel in kernels {
                    let id = kernel.kernel_module_id;
                    for (index, executable) in kernel.executables.iter().enumerate() {
                        let _ = writeln!(
                            out,
                            "profile pipeline {id} executable {index} name {:?} subgroup_size {} \
                             description {:?}",
                            executable.name, executable.subgroup_size, executable.description
                        );
                        for statistic in &executable.statistics {
                            let _ = writeln!(
                                out,
                                "profile pipeline {id} executable {index} statistic {:?} {} \
                                 description {:?}",
                                statistic.name, statistic.value, statistic.description
                            );
                        }
                    }
                }
            }
            PipelineStatistics::NotCaptured => {
                out.push_str("profile pipeline statistics: not reported\n");
            }
            PipelineStatistics::Failed(reason) => {
                let _ = writeln!(out, "profile pipeline statistics: not reported ({reason})");
            }
        }
        let us = |d: Duration| d.as_secs_f64() * 1e6;
        let _ = writeln!(out, "profile dispatches {}", stats.dispatches);
        let _ = writeln!(out, "profile submits {}", stats.submits);
        let _ = writeln!(out, "profile host_waits {}", stats.host_waits);
        let _ = writeln!(out, "profile barriers_per_step {}", stats.barriers_per_step);
        let _ = writeln!(out, "profile bytes_uploaded {}", stats.bytes_uploaded);
        let _ = writeln!(out, "profile bytes_downloaded {}", stats.bytes_downloaded);
        let _ = writeln!(
            out,
            "profile device_allocations {}",
            stats.device_allocations
        );
        let _ = writeln!(out, "profile peak_device_bytes {}", stats.peak_device_bytes);
        let _ = writeln!(
            out,
            "profile pipeline_creation_us {:.3}",
            us(self.pipeline_creation)
        );
        let _ = writeln!(out, "profile compile_us {:.3}", us(self.compile));
        match self.timings.as_ref().and_then(|t| t.steady_us_per_step) {
            Some(steady) => {
                let _ = writeln!(out, "profile steady_us_per_step {steady:.3}");
            }
            None => out.push_str("profile steady_us_per_step unavailable\n"),
        }
        match &self.timings {
            Some(timings) => {
                let (mut flop, mut bytes, mut uncounted) = (0u128, 0u128, 0usize);
                for site in timings.sites.iter().map(|time| &time.site) {
                    match (site.plan, site.count) {
                        (PlanName::Step, DispatchCount::Counted { flop: f, bytes: b }) => {
                            flop += u128::from(f);
                            bytes += u128::from(b);
                        }
                        (PlanName::Step, DispatchCount::Uncounted) => uncounted += 1,
                        (PlanName::Final, _) => {}
                    }
                }
                out.push_str("profile step_total ");
                write_throughput(&mut out, Some((flop, bytes)), timings.steady_us_per_step);
                let _ = writeln!(out, " uncounted_sites {uncounted}");
            }
            None => out.push_str("profile step_total unavailable\n"),
        }
        out
    }
}
