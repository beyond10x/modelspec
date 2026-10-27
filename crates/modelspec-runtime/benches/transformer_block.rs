//! The transformer-block benchmark on one Vulkan device (story:transformer-block-benchmark): two
//! BERT-tiny encoder layers (hidden 128, 2 heads, feed-forward 512, sequences of 128, post
//! layer-norm, no dropout), an MSE loss and SGD, at batches of 8, 32 and 128 sequences. Each
//! point runs `examples/transformer_block.mspec` itself, read into the binary, with its batch
//! substituted ([`block::Point::source`]); the data is the runner's synthetic batch, `x` drawn,
//! the key mask all ones and the target the teacher's prediction.
//!
//! The run discipline is the MLP sweep's (`benches/mlp_sweep.rs`), unchanged: per point a
//! [`sweep::WARMUP_STEPS`]-step warm-up run, calibration runs until one lasts at least
//! `--min-seconds`, then [`sweep::RUNS`] timed runs of that many steps, all one series on one
//! setup (`vulkan::runner::train_repeatedly`), a run's time `RunStats::elapsed` and its span
//! `[return − elapsed, return]`. The point is compiled once, at the warm-up's step count.
//!
//! Each point prints one JSON line on stdout ([`block::BlockResult`]): the batch, the sequence
//! length, tokens and matmul-only FLOP per step (`339,738,624·B`, the closed form in
//! `benches/transformer_block/block.rs`), the device, the build it ran from, the timed runs,
//! their median µs per step, tokens per second and GFLOP/s. A point that does not compile or
//! run prints its refusal and the bench goes on. Progress goes to stderr. The opponent
//! `opponents/transformer_block/transformer_block.py` writes the same lines for PyTorch.
//!
//! Run it with
//! `cargo bench -p modelspec-runtime --features vulkan --bench transformer_block > modelspec.jsonl`;
//! it is not part of `task ci`.

#[allow(dead_code)]
#[path = "mlp_sweep/sweep.rs"]
mod sweep;

#[allow(dead_code)]
#[path = "transformer_block/block.rs"]
mod block;

use std::io::Write;
use std::process::{Command, ExitCode};
use std::time::{SystemTime, UNIX_EPOCH};

use block::{BlockResult, Point};
use clap::Parser;
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{CompiledTraining, compile_artifact_for, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::vulkan::negotiate::{Negotiated, negotiate};
use modelspec_runtime::vulkan::runner::train_repeatedly;
use modelspec_runtime::vulkan::select::select_device;
use modelspec_runtime::vulkan::{Device, Vulkan};
use sweep::{Build, TimedRun};

/// The side this bench writes.
const SIDE: &str = "modelspec-vulkan";

/// The calibration runs a point gets before its step count is taken as it stands.
const CALIBRATIONS: usize = 4;

/// The transformer-block benchmark on one Vulkan device.
#[derive(Parser)]
#[command(name = "transformer_block")]
struct Cli {
    /// The Vulkan device to run on.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// The shortest timed run, in seconds; calibration aims 20 % above it.
    #[arg(long, default_value_t = 5.0)]
    min_seconds: f64,
    /// Run only this batch, in sequences of 128; repeatable. Default: 8, 32 and 128.
    #[arg(long = "point", value_name = "BATCH", value_parser = clap::value_parser!(u64).range(1..))]
    points: Vec<u64>,
    /// Passed by `cargo bench`; ignored.
    #[arg(long, hide = true)]
    bench: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let points = if cli.points.is_empty() {
        block::points()
    } else {
        cli.points.iter().map(|&batch| Point { batch }).collect()
    };
    run_on_device(cli.device, cli.min_seconds, &points)
}

/// The opened device a point runs on.
struct On<'d> {
    device: &'d Device<'d>,
    profile: &'d DeviceProfile,
    negotiated: &'d Negotiated,
}

/// Opens device `index` and runs every point of `points` on it.
fn run_on_device(index: usize, min_seconds: f64, points: &[Point]) -> ExitCode {
    let opened = (|| {
        let vulkan = Vulkan::load()?;
        let candidates = vulkan.candidates()?;
        let selected = select_device(&candidates, Some(index))?[0];
        let profile = candidates[selected].profile.clone();
        let negotiated = negotiate(&profile)?;
        Ok::<_, modelspec_core::diag::Diagnostic>((vulkan, profile, negotiated))
    })();
    let (vulkan, profile, negotiated) = match opened {
        Ok(opened) => opened,
        Err(diagnostic) => {
            eprintln!("transformer_block: {diagnostic}");
            return ExitCode::FAILURE;
        }
    };
    let device = match vulkan.create_device(index, &negotiated) {
        Ok(device) => device,
        Err(diagnostic) => {
            eprintln!("transformer_block: {diagnostic}");
            return ExitCode::FAILURE;
        }
    };
    let named = format!(
        "vulkan {:04x}:{:04x} driver_version {}",
        profile.vendor_id, profile.device_id, profile.driver_version
    );
    let build = build();
    if build.dirty != Some(false) {
        eprintln!("transformer_block: warning: not a clean checkout: {build:?}");
    }
    let on = On {
        device: &device,
        profile: &profile,
        negotiated: &negotiated,
    };
    let mut stdout = std::io::stdout();
    for &point in points {
        eprintln!(
            "transformer_block: batch {} ({} tokens): {} FLOP per step",
            point.batch,
            point.tokens_per_step(),
            point.flop_per_step()
        );
        let result = match measure(&on, point, min_seconds) {
            Ok((steps, runs)) => {
                BlockResult::measured(SIDE, point, steps, runs, Some(named.clone()))
            }
            Err(refusal) => {
                eprintln!("transformer_block: refused: {refusal}");
                BlockResult::refused(SIDE, point, refusal, Some(named.clone()))
            }
        }
        .with_build(build.clone());
        if writeln!(stdout, "{}", result.to_line())
            .and_then(|()| stdout.flush())
            .is_err()
        {
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

/// The point's source compiled for training, its step count set to the warm-up's, and for
/// Vulkan, its plan scheduled within the limits of the device `profile` as
/// `train --backend vulkan` schedules it. The plan does not depend on the step count
/// (`tests/transformer_block.rs`).
fn compile(
    point: Point,
    profile: &DeviceProfile,
) -> Result<(CompiledTraining, CompiledArtifact), String> {
    let name = point.file_name();
    let source = point.source()?;
    let refusal = |diagnostics: Vec<modelspec_core::diag::Diagnostic>| {
        diagnostics
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut compiled = compile_training(&name, &source).map_err(refusal)?;
    compiled.config.steps = sweep::WARMUP_STEPS;
    let artifact =
        compile_artifact_for(&name, &source, Backend::Vulkan, Some(profile)).map_err(refusal)?;
    Ok((compiled, artifact))
}

/// Where a point's series of runs is.
#[derive(Clone, Copy)]
enum Phase {
    Warmup,
    Calibration(usize),
    Timed,
}

/// The warm-up, the calibration and the timed runs of `point`: the steps of each timed run and
/// the runs, as the MLP sweep measures a point (`benches/mlp_sweep/runs.rs`).
fn measure(on: &On<'_>, point: Point, min_seconds: f64) -> Result<(u64, Vec<TimedRun>), String> {
    let target = min_seconds * 1.2;
    let (compiled, artifact) = compile(point, on.profile)?;
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
                    eprintln!("transformer_block:   warm-up: {steps} steps in {seconds:.3} s");
                    steps = sweep::steps_for(seconds, stats.steps, target);
                    phase = Phase::Calibration(1);
                }
                Phase::Calibration(made) => {
                    eprintln!("transformer_block:   calibration: {steps} steps in {seconds:.3} s");
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
                        "transformer_block:   timed: {steps} steps in {seconds:.3} s, {:.3} us per step",
                        stats.us_per_step()
                    );
                    if seconds < min_seconds {
                        eprintln!(
                            "transformer_block:   warning: the run was shorter than {min_seconds} s"
                        );
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

/// The package version and the git commit and state of the checkout this bench was built from,
/// read when the bench starts.
fn build() -> Build {
    let git = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(env!("CARGO_MANIFEST_DIR"))
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
    };
    Build::from_git(
        Some(env!("CARGO_PKG_VERSION")),
        git(&["rev-parse", "HEAD"]).as_deref(),
        git(&["status", "--porcelain", "--untracked-files=no"]).as_deref(),
    )
}
