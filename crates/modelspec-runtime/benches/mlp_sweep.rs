//! The MLP size sweep on one Vulkan device (`story:mlp-size-sweep-vs-pytorch-cuda`): 784 -> H
//! ReLU -> 10, no bias, `f32`, `mse`, SGD, at hidden widths 64, 256, 1024, 4096 and batches 100,
//! 1000, 8192. Each point's source is formatted here ([`sweep::Point::source`]) and compiled; no
//! `.mspec` file is read. The data is the runner's synthetic teacher-student batch.
//!
//! Per point: a [`sweep::WARMUP_STEPS`]-step warm-up run, calibration runs until one lasts at
//! least `--min-seconds`, then [`sweep::RUNS`] timed runs of that many steps ([`runs::measure`]).
//! A run's time is `RunStats::elapsed`, host wall time from the first submit to the end of the
//! one host wait: compilation, SPIR-V emission, pipeline creation, buffer allocation and the
//! parameter draws run before it. The span holds the one upload of the parameters and the batch,
//! the final loss evaluation and the read-back of the parameters and losses, once per run.
//!
//! The GPU between runs: all runs of a point are one series on one setup
//! (`vulkan::runner::train_repeatedly`). The point is compiled once, at the warm-up's step count,
//! and its batch, its CPU teacher forward, the parameter draws, the SPIR-V, the pipelines, the
//! buffers and the upload staging buffer are made once, before the warm-up; each later run is
//! given its step count, and starts from the same initial parameters and batch, copied up inside
//! its span as before. No extra GPU warm-up runs before a timed run. What runs on the CPU between
//! two runs, with the GPU idle, is the read-back and check of one run, the next run's losses and
//! read-back buffers and the recording of its first command buffer: 26 ms between two timed
//! spans at batch 8192, hidden 4096, where a whole setup per run took about 150 ms, enough for
//! the driver to lower the clock 0.3 s into every run and take two to three seconds to raise it
//! (story:sweep-run-starts-busy, `benchmarks/raw/2026-10-25-sweep-idle/diagnosis.md`). The
//! PyTorch side runs its timed runs back to back, a synchronize between them.
//!
//! `span_unix_ms` is not read from the span itself: `RunStats` has its length but not its start.
//! A run's span is written as `[return − elapsed, return]`, `return` being the wall clock when
//! the series handed the finished run to the harness, so both ends sit later than the span by
//! what follows the last wait: the host read of the downloaded values and the divergence check.
//!
//! Each point prints one JSON line on stdout ([`sweep::PointResult`]): FLOP per step
//! (matmul-only, `3196·B·H`), µs per step of each timed run, their median and GFLOP/s, and the
//! build it ran from: the package version, the git commit of the checkout and whether its
//! tracked files differed from that commit (`git status --porcelain --untracked-files=no`,
//! read when the sweep starts; left out where git does not answer). A point that does not
//! compile or run prints its refusal and the sweep goes on. Progress goes to stderr. Every point is
//! scheduled within the workgroup invocations and workgroup memory of the device it runs on
//! (`compile_artifact_for`), as `train --backend vulkan` schedules it.
//!
//! `--fast-math` compiles every point with `fast_math` set (DESIGN.md § 12 ADR-008) and writes
//! its lines as the side `modelspec-vulkan-fast-math`, a figure reported beside the
//! deterministic `modelspec-vulkan` one, never as the headline.
//!
//! `--crossover <file>...` reads result lines of any side (this bench's and those of
//! `opponents/mlp_sweep/mlp_sweep.py`) and prints, per side and batch, the first width whose
//! median µs per step is more than 10 % above the previous width's ([`sweep::crossover_lines`]),
//! without opening a device.
//!
//! Run it with
//! `cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep > modelspec.jsonl`;
//! it is not part of `task ci`.

#[path = "mlp_sweep/sweep.rs"]
mod sweep;

#[path = "mlp_sweep/runs.rs"]
mod runs;

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

use clap::Parser;
use modelspec_runtime::vulkan::Vulkan;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::select::select_device;
use runs::{On, measure};
use sweep::{Build, Point, PointResult};

/// The side this bench writes.
const SIDE: &str = "modelspec-vulkan";

/// The side this bench writes with `--fast-math`: a labelled figure beside the deterministic one,
/// never the headline (DESIGN.md § 12 ADR-008).
const SIDE_FAST_MATH: &str = "modelspec-vulkan-fast-math";

/// The MLP size sweep on one Vulkan device, or the crossover of result files.
#[derive(Parser)]
#[command(name = "mlp_sweep")]
struct Cli {
    /// The Vulkan device to run on.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// The shortest timed run, in seconds; calibration aims 20 % above it.
    #[arg(long, default_value_t = 5.0)]
    min_seconds: f64,
    /// Run only this point, `BATCH:HIDDEN`; repeatable. Default: all 12.
    #[arg(long = "point", value_parser = parse_point)]
    points: Vec<Point>,
    /// Print the crossover lines of these result files instead of running; opens no device.
    #[arg(long, num_args = 1.., value_name = "RESULTS")]
    crossover: Vec<PathBuf>,
    /// Compile every point with fast_math (DESIGN.md ADR-008) and write the side
    /// `modelspec-vulkan-fast-math`.
    #[arg(long)]
    fast_math: bool,
    /// Passed by `cargo bench`; ignored.
    #[arg(long, hide = true)]
    bench: bool,
}

fn parse_point(text: &str) -> Result<Point, String> {
    let (batch, hidden) = text
        .split_once(':')
        .ok_or_else(|| format!("`{text}` is not BATCH:HIDDEN"))?;
    let number = |part: &str| {
        part.parse::<u64>()
            .map_err(|error| format!("`{part}` in `{text}`: {error}"))
    };
    Ok(Point {
        batch: number(batch)?,
        hidden: number(hidden)?,
    })
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if !cli.crossover.is_empty() {
        return crossover(&cli.crossover);
    }
    let points = if cli.points.is_empty() {
        sweep::points()
    } else {
        cli.points.clone()
    };
    sweep_on_device(cli.device, cli.min_seconds, cli.fast_math, &points)
}

/// Prints the crossover lines of the result lines in `files`.
fn crossover(files: &[PathBuf]) -> ExitCode {
    let mut results = Vec::new();
    for file in files {
        let text = match std::fs::read_to_string(file) {
            Ok(text) => text,
            Err(error) => {
                eprintln!("mlp_sweep: {}: {error}", file.display());
                return ExitCode::FAILURE;
            }
        };
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            match PointResult::from_line(line) {
                Ok(result) => results.push(result),
                Err(error) => {
                    eprintln!("mlp_sweep: {}:{}: {error}", file.display(), index + 1);
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    for line in sweep::crossover_lines(&results) {
        println!("{}", line.to_line());
    }
    ExitCode::SUCCESS
}

/// Opens device `index` and runs every point of `points` on it.
fn sweep_on_device(index: usize, min_seconds: f64, fast_math: bool, points: &[Point]) -> ExitCode {
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
            eprintln!("mlp_sweep: {diagnostic}");
            return ExitCode::FAILURE;
        }
    };
    let device = match vulkan.create_device(index, &negotiated) {
        Ok(device) => device,
        Err(diagnostic) => {
            eprintln!("mlp_sweep: {diagnostic}");
            return ExitCode::FAILURE;
        }
    };
    let named = format!(
        "vulkan {:04x}:{:04x} driver_version {}",
        profile.vendor_id, profile.device_id, profile.driver_version
    );
    let build = build();
    if build.dirty != Some(false) {
        eprintln!("mlp_sweep: warning: not a clean checkout: {build:?}");
    }
    let on = On {
        device: &device,
        profile: &profile,
        negotiated: &negotiated,
        fast_math,
    };
    let side = if fast_math { SIDE_FAST_MATH } else { SIDE };
    let mut stdout = std::io::stdout();
    for &point in points {
        eprintln!(
            "mlp_sweep: batch {} hidden {}: {} FLOP per step",
            point.batch,
            point.hidden,
            point.flop_per_step()
        );
        let result = match measure(&on, point, min_seconds) {
            Ok((steps, runs)) => {
                PointResult::measured(side, point, steps, runs, Some(named.clone()))
            }
            Err(refusal) => {
                eprintln!("mlp_sweep: refused: {refusal}");
                PointResult::refused(side, point, refusal, Some(named.clone()))
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

/// The package version and the git commit and state of the checkout this bench was built from,
/// read when the sweep starts. `cargo bench` rebuilds a changed tree before it runs the bench, so
/// there the checkout's state is the build's.
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
