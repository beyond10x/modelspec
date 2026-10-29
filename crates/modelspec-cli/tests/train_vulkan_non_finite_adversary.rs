//! Adversary cases for story:train-vulkan-refuses-non-finite-loss: the windows of losses a
//! `--checkpoint-every` stop checks, the step it names, the checkpoints written before it, chunk
//! boundaries of `CHUNK_STEPS = 256`, resumed runs, the final loss, and the `--serve` stream.
//!
//! Every case runs one file on `--backend cpu` and on `--backend vulkan` and compares the exit
//! status, stdout, stderr and the names of the checkpoint files written, each finite printed loss
//! within the run tolerance of DESIGN.md § 9 gate 5 and the rest byte for byte. The fixtures are
//! the linear regression example, whose device run updates the parameters bit for bit as the CPU
//! run does, at learning rates whose divergence the device can name at the CPU's step: the loss
//! overflows while every parameter is finite (`lr = 100.0` at step 8, [`PAST_CHUNK`] at step
//! 372, [`PAST_CHUNK_8`] at step 382), or the parameter overflows in the update of a step whose
//! parameters the device reads back (`lr = 1e38` with one step, or a checkpoint after every
//! step).
//!
//! Since release 0.21.0 the CPU sums an `mse` loss in `f64` and divides it there, while the
//! device sums in `f32` (DESIGN.md § 5.4, coordinator decision D84, commit a123c1a5), so the
//! printed losses differ in their last bits. Until story:vulkan-loss-sum-overflow-parity the
//! device divided its `f32` sum by the element count afterwards, so over the example's 8 rows the
//! sum overflowed before the CPU's mean did: at `lr = 0.6`, which overflowed at step 372 on both
//! before 0.21.0, the CPU named step 382 and the device 372 (RTX 3090). The device now also sums
//! each difference times the reciprocal of the count times the difference, the mean, and writes
//! it where its plain sum overflows, so its loss overflows where the CPU's does; divergence past
//! the first chunk is tested on the 8 rows at `lr = 0.6` ([`PAST_CHUNK_8`]) and on one row,
//! whose sum is its mean ([`PAST_CHUNK`]).
//!
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-cli --test train_vulkan_non_finite_adversary -- --ignored`
#![cfg(feature = "vulkan")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Duration;

use serde_json::Value;

#[allow(dead_code)]
#[path = "../../modelspec-runtime/tests/common/mod.rs"]
mod common;

use common::{GPU_RUN_ATOL, GPU_RUN_RTOL, within};

const EXAMPLE: &str = "examples/linear_regression.mspec";

/// A fixture: the example at learning rate `lr` with `rows` rows in its input and target.
#[derive(Clone, Copy, Debug)]
struct Fit {
    lr: &'static str,
    rows: u32,
}

/// The example's 8 rows at `lr = 100.0`: the loss of step 8 overflows.
const LR_100: Fit = Fit {
    lr: "100.0",
    rows: 8,
};

/// The example's 8 rows at `lr = 1e38`: the update of step 0 overflows `LinearReg.w`.
const LR_1E38: Fit = Fit {
    lr: "1e38",
    rows: 8,
};

/// One row at `lr = 0.17723`: the loss of step 372 overflows on the CPU and the RTX 3090, past
/// the first chunk of 256 steps, as the 8 rows at `lr = 0.6` did before release 0.21.0 (module
/// docs). The rate is the one in `0.1772` to `0.1773` whose loss overflows at step 372, so the
/// windows below keep their steps; with one row the CPU's `f64` mean is the `f32` sum the
/// device makes, up to the rounding of the difference.
const PAST_CHUNK: Fit = Fit {
    lr: "0.17723",
    rows: 1,
};

/// The example's 8 rows at `lr = 0.6`: the CPU's `f64` mean of the squared differences passes
/// `f32::MAX` at step 382, past the first chunk of 256 steps, and the device's sum of the
/// squared differences scaled by `1/8`, which it writes where its plain sum overflows, with it
/// (story:vulkan-loss-sum-overflow-parity).
const PAST_CHUNK_8: Fit = Fit { lr: "0.6", rows: 8 };

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch() -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join("train_vulkan_non_finite_adversary")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The example with its learning rate replaced by `fit.lr` and its 8 rows by `fit.rows`, written
/// once per fixture.
fn with_lr(fit: Fit) -> String {
    let Fit { lr, rows } = fit;
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example exists");
    let mut changed = source.replace("sgd(lr = 0.1)", &format!("sgd(lr = {lr})"));
    assert_ne!(
        changed, source,
        "the example's optimizer is `sgd(lr = 0.1)`"
    );
    if rows != 8 {
        let before = changed.clone();
        changed = changed
            .replace("[8, 4]", &format!("[{rows}, 4]"))
            .replace("[8, 1]", &format!("[{rows}, 1]"));
        assert_ne!(changed, before, "the example has 8 rows of `x` and `y`");
    }
    std::fs::create_dir_all(scratch()).expect("the temp dir is writable");
    let path = scratch().join(format!("r{rows}-lr-{lr}.mspec"));
    // Tests run in parallel and two write the same fixture: write a file of this thread's own and
    // rename it, so no reader sees the truncated file.
    let staged = scratch().join(format!(
        "r{rows}-lr-{lr}.{:?}.tmp",
        std::thread::current().id()
    ));
    std::fs::write(&staged, changed).expect("the source is written");
    std::fs::rename(&staged, &path).expect("the source is moved into place");
    path.display().to_string()
}

/// One `train` run: status, stdout, stderr and the sorted names of the files in its own
/// checkpoint directory.
#[derive(Debug, PartialEq)]
struct Outcome {
    status: Option<i32>,
    stdout: String,
    stderr: String,
    files: Vec<String>,
}

/// Runs `train <file> --backend <backend> <extra...>` with `--checkpoint <dir>/run.msckpt` when
/// `checkpoint`, in a fresh directory named by `label` and `backend`.
fn run(label: &str, file: &str, backend: &str, checkpoint: bool, extra: &[String]) -> Outcome {
    let dir = scratch().join(format!("{label}-{backend}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let mut command = Command::new(env!("CARGO_BIN_EXE_modelspec"));
    command
        .args(["train", file, "--backend", backend])
        .args(extra)
        .current_dir(repo_root());
    if checkpoint {
        command
            .arg("--checkpoint")
            .arg(dir.join("run.msckpt").display().to_string());
    }
    let output: Output = command.output().expect("modelspec runs");
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .expect("the dir exists")
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    Outcome {
        status: output.status.code(),
        stdout: text(&output.stdout),
        stderr: text(&output.stderr),
        files,
    }
}

/// Whether `g` passes against the CPU's `c` with the run tolerance
/// `GPU_RUN_ATOL + GPU_RUN_RTOL·|c|` of DESIGN.md § 9 gate 5, a non-finite `c` only by `g`'s
/// own bits.
fn loss_as_the_cpu(g: f32, c: f32) -> bool {
    if c.is_finite() {
        within(g, c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL)
    } else {
        g.to_bits() == c.to_bits()
    }
}

/// A `step <k> loss <v>` line split at its loss: the text before it, the value, and the text
/// after it.
fn loss_line(line: &str) -> Option<(&str, f32, &str)> {
    let (head, tail) = line.split_once(" loss ")?;
    if !head.starts_with("step ") {
        return None;
    }
    let (value, rest) = tail.split_once(' ').unwrap_or((tail, ""));
    Some((head, value.parse().ok()?, rest))
}

/// Whether the device run's stdout `g` is the CPU run's `c` as DESIGN.md § 8 holds it: the same
/// lines, each loss of a `step` line by [`loss_as_the_cpu`] and every other line, and the rest of
/// a `step` line, byte for byte (coordinator decisions D84, D88).
fn stdout_as_the_cpu(g: &str, c: &str) -> bool {
    let (g, c): (Vec<&str>, Vec<&str>) = (g.lines().collect(), c.lines().collect());
    g.len() == c.len()
        && g.iter()
            .zip(&c)
            .all(|(g, c)| match (loss_line(g), loss_line(c)) {
                (Some((g_head, g_loss, g_rest)), Some((c_head, c_loss, c_rest))) => {
                    (g_head, g_rest) == (c_head, c_rest) && loss_as_the_cpu(g_loss, c_loss)
                }
                _ => g == c,
            })
}

/// Whether the device run's [`Outcome`] is the CPU run's: status, stderr and checkpoint files
/// equal, stdout by [`stdout_as_the_cpu`].
fn outcome_as_the_cpu(g: &Outcome, c: &Outcome) -> bool {
    (g.status, &g.stderr, &g.files) == (c.status, &c.stderr, &c.files)
        && stdout_as_the_cpu(&g.stdout, &c.stdout)
}

/// Whether the device run's event body `g` is the CPU run's `c`: every field equal except the
/// loss of a `StepCompleted`, which is `null` where the CPU's is and otherwise passes
/// [`loss_as_the_cpu`].
fn body_as_the_cpu(g: &Value, c: &Value) -> bool {
    if g["kind"] != "StepCompleted" || c["kind"] != "StepCompleted" {
        return g == c;
    }
    let (mut g_rest, mut c_rest) = (g.clone(), c.clone());
    g_rest["value"]["loss"] = Value::Null;
    c_rest["value"]["loss"] = Value::Null;
    if g_rest != c_rest {
        return false;
    }
    match (g["value"]["loss"].as_f64(), c["value"]["loss"].as_f64()) {
        (Some(g), Some(c)) => loss_as_the_cpu(g as f32, c as f32),
        (None, None) => g["value"]["loss"] == c["value"]["loss"],
        _ => false,
    }
}

/// The E-TRAIN-001 line of a stderr, or the first line.
fn headline(stderr: &str) -> &str {
    stderr
        .lines()
        .find(|line| line.starts_with("E-TRAIN-001"))
        .or_else(|| stderr.lines().next())
        .unwrap_or("")
}

/// Runs each case on both backends and returns one line per case whose device run differs from
/// the CPU run, after checking the CPU run is the divergence the case was built for.
fn mismatches(cases: &[(&str, Fit, bool, Vec<String>, &str)]) -> Vec<String> {
    let mut found = Vec::new();
    for (label, fit, checkpoint, extra, cpu_names) in cases {
        let file = with_lr(*fit);
        let cpu = run(label, &file, "cpu", *checkpoint, extra);
        assert_eq!(
            cpu.status,
            Some(1),
            "{label}: the CPU run diverges: {}",
            cpu.stderr
        );
        assert!(
            headline(&cpu.stderr).starts_with(cpu_names),
            "{label}: the CPU run names {cpu_names}: {}",
            cpu.stderr
        );
        let gpu = run(label, &file, "vulkan", *checkpoint, extra);
        if !outcome_as_the_cpu(&gpu, &cpu) {
            found.push(format!(
                "{label} ({fit:?}, {extra:?}): cpu [{:?}, {}, files {:?}] vulkan [{:?}, {}, \
                 files {:?}], stdout equal: {}",
                cpu.status,
                headline(&cpu.stderr),
                cpu.files,
                gpu.status,
                headline(&gpu.stderr),
                gpu.files,
                gpu.stdout == cpu.stdout,
            ));
        }
    }
    found
}

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| (*s).to_owned()).collect()
}

const LOSS_8: &str = "E-TRAIN-001: the loss of step 8 is not finite";
const LOSS_372: &str = "E-TRAIN-001: the loss of step 372 is not finite";
const LOSS_382: &str = "E-TRAIN-001: the loss of step 382 is not finite";
const UPDATE_0: &str = "E-TRAIN-001: the update of step 0 leaves a parameter not finite";

/// The windows of losses each `--checkpoint-every` stop checks, around the failing step 8 of
/// `lr = 100.0`: a stop before, at and after it, every step a stop, and the failing loss as the
/// final loss of the run (`--steps 8`) with and without a stop before it.
#[test]
#[ignore = "needs a Vulkan device"]
fn the_windows_around_a_divergent_loss_name_the_cpu_step_and_write_the_cpu_checkpoints() {
    let cases = vec![
        (
            "every-1",
            LR_100,
            true,
            args(&["--checkpoint-every", "1"]),
            LOSS_8,
        ),
        (
            "every-2",
            LR_100,
            true,
            args(&["--checkpoint-every", "2"]),
            LOSS_8,
        ),
        (
            "every-7",
            LR_100,
            true,
            args(&["--checkpoint-every", "7"]),
            LOSS_8,
        ),
        (
            "every-8",
            LR_100,
            true,
            args(&["--checkpoint-every", "8"]),
            LOSS_8,
        ),
        (
            "every-9",
            LR_100,
            true,
            args(&["--checkpoint-every", "9"]),
            LOSS_8,
        ),
        ("steps-8", LR_100, false, args(&["--steps", "8"]), LOSS_8),
        (
            "steps-8-every-4",
            LR_100,
            true,
            args(&["--steps", "8", "--checkpoint-every", "4"]),
            LOSS_8,
        ),
        (
            "steps-9-every-8",
            LR_100,
            true,
            args(&["--steps", "9", "--checkpoint-every", "8"]),
            LOSS_8,
        ),
        (
            "steps-8-every-7",
            LR_100,
            true,
            args(&["--steps", "8", "--checkpoint-every", "7"]),
            LOSS_8,
        ),
    ];
    let found = mismatches(&cases);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// A parameter that overflows in the update of a step the device reads back is named at that
/// step, as the CPU names it: `lr = 1e38` with one step (the final read-back) and with a
/// checkpoint after every step.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_parameter_read_back_at_its_own_step_is_named_as_the_cpu_names_it() {
    let cases = vec![
        (
            "p-steps-1",
            LR_1E38,
            false,
            args(&["--steps", "1"]),
            UPDATE_0,
        ),
        (
            "p-every-1",
            LR_1E38,
            true,
            args(&["--checkpoint-every", "1"]),
            UPDATE_0,
        ),
        (
            "p-steps-1-every-1",
            LR_1E38,
            true,
            args(&["--steps", "1", "--checkpoint-every", "1"]),
            UPDATE_0,
        ),
    ];
    let found = mismatches(&cases);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// [`PAST_CHUNK`] overflows its loss at step 372, past the first command buffer of 256 steps: no
/// stop, a stop at the chunk boundary, windows that cross it, a stop at the failing step and one
/// after it, and the failing loss as the run's final loss.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_divergence_past_the_first_chunk_names_the_cpu_step() {
    let cases = vec![
        ("c-plain", PAST_CHUNK, false, args(&[]), LOSS_372),
        (
            "c-every-100",
            PAST_CHUNK,
            true,
            args(&["--checkpoint-every", "100"]),
            LOSS_372,
        ),
        (
            "c-every-124",
            PAST_CHUNK,
            true,
            args(&["--checkpoint-every", "124"]),
            LOSS_372,
        ),
        (
            "c-every-128",
            PAST_CHUNK,
            true,
            args(&["--checkpoint-every", "128"]),
            LOSS_372,
        ),
        (
            "c-every-186",
            PAST_CHUNK,
            true,
            args(&["--checkpoint-every", "186"]),
            LOSS_372,
        ),
        (
            "c-every-255",
            PAST_CHUNK,
            true,
            args(&["--checkpoint-every", "255"]),
            LOSS_372,
        ),
        (
            "c-every-256",
            PAST_CHUNK,
            true,
            args(&["--checkpoint-every", "256"]),
            LOSS_372,
        ),
        (
            "c-every-257",
            PAST_CHUNK,
            true,
            args(&["--checkpoint-every", "257"]),
            LOSS_372,
        ),
        (
            "c-every-372",
            PAST_CHUNK,
            true,
            args(&["--checkpoint-every", "372"]),
            LOSS_372,
        ),
        (
            "c-every-373",
            PAST_CHUNK,
            true,
            args(&["--checkpoint-every", "373"]),
            LOSS_372,
        ),
        (
            "c-steps-372",
            PAST_CHUNK,
            false,
            args(&["--steps", "372"]),
            LOSS_372,
        ),
        (
            "c-steps-372-every-256",
            PAST_CHUNK,
            true,
            args(&["--steps", "372", "--checkpoint-every", "256"]),
            LOSS_372,
        ),
    ];
    let found = mismatches(&cases);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// [`PAST_CHUNK_8`] overflows its loss at step 382, the windows of
/// [`a_divergence_past_the_first_chunk_names_the_cpu_step`] around that step: no stop, the
/// chunk boundary, windows that cross it, a stop the step before the failing one (`127`), at it
/// (`191`, `382`) and after it, and the failing loss as the run's final loss.
#[test]
#[ignore = "needs a Vulkan device"]
fn an_eight_row_divergence_past_the_first_chunk_names_the_cpu_step() {
    let every = |label, steps| {
        (
            label,
            PAST_CHUNK_8,
            true,
            args(&["--checkpoint-every", steps]),
            LOSS_382,
        )
    };
    let cases = vec![
        ("c8-plain", PAST_CHUNK_8, false, args(&[]), LOSS_382),
        every("c8-every-100", "100"),
        every("c8-every-127", "127"),
        every("c8-every-128", "128"),
        every("c8-every-191", "191"),
        every("c8-every-255", "255"),
        every("c8-every-256", "256"),
        every("c8-every-257", "257"),
        every("c8-every-382", "382"),
        every("c8-every-383", "383"),
        (
            "c8-steps-382",
            PAST_CHUNK_8,
            false,
            args(&["--steps", "382"]),
            LOSS_382,
        ),
        (
            "c8-steps-382-every-256",
            PAST_CHUNK_8,
            true,
            args(&["--steps", "382", "--checkpoint-every", "256"]),
            LOSS_382,
        ),
    ];
    let found = mismatches(&cases);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// A run resumed from the CPU's checkpoint of step 256 of [`PAST_CHUNK`] numbers its steps from 256
/// and its stops from the absolute step: `--checkpoint-every 100` stops at 300 and 400, and the
/// loss of step 372 lies in the window the stop at 400 reads.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_resumed_divergent_run_names_the_cpu_step() {
    let file = with_lr(PAST_CHUNK);
    let seed = run(
        "r-seed",
        &file,
        "cpu",
        true,
        &args(&["--checkpoint-every", "256"]),
    );
    assert_eq!(seed.files, ["run-256.msckpt"], "{}", seed.stderr);
    let from = scratch().join("r-seed-cpu").join("run-256.msckpt");
    let kept = scratch().join("resume-from-256.msckpt");
    std::fs::copy(&from, &kept).expect("the checkpoint is copied");
    let kept = kept.display().to_string();
    let cases = vec![
        (
            "r-plain",
            PAST_CHUNK,
            false,
            args(&["--resume", &kept]),
            LOSS_372,
        ),
        (
            "r-every-100",
            PAST_CHUNK,
            true,
            args(&["--resume", &kept, "--checkpoint-every", "100"]),
            LOSS_372,
        ),
        (
            "r-every-58",
            PAST_CHUNK,
            true,
            args(&["--resume", &kept, "--checkpoint-every", "58"]),
            LOSS_372,
        ),
        (
            "r-steps-116",
            PAST_CHUNK,
            false,
            args(&["--resume", &kept, "--steps", "116"]),
            LOSS_372,
        ),
    ];
    let found = mismatches(&cases);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// [`a_resumed_divergent_run_names_the_cpu_step`] on [`PAST_CHUNK_8`], whose loss of step 382
/// lies in the window the stop at 400 reads; `--checkpoint-every 191` stops at the failing
/// step, and `--steps 126` ends the resumed run there.
#[test]
#[ignore = "needs a Vulkan device"]
fn an_eight_row_resumed_divergent_run_names_the_cpu_step() {
    let file = with_lr(PAST_CHUNK_8);
    let seed = run(
        "r8-seed",
        &file,
        "cpu",
        true,
        &args(&["--checkpoint-every", "256"]),
    );
    assert_eq!(seed.files, ["run-256.msckpt"], "{}", seed.stderr);
    let from = scratch().join("r8-seed-cpu").join("run-256.msckpt");
    let kept = scratch().join("resume8-from-256.msckpt");
    std::fs::copy(&from, &kept).expect("the checkpoint is copied");
    let kept = kept.display().to_string();
    let cases = vec![
        (
            "r8-plain",
            PAST_CHUNK_8,
            false,
            args(&["--resume", &kept]),
            LOSS_382,
        ),
        (
            "r8-every-100",
            PAST_CHUNK_8,
            true,
            args(&["--resume", &kept, "--checkpoint-every", "100"]),
            LOSS_382,
        ),
        (
            "r8-every-191",
            PAST_CHUNK_8,
            true,
            args(&["--resume", &kept, "--checkpoint-every", "191"]),
            LOSS_382,
        ),
        (
            "r8-steps-126",
            PAST_CHUNK_8,
            false,
            args(&["--resume", &kept, "--steps", "126"]),
            LOSS_382,
        ),
    ];
    let found = mismatches(&cases);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// Starts `train <file> --backend <backend> --serve 127.0.0.1:0 <extra...>` and returns the body
/// of each event of the whole `/events` stream.
fn served_bodies(file: &str, backend: &str, extra: &[String]) -> Vec<Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args([
            "train",
            file,
            "--backend",
            backend,
            "--serve",
            "127.0.0.1:0",
        ])
        .args(extra)
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("modelspec starts");
    let stderr = child.stderr.take().expect("stderr is piped");
    let mut lines = BufReader::new(stderr).lines();
    let address = loop {
        let line = lines
            .next()
            .unwrap_or_else(|| panic!("no `serving` line from --backend {backend}"))
            .expect("stderr is UTF-8");
        if let Some(rest) = line.strip_prefix("serving http://") {
            break rest
                .strip_suffix("/events")
                .expect("`serving http://<addr>/events`")
                .to_owned();
        }
    };
    let mut stream = TcpStream::connect(&address).expect("the server accepts a connection");
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .expect("a read timeout");
    write!(
        stream,
        "GET /events HTTP/1.1\r\nHost: {address}\r\nAccept: text/event-stream\r\n\r\n"
    )
    .expect("the request is sent");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("the server ends the response");
    let _ = child.kill();
    let _ = child.wait();
    let (_, body) = response
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("a response head: {response:?}"));
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str::<Value>(data).expect("data is JSON")["body"].clone())
        .collect()
}

/// Under `--serve`, a divergence the device names at a checkpoint stop, past the first chunk,
/// and in a resumed-style numbering ends the stream as the CPU's does, event for event, each
/// `parameters` list the CPU's as a device run carries it ([`as_on_device`]) and each finite loss
/// within the run tolerance of the CPU's ([`body_as_the_cpu`]).
#[test]
#[ignore = "needs a Vulkan device"]
fn a_divergent_stream_with_checkpoint_stops_is_the_cpu_stream() {
    let mut found = Vec::new();
    for (label, fit, extra) in [
        ("s-every-4", LR_100, args(&["--checkpoint-every", "4"])),
        ("s-every-9", LR_100, args(&["--checkpoint-every", "9"])),
        ("s-chunk", PAST_CHUNK, args(&[])),
        (
            "s-chunk-every-128",
            PAST_CHUNK,
            args(&["--checkpoint-every", "128"]),
        ),
        ("s8-chunk", PAST_CHUNK_8, args(&[])),
        (
            "s8-chunk-every-191",
            PAST_CHUNK_8,
            args(&["--checkpoint-every", "191"]),
        ),
        ("s-p-every-1", LR_1E38, args(&["--checkpoint-every", "1"])),
    ] {
        let file = with_lr(fit);
        let dir = scratch().join(format!("{label}-serve"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("the temp dir is writable");
        let mut extra = extra.clone();
        if extra.iter().any(|a| a == "--checkpoint-every") {
            extra.push("--checkpoint".into());
            extra.push(dir.join("run.msckpt").display().to_string());
        }
        let cpu = as_on_device(&served_bodies(&file, "cpu", &extra));
        let gpu = served_bodies(&file, "vulkan", &extra);
        if gpu.len() != cpu.len() || gpu.iter().zip(&cpu).any(|(g, c)| !body_as_the_cpu(g, c)) {
            let first = cpu
                .iter()
                .zip(&gpu)
                .position(|(c, g)| !body_as_the_cpu(g, c))
                .unwrap_or(cpu.len().min(gpu.len()));
            found.push(format!(
                "{label}: {} cpu events, {} vulkan events, first difference at {first}: cpu {:?} \
                 vulkan {:?}",
                cpu.len(),
                gpu.len(),
                cpu.get(first),
                gpu.get(first)
            ));
        }
    }
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// A finite run with checkpoints waits once per checkpoint and once at the end, as the base did:
/// the check copies the window's losses into the checkpoint's read-back without a wait of its
/// own. `--checkpoint-every 100` of 500 steps and `--checkpoint-every 256` across the chunk.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_finite_checked_run_waits_as_the_base_did() {
    let file = repo_root().join(EXAMPLE).display().to_string();
    for (every, waits) in [("100", "5"), ("256", "2"), ("1", "500")] {
        let out = run(
            &format!("w-every-{every}"),
            &file,
            "vulkan",
            true,
            &args(&["--checkpoint-every", every, "--stats"]),
        );
        assert_eq!(out.status, Some(0), "{}", out.stderr);
        let line = out
            .stderr
            .lines()
            .find(|line| line.starts_with("host waits: "))
            .unwrap_or_else(|| panic!("no host waits line: {}", out.stderr));
        assert_eq!(line, format!("host waits: {waits}"), "every {every}");
    }
}

/// DESIGN.md § 8: "`--stats` prints the run's counters on stderr after it". A diverged device
/// run is a run, and before this change the same `--stats` run printed `host waits: 1`: the
/// counters follow the E-TRAIN-001 diagnostic, so the waits of a refused run can be compared with
/// the base's.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_diverged_run_with_stats_still_prints_its_counters() {
    let file = with_lr(LR_100);
    let out = run(
        "stats-diverged",
        &file,
        "vulkan",
        false,
        &args(&["--stats"]),
    );
    assert_eq!(out.status, Some(1), "{}", out.stderr);
    assert!(out.stderr.starts_with(LOSS_8), "{}", out.stderr);
    assert!(
        out.stderr.lines().any(|line| line == "host waits: 1"),
        "no `host waits: 1` line after the diagnostic:\n{}",
        out.stderr
    );
}

/// The event bodies `cpu` of a CPU run as a device run sends them (`spec/domains/observatory.yaml`,
/// `StepCompleted.parameters`): its first step and its post-loop step `start_step + steps` keep
/// their `Value` entries only, and every other step's list is empty.
fn as_on_device(cpu: &[Value]) -> Vec<Value> {
    let started = cpu
        .iter()
        .find(|body| body["kind"] == "RunStarted")
        .expect("a RunStarted");
    let start = started["value"]["start_step"]
        .as_u64()
        .expect("a start_step");
    let end = start + started["value"]["steps"].as_u64().expect("a steps");
    cpu.iter()
        .map(|body| {
            let mut body = body.clone();
            if body["kind"] == "StepCompleted" {
                let step = body["value"]["step"].as_u64().expect("a step");
                let parameters = body["value"]["parameters"]
                    .as_array_mut()
                    .expect("a parameters list");
                if step == start || step == end {
                    parameters.retain(|entry| entry["subject"] == "Value");
                } else {
                    parameters.clear();
                }
            }
            body
        })
        .collect()
}
