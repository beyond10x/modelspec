//! Adversary cases for story:train-vulkan-refuses-non-finite-loss: the windows of losses a
//! `--checkpoint-every` stop checks, the step it names, the checkpoints written before it, chunk
//! boundaries of `CHUNK_STEPS = 256`, resumed runs, the final loss, and the `--serve` stream.
//!
//! Every case runs one file on `--backend cpu` and on `--backend vulkan` and compares the exit
//! status, stdout, stderr and the names of the checkpoint files written. The fixtures are the
//! linear regression example, whose device run is bit for bit the CPU run's, at learning rates
//! whose divergence the device can name at the CPU's step: the loss overflows while every
//! parameter is finite (`lr = 100.0` at step 8, `lr = 0.6` at step 372), or the parameter
//! overflows in the update of a step whose parameters the device reads back (`lr = 1e38` with one
//! step, or a checkpoint after every step).
//!
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-cli --test train_vulkan_non_finite_adversary -- --ignored`
#![cfg(feature = "vulkan")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Duration;

use serde_json::Value;

const EXAMPLE: &str = "examples/linear_regression.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch() -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join("train_vulkan_non_finite_adversary")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The example with its learning rate replaced by `lr`, written once per `lr`.
fn with_lr(lr: &str) -> String {
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example exists");
    let changed = source.replace("sgd(lr = 0.1)", &format!("sgd(lr = {lr})"));
    assert_ne!(
        changed, source,
        "the example's optimizer is `sgd(lr = 0.1)`"
    );
    std::fs::create_dir_all(scratch()).expect("the temp dir is writable");
    let path = scratch().join(format!("lr-{lr}.mspec"));
    std::fs::write(&path, changed).expect("the source is written");
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
fn mismatches(cases: &[(&str, &str, bool, Vec<String>, &str)]) -> Vec<String> {
    let mut found = Vec::new();
    for (label, lr, checkpoint, extra, cpu_names) in cases {
        let file = with_lr(lr);
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
        if gpu != cpu {
            found.push(format!(
                "{label} (lr {lr}, {extra:?}): cpu [{:?}, {}, files {:?}] vulkan [{:?}, {}, \
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
            "100.0",
            true,
            args(&["--checkpoint-every", "1"]),
            LOSS_8,
        ),
        (
            "every-2",
            "100.0",
            true,
            args(&["--checkpoint-every", "2"]),
            LOSS_8,
        ),
        (
            "every-7",
            "100.0",
            true,
            args(&["--checkpoint-every", "7"]),
            LOSS_8,
        ),
        (
            "every-8",
            "100.0",
            true,
            args(&["--checkpoint-every", "8"]),
            LOSS_8,
        ),
        (
            "every-9",
            "100.0",
            true,
            args(&["--checkpoint-every", "9"]),
            LOSS_8,
        ),
        ("steps-8", "100.0", false, args(&["--steps", "8"]), LOSS_8),
        (
            "steps-8-every-4",
            "100.0",
            true,
            args(&["--steps", "8", "--checkpoint-every", "4"]),
            LOSS_8,
        ),
        (
            "steps-9-every-8",
            "100.0",
            true,
            args(&["--steps", "9", "--checkpoint-every", "8"]),
            LOSS_8,
        ),
        (
            "steps-8-every-7",
            "100.0",
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
            "1e38",
            false,
            args(&["--steps", "1"]),
            UPDATE_0,
        ),
        (
            "p-every-1",
            "1e38",
            true,
            args(&["--checkpoint-every", "1"]),
            UPDATE_0,
        ),
        (
            "p-steps-1-every-1",
            "1e38",
            true,
            args(&["--steps", "1", "--checkpoint-every", "1"]),
            UPDATE_0,
        ),
    ];
    let found = mismatches(&cases);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// `lr = 0.6` overflows its loss at step 372, past the first command buffer of 256 steps: no
/// stop, a stop at the chunk boundary, windows that cross it, a stop at the failing step and one
/// after it, and the failing loss as the run's final loss.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_divergence_past_the_first_chunk_names_the_cpu_step() {
    let cases = vec![
        ("c-plain", "0.6", false, args(&[]), LOSS_372),
        (
            "c-every-100",
            "0.6",
            true,
            args(&["--checkpoint-every", "100"]),
            LOSS_372,
        ),
        (
            "c-every-124",
            "0.6",
            true,
            args(&["--checkpoint-every", "124"]),
            LOSS_372,
        ),
        (
            "c-every-128",
            "0.6",
            true,
            args(&["--checkpoint-every", "128"]),
            LOSS_372,
        ),
        (
            "c-every-186",
            "0.6",
            true,
            args(&["--checkpoint-every", "186"]),
            LOSS_372,
        ),
        (
            "c-every-255",
            "0.6",
            true,
            args(&["--checkpoint-every", "255"]),
            LOSS_372,
        ),
        (
            "c-every-256",
            "0.6",
            true,
            args(&["--checkpoint-every", "256"]),
            LOSS_372,
        ),
        (
            "c-every-257",
            "0.6",
            true,
            args(&["--checkpoint-every", "257"]),
            LOSS_372,
        ),
        (
            "c-every-372",
            "0.6",
            true,
            args(&["--checkpoint-every", "372"]),
            LOSS_372,
        ),
        (
            "c-every-373",
            "0.6",
            true,
            args(&["--checkpoint-every", "373"]),
            LOSS_372,
        ),
        (
            "c-steps-372",
            "0.6",
            false,
            args(&["--steps", "372"]),
            LOSS_372,
        ),
        (
            "c-steps-372-every-256",
            "0.6",
            true,
            args(&["--steps", "372", "--checkpoint-every", "256"]),
            LOSS_372,
        ),
    ];
    let found = mismatches(&cases);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// A run resumed from the CPU's checkpoint of step 256 of `lr = 0.6` numbers its steps from 256
/// and its stops from the absolute step: `--checkpoint-every 100` stops at 300 and 400, and the
/// loss of step 372 lies in the window the stop at 400 reads.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_resumed_divergent_run_names_the_cpu_step() {
    let file = with_lr("0.6");
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
            "0.6",
            false,
            args(&["--resume", &kept]),
            LOSS_372,
        ),
        (
            "r-every-100",
            "0.6",
            true,
            args(&["--resume", &kept, "--checkpoint-every", "100"]),
            LOSS_372,
        ),
        (
            "r-every-58",
            "0.6",
            true,
            args(&["--resume", &kept, "--checkpoint-every", "58"]),
            LOSS_372,
        ),
        (
            "r-steps-116",
            "0.6",
            false,
            args(&["--resume", &kept, "--steps", "116"]),
            LOSS_372,
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
/// and in a resumed-style numbering ends the stream as the CPU's does, event for event.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_divergent_stream_with_checkpoint_stops_is_the_cpu_stream() {
    let mut found = Vec::new();
    for (label, lr, extra) in [
        ("s-every-4", "100.0", args(&["--checkpoint-every", "4"])),
        ("s-every-9", "100.0", args(&["--checkpoint-every", "9"])),
        ("s-chunk", "0.6", args(&[])),
        (
            "s-chunk-every-128",
            "0.6",
            args(&["--checkpoint-every", "128"]),
        ),
        ("s-p-every-1", "1e38", args(&["--checkpoint-every", "1"])),
    ] {
        let file = with_lr(lr);
        let dir = scratch().join(format!("{label}-serve"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("the temp dir is writable");
        let mut extra = extra.clone();
        if extra.iter().any(|a| a == "--checkpoint-every") {
            extra.push("--checkpoint".into());
            extra.push(dir.join("run.msckpt").display().to_string());
        }
        let cpu = served_bodies(&file, "cpu", &extra);
        let gpu = served_bodies(&file, "vulkan", &extra);
        if gpu != cpu {
            let first = cpu
                .iter()
                .zip(&gpu)
                .position(|(c, g)| c != g)
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
    let file = with_lr("100.0");
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
