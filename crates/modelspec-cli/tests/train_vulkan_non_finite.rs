//! `modelspec train --backend vulkan` refuses a run whose loss or parameters stop being finite
//! with E-TRAIN-001, as `--backend cpu` does (story:train-vulkan-refuses-non-finite-loss;
//! DESIGN.md § 8).
//!
//! The device reads its parameters back only after the last step and at each requested
//! checkpoint, so it checks every step's loss and the parameters where it reads them. The
//! `lr = 100.0` fixture overflows its loss while every parameter is finite: the device run prints
//! what the CPU run prints, stdout, stderr and status. The `lr = 1e38` fixture takes `LinearReg.w`
//! past `f32::MAX` in the update of step 0 while that step's loss is finite: the CPU names step 0
//! and the parameter, the device the first loss the infinite parameter makes non-finite, step 1.
//!
//! The tests create a Vulkan instance and are ignored by default; run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-cli --test train_vulkan_non_finite -- --ignored`.
#![cfg(feature = "vulkan")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Duration;

use serde_json::Value;

const EXAMPLE: &str = "examples/linear_regression.mspec";
const SCHEMA: &str =
    "crates/modelspec-observatory/schema/entities/modelspec.observatory.RunEvent.schema.json";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

/// The example with its learning rate replaced by `lr`, written to a fresh file `name`.
fn with_lr(name: &str, lr: &str) -> String {
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example exists");
    let changed = source.replace("sgd(lr = 0.1)", &format!("sgd(lr = {lr})"));
    assert_ne!(
        changed, source,
        "the example's optimizer is `sgd(lr = 0.1)`"
    );
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("train_vulkan_non_finite");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join(name);
    std::fs::write(&path, changed).expect("the source is written");
    path.display().to_string()
}

/// Runs `modelspec train <path> --backend <backend> <extra...>` from the repository root.
fn train(path: &str, backend: &str, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["train", path, "--backend", backend])
        .args(extra)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

/// Acceptance 1, `lr = 100.0`: the device run's stdout, stderr and status are the CPU run's.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_divergent_loss_on_vulkan_prints_what_the_cpu_prints() {
    let path = with_lr("lr100.mspec", "100.0");
    let cpu = train(&path, "cpu", &[]);
    let gpu = train(&path, "vulkan", &[]);
    assert_eq!(cpu.status.code(), Some(1), "{}", text(&cpu.stderr));
    assert_eq!(
        (gpu.status.code(), text(&gpu.stdout), text(&gpu.stderr)),
        (cpu.status.code(), text(&cpu.stdout), text(&cpu.stderr)),
        "status, stdout and stderr of --backend vulkan against --backend cpu"
    );
}

/// Acceptance 1, `lr = 1e38`, as the device can check it without reading the parameters back
/// after every step: exit 1, one E-TRAIN-001 line naming the loss of step 1, which the infinite
/// `LinearReg.w` makes non-finite, and the CPU run's stdout up to step 0 followed by that step's
/// line, with no `param` lines.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_non_finite_parameter_on_vulkan_stops_at_the_first_loss_it_makes_non_finite() {
    let path = with_lr("lr1e38.mspec", "1e38");
    let cpu = train(&path, "cpu", &[]);
    let gpu = train(&path, "vulkan", &[]);
    let (stdout, stderr) = (text(&gpu.stdout), text(&gpu.stderr));
    assert_eq!(
        gpu.status.code(),
        Some(1),
        "stdout: {stdout}\nstderr: {stderr}"
    );
    let lines: Vec<&str> = stderr
        .lines()
        .filter(|line| line.starts_with("E-TRAIN-001: "))
        .collect();
    assert_eq!(
        lines,
        ["E-TRAIN-001: the loss of step 1 is not finite"],
        "{stderr}"
    );
    let printed: Vec<&str> = stdout.lines().collect();
    let cpu_out = text(&cpu.stdout);
    let expected: Vec<&str> = cpu_out.lines().collect();
    assert_eq!(printed[..printed.len() - 1], expected[..], "{stdout}");
    let last = printed[printed.len() - 1];
    assert!(
        last == "step 1 loss NaN" || last == "step 1 loss inf",
        "{stdout}"
    );
}

/// The `lr = 100.0` fixture with `--checkpoint-every 4` of its 500 steps: the checkpoints of
/// steps 4 and 8 are written, as the CPU run writes them; the loss of step 8, read back at the
/// stop after step 12, is `inf`, so the run stops there without writing `run-12.msckpt` and prints
/// what the CPU prints.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_divergent_run_writes_the_checkpoints_the_cpu_writes() {
    let path = with_lr("lr100-every.mspec", "100.0");
    let files = |backend: &str| {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("train_vulkan_non_finite")
            .join(format!("every-{backend}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("the temp dir is writable");
        let run = dir.join("run.msckpt").display().to_string();
        let output = train(
            &path,
            backend,
            &["--checkpoint", &run, "--checkpoint-every", "4"],
        );
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .expect("the dir exists")
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        (output, names)
    };
    let (cpu, cpu_files) = files("cpu");
    let (gpu, gpu_files) = files("vulkan");
    assert_eq!(cpu.status.code(), Some(1), "{}", text(&cpu.stderr));
    assert_eq!(cpu_files, ["run-4.msckpt", "run-8.msckpt"]);
    assert_eq!(gpu_files, cpu_files);
    assert_eq!(
        (gpu.status.code(), text(&gpu.stdout), text(&gpu.stderr)),
        (cpu.status.code(), text(&cpu.stdout), text(&cpu.stderr)),
    );
}

/// DESIGN.md § 8: `--stats` and `--profile` print the run's counters on stderr after it, and a
/// refused run is a run. The `lr = 100.0` fixture with `--checkpoint-every 4` stops after step 12
/// (above): the counters follow the E-TRAIN-001 line and count what the device ran, 12 steps and
/// 3 host waits, and the profile times the step plan's sites over those 12 steps, without the
/// final loss evaluation the stopped run never recorded.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_run_refused_at_a_checkpoint_stop_prints_the_counters_of_what_it_ran() {
    let path = with_lr("lr100-every-stats.mspec", "100.0");
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("train_vulkan_non_finite")
        .join("every-stats");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let run = dir.join("run.msckpt").display().to_string();
    let output = train(
        &path,
        "vulkan",
        &[
            "--checkpoint",
            &run,
            "--checkpoint-every",
            "4",
            "--stats",
            "--profile",
        ],
    );
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.starts_with("E-TRAIN-001: the loss of step 8 is not finite"),
        "{stderr}"
    );
    let lines: Vec<&str> = stderr.lines().collect();
    for expected in ["steps: 12", "host waits: 3", "profile host_waits 3"] {
        assert!(lines.contains(&expected), "no `{expected}` line:\n{stderr}");
    }
    if !lines.contains(&"profile kernel timings unavailable: no timestamp queries") {
        let sites: Vec<&str> = lines
            .iter()
            .copied()
            .filter(|line| line.starts_with("profile dispatch "))
            .collect();
        assert!(!sites.is_empty(), "no `profile dispatch` line:\n{stderr}");
        for site in sites {
            assert!(
                site.starts_with("profile dispatch step ") && site.contains(" calls 12 "),
                "{site}:\n{stderr}"
            );
        }
    }
}

/// Starts `modelspec train <path> --backend <backend> --serve 127.0.0.1:0`, reads the whole
/// `/events` stream once the `serving` line names the address, then kills the process. Returns
/// each event's JSON `data`.
fn served_events(path: &str, backend: &str) -> Vec<Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args([
            "train",
            path,
            "--backend",
            backend,
            "--serve",
            "127.0.0.1:0",
        ])
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
        .map(|data| serde_json::from_str(data).expect("data is JSON"))
        .collect()
}

/// Acceptance 2: under `--serve` the device run's stream validates against the run event schema
/// and carries the CPU run's event bodies in the CPU run's order: `RunStarted`, one
/// `StepCompleted` per step through the failing one, whose loss is `null`, then `RunFailed` with
/// E-TRAIN-001.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_divergent_vulkan_run_ends_its_stream_as_the_cpu_run_does() {
    let path = with_lr("lr100-serve.mspec", "100.0");
    let cpu = served_events(&path, "cpu");
    let gpu = served_events(&path, "vulkan");
    let schema: Value = serde_json::from_str(
        &std::fs::read_to_string(repo_root().join(SCHEMA)).expect("the schema is committed"),
    )
    .expect("the schema is JSON");
    let validator = jsonschema::validator_for(&schema).expect("the schema compiles");
    for event in &gpu {
        let errors: Vec<String> = validator
            .iter_errors(event)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "{event}: {errors:?}");
    }
    let bodies = |events: &[Value]| -> Vec<Value> {
        events.iter().map(|event| event["body"].clone()).collect()
    };
    let gpu_bodies = bodies(&gpu);
    assert_eq!(gpu_bodies, bodies(&cpu));
    let last = &gpu_bodies[gpu_bodies.len() - 1];
    assert_eq!(last["kind"], "RunFailed", "{last}");
    assert_eq!(last["value"]["code"], "E-TRAIN-001", "{last}");
    let failing = &gpu_bodies[gpu_bodies.len() - 2];
    assert_eq!(failing["kind"], "StepCompleted", "{failing}");
    assert_eq!(failing["value"]["loss"], Value::Null, "{failing}");
}
