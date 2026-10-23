//! Adversary pass on story:train-refuses-non-finite-loss (DESIGN.md § 8, `E-TRAIN-001`).
//!
//! Fixtures are `examples/linear_regression.mspec` with its learning rate replaced: `lr = 100.0`
//! overflows the loss to `inf` while every parameter is still finite; `lr = 1e38` takes
//! `LinearReg.w` past `f32::MAX` in the update of step 0 while the loss of step 0 is finite.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use modelspec_core::compile::compile_training;
use modelspec_runtime::train::checkpoint;
use serde_json::Value;

const EXAMPLE: &str = "examples/linear_regression.mspec";
const WAIT: Duration = Duration::from_secs(120);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

fn dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_train_non_finite")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    dir
}

fn with_lr(lr: &str) -> String {
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example exists");
    let changed = source.replace("sgd(lr = 0.1)", &format!("sgd(lr = {lr})"));
    assert_ne!(changed, source);
    changed
}

/// Writes the example at `lr` into a fresh dir `name` and returns (dir, source path, source).
fn fixture(name: &str, lr: &str) -> (PathBuf, String, String) {
    let dir = dir(name);
    let source = with_lr(lr);
    let path = dir.join("fit.mspec");
    std::fs::write(&path, &source).expect("the source is written");
    (dir, path.display().to_string(), source)
}

fn train(path: &str, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["train", path, "--backend", "cpu"])
        .args(extra)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

/// The step the single `E-TRAIN-001: ` line of `stderr` names.
fn failing_step(stderr: &str) -> u64 {
    let lines: Vec<&str> = stderr
        .lines()
        .filter(|l| l.starts_with("E-TRAIN-001: "))
        .collect();
    assert_eq!(lines.len(), 1, "one E-TRAIN-001 line: {stderr}");
    let (_, rest) = lines[0].split_once("step ").expect("names a step");
    rest.split(|c: char| !c.is_ascii_digit())
        .next()
        .unwrap()
        .parse()
        .expect("a step number")
}

fn checkpoints(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "msckpt"))
        .collect();
    found.sort();
    found
}

/// Every parameter value of the checkpoint at `path`, read for the compiled `source`.
fn checkpoint_values(source: &str, path: &Path) -> Vec<f32> {
    let compiled = compile_training("fit.mspec", source).expect("it compiles");
    let read = checkpoint::read(path).expect("a readable checkpoint");
    read.parameters_for(&compiled)
        .expect("it matches the file")
        .values()
        .flat_map(|t| t.data().to_vec())
        .collect()
}

/// A run that reaches the final evaluation `end` with finite parameters and an infinite loss is
/// refused at `end`: exit 1, stdout ends with `step <end> loss inf`, no `param` line, no
/// diagnostic on stdout, and the final `--checkpoint` is not written.
#[test]
fn the_loss_of_the_final_evaluation_is_checked() {
    let (dir, path, _) = fixture("final", "100.0");
    let first = train(&path, &[]);
    let k = failing_step(&text(&first.stderr));
    assert!(k > 0, "the loss case fails after step 0: {k}");
    let ckpt = dir.join("c.msckpt");
    let output = train(
        &path,
        &[
            "--steps",
            &k.to_string(),
            "--checkpoint",
            ckpt.to_str().unwrap(),
        ],
    );
    let (stdout, stderr) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(output.status.code(), Some(1), "{stdout}\n{stderr}");
    assert_eq!(failing_step(&stderr), k, "{stderr}");
    assert_eq!(
        stdout.lines().last(),
        Some(format!("step {k} loss inf").as_str())
    );
    assert!(!stdout.lines().any(|l| l.starts_with("param ")), "{stdout}");
    assert!(
        !stdout.contains("E-TRAIN"),
        "the diagnostic is on stderr only: {stdout}"
    );
    assert!(!stderr.lines().any(|l| l.starts_with("step ")), "{stderr}");
    assert!(!ckpt.exists(), "a failed run writes no final checkpoint");
}

/// `--checkpoint-every 1` never writes a checkpoint holding a non-finite parameter, and never one
/// at or after the failing step's update.
#[test]
fn no_checkpoint_holds_a_non_finite_parameter() {
    for (name, lr) in [("every_param", "1e38"), ("every_loss", "100.0")] {
        let (dir, path, source) = fixture(name, lr);
        let ckpt = dir.join("c.msckpt");
        let output = train(
            &path,
            &[
                "--checkpoint",
                ckpt.to_str().unwrap(),
                "--checkpoint-every",
                "1",
            ],
        );
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(1), "{lr}: {stderr}");
        let k = failing_step(&stderr);
        assert!(!ckpt.exists(), "{lr}: no final checkpoint");
        for file in checkpoints(&dir) {
            let values = checkpoint_values(&source, &file);
            assert!(
                values.iter().all(|v| v.is_finite()),
                "{lr}: {} holds {values:?}",
                file.display()
            );
            let step: u64 = file
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .trim_start_matches("c-")
                .parse()
                .unwrap();
            assert!(
                step <= k,
                "{lr}: {} is past failing step {k}",
                file.display()
            );
        }
        if lr == "1e38" {
            assert!(
                checkpoints(&dir).is_empty(),
                "step 0's update is not written"
            );
        }
    }
}

/// Resuming from the last checkpoint before the failing step fails at the same step with the
/// same last stdout line as the uninterrupted run.
#[test]
fn a_resumed_run_fails_at_the_same_step() {
    let (dir, path, _) = fixture("resume", "100.0");
    let ckpt = dir.join("c.msckpt");
    let whole = train(
        &path,
        &[
            "--checkpoint",
            ckpt.to_str().unwrap(),
            "--checkpoint-every",
            "1",
        ],
    );
    let k = failing_step(&text(&whole.stderr));
    let from = dir.join(format!("c-{}.msckpt", k - 1));
    assert!(from.exists(), "{} was written", from.display());
    let resumed = train(
        &path,
        &["--resume", from.to_str().unwrap(), "--steps", "10"],
    );
    let (stdout, stderr) = (text(&resumed.stdout), text(&resumed.stderr));
    assert_eq!(resumed.status.code(), Some(1), "{stdout}\n{stderr}");
    assert_eq!(failing_step(&stderr), k);
    assert_eq!(
        stdout.lines().last(),
        text(&whole.stdout).lines().last(),
        "{stdout}"
    );
    assert!(!stdout.lines().any(|l| l.starts_with("param ")), "{stdout}");
}

struct Served {
    child: Child,
    address: String,
}

impl Served {
    fn start(path: &str) -> Served {
        let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(["train", path, "--backend", "cpu", "--serve", "127.0.0.1:0"])
            .current_dir(repo_root())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("modelspec starts");
        let err = child.stderr.take().unwrap();
        let (send, receive) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines() {
                let Ok(line) = line else { break };
                let _ = send.send(line);
            }
        });
        loop {
            let line = receive.recv_timeout(WAIT).expect("a `serving` line");
            if let Some(rest) = line.strip_prefix("serving http://") {
                let address = rest.strip_suffix("/events").unwrap().to_owned();
                return Served { child, address };
            }
        }
    }

    /// The `data:` JSON of every frame of `GET /events`, read to its end.
    fn events(&self) -> Vec<Value> {
        let mut stream = TcpStream::connect(&self.address).unwrap();
        stream.set_read_timeout(Some(WAIT)).unwrap();
        write!(
            stream,
            "GET /events HTTP/1.1\r\nHost: {}\r\nAccept: text/event-stream\r\n\r\n",
            self.address
        )
        .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .map(|d| serde_json::from_str(d).unwrap())
            .collect()
    }

    fn terminate(&mut self) -> ExitStatus {
        let pid = self.child.id().to_string();
        assert!(
            Command::new("kill")
                .args(["-s", "TERM", &pid])
                .status()
                .unwrap()
                .success()
        );
        let start = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(start.elapsed() < WAIT, "still serving after SIGTERM");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// DESIGN.md § 8, the unit's own contract for E-TRAIN-001: "`--serve` ends the stream with that
/// step's `StepCompleted` (its loss `null` when the loss is the non-finite value), then
/// `RunFailed`". Driven with the parameter fixture, whose failing step has a finite loss.
#[test]
fn design_8_a_parameter_divergence_ends_the_stream_with_its_finite_loss() {
    let (_, path, _) = fixture("serve_param", "1e38");
    let mut served = Served::start(&path);
    let events = served.events();
    let status = served.terminate();
    assert_eq!(status.code(), Some(1), "exits with the run's status");
    let kinds: Vec<&str> = events
        .iter()
        .map(|e| e["body"]["kind"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds[kinds.len() - 2..],
        ["StepCompleted", "RunFailed"],
        "{kinds:?}"
    );
    assert_eq!(
        events.last().unwrap()["body"]["value"]["code"],
        "E-TRAIN-001"
    );
    let last_step = &events[events.len() - 2]["body"]["value"];
    assert_eq!(last_step["step"], 0, "{last_step}");
    assert!(
        last_step["loss"].as_f64().is_some_and(f64::is_finite),
        "DESIGN.md § 8: a parameter divergence keeps the step's finite loss: {last_step}"
    );
}

/// The loss fixture under `--serve` exits 1 once interrupted, the stream ends with `RunFailed`.
#[test]
fn a_diverged_serve_run_exits_1() {
    let (_, path, _) = fixture("serve_loss", "100.0");
    let mut served = Served::start(&path);
    let events = served.events();
    let status = served.terminate();
    assert_eq!(status.code(), Some(1));
    assert_eq!(events.last().unwrap()["body"]["kind"], "RunFailed");
    assert_eq!(
        events[events.len() - 2]["body"]["value"]["loss"],
        Value::Null
    );
}
