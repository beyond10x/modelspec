//! `modelspec train --backend vulkan --data <packed-dir>` on the hidden-32 twin of BERT-tiny
//! against `--backend cpu` (story:bert-vulkan-kernels-attention-loss-adamw, coordinator decision
//! D73, 2026-10-25): ten training steps on the GPU, with dropout, attention, cross-entropy and
//! AdamW, print `step` and `eval` losses and final parameters within the run tolerance of
//! DESIGN.md § 9 gate 5, `|g - c| <= GPU_RUN_ATOL + GPU_RUN_RTOL * |c|`, accuracies within
//! `1 / M` of the CPU's, `M` the selected `HeldOut` rows, and stream them as schema-valid events.
//!
//! The run is the twin with `steps = 10` and `warmup_steps = 2`, written beside the packed data:
//! the fixture's warm-up of 80 steps is refused by a run of 10 (E-TRAIN schedule refusal). The
//! packed directory is made as `bert_cpu_train.rs` makes it, from the synthetic corpus of
//! `common/markov_corpus.rs`, under this crate's build directory.
//!
//! The device tests are ignored by default; run them with
//! `cargo test -p modelspec-cli --features vulkan --test vulkan_bert_cpu_steps -- --ignored`.
//! Their tolerances are those of `crates/modelspec-runtime/tests/common/mod.rs`.
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use modelspec_core::compile::compile_training;
use modelspec_runtime::dataset::SplitRole;
use modelspec_runtime::train::data::{self, DataSource};
use serde_json::Value;

#[allow(dead_code)]
#[path = "../../modelspec-runtime/tests/common/mod.rs"]
mod common;

#[path = "common/markov_corpus.rs"]
mod markov_corpus;

use common::{GPU_RUN_ATOL, GPU_RUN_RTOL, within};

const TWIN: &str = "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec";
const RUN_EVENT_SCHEMA: &str =
    "crates/modelspec-observatory/schema/entities/modelspec.observatory.RunEvent.schema.json";
const STEPS: u64 = 10;
const EVAL_EVERY: &str = "5";
const WAIT: Duration = Duration::from_secs(600);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh, empty directory under this crate's build directory, named for the test.
fn scratch(name: &str) -> PathBuf {
    let dir =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("vulkan_bert_cpu_steps_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// Runs `modelspec <args>` from the repository root.
fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout: {}\nstderr: {}",
        stdout(output),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

/// The synthetic corpus written and prepared, a 300-token tokenizer trained on it and the two
/// packed into sequences of 16, all under `case`; returns the packed directory.
fn packed(case: &Path) -> PathBuf {
    let corpus = case.join("corpus");
    markov_corpus::write(&corpus);
    let dataset = case.join("dataset");
    assert_exit(
        &modelspec(&["data", "prepare", path(&corpus), "--out", path(&dataset)]),
        0,
    );
    let tokenizer = case.join("tokenizer.json");
    assert_exit(
        &modelspec(&[
            "data",
            "tokenizer",
            "--dataset",
            path(&dataset),
            "--vocab-size",
            "300",
            "--out",
            path(&tokenizer),
        ]),
        0,
    );
    let packed = case.join("packed");
    assert_exit(
        &modelspec(&[
            "data",
            "pack",
            "--dataset",
            path(&dataset),
            "--tokenizer",
            path(&tokenizer),
            "--seq-len",
            "16",
            "--out",
            path(&packed),
        ]),
        0,
    );
    packed
}

/// The twin with `steps = 10` and `warmup_steps = 2`, written to `<case>/twin10.mspec`.
fn twin10(case: &Path) -> String {
    let mut source = std::fs::read_to_string(repo_root().join(TWIN)).expect("the fixture");
    for (from, to) in [
        ("steps = 800;", "steps = 10;"),
        ("warmup_steps = 80)", "warmup_steps = 2)"),
    ] {
        assert!(source.contains(from), "the fixture holds {from:?}");
        source = source.replace(from, to);
    }
    let file = case.join("twin10.mspec");
    std::fs::write(&file, source).expect("the variant is written");
    path(&file).to_owned()
}

/// `modelspec train <file> --backend <backend> --data <data> <extra...>`.
fn train(file: &str, backend: &str, data: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["train", file, "--backend", backend, "--data", path(data)];
    args.extend_from_slice(extra);
    modelspec(&args)
}

/// The `(step, loss text)` of every `step` line.
fn step_lines(out: &str) -> Vec<(u64, String)> {
    out.lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (step.parse().unwrap(), loss.to_owned())
        })
        .collect()
}

/// The `(step, loss, accuracy)` texts of every `eval <k> loss <v> accuracy <a>` line.
fn eval_lines(out: &str) -> Vec<(u64, String, String)> {
    out.lines()
        .filter_map(|line| line.strip_prefix("eval "))
        .map(|rest| {
            let words: Vec<&str> = rest.split_whitespace().collect();
            assert_eq!(words.len(), 5, "{rest}");
            assert_eq!((words[1], words[3]), ("loss", "accuracy"), "{rest}");
            (
                words[0].parse().unwrap(),
                words[2].to_owned(),
                words[4].to_owned(),
            )
        })
        .collect()
}

/// The values of every `<label> @<path>: <type> = [v, ...]` line, by path.
fn tensors(out: &str, label: &str) -> BTreeMap<String, Vec<f32>> {
    let prefix = format!("{label} @");
    out.lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        .map(|rest| {
            let (path, rest) = rest.split_once(": ").expect("`@<path>: <type> = [..]`");
            let (_, list) = rest.split_once(" = [").expect("a value list");
            let list = list.strip_suffix(']').expect("a closed list");
            let values = list
                .split(", ")
                .map(|v| v.parse::<f32>().expect("an f32"))
                .collect();
            (path.to_owned(), values)
        })
        .collect()
}

/// The loss of every `StepCompleted` of a recorded run file, by step.
fn recorded_losses(file: &Path) -> BTreeMap<u64, f64> {
    std::fs::read_to_string(file)
        .expect("the recording")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("a JSON event"))
        .filter(|event| event["body"]["kind"] == "StepCompleted")
        .map(|event| {
            let value = &event["body"]["value"];
            (
                value["step"].as_u64().expect("a step"),
                value["loss"].as_f64().expect("a loss"),
            )
        })
        .collect()
}

fn bound(c: f32) -> f64 {
    f64::from(GPU_RUN_ATOL) + f64::from(GPU_RUN_RTOL) * f64::from(c).abs()
}

/// Scenario 4: ten GPU steps of the twin on packed data print `step` losses and final
/// parameters within the run tolerance of the CPU run's, every one of the eleven recorded losses
/// too, and two GPU runs with one seed print identical stdout.
#[test]
#[ignore = "needs a Vulkan device"]
fn steps() {
    let case = scratch("steps");
    let data = packed(&case);
    let file = twin10(&case);
    let cpu_record = case.join("cpu.run.jsonl");
    let gpu_record = case.join("gpu.run.jsonl");
    let cpu = train(&file, "cpu", &data, &["--record", path(&cpu_record)]);
    assert_exit(&cpu, 0);
    let gpu = train(&file, "vulkan", &data, &["--record", path(&gpu_record)]);
    assert_exit(&gpu, 0);
    let again = train(&file, "vulkan", &data, &[]);
    assert_exit(&again, 0);
    let (cpu_out, gpu_out) = (stdout(&cpu), stdout(&gpu));
    assert_eq!(gpu_out, stdout(&again), "two GPU runs from one seed differ");
    assert_eq!(
        cpu_out.lines().next(),
        gpu_out.lines().next(),
        "the data line"
    );
    assert_eq!(tensors(&cpu_out, "init"), tensors(&gpu_out, "init"));

    let mut outside = Vec::new();
    let (cpu_steps, gpu_steps) = (step_lines(&cpu_out), step_lines(&gpu_out));
    assert_eq!(
        cpu_steps.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        gpu_steps.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        "{cpu_out}\n{gpu_out}"
    );
    for ((k, c), (_, g)) in cpu_steps.iter().zip(&gpu_steps) {
        let (c, g): (f32, f32) = (c.parse().unwrap(), g.parse().unwrap());
        if !within(g, c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) {
            outside.push(format!("step {k}: gpu {g} cpu {c}"));
        }
    }
    let (cpu_losses, gpu_losses) = (recorded_losses(&cpu_record), recorded_losses(&gpu_record));
    assert_eq!(cpu_losses.len() as u64, STEPS + 1);
    assert_eq!(
        cpu_losses.keys().collect::<Vec<_>>(),
        gpu_losses.keys().collect::<Vec<_>>()
    );
    eprintln!("step | CPU loss | GPU loss | |diff| | bound");
    for (k, &c) in &cpu_losses {
        let (c, g) = (c as f32, gpu_losses[k] as f32);
        let diff = (f64::from(g) - f64::from(c)).abs();
        eprintln!("{k} | {c} | {g} | {diff:.3e} | {:.3e}", bound(c));
        if !within(g, c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) {
            outside.push(format!("recorded step {k}: gpu {g} cpu {c}"));
        }
    }

    let (cpu_params, gpu_params) = (tensors(&cpu_out, "param"), tensors(&gpu_out, "param"));
    assert_eq!(cpu_params.len(), 41, "every parameter of the twin");
    assert_eq!(
        cpu_params.keys().collect::<Vec<_>>(),
        gpu_params.keys().collect::<Vec<_>>()
    );
    let (mut elements, mut max_abs, mut max_rel) = (0usize, 0.0f64, 0.0f64);
    for (name, c) in &cpu_params {
        let g = &gpu_params[name];
        assert_eq!(g.len(), c.len(), "{name}");
        for (i, (&g, &c)) in g.iter().zip(c).enumerate() {
            let diff = (f64::from(g) - f64::from(c)).abs();
            max_abs = max_abs.max(diff);
            if c != 0.0 {
                max_rel = max_rel.max(diff / f64::from(c).abs());
            }
            elements += 1;
            if !within(g, c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) && outside.len() < 16 {
                outside.push(format!("param @{name} [{i}]: gpu {g} cpu {c}"));
            }
        }
    }
    eprintln!(
        "final parameters: {elements} elements, max |g-c| {max_abs:.3e}, max relative {max_rel:.3e}"
    );
    assert_eq!(elements, 28_684);
    assert!(outside.is_empty(), "{outside:#?}");
}

/// The `M` of the accuracy rule: the rows of positive weight of every full `HeldOut` batch of
/// `dir` for `file`, the rows the held-out accuracy counts.
fn selected_rows(file: &str, dir: &Path) -> usize {
    let source = std::fs::read_to_string(repo_root().join(file)).expect("the variant");
    let compiled = compile_training(file, &source).expect("the twin compiles");
    let binding = match data::source(dir, &compiled.config).expect("the packed directory") {
        DataSource::Packed(binding) => binding,
        DataSource::Dense => panic!("{} is packed", dir.display()),
    };
    let packed = data::packed(dir, binding, &compiled.config, &compiled.forward)
        .expect("the packed data reads");
    let batches = packed.sequences(SplitRole::HeldOut) / packed.batch();
    assert!(batches > 0, "a HeldOut batch");
    (0..batches as u64)
        .map(|b| {
            packed
                .draw(SplitRole::HeldOut, b)
                .expect("a HeldOut batch")
                .weights
                .iter()
                .filter(|&&w| w > 0.0)
                .count()
        })
        .sum()
}

/// Scenario 5: with `--eval-every 5` both backends print `eval` lines for steps 0, 5 and 10,
/// each GPU loss within the run tolerance of the CPU's and each accuracy within `1 / M`.
#[test]
#[ignore = "needs a Vulkan device"]
fn heldout_eval() {
    let case = scratch("heldout_eval");
    let data = packed(&case);
    let file = twin10(&case);
    let cpu = train(&file, "cpu", &data, &["--eval-every", EVAL_EVERY]);
    assert_exit(&cpu, 0);
    let gpu = train(&file, "vulkan", &data, &["--eval-every", EVAL_EVERY]);
    assert_exit(&gpu, 0);
    let (cpu_out, gpu_out) = (stdout(&cpu), stdout(&gpu));
    let (cpu_evals, gpu_evals) = (eval_lines(&cpu_out), eval_lines(&gpu_out));
    eprintln!("cpu:");
    for line in cpu_out.lines().filter(|l| l.starts_with("eval ")) {
        eprintln!("{line}");
    }
    eprintln!("vulkan:");
    for line in gpu_out.lines().filter(|l| l.starts_with("eval ")) {
        eprintln!("{line}");
    }
    let steps: Vec<u64> = gpu_evals.iter().map(|(k, _, _)| *k).collect();
    assert_eq!(steps, [0, 5, 10], "{gpu_out}");
    assert_eq!(
        cpu_evals.iter().map(|(k, _, _)| *k).collect::<Vec<_>>(),
        steps
    );
    let m = selected_rows(&file, &data);
    eprintln!(
        "M = {m} selected HeldOut rows, 1 / M = {:.3e}",
        1.0 / m as f64
    );
    let mut outside = Vec::new();
    for ((k, c_loss, c_accuracy), (_, g_loss, g_accuracy)) in cpu_evals.iter().zip(&gpu_evals) {
        let (c, g): (f32, f32) = (c_loss.parse().unwrap(), g_loss.parse().unwrap());
        if !within(g, c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) {
            outside.push(format!("eval {k} loss: gpu {g} cpu {c}"));
        }
        let (c, g): (f64, f64) = (c_accuracy.parse().unwrap(), g_accuracy.parse().unwrap());
        if (g - c).abs() > 1.0 / m as f64 {
            outside.push(format!("eval {k} accuracy: gpu {g} cpu {c}, M {m}"));
        }
    }
    assert!(outside.is_empty(), "{outside:#?}");
}

/// A serving `modelspec` process, killed when dropped.
struct Served {
    child: Child,
    address: String,
    stdout: Option<JoinHandle<Vec<u8>>>,
    stderr: Option<JoinHandle<String>>,
}

impl Served {
    /// Starts `modelspec <args>` and waits for the `serving http://<addr>/events` line.
    fn start(args: &[&str]) -> Served {
        let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(args)
            .current_dir(repo_root())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("modelspec starts");
        let mut out = child.stdout.take().expect("stdout is piped");
        let stdout = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = out.read_to_end(&mut bytes);
            bytes
        });
        let (lines, received) = mpsc::channel();
        let err = child.stderr.take().expect("stderr is piped");
        let stderr = std::thread::spawn(move || {
            let mut all = String::new();
            for line in BufReader::new(err).lines() {
                let Ok(line) = line else { break };
                all.push_str(&line);
                all.push('\n');
                let _ = lines.send(line);
            }
            all
        });
        let mut served = Served {
            child,
            address: String::new(),
            stdout: Some(stdout),
            stderr: Some(stderr),
        };
        loop {
            match received.recv_timeout(WAIT) {
                Ok(line) => {
                    if let Some(rest) = line.strip_prefix("serving http://") {
                        let address = rest
                            .strip_suffix("/events")
                            .unwrap_or_else(|| panic!("`serving http://<addr>/events`: {line}"));
                        served.address = address.to_owned();
                        return served;
                    }
                }
                Err(error) => {
                    let status = served.child.try_wait();
                    let (_, stderr) = served.stop();
                    panic!("no `serving` line ({error}, status {status:?}): {stderr}");
                }
            }
        }
    }

    /// The `data:` lines of every frame of `GET /events`, read until the server ends it.
    fn frames(&self) -> Vec<String> {
        let mut stream = TcpStream::connect(&self.address).expect("a connection");
        stream.set_read_timeout(Some(WAIT)).unwrap();
        write!(
            stream,
            "GET /events HTTP/1.1\r\nHost: {}\r\nAccept: text/event-stream\r\n\r\n",
            self.address
        )
        .unwrap();
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("the response ends");
        let (_, body) = response.split_once("\r\n\r\n").expect("a response head");
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(str::to_owned)
            .collect()
    }

    /// Sends SIGTERM, waits for the process to exit and returns its status, stdout and stderr.
    fn terminate(&mut self) -> (ExitStatus, String, String) {
        let pid = self.child.id().to_string();
        let sent = Command::new("kill")
            .args(["-s", "TERM", &pid])
            .status()
            .expect("kill runs");
        assert!(sent.success(), "kill -s TERM {pid}: {sent}");
        let start = Instant::now();
        let status = loop {
            if let Some(status) = self.child.try_wait().expect("the status is read") {
                break status;
            }
            assert!(start.elapsed() < WAIT, "the process serves after SIGTERM");
            std::thread::sleep(Duration::from_millis(10));
        };
        let (stdout, stderr) = self.stop();
        (status, String::from_utf8(stdout).expect("UTF-8"), stderr)
    }

    /// Kills the process and returns its stdout and stderr.
    fn stop(&mut self) -> (Vec<u8>, String) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let stdout = self
            .stdout
            .take()
            .map(|h| h.join().expect("stdout reader"))
            .unwrap_or_default();
        let stderr = self
            .stderr
            .take()
            .map(|h| h.join().expect("stderr reader"))
            .unwrap_or_default();
        (stdout, stderr)
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Scenario 6: a GPU run with `--eval-every 5`, `--serve` and `--record` streams events that
/// validate against the observatory JSON Schema: `RunStarted`, one `StepCompleted` per step whose
/// loss is its `step` line as `f32` shortest round-trip text, the events of steps 0, 5 and 10
/// with `heldout_mlm_loss` and `heldout_mlm_accuracy` equal to the `eval` lines and the others
/// with neither, then `RunFinished`; the record holds the streamed events line for line, and
/// replaying it serves the same events.
#[test]
#[ignore = "needs a Vulkan device"]
fn streamed() {
    let case = scratch("streamed");
    let data = packed(&case);
    let file = twin10(&case);
    let record = case.join("gpu.run.jsonl");
    let mut served = Served::start(&[
        "train",
        &file,
        "--backend",
        "vulkan",
        "--data",
        path(&data),
        "--eval-every",
        EVAL_EVERY,
        "--record",
        path(&record),
        "--serve",
        "127.0.0.1:0",
    ]);
    let frames = served.frames();
    let (status, out, stderr) = served.terminate();
    assert_eq!(status.code(), Some(0), "{stderr}");

    let schema = std::fs::read_to_string(repo_root().join(RUN_EVENT_SCHEMA)).unwrap();
    let schema: Value = serde_json::from_str(&schema).unwrap();
    let validator = jsonschema::validator_for(&schema).expect("the schema compiles");
    let events: Vec<Value> = frames
        .iter()
        .map(|data| serde_json::from_str(data).expect("data is JSON"))
        .collect();
    for event in &events {
        let errors: Vec<String> = validator
            .iter_errors(event)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "{event}: {errors:?}");
    }
    let kinds: Vec<&str> = events
        .iter()
        .map(|e| e["body"]["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds.first(), Some(&"RunStarted"));
    assert_eq!(kinds.last(), Some(&"RunFinished"));

    let printed: BTreeMap<u64, String> = step_lines(&out).into_iter().collect();
    let evals: BTreeMap<u64, (String, String)> = eval_lines(&out)
        .into_iter()
        .map(|(k, loss, accuracy)| (k, (loss, accuracy)))
        .collect();
    assert_eq!(
        evals.keys().copied().collect::<Vec<_>>(),
        [0, 5, 10],
        "{out}"
    );
    let steps: Vec<&Value> = events
        .iter()
        .filter(|e| e["body"]["kind"] == "StepCompleted")
        .map(|e| &e["body"]["value"])
        .collect();
    assert_eq!(steps.len() as u64, STEPS + 1);
    for (k, step) in steps.iter().enumerate() {
        let k = k as u64;
        assert_eq!(step["step"], k);
        let loss = (step["loss"].as_f64().expect("a finite loss") as f32).to_string();
        if let Some(line) = printed.get(&k) {
            assert_eq!(&loss, line, "step {k}");
        }
        let metrics = step["metrics"].as_array().expect("a metrics list");
        match evals.get(&k) {
            None => assert!(metrics.is_empty(), "step {k}: {step}"),
            Some((loss, accuracy)) => {
                let named: Vec<(&str, String)> = metrics
                    .iter()
                    .map(|m| {
                        let value = m["value"].as_f64().expect("a finite value") as f32;
                        (m["name"].as_str().unwrap(), value.to_string())
                    })
                    .collect();
                assert_eq!(
                    named,
                    [
                        ("heldout_mlm_loss", loss.clone()),
                        ("heldout_mlm_accuracy", accuracy.clone()),
                    ],
                    "step {k}"
                );
            }
        }
    }

    let recorded: Vec<String> = std::fs::read_to_string(&record)
        .expect("the recording")
        .lines()
        .map(str::to_owned)
        .collect();
    assert_eq!(recorded, frames, "the record holds the streamed events");
    let mut replay = Served::start(&["observe", path(&record), "--serve", "127.0.0.1:0"]);
    let replayed = replay.frames();
    let (status, _, stderr) = replay.terminate();
    assert_eq!(status.code(), Some(0), "{stderr}");
    assert_eq!(replayed, frames, "the replay serves the same events");
}
