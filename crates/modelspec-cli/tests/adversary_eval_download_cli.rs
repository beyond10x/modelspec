//! Adversary pass 1 on story:vulkan-eval-download-per-batch (coordinator decision D76): `modelspec
//! train --backend vulkan --data <packed-dir> --eval-every <n>` against `--backend cpu` at the
//! edges of the per-batch evaluation: one full `HeldOut` batch and a remainder, too few
//! sequences for one batch, a resumed run, the recorded `StepCompleted` metrics, a run that
//! diverges at an evaluated step, three runs of the twin over 123 batches, and the device memory
//! the evaluation adds. Since story:vulkan-eval-metrics-on-device the CLI reduces each
//! evaluation on the device in one command buffer, and since
//! story:vulkan-stepwise-multi-step-submit a host wait ends each segment of steps and each
//! evaluation, so the counts here are those host waits and the reduction's state; the per-batch
//! reduction is held by the runtime crate's `tests/adversary_eval_download.rs`.
//!
//! Every case is a device case, `#[ignore]` as "device"; run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-cli --features vulkan --test adversary_eval_download_cli -- --ignored`.
//! The `mlm_bag` cases pack the original-text fixture `corpus-tokenizer` of the runtime crate at
//! 300 tokens and seq-len 16 (9 `HeldOut` sequences); the twin cases pack the synthetic corpus of
//! `common/markov_corpus.rs` with 30 held-out articles, as `vulkan_eval_per_batch.rs` does.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::compile::compile_training;
use modelspec_runtime::dataset::SplitRole;
use modelspec_runtime::spirv::emit::HELD_OUT_STATE;
use modelspec_runtime::train::data::{self, DataSource, PackedData};
use serde_json::Value;

#[allow(dead_code)]
#[path = "../../modelspec-runtime/tests/common/mod.rs"]
mod common;

#[allow(dead_code)]
#[path = "common/markov_corpus.rs"]
mod markov_corpus;

use common::{GPU_RUN_ATOL, GPU_RUN_RTOL, within};

const TWIN: &str = "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec";
const BAG: &str = "crates/modelspec-cli/tests/fixtures/mlm_bag.mspec";
const FIXTURE: &str = "crates/modelspec-runtime/tests/fixtures/corpus-tokenizer";
const SEQ_LEN: usize = 16;
const VOCAB: u64 = 300;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adv_eval_download_cli_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

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

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout: {}\nstderr: {}",
        stdout(output),
        stderr(output)
    );
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

/// `data prepare`, `data tokenizer --vocab-size 300` and `data pack --seq-len 16` of the corpus
/// directory `corpus` under `case`; returns the packed directory.
fn pack(case: &Path, corpus: &Path) -> PathBuf {
    let dataset = case.join("dataset");
    assert_exit(
        &modelspec(&["data", "prepare", path(corpus), "--out", path(&dataset)]),
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
            &VOCAB.to_string(),
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
            &SEQ_LEN.to_string(),
            "--out",
            path(&packed),
        ]),
        0,
    );
    packed
}

/// The runtime crate's `corpus-tokenizer` fixture packed under `case`.
fn fixture_packed(case: &Path) -> PathBuf {
    pack(case, &repo_root().join(FIXTURE))
}

/// The synthetic corpus with 30 held-out articles packed under `case`.
fn markov_packed(case: &Path) -> PathBuf {
    let corpus = case.join("corpus");
    markov_corpus::write_with(&corpus, 30);
    pack(case, &corpus)
}

/// `mlm_bag.mspec` at batch `b`, `steps` steps and SGD rate `lr`, written to `<case>/bag.mspec`.
fn bag(case: &Path, b: usize, steps: u64, lr: &str) -> String {
    let mut source = std::fs::read_to_string(repo_root().join(BAG)).expect("the fixture");
    let rows = b * SEQ_LEN;
    for (from, to) in [
        ("[64, 300]".to_owned(), format!("[{rows}, 300]")),
        ("[64]".to_owned(), format!("[{rows}]")),
        ("steps = 200;".to_owned(), format!("steps = {steps};")),
        ("sgd(lr = 1.0)".to_owned(), format!("sgd(lr = {lr})")),
    ] {
        assert!(source.contains(&from), "the fixture holds {from:?}");
        source = source.replace(&from, &to);
    }
    let file = case.join("bag.mspec");
    std::fs::write(&file, source).expect("the variant is written");
    path(&file).to_owned()
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

fn packed_data(file: &str, dir: &Path) -> PackedData {
    let source = std::fs::read_to_string(file).expect("the variant");
    let compiled = compile_training(file, &source).expect("the variant compiles");
    let binding = match data::source(dir, &compiled.config).expect("the packed directory") {
        DataSource::Packed(binding) => binding,
        DataSource::Dense => panic!("{} is packed", dir.display()),
    };
    data::packed(dir, binding, &compiled.config, &compiled.forward).expect("the packed data reads")
}

/// The full `HeldOut` batches of `dir` for `file`, the sequences left over, and the rows of
/// positive weight over the full batches.
fn heldout(file: &str, dir: &Path) -> (usize, usize, usize) {
    let packed = packed_data(file, dir);
    let sequences = packed.sequences(SplitRole::HeldOut);
    let batches = sequences / packed.batch();
    let selected = (0..batches as u64)
        .map(|b| {
            packed
                .draw(SplitRole::HeldOut, b)
                .expect("a HeldOut batch")
                .weights
                .iter()
                .filter(|&&w| w > 0.0)
                .count()
        })
        .sum();
    (batches, sequences % packed.batch(), selected)
}

fn train(file: &str, backend: &str, data: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["train", file, "--backend", backend, "--data", path(data)];
    args.extend_from_slice(extra);
    modelspec(&args)
}

/// The `(step, loss, accuracy)` texts of every `eval <k> loss <v> accuracy <a>` line.
fn eval_lines(out: &str) -> Vec<(u64, String, String)> {
    out.lines()
        .filter_map(|line| line.strip_prefix("eval "))
        .map(|rest| {
            let words: Vec<&str> = rest.split_whitespace().collect();
            assert_eq!(words.len(), 5, "{rest}");
            (
                words[0].parse().expect("a step"),
                words[2].to_owned(),
                words[4].to_owned(),
            )
        })
        .collect()
}

fn eval_steps(out: &str) -> Vec<u64> {
    eval_lines(out).into_iter().map(|(k, _, _)| k).collect()
}

/// The value of the `<name>: <value>` line of a `--stats` stderr.
fn counter(stderr: &str, name: &str) -> u64 {
    let prefix = format!("{name}: ");
    stderr
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no `{name}` line: {stderr}"))
        .parse()
        .unwrap_or_else(|error| panic!("`{name}`: {error}: {stderr}"))
}

/// The `eval` lines of `gpu` that are not within the run tolerance of `cpu`'s (loss) or `1 / m`
/// (accuracy), and a different set of evaluated steps.
fn outside(cpu: &str, gpu: &str, m: usize) -> Vec<String> {
    let (c, g) = (eval_lines(cpu), eval_lines(gpu));
    let mut out = Vec::new();
    if eval_steps(cpu) != eval_steps(gpu) {
        out.push(format!(
            "evaluated steps: cpu {:?} gpu {:?}",
            eval_steps(cpu),
            eval_steps(gpu)
        ));
    }
    for ((k, cl, ca), (_, gl, ga)) in c.iter().zip(&g) {
        let (cl, gl): (f32, f32) = (cl.parse().unwrap(), gl.parse().unwrap());
        if !within(gl, cl, cl.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) {
            out.push(format!("eval {k} loss: gpu {gl} cpu {cl}"));
        }
        let (ca, ga): (f64, f64) = (ca.parse().unwrap(), ga.parse().unwrap());
        if (ga - ca).abs() > 1.0 / m as f64 {
            out.push(format!("eval {k} accuracy: gpu {ga} cpu {ca}, M {m}"));
        }
    }
    out
}

/// A `HeldOut` split of exactly one full batch and a remainder (`mlm_bag` at `B = 5` over the
/// fixture's 9 sequences): `--eval-every 1 --steps 3` on the device evaluates steps 0 to 3
/// within the run tolerance of the CPU, with `4 + 4` host waits and submits: each step a segment
/// of its own, since the next is evaluated, and one per evaluation, none per batch
/// (story:vulkan-eval-metrics-on-device, story:vulkan-stepwise-multi-step-submit).
#[test]
#[ignore = "device: needs a Vulkan device"]
fn adv_ed_cli_one_full_batch_and_a_remainder_matches_the_cpu() {
    let case = scratch("one_batch");
    let data = fixture_packed(&case);
    let file = bag(&case, 5, 3, "1.0");
    let (batches, left, m) = heldout(&file, &data);
    assert_eq!(
        (batches, left),
        (1, 4),
        "one full batch of 5 and 4 sequences left"
    );
    let cpu = train(&file, "cpu", &data, &["--eval-every", "1"]);
    let gpu = train(&file, "vulkan", &data, &["--eval-every", "1", "--stats"]);
    assert_exit(&cpu, 0);
    assert_exit(&gpu, 0);
    let wrong = outside(&stdout(&cpu), &stdout(&gpu), m);
    assert!(
        wrong.is_empty(),
        "{wrong:#?}\ncpu {}\ngpu {}",
        stdout(&cpu),
        stdout(&gpu)
    );
    assert_eq!(eval_steps(&stdout(&gpu)), [0, 1, 2, 3]);
    let stats = stderr(&gpu);
    assert_eq!(counter(&stats, "host waits"), 4 + 4, "{stats}");
    assert_eq!(counter(&stats, "submits"), 4 + 4, "{stats}");
}

/// A `HeldOut` split of fewer sequences than one batch is refused on the device as on the CPU:
/// the same exit code and the same refusal line, no panic and no device run.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn adv_ed_cli_too_few_heldout_sequences_is_refused_as_on_the_cpu() {
    let case = scratch("too_few");
    let data = fixture_packed(&case);
    let probe = bag(&case, 1, 2, "1.0");
    let packed = packed_data(&probe, &data);
    let (held, trained) = (
        packed.sequences(SplitRole::HeldOut),
        packed.sequences(SplitRole::Train),
    );
    let b = held + 1;
    assert!(
        trained >= b,
        "{trained} Train sequences fill a batch of {b}"
    );
    let file = bag(&case, b, 2, "1.0");
    let cpu = train(&file, "cpu", &data, &["--eval-every", "1"]);
    let gpu = train(&file, "vulkan", &data, &["--eval-every", "1"]);
    let refusal = |output: &Output| {
        stderr(output)
            .lines()
            .find(|line| line.contains("fewer than the batch"))
            .map(str::to_owned)
    };
    assert_eq!(cpu.status.code(), Some(1), "cpu stderr: {}", stderr(&cpu));
    assert_eq!(
        gpu.status.code(),
        cpu.status.code(),
        "gpu stderr: {}",
        stderr(&gpu)
    );
    assert!(refusal(&cpu).is_some(), "cpu stderr: {}", stderr(&cpu));
    assert_eq!(refusal(&gpu), refusal(&cpu), "gpu stderr: {}", stderr(&gpu));
    assert!(!stderr(&gpu).contains("panicked"), "{}", stderr(&gpu));
}

/// A run resumed at step 4 with `--steps 6 --eval-every 5` evaluates steps 5 and 10 and no
/// step of its own start, on the device as on the CPU, within the run tolerance, with `3 + 2`
/// host waits: the segments 4, 5 to 9 and 10, and the evaluations at 5 and 10, none per batch
/// (story:vulkan-eval-metrics-on-device, story:vulkan-stepwise-multi-step-submit).
#[test]
#[ignore = "device: needs a Vulkan device"]
fn adv_ed_cli_resumed_run_evaluates_from_the_next_multiple() {
    let case = scratch("resume");
    let data = fixture_packed(&case);
    let file = bag(&case, 2, 4, "1.0");
    let (_, _, m) = heldout(&file, &data);
    let checkpoint = case.join("at4.msckpt");
    assert_exit(
        &train(&file, "cpu", &data, &["--checkpoint", path(&checkpoint)]),
        0,
    );
    let resumed = [
        "--resume",
        path(&checkpoint),
        "--steps",
        "6",
        "--eval-every",
        "5",
    ];
    let cpu = train(&file, "cpu", &data, &resumed);
    let mut on_device = resumed.to_vec();
    on_device.push("--stats");
    let gpu = train(&file, "vulkan", &data, &on_device);
    assert_exit(&cpu, 0);
    assert_exit(&gpu, 0);
    assert_eq!(eval_steps(&stdout(&cpu)), [5, 10], "{}", stdout(&cpu));
    let wrong = outside(&stdout(&cpu), &stdout(&gpu), m);
    assert!(
        wrong.is_empty(),
        "{wrong:#?}\ncpu {}\ngpu {}",
        stdout(&cpu),
        stdout(&gpu)
    );
    let stats = stderr(&gpu);
    assert_eq!(counter(&stats, "steps"), 6, "{stats}");
    assert_eq!(counter(&stats, "host waits"), 3 + 2, "{stats}");
}

/// The `StepCompleted` of an evaluated step of a recorded device run carries
/// `heldout_mlm_loss` and `heldout_mlm_accuracy` whose `f32` texts are the printed `eval`
/// line's; every other step's metrics are empty.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn adv_ed_cli_record_carries_the_printed_eval() {
    let case = scratch("record");
    let data = fixture_packed(&case);
    let file = bag(&case, 2, 4, "1.0");
    let record = case.join("run.jsonl");
    let gpu = train(
        &file,
        "vulkan",
        &data,
        &["--eval-every", "2", "--record", path(&record)],
    );
    assert_exit(&gpu, 0);
    let printed: Vec<(u64, String, String)> = eval_lines(&stdout(&gpu));
    assert_eq!(
        printed.iter().map(|(k, _, _)| *k).collect::<Vec<_>>(),
        [0, 2, 4]
    );
    let events: Vec<Value> = std::fs::read_to_string(&record)
        .expect("the recording")
        .lines()
        .map(|line| serde_json::from_str(line).expect("a JSON event"))
        .filter(|event: &Value| event["body"]["kind"] == "StepCompleted")
        .collect();
    let steps: Vec<u64> = events
        .iter()
        .map(|e| e["body"]["value"]["step"].as_u64().expect("a step"))
        .collect();
    assert_eq!(steps, [0, 1, 2, 3, 4]);
    let mut wrong = Vec::new();
    for event in &events {
        let value = &event["body"]["value"];
        let k = value["step"].as_u64().unwrap();
        let metrics = value["metrics"].as_array().expect("a metrics list");
        let named = |name: &str| {
            metrics
                .iter()
                .find(|m| m["name"] == name)
                .and_then(|m| m["value"].as_f64())
                .map(|v| (v as f32).to_string())
        };
        match printed.iter().find(|(p, _, _)| *p == k) {
            Some((_, loss, accuracy)) => {
                if named("heldout_mlm_loss").as_ref() != Some(loss)
                    || named("heldout_mlm_accuracy").as_ref() != Some(accuracy)
                    || metrics.len() != 2
                {
                    wrong.push(format!(
                        "step {k}: printed {loss} {accuracy}, recorded {metrics:?}"
                    ));
                }
            }
            None => {
                if !metrics.is_empty() {
                    wrong.push(format!("step {k}: not evaluated, recorded {metrics:?}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// SGD at `lr = 1e30` overflows the parameters after step 0, so step 1 is a non-finite loss at
/// an evaluated step (`--eval-every 1`): the device run stops as the CPU run does, with the
/// same exit code, an E-TRAIN-001 diagnostic, the same `step` and `eval` steps printed, and no
/// held-out error in place of the diagnostic.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn adv_ed_cli_divergence_at_an_evaluated_step_stops_as_on_the_cpu() {
    let case = scratch("diverge");
    let data = fixture_packed(&case);
    let file = bag(&case, 2, 3, "1e30");
    let cpu = train(&file, "cpu", &data, &["--eval-every", "1"]);
    let gpu = train(&file, "vulkan", &data, &["--eval-every", "1"]);
    let steps = |out: &str, prefix: &str| -> Vec<String> {
        out.lines()
            .filter_map(|line| line.strip_prefix(prefix))
            .map(|rest| {
                rest.split_whitespace()
                    .next()
                    .unwrap_or_default()
                    .to_owned()
            })
            .collect()
    };
    let report = format!(
        "cpu exit {:?}\n{}{}\ngpu exit {:?}\n{}{}",
        cpu.status.code(),
        stdout(&cpu),
        stderr(&cpu),
        gpu.status.code(),
        stdout(&gpu),
        stderr(&gpu)
    );
    assert_eq!(cpu.status.code(), Some(1), "{report}");
    assert_eq!(gpu.status.code(), cpu.status.code(), "{report}");
    assert!(stderr(&cpu).contains("E-TRAIN-001"), "{report}");
    assert!(stderr(&gpu).contains("E-TRAIN-001"), "{report}");
    assert_eq!(
        steps(&stdout(&gpu), "step "),
        steps(&stdout(&cpu), "step "),
        "{report}"
    );
    assert_eq!(
        steps(&stdout(&gpu), "eval "),
        steps(&stdout(&cpu), "eval "),
        "{report}"
    );
}

/// The twin over 123 full `HeldOut` batches, `--eval-every 5 --steps 10`, three times on the
/// device: the same stdout, byte for byte.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn adv_ed_cli_three_twin_runs_print_the_same_bytes() {
    let case = scratch("three_runs");
    let data = markov_packed(&case);
    let file = twin10(&case);
    let outs: Vec<String> = (0..3)
        .map(|_| {
            let gpu = train(&file, "vulkan", &data, &["--eval-every", "5"]);
            assert_exit(&gpu, 0);
            stdout(&gpu)
        })
        .collect();
    assert_eq!(eval_steps(&outs[0]), [0, 5, 10]);
    assert_eq!(outs[0], outs[1]);
    assert_eq!(outs[0], outs[2]);
}

/// The twin over 123 full `HeldOut` batches: `--eval-every 5` adds to the run's peak device bytes
/// one prediction and the held-out batches' data (each batch's five tensors of `B·S` 4-byte
/// elements at most), within 1 MiB of allocation rounding, not a prediction per batch; and to
/// its bytes downloaded exactly the device reduction's state per evaluated step, no prediction
/// (story:vulkan-eval-metrics-on-device).
#[test]
#[ignore = "device: needs a Vulkan device"]
fn adv_ed_cli_eval_adds_one_prediction_of_device_memory() {
    let case = scratch("memory");
    let data = markov_packed(&case);
    let file = twin10(&case);
    let (batches, _, _) = heldout(&file, &data);
    assert!(batches >= 64, "{batches} batches");
    let plain = train(&file, "vulkan", &data, &["--stats"]);
    let evaluated = train(&file, "vulkan", &data, &["--eval-every", "5", "--stats"]);
    assert_exit(&plain, 0);
    assert_exit(&evaluated, 0);
    let (p, e) = (stderr(&plain), stderr(&evaluated));
    let rows = 2 * SEQ_LEN as u64;
    let prediction = rows * VOCAB * 4;
    let held_data = batches as u64 * 5 * rows * 4;
    let added = counter(&e, "peak device bytes") - counter(&p, "peak device bytes");
    let bound = prediction + held_data + (1 << 20);
    eprintln!(
        "{batches} batches: peak device bytes {} without eval, {} with, {added} added; bound \
         {bound}; a prediction per batch is {}",
        counter(&p, "peak device bytes"),
        counter(&e, "peak device bytes"),
        batches as u64 * prediction
    );
    assert!(
        added <= bound,
        "{added} peak device bytes added, above {bound}\n{p}\n{e}"
    );
    assert_eq!(
        counter(&e, "bytes downloaded") - counter(&p, "bytes downloaded"),
        3 * 4 * HELD_OUT_STATE,
        "{p}\n{e}"
    );
}
