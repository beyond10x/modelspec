//! story:vulkan-stepwise-multi-step-submit: a stepwise Vulkan run records several steps into one
//! command buffer and waits once per segment, and prints the bytes of the run that records one
//! step per command buffer.
//!
//! Each case trains the hidden-32 twin of BERT-tiny, `crates/modelspec-cli/tests/fixtures/bert_cpu.mspec`
//! (dropout, `adamw`), or its variant with `sgd`, which takes `--checkpoint-every`, on a packed
//! synthetic corpus made as `bert_cpu_train.rs` makes it (`common/markov_corpus.rs`), with the
//! hidden `--submit-steps 1`, `--submit-steps 3` and the runner's own choice, and compares stdout,
//! the recorded `StepCompleted` bodies, the checkpoint files and, for a run that diverges, stderr.
//!
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-cli --test train_vulkan_submit_steps -- --ignored`
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

#[path = "common/markov_corpus.rs"]
mod markov_corpus;

const TWIN: &str = "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("submit_steps_{name}"));
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

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout: {}\nstderr: {}",
        text(&output.stdout),
        text(&output.stderr)
    );
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

/// The synthetic corpus prepared, a 300-token tokenizer trained on it and the two packed into
/// sequences of 16 under `case`; returns the packed directory.
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

/// The twin with its optimizer line replaced by `optimizer`, written to `<dir>/<name>`.
fn with_optimizer(dir: &Path, name: &str, optimizer: &str) -> String {
    let source = std::fs::read_to_string(repo_root().join(TWIN)).expect("the twin");
    let adamw = source
        .lines()
        .find(|line| line.trim_start().starts_with("optimizer = adamw("))
        .expect("the twin trains with adamw");
    let file = dir.join(name);
    std::fs::write(&file, source.replace(adamw, &format!("    {optimizer}"))).expect("written");
    file.to_str().expect("UTF-8").to_owned()
}

/// The `submit` settings each case compares: one step per command buffer, three, and the
/// runner's own choice.
const SUBMITS: [Option<&str>; 3] = [Some("1"), Some("3"), None];

/// `train <file> --backend vulkan --stats` with `args` and, when given, `--submit-steps <n>`.
fn train(file: &str, args: &[&str], submit: Option<&str>) -> Output {
    let mut all = vec!["train", file, "--backend", "vulkan", "--stats"];
    all.extend_from_slice(args);
    if let Some(n) = submit {
        all.extend_from_slice(&["--submit-steps", n]);
    }
    modelspec(&all)
}

/// The value of the `--stats` counter `name`.
fn counter(stderr: &str, name: &str) -> u64 {
    stderr
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{name}: ")))
        .unwrap_or_else(|| panic!("no `{name}` counter:\n{stderr}"))
        .trim()
        .parse()
        .expect("a count")
}

/// The bodies of the `StepCompleted` events of a recorded run, in order.
fn step_bodies(record: &Path) -> Vec<Value> {
    std::fs::read_to_string(record)
        .expect("the recording")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("a JSON event"))
        .filter(|event| event["body"]["kind"] == "StepCompleted")
        .map(|event| event["body"].clone())
        .collect()
}

/// Dropout, `adamw`, packed data and `--eval-every 30` over 100 steps: every submit setting
/// prints the bytes of one step per command buffer and records the same `StepCompleted` bodies,
/// and the runner's choice waits fewer times than one step per command buffer, which waits once
/// per step and once per evaluation.
#[test]
#[ignore = "needs a Vulkan device"]
fn device_adamw_dropout_and_held_out_print_the_lines_of_one_step_per_submit() {
    let case = scratch("adamw");
    let packed = packed(&case);
    let mut runs = Vec::new();
    for submit in SUBMITS {
        let record = case.join(format!("run-{}.jsonl", submit.unwrap_or("auto")));
        let output = train(
            TWIN,
            &[
                "--data",
                path(&packed),
                "--steps",
                "100",
                "--eval-every",
                "30",
                "--record",
                path(&record),
            ],
            submit,
        );
        assert_exit(&output, 0);
        runs.push((submit, output, step_bodies(&record)));
    }
    let (_, one, one_bodies) = &runs[0];
    let printed = text(&one.stdout);
    assert!(printed.contains("\neval 30 loss "), "{printed}");
    assert!(printed.contains("\neval 100 loss "), "{printed}");
    assert_eq!(one_bodies.len(), 101);
    for (submit, output, bodies) in &runs[1..] {
        assert_eq!(text(&output.stdout), printed, "--submit-steps {submit:?}");
        assert_eq!(bodies, one_bodies, "--submit-steps {submit:?}");
    }
    let waits = |output: &Output| counter(&text(&output.stderr), "host waits");
    // One per step and the final loss, and one per evaluation at steps 0, 30, 60, 90 and 100.
    assert_eq!(waits(one), 101 + 5, "{}", text(&one.stderr));
    let chosen = waits(&runs[2].1);
    assert!(chosen < 40, "the runner's choice waits {chosen} times");
}

/// Dropout, `sgd`, packed data, `--eval-every 30` and `--checkpoint-every 25` over 100 steps:
/// every submit setting prints the same bytes and writes the same checkpoint files, byte for
/// byte.
#[test]
#[ignore = "needs a Vulkan device"]
fn device_checkpoints_inside_the_run_are_the_files_of_one_step_per_submit() {
    let case = scratch("checkpoints");
    let packed = packed(&case);
    let file = with_optimizer(&case, "twin_sgd.mspec", "optimizer = sgd(lr = 0.05);");
    let mut runs = Vec::new();
    for submit in SUBMITS {
        let dir = case.join(format!("ckpt-{}", submit.unwrap_or("auto")));
        std::fs::create_dir_all(&dir).expect("a checkpoint directory");
        let output = train(
            &file,
            &[
                "--data",
                path(&packed),
                "--steps",
                "100",
                "--eval-every",
                "30",
                "--checkpoint",
                path(&dir.join("run.msckpt")),
                "--checkpoint-every",
                "25",
            ],
            submit,
        );
        assert_exit(&output, 0);
        let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(&dir)
            .expect("the checkpoints")
            .map(|entry| {
                let entry = entry.expect("an entry");
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    std::fs::read(entry.path()).expect("a checkpoint"),
                )
            })
            .collect();
        files.sort();
        runs.push((submit, text(&output.stdout), files));
    }
    let (_, printed, files) = &runs[0];
    let names: Vec<&str> = files.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        [
            "run-25.msckpt",
            "run-50.msckpt",
            "run-75.msckpt",
            "run.msckpt"
        ]
    );
    for (submit, other, other_files) in &runs[1..] {
        assert_eq!(other, printed, "--submit-steps {submit:?}");
        assert!(
            other_files == files,
            "--submit-steps {submit:?}: the checkpoints differ"
        );
    }
}

/// A run whose loss stops being finite stops at the same step with the same stdout and the same
/// diagnostic, whatever the submit setting.
#[test]
#[ignore = "needs a Vulkan device"]
fn device_a_divergent_run_stops_where_one_step_per_submit_stops() {
    let case = scratch("diverged");
    let packed = packed(&case);
    let file = with_optimizer(&case, "twin_diverges.mspec", "optimizer = sgd(lr = 1e30);");
    let mut runs = Vec::new();
    for submit in SUBMITS {
        let mut args = vec!["train", &file, "--backend", "vulkan"];
        args.extend_from_slice(&["--data", path(&packed), "--steps", "100"]);
        if let Some(n) = submit {
            args.extend_from_slice(&["--submit-steps", n]);
        }
        let output = modelspec(&args);
        assert_exit(&output, 1);
        runs.push((submit, text(&output.stdout), text(&output.stderr)));
    }
    let (_, printed, diagnostic) = &runs[0];
    assert!(diagnostic.starts_with("E-TRAIN-001"), "{diagnostic}");
    for (submit, other, other_diagnostic) in &runs[1..] {
        assert_eq!(other, printed, "--submit-steps {submit:?}");
        assert_eq!(other_diagnostic, diagnostic, "--submit-steps {submit:?}");
    }
}
