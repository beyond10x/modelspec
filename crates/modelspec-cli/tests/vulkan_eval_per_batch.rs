//! `modelspec train --backend vulkan --data <packed-dir> --eval-every 5` on the hidden-32 twin of
//! BERT-tiny over a `HeldOut` split of at least 64 full batches (story:vulkan-eval-download-per-batch,
//! coordinator decision D76, 2026-10-26; story:vulkan-eval-metrics-on-device, 2026-10-28): the
//! stepwise run evaluates every held-out batch in one command buffer before the evaluated step's
//! and reduces its logits on the device, so its `eval` lines are within the run tolerance of
//! DESIGN.md § 9 gate 5 of the CPU run's, `|g - c| <= GPU_RUN_ATOL + GPU_RUN_RTOL * |c|` for each
//! loss and `1 / M` for each accuracy, `M` the selected `HeldOut` rows, and `--stats` counts one
//! host wait and submit per segment of steps and one per evaluation, none for the batches
//! (story:vulkan-stepwise-multi-step-submit). The per-batch host waits of D76 remain the
//! host reduction of `HeldOutReduction::Host`, which `modelspec-runtime`'s
//! `tests/adversary_eval_download.rs` holds.
//!
//! The run is the twin with `steps = 10` and `warmup_steps = 2`, written beside the packed data,
//! as `vulkan_bert_cpu_steps.rs` writes it: the fixture's warm-up of 80 steps is refused by a run
//! of 10. The packed directory is made from the synthetic corpus of `common/markov_corpus.rs`
//! with [`VALID_ARTICLES`] held-out articles instead of its usual 10, whose split holds fewer
//! than 64 batches of the twin's 2 sequences, under this crate's build directory.
//!
//! The device test is ignored by default; run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-cli --features vulkan --test vulkan_eval_per_batch -- --include-ignored`.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::compile::compile_training;
use modelspec_runtime::dataset::SplitRole;
use modelspec_runtime::train::data::{self, DataSource};

#[allow(dead_code)]
#[path = "../../modelspec-runtime/tests/common/mod.rs"]
mod common;

#[allow(dead_code)]
#[path = "common/markov_corpus.rs"]
mod markov_corpus;

use common::{GPU_RUN_ATOL, GPU_RUN_RTOL, within};

const TWIN: &str = "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec";
const STEPS: u64 = 10;
const EVAL_EVERY: u64 = 5;
/// The evaluated steps of a run from step 0 to [`STEPS`] every [`EVAL_EVERY`]: 0, 5 and 10.
const EVALUATED: [u64; 3] = [0, 5, 10];
/// The held-out articles of the synthetic corpus: enough for more than [`MIN_BATCHES`] batches.
const VALID_ARTICLES: usize = 30;
/// The full `HeldOut` batches the run evaluates at least (the story's scenario 2).
const MIN_BATCHES: usize = 64;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh, empty directory under this crate's build directory, named for the test.
fn scratch(name: &str) -> PathBuf {
    let dir =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("vulkan_eval_per_batch_{name}"));
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

/// The synthetic corpus with [`VALID_ARTICLES`] held-out articles written and prepared, a
/// 300-token tokenizer trained on its `Train` split and the two packed into sequences of 16, all
/// under `case`; returns the packed directory.
fn packed(case: &Path) -> PathBuf {
    let corpus = case.join("corpus");
    markov_corpus::write_with(&corpus, VALID_ARTICLES);
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

/// The full `HeldOut` batches of `dir` for `file` and the `M` of the accuracy rule, the rows of
/// positive weight over all of them.
fn heldout(file: &str, dir: &Path) -> (usize, usize) {
    let source = std::fs::read_to_string(file).expect("the variant");
    let compiled = compile_training(file, &source).expect("the twin compiles");
    let binding = match data::source(dir, &compiled.config).expect("the packed directory") {
        DataSource::Packed(binding) => binding,
        DataSource::Dense => panic!("{} is packed", dir.display()),
    };
    let packed = data::packed(dir, binding, &compiled.config, &compiled.forward)
        .expect("the packed data reads");
    let batches = packed.sequences(SplitRole::HeldOut) / packed.batch();
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
    (batches, selected)
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

fn bound(c: f32) -> f64 {
    f64::from(GPU_RUN_ATOL) + f64::from(GPU_RUN_RTOL) * f64::from(c).abs()
}

/// Scenario 2: over at least 64 full `HeldOut` batches, `--eval-every 5 --steps 10` on the
/// device exits 0, prints `eval` lines for steps 0, 5 and 10, each loss within the run tolerance
/// of the CPU's and each accuracy within `1 / M`, and its `--stats` count one host wait and one
/// submit per step and one for the final loss, none per batch of an evaluated step.
#[test]
#[ignore = "needs a Vulkan device"]
fn every_evaluation_rides_its_steps_host_wait() {
    let case = scratch("eval");
    let data = packed(&case);
    let file = twin10(&case);
    let (batches, m) = heldout(&file, &data);
    assert!(
        batches >= MIN_BATCHES,
        "{batches} full HeldOut batches, fewer than {MIN_BATCHES}"
    );
    let every = EVAL_EVERY.to_string();
    let steps = STEPS.to_string();
    let data_dir = path(&data);
    let cpu = modelspec(&[
        "train",
        &file,
        "--backend",
        "cpu",
        "--data",
        data_dir,
        "--eval-every",
        &every,
        "--steps",
        &steps,
    ]);
    assert_exit(&cpu, 0);
    let gpu = modelspec(&[
        "train",
        &file,
        "--backend",
        "vulkan",
        "--data",
        data_dir,
        "--eval-every",
        &every,
        "--steps",
        &steps,
        "--stats",
    ]);
    assert_exit(&gpu, 0);
    let (cpu_out, gpu_out) = (stdout(&cpu), stdout(&gpu));
    let (cpu_evals, gpu_evals) = (eval_lines(&cpu_out), eval_lines(&gpu_out));
    assert_eq!(
        gpu_evals.iter().map(|(k, _, _)| *k).collect::<Vec<_>>(),
        EVALUATED,
        "{gpu_out}"
    );
    assert_eq!(
        cpu_evals.iter().map(|(k, _, _)| *k).collect::<Vec<_>>(),
        EVALUATED,
        "{cpu_out}"
    );
    eprintln!(
        "{batches} full HeldOut batches, M = {m} selected rows, 1 / M = {:.3e}",
        1.0 / m as f64
    );
    eprintln!(
        "step | CPU loss | GPU loss | loss ratio to bound | CPU accuracy | GPU accuracy | accuracy ratio to 1 / M"
    );
    let mut outside = Vec::new();
    let (mut worst_loss, mut worst_accuracy) = (0.0f64, 0.0f64);
    for ((k, c_loss, c_accuracy), (_, g_loss, g_accuracy)) in cpu_evals.iter().zip(&gpu_evals) {
        let (c, g): (f32, f32) = (c_loss.parse().unwrap(), g_loss.parse().unwrap());
        let loss_ratio = (f64::from(g) - f64::from(c)).abs() / bound(c);
        worst_loss = worst_loss.max(loss_ratio);
        if !within(g, c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) {
            outside.push(format!("eval {k} loss: gpu {g} cpu {c}"));
        }
        let (ca, ga): (f64, f64) = (c_accuracy.parse().unwrap(), g_accuracy.parse().unwrap());
        let accuracy_ratio = (ga - ca).abs() * m as f64;
        worst_accuracy = worst_accuracy.max(accuracy_ratio);
        if (ga - ca).abs() > 1.0 / m as f64 {
            outside.push(format!("eval {k} accuracy: gpu {ga} cpu {ca}, M {m}"));
        }
        eprintln!("{k} | {c} | {g} | {loss_ratio:.3e} | {ca} | {ga} | {accuracy_ratio:.3e}");
    }
    eprintln!("worst loss ratio {worst_loss:.3e}, worst accuracy ratio {worst_accuracy:.3e}");
    assert!(outside.is_empty(), "{outside:#?}");

    let stats = stderr(&gpu);
    eprintln!("{stats}");
    // The segments 0 to 4, 5 to 9 and 10, and the evaluations at 0, 5 and 10.
    let expected = 3 + EVALUATED.len() as u64;
    assert_eq!(counter(&stats, "steps"), STEPS, "{stats}");
    assert_eq!(counter(&stats, "host waits"), expected, "{stats}");
    assert_eq!(counter(&stats, "submits"), expected, "{stats}");
}
