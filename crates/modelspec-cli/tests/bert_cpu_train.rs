//! `modelspec train --backend cpu --data <packed-dir> --eval-every <n>` on the hidden-32 twin of
//! BERT-tiny, `crates/modelspec-cli/tests/fixtures/bert_cpu.mspec` (story:bert-tiny-cpu-model).
//!
//! The packed directory is made as `train_data.rs` makes it, with `modelspec data prepare`,
//! `data tokenizer --vocab-size 300` and `data pack --seq-len 16`, from a synthetic corpus the
//! test writes (`common/markov_corpus.rs`, coordinator decision D72), all in a directory under
//! this crate's build directory (`CARGO_TARGET_TMPDIR`, under the tree's `target/tmp/`). The
//! corpus fixture of story:bert-corpus-splits packs to 20 training sequences, too few for the
//! twin's held-out loss to show it learns.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[path = "common/markov_corpus.rs"]
mod markov_corpus;

const TWIN: &str = "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec";
/// How far below the held-out unigram baseline the last held-out loss ends, in nats.
const MARGIN: f64 = 0.2;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh, empty directory under this crate's build directory, named for the test.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("bert_cpu_train_{name}"));
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

/// The `(step, loss)` of every `step` line.
fn losses(stdout: &str) -> Vec<(u64, f32)> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (step.parse().unwrap(), loss.parse().unwrap())
        })
        .collect()
}

/// The `--eval-every` of the run.
const EVAL_EVERY: &str = "100";

/// Scenario 4: the twin, trained twice from its one seed on the packed synthetic corpus with
/// `--eval-every`, exits 0 both times and prints identical stdout, `step` and `eval` lines
/// included, and the held-out loss of its last `eval` line is at least [`MARGIN`] nats below the
/// held-out unigram baseline `data pack` records in `packed.json`: the twin learns more than
/// token frequencies. A `step` loss is one batch of 2 sequences, so a twin that does not learn
/// can end below its step 0 (coordinator decisions D71 and D72).
#[test]
fn reproducible() {
    let case = scratch("reproducible");
    let packed = packed(&case);
    let run = || {
        modelspec(&[
            "train",
            TWIN,
            "--backend",
            "cpu",
            "--data",
            path(&packed),
            "--eval-every",
            EVAL_EVERY,
        ])
    };
    let first = run();
    assert_exit(&first, 0);
    let second = run();
    assert_exit(&second, 0);
    let out = stdout(&first);
    assert_eq!(out, stdout(&second), "two runs from one seed differ");

    assert!(
        out.lines().next().unwrap().starts_with("data dataset "),
        "{out}"
    );
    let losses = losses(&out);
    assert!(losses.len() >= 2, "{out}");
    let evals = out.lines().filter(|l| l.starts_with("eval ")).count();
    assert!(evals >= 2, "{out}");
    let (k0, first_loss) = losses[0];
    let (_, last_loss) = *losses.last().unwrap();
    assert_eq!(k0, 0, "{out}");
    assert!(
        first_loss.is_finite() && last_loss.is_finite(),
        "{losses:?}"
    );

    let record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(packed.join("packed.json")).expect("packed.json"),
    )
    .expect("a PackedDataset record");
    let baseline = record["heldout_unigram_baseline_nats"]
        .as_f64()
        .expect("heldout_unigram_baseline_nats");
    let (last_eval, heldout) = *held_out(&out).last().unwrap();
    assert!(
        heldout <= baseline - MARGIN,
        "held-out loss {heldout} at eval {last_eval} not {MARGIN} nats below the held-out unigram \
         baseline {baseline}:\n{out}"
    );
}

/// The `(step, loss)` of every `eval <k> loss <v> accuracy <a>` line.
fn held_out(stdout: &str) -> Vec<(u64, f64)> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("eval "))
        .map(|rest| {
            let mut words = rest.split_whitespace();
            let step = words.next().unwrap().parse().unwrap();
            assert_eq!(words.next(), Some("loss"), "{rest}");
            (step, words.next().unwrap().parse().unwrap())
        })
        .collect()
}
