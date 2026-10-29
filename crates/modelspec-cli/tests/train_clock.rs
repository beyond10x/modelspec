//! The training clock of `modelspec train` (story:bert-tiny-vs-pytorch-cuda, coordinator decision
//! D87): with `--clock` a finished run prints `train_seconds <s> eval_seconds <s> steps <n>`
//! after its steps, the training-loop time without its evaluations, checkpoint writes or events,
//! and each `eval` line ends in the cumulative `train_seconds` at its step, so the time at which
//! the held-out loss first falls below a target can be read from stdout alone. Without the flag
//! stdout is unchanged, so two runs of one seed still print the same bytes.
//!
//! The runs train the hidden-32 twin of BERT-tiny, `crates/modelspec-cli/tests/fixtures/bert_cpu.mspec`,
//! on the CPU over a packed synthetic corpus made as `bert_cpu_train.rs` makes it
//! (`common/markov_corpus.rs`), under this crate's build directory.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::Instant;

#[path = "common/markov_corpus.rs"]
mod markov_corpus;

const TWIN: &str = "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("train_clock_{name}"));
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

/// The `(train_seconds, eval_seconds, steps)` of the run's one clock line, which follows its
/// `step` and `eval` lines and precedes its `param` lines.
fn clock(out: &str) -> (f64, f64, u64) {
    let lines: Vec<&str> = out.lines().collect();
    let at: Vec<usize> = (0..lines.len())
        .filter(|&i| lines[i].starts_with("train_seconds "))
        .collect();
    assert_eq!(at.len(), 1, "one clock line:\n{out}");
    let at = at[0];
    assert!(
        lines[..at].iter().all(|line| !line.starts_with("param ")),
        "the clock line precedes the param lines:\n{out}"
    );
    assert!(
        lines[at + 1..]
            .iter()
            .all(|line| !line.starts_with("step ") && !line.starts_with("eval ")),
        "the clock line follows the step and eval lines:\n{out}"
    );
    let words: Vec<&str> = lines[at].split_whitespace().collect();
    match words[..] {
        ["train_seconds", train, "eval_seconds", eval, "steps", steps] => (
            train.parse().expect("train_seconds"),
            eval.parse().expect("eval_seconds"),
            steps.parse().expect("steps"),
        ),
        _ => panic!("the last line is not `train_seconds <s> eval_seconds <s> steps <n>`:\n{out}"),
    }
}

/// The `(step, train_seconds)` of every `eval <k> loss <v> accuracy <a> train_seconds <s>` line.
fn evals(out: &str) -> Vec<(u64, f64)> {
    out.lines()
        .filter_map(|line| line.strip_prefix("eval "))
        .map(|rest| {
            let words: Vec<&str> = rest.split_whitespace().collect();
            match words[..] {
                [step, "loss", _, "accuracy", _, "train_seconds", seconds] => (
                    step.parse().expect("the step"),
                    seconds.parse().expect("train_seconds"),
                ),
                _ => panic!("not an eval line with train_seconds: eval {rest}"),
            }
        })
        .collect()
}

/// The twin for 200 steps on the CPU with `--clock` and `extra`, timed from outside: its stdout
/// and wall seconds.
fn timed(packed: &Path, extra: &[&str]) -> (String, f64) {
    timed_on("cpu", packed, extra)
}

/// [`timed`] on `backend`.
fn timed_on(backend: &str, packed: &Path, extra: &[&str]) -> (String, f64) {
    let mut args = vec![
        "train",
        TWIN,
        "--backend",
        backend,
        "--data",
        path(packed),
        "--steps",
        "200",
        "--clock",
    ];
    args.extend_from_slice(extra);
    let started = Instant::now();
    let output = modelspec(&args);
    let wall = started.elapsed().as_secs_f64();
    assert_exit(&output, 0);
    (stdout(&output), wall)
}

/// With `--eval-every 100` the run evaluates at steps 0, 100 and 200: its training and evaluation
/// seconds are both positive and together at most the run's wall time, its steps are the 200
/// updates, and the `eval` lines carry the cumulative training seconds, 0 at step 0, rising, and
/// at most the run's total.
#[test]
fn train_and_eval_seconds_split_the_run_and_each_eval_line_carries_the_training_clock() {
    let case = scratch("split");
    let packed = packed(&case);
    let (out, wall) = timed(&packed, &["--eval-every", "100"]);
    let (train, eval, steps) = clock(&out);
    assert_eq!(steps, 200, "{out}");
    assert!(train > 0.0 && eval > 0.0, "{out}");
    assert!(
        train + eval <= wall,
        "train {train} + eval {eval} is more than the wall time {wall}"
    );
    let evals = evals(&out);
    assert_eq!(
        evals.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        [0, 100, 200],
        "{out}"
    );
    assert_eq!(evals[0].1, 0.0, "no training before step 0:\n{out}");
    assert!(
        evals.windows(2).all(|pair| pair[0].1 < pair[1].1),
        "the clock rises from eval to eval:\n{out}"
    );
    assert!(evals[2].1 <= train, "{out}");
}

/// The split above on a stepwise Vulkan run, whose held-out evaluations the device reduces: the
/// evaluations are timed into `eval_seconds`, not into the training clock
/// (story:vulkan-stepwise-multi-step-submit; `eval_seconds 0.000000` before it).
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn device_train_and_eval_seconds_split_a_stepwise_run() {
    let case = scratch("split_vulkan");
    let packed = packed(&case);
    let (out, wall) = timed_on("vulkan", &packed, &["--eval-every", "100"]);
    let (train, eval, steps) = clock(&out);
    assert_eq!(steps, 200, "{out}");
    assert!(train > 0.0 && eval > 0.0, "{out}");
    assert!(
        train + eval <= wall,
        "train {train} + eval {eval} is more than the wall time {wall}"
    );
    let evals = evals(&out);
    assert_eq!(
        evals.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        [0, 100, 200],
        "{out}"
    );
    assert_eq!(evals[0].1, 0.0, "no training before step 0:\n{out}");
    assert!(
        evals.windows(2).all(|pair| pair[0].1 < pair[1].1),
        "the clock rises from eval to eval:\n{out}"
    );
    assert!(evals[2].1 <= train, "{out}");
}

/// Without `--eval-every` nothing is evaluated: `eval_seconds` is 0 and no `eval` line is printed.
#[test]
fn eval_seconds_is_0_without_eval_every() {
    let case = scratch("plain");
    let packed = packed(&case);
    let (out, wall) = timed(&packed, &[]);
    let (train, eval, steps) = clock(&out);
    assert_eq!((eval, steps), (0.0, 200), "{out}");
    assert!(train > 0.0 && train <= wall, "train {train}, wall {wall}");
    assert!(evals(&out).is_empty(), "{out}");
}

/// Without `--clock` stdout has no clock: no `train_seconds` line, `eval` lines of four fields, and
/// two runs of one seed print the same bytes.
#[test]
fn without_clock_stdout_is_unchanged_and_reproducible() {
    let case = scratch("unclocked");
    let packed = packed(&case);
    let run = || {
        let output = modelspec(&[
            "train",
            TWIN,
            "--backend",
            "cpu",
            "--data",
            path(&packed),
            "--steps",
            "100",
            "--eval-every",
            "50",
        ]);
        assert_exit(&output, 0);
        stdout(&output)
    };
    let out = run();
    assert!(!out.contains("train_seconds"), "{out}");
    let evals: Vec<&str> = out
        .lines()
        .filter(|line| line.starts_with("eval "))
        .collect();
    assert_eq!(evals.len(), 3, "{out}");
    for line in evals {
        assert_eq!(line.split_whitespace().count(), 6, "{line}");
    }
    assert_eq!(run(), out, "a second run of the seed");
}
