//! `modelspec train --backend cpu` refuses a run whose loss or parameters stop being finite
//! (story:train-refuses-non-finite-loss; DESIGN.md § 8, `E-TRAIN-001`).
//!
//! The divergent fixture is `examples/linear_regression.mspec` with `sgd(lr = 100.0)`: its loss
//! overflows to `inf` while every parameter is still finite. The parameter fixture uses
//! `sgd(lr = 1e38)`: the update of step 0 takes `LinearReg.w` past `f32::MAX` while the loss of
//! step 0 is still finite. The golden `tests/golden/linear_regression.train.stdout` is the stdout
//! of the release 0.5.0 binary on the unchanged example.
//!
//! Acceptance scenarios map to the tests: 1 is [`a_divergent_run_exits_1_with_e_train_001`], 2 is
//! [`a_finite_run_prints_the_golden`] and 3 is
//! [`a_non_finite_parameter_is_named_by_its_canonical_path`]; scenario 5, the live stream, is
//! `a_divergent_run_ends_its_stream_with_a_null_loss` in `observatory_stream.rs`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const EXAMPLE: &str = "examples/linear_regression.mspec";
const GOLDEN: &str = include_str!("golden/linear_regression.train.stdout");

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

/// Runs `modelspec train <path> --backend cpu` from the repository root.
fn train(path: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["train", path, "--backend", "cpu"])
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

/// The example with its learning rate replaced by `lr`, written to a fresh file `name`.
fn with_lr(name: &str, lr: &str) -> String {
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example exists");
    let changed = source.replace("sgd(lr = 0.1)", &format!("sgd(lr = {lr})"));
    assert_ne!(
        changed, source,
        "the example's optimizer is `sgd(lr = 0.1)`"
    );
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("train_non_finite");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join(name);
    std::fs::write(&path, changed).expect("the source is written");
    path.display().to_string()
}

/// The lines of `stderr` that start with `E-TRAIN-001`, the diagnostic layout of DESIGN.md § 8.
fn e_train_001(stderr: &str) -> Vec<&str> {
    stderr
        .lines()
        .filter(|line| line.starts_with("E-TRAIN-001: "))
        .collect()
}

/// The step `step <k>` that a diagnostic line names.
fn named_step(line: &str) -> u64 {
    let (_, rest) = line
        .split_once("step ")
        .unwrap_or_else(|| panic!("the diagnostic names a step: {line}"));
    rest.split(|c: char| !c.is_ascii_digit())
        .next()
        .and_then(|digits| digits.parse().ok())
        .unwrap_or_else(|| panic!("a step number: {line}"))
}

#[test]
fn a_divergent_run_exits_1_with_e_train_001() {
    let output = train(&with_lr("lr100.mspec", "100.0"));
    let (stdout, stderr) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout: {stdout}\nstderr: {stderr}"
    );
    let lines = e_train_001(&stderr);
    assert_eq!(lines.len(), 1, "one E-TRAIN-001 line: {stderr}");
    assert!(lines[0].contains("loss"), "{}", lines[0]);
    let step = named_step(lines[0]);
    assert!(
        !stdout.lines().any(|line| line.starts_with("param ")),
        "no param lines: {stdout}"
    );
    let last = stdout.lines().last().expect("stdout has lines");
    assert!(
        last == format!("step {step} loss NaN")
            || last == format!("step {step} loss inf")
            || last == format!("step {step} loss -inf"),
        "the last stdout line is the failing step's: {stdout}"
    );
    let golden: Vec<&str> = GOLDEN.lines().collect();
    let printed: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        printed[..printed.len() - 1],
        golden[..2],
        "the init line and the step 0 line, as a finite run prints them"
    );
}

#[test]
fn a_finite_run_prints_the_golden() {
    let output = train(EXAMPLE);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "stderr: {}", text(&output.stderr));
    assert_eq!(text(&output.stdout), GOLDEN);
}

#[test]
fn a_non_finite_parameter_is_named_by_its_canonical_path() {
    let output = train(&with_lr("lr1e38.mspec", "1e38"));
    let (stdout, stderr) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout: {stdout}\nstderr: {stderr}"
    );
    let lines = e_train_001(&stderr);
    assert_eq!(lines.len(), 1, "one E-TRAIN-001 line: {stderr}");
    assert_eq!(named_step(lines[0]), 0, "{}", lines[0]);
    assert!(stderr.contains("@LinearReg.w"), "{stderr}");
    let golden: Vec<&str> = GOLDEN.lines().collect();
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        golden[..2],
        "the init line and the finite loss of step 0, no param lines"
    );
}
