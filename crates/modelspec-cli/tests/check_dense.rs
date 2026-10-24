//! `modelspec check` and `modelspec train --backend cpu` on dense layers with a bias row and GELU
//! (DESIGN.md § 2.2; story:bert-dense-gelu).
//!
//! `add_rows(x, b)` takes `x: [N, H]` and `b: [H]`. A `b` of any other length is refused with
//! `E-TYPE-003` at the call, and an `x` of rank 1 with `E-TYPE-004` at the call. A file that
//! uses `add_rows` and `gelu` correctly checks clean and trains on the CPU interpreter. The
//! fixtures are under `tests/fixtures/` of this crate.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FIXTURES: &str = "crates/modelspec-cli/tests/fixtures";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs `modelspec <args> <fixture>` from the repository root, the fixture path first.
fn modelspec(command: &str, fixture: &str, more: &[&str]) -> (Output, String) {
    let relative = format!("{FIXTURES}/{fixture}");
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .arg(command)
        .arg(&relative)
        .args(more)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    (output, relative)
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("UTF-8")
}

/// `check` exits 1 with exactly one diagnostic, `code` at `line:column`, whose first line names
/// `add_rows`.
fn refused_at(fixture: &str, code: &str, line: u32, column: u32) {
    let (output, relative) = modelspec("check", fixture, &[]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{fixture}:\n{stderr}");
    assert!(output.stdout.is_empty(), "{fixture}: check wrote to stdout");
    let lines: Vec<&str> = stderr.lines().collect();
    let heads = lines.iter().filter(|l| !l.starts_with("  ")).count();
    assert_eq!(heads, 1, "{fixture}: one diagnostic expected:\n{stderr}");
    assert!(
        lines[0].starts_with(&format!("{code}: ")) && code.starts_with("E-TYPE-"),
        "{fixture}: expected {code}:\n{stderr}"
    );
    assert!(
        lines[0].contains("`add_rows`"),
        "{fixture}: {:?} does not name add_rows",
        lines[0]
    );
    assert_eq!(
        lines.get(1).copied(),
        Some(format!("  {relative}:{line}:{column}").as_str()),
        "{fixture}: location\n{stderr}"
    );
}

/// A bias of shape `[H + 1]` for an `x` of shape `[N, H]`.
#[test]
fn add_rows_with_a_bias_of_h_plus_1_is_e_type_003_at_the_call() {
    refused_at("add_rows_bias_too_long.mspec", "E-TYPE-003", 5, 16);
    let (output, _) = modelspec("check", "add_rows_bias_too_long.mspec", &[]);
    let stderr = text(&output.stderr);
    assert!(
        stderr.contains("lhs: Tensor<f32, [8, 3]>") && stderr.contains("rhs: Tensor<f32, [4]>"),
        "both operand types are named:\n{stderr}"
    );
    assert!(
        !stderr.contains("one of them rank 0"),
        "the note is add_rows's rule, not add's:\n{stderr}"
    );
}

/// An `x` of rank 1.
#[test]
fn add_rows_with_an_x_of_rank_1_is_e_type_004_at_the_call() {
    refused_at("add_rows_rank1_x.mspec", "E-TYPE-004", 5, 16);
}

#[test]
fn dense_gelu_checks_clean() {
    let (output, _) = modelspec("check", "dense_gelu.mspec", &[]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    assert!(output.stdout.is_empty());
    assert_eq!(stderr, "", "a clean check prints nothing");
}

/// `gelu(add_rows(matmul(v, w), b))` lowers, differentiates and runs on the CPU interpreter: the
/// run exits 0 and prints a finite loss for step 0 and for its last step. M0 synthetic data
/// supplies exactly one input and a parameter is rank 2 (`xavier_uniform`), so the fixture's one
/// input is the bias row `b` and both matmul operands are parameters.
#[test]
fn dense_gelu_trains_on_the_cpu() {
    let (output, _) = modelspec("train", "dense_gelu_train.mspec", &["--backend", "cpu"]);
    let stdout = text(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{stdout}\n{}",
        text(&output.stderr)
    );
    let losses: Vec<(u64, f32)> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (step.parse().unwrap(), loss.parse().unwrap())
        })
        .collect();
    let steps: Vec<u64> = losses.iter().map(|&(step, _)| step).collect();
    assert_eq!(steps.first(), Some(&0), "{stdout}");
    assert_eq!(steps.last(), Some(&3), "{stdout}");
    assert!(
        losses
            .iter()
            .all(|&(_, loss)| loss.is_finite() && loss > 0.0),
        "{stdout}"
    );
}
