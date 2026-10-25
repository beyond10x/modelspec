//! `modelspec check` and `ir` on weighted cross-entropy over integer labels (DESIGN.md § 2.2,
//! § 4; story:bert-mlm-cross-entropy).
//!
//! `cross_entropy(logits, labels, weights)` takes `logits: f32[N, V]`, `labels: i32[N]` (the
//! second permitted use of an `i32` tensor) and `weights: f32[N]`, and is rank 0. Each refused
//! call exits 1 with one diagnostic at the call, or at the argument it names. The tied decoder of
//! an MLM head checks clean and lowers to a training program holding `CrossEntropy`, its
//! gradient op `CrossEntropyGrad` and the `EmbeddingGrad` of the table's second read. The
//! fixtures are under `tests/fixtures/` of this crate.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FIXTURES: &str = "crates/modelspec-cli/tests/fixtures";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs `modelspec <command> <fixture> <more>` from the repository root.
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
/// `named`; returns stderr.
fn refused_at(fixture: &str, named: &str, code: &str, line: u32, column: u32) -> String {
    let (output, relative) = modelspec("check", fixture, &[]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{fixture}:\n{stderr}");
    assert!(output.stdout.is_empty(), "{fixture}: check wrote to stdout");
    let lines: Vec<&str> = stderr.lines().collect();
    let heads = lines.iter().filter(|l| !l.starts_with("  ")).count();
    assert_eq!(heads, 1, "{fixture}: one diagnostic expected:\n{stderr}");
    assert!(
        lines[0].starts_with(&format!("{code}: ")),
        "{fixture}: expected {code}:\n{stderr}"
    );
    assert!(
        lines[0].contains(named),
        "{fixture}: {:?} does not name {named}",
        lines[0]
    );
    assert_eq!(
        lines.get(1).copied(),
        Some(format!("  {relative}:{line}:{column}").as_str()),
        "{fixture}: location\n{stderr}"
    );
    stderr
}

/// The line of every fixture's `cross_entropy` call, and the column the call starts at.
const CALL: (u32, u32) = (14, 12);

/// `f32` labels: `E-TYPE-018` at the call, the code of every other misuse of an element type,
/// naming the operand and `i32`.
#[test]
fn check_cross_entropy_refuses_f32_labels_with_e_type_018_at_the_call() {
    let stderr = refused_at(
        "cross_entropy_f32_labels.mspec",
        "`cross_entropy` operand 2 must be i32",
        "E-TYPE-018",
        CALL.0,
        CALL.1,
    );
    assert!(stderr.contains("Tensor<f32, [6]>"), "{stderr}");
    assert!(stderr.contains("`cross_entropy`"), "{stderr}");
}

/// Labels of length `N + 1`: the shape code `E-TYPE-003` at the call, naming both types and the
/// operand that differs.
#[test]
fn check_cross_entropy_refuses_labels_of_length_n_plus_1_with_e_type_003_at_the_call() {
    let stderr = refused_at(
        "cross_entropy_labels_too_long.mspec",
        "`cross_entropy`",
        "E-TYPE-003",
        CALL.0,
        CALL.1,
    );
    assert!(
        stderr.contains("Tensor<f32, [6, 5]>") && stderr.contains("Tensor<i32, [7]>"),
        "{stderr}"
    );
    assert!(
        stderr.contains("`labels` has type Tensor<i32, [7]>"),
        "{stderr}"
    );
}

/// `logits` of rank 1: the rank code `E-TYPE-004` at the call, naming `logits`.
#[test]
fn check_cross_entropy_refuses_logits_of_rank_1_with_e_type_004_at_the_call() {
    let stderr = refused_at(
        "cross_entropy_logits_rank1.mspec",
        "`logits` has type Tensor<f32, [6]>",
        "E-TYPE-004",
        CALL.0,
        CALL.1,
    );
    assert!(stderr.contains("`cross_entropy`"), "{stderr}");
}

/// Weights that read a parameter through a model function: the weights take no gradient, so the
/// checker refuses them with `E-TYPE-021` at the argument, naming `weights`.
#[test]
fn check_cross_entropy_refuses_weights_that_read_a_parameter_with_e_type_021_at_the_argument() {
    let fixture = "cross_entropy_param_weights.mspec";
    let source = std::fs::read_to_string(repo_root().join(FIXTURES).join(fixture))
        .expect("the fixture reads");
    let line = source
        .lines()
        .position(|l| l.contains("loss ="))
        .expect("a loss line");
    let column = source
        .lines()
        .nth(line)
        .unwrap()
        .find("model.scaled")
        .unwrap() as u32
        + 1;
    refused_at(fixture, "`weights`", "E-TYPE-021", line as u32 + 1, column);
}

#[test]
fn check_cross_entropy_accepts_the_tied_decoder() {
    let (output, _) = modelspec("check", "cross_entropy_train.mspec", &[]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    assert!(output.stdout.is_empty());
    assert_eq!(stderr, "", "a clean check prints nothing");
}

/// The training program of the tied decoder holds `CrossEntropy` over the logits, the labels and
/// the weights, its gradient op over the same three and the seed, and the `EmbeddingGrad` and the
/// `Transpose` of the table's two reads, whose sum is the table's gradient.
#[test]
fn check_cross_entropy_training_ir_holds_the_loss_its_gradient_and_both_table_adjoints() {
    let (output, _) = modelspec("ir", "cross_entropy_train.mspec", &["--stage", "training"]);
    let stdout = text(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{stdout}\n{}",
        text(&output.stderr)
    );
    let line = |op: &str| {
        stdout
            .lines()
            .find(|l| l.contains(&format!("= {op} %")))
            .unwrap_or_else(|| panic!("{op}:\n{stdout}"))
            .to_owned()
    };
    let loss = line("CrossEntropy");
    assert!(loss.contains("Tensor<f32, []>"), "{loss}");
    let grad = line("CrossEntropyGrad");
    assert!(grad.contains("Tensor<f32, [6, 5]>"), "{grad}");
    line("EmbeddingGrad [5]");
    assert!(
        stdout.contains(" = Add %"),
        "the two adjoints of the table add up:\n{stdout}"
    );
}
