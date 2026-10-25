//! `modelspec check`, `ir` and `train --backend cpu` on layer norm and softmax (DESIGN.md § 2.2,
//! § 4; story:bert-layer-norm-softmax).
//!
//! `layer_norm(x, gamma, beta)` takes `x: [N, H]` and `gamma`, `beta` of shape `[H]`; a `gamma`
//! of shape `[H + 1]` is refused with `E-TYPE-003` at the call. `softmax(x)` takes a rank-2 `x`;
//! a rank-1 operand is refused with `E-TYPE-004` at the call. A file that uses both correctly
//! checks clean, lowers to a training program holding their gradient ops and trains on the CPU
//! interpreter. The fixtures are under `tests/fixtures/` of this crate.

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
/// `operator`; returns stderr.
fn refused_at(fixture: &str, operator: &str, code: &str, line: u32, column: u32) -> String {
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
        lines[0].contains(&format!("`{operator}`")),
        "{fixture}: {:?} does not name {operator}",
        lines[0]
    );
    assert_eq!(
        lines.get(1).copied(),
        Some(format!("  {relative}:{line}:{column}").as_str()),
        "{fixture}: location\n{stderr}"
    );
    stderr
}

/// A `gamma` of shape `[H + 1]` for an `x` of shape `[N, H]` is `E-TYPE-003` at the call, both
/// operand types named, with `layer_norm`'s rule as the note.
#[test]
fn layer_norm_with_a_gamma_of_h_plus_1_is_e_type_003_at_the_call() {
    let stderr = refused_at(
        "layer_norm_gamma_too_long.mspec",
        "layer_norm",
        "E-TYPE-003",
        7,
        16,
    );
    assert!(
        stderr.contains("lhs: Tensor<f32, [8, 3]>") && stderr.contains("rhs: Tensor<f32, [4]>"),
        "both operand types are named:\n{stderr}"
    );
    assert!(
        stderr.contains("`gamma` and `beta` of shape [H]"),
        "the note is layer_norm's rule:\n{stderr}"
    );
}

/// `softmax` of a rank-1 tensor is `E-TYPE-004` at the call.
#[test]
fn softmax_of_a_rank_1_tensor_is_e_type_004_at_the_call() {
    let stderr = refused_at("softmax_rank1.mspec", "softmax", "E-TYPE-004", 6, 17);
    assert!(stderr.contains("Tensor<f32, [3]>"), "{stderr}");
}

#[test]
fn norm_softmax_checks_clean() {
    let (output, _) = modelspec("check", "norm_softmax_train.mspec", &[]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    assert!(output.stdout.is_empty());
    assert_eq!(stderr, "", "a clean check prints nothing");
}

/// The training program of `softmax(layer_norm(matmul(x, w), gamma, beta))` holds each internal
/// gradient op: `SoftmaxGrad`, `LayerNormGradX`, `LayerNormGradGamma` and the `SumRows` of `beta`.
#[test]
fn the_training_ir_holds_the_norm_and_softmax_gradient_ops() {
    let (output, _) = modelspec("ir", "norm_softmax_train.mspec", &["--stage", "training"]);
    let stdout = text(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{stdout}\n{}",
        text(&output.stderr)
    );
    for op in [
        "= LayerNorm ",
        "= Softmax ",
        "= SoftmaxGrad ",
        "= LayerNormGradX ",
        "= LayerNormGradGamma ",
        "= SumRows ",
    ] {
        assert!(stdout.contains(op), "{op}:\n{stdout}");
    }
}

/// The model trains on synthetic data on the CPU interpreter: every loss is finite, the last
/// below the first, and `gamma` and `beta` move from `ones()` and `zeros()`.
#[test]
fn norm_softmax_trains_on_the_cpu() {
    let (output, _) = modelspec("train", "norm_softmax_train.mspec", &["--backend", "cpu"]);
    let stdout = text(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{stdout}\n{}",
        text(&output.stderr)
    );
    let losses: Vec<f32> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (_, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            loss.parse().unwrap()
        })
        .collect();
    assert!(losses.len() > 1, "{stdout}");
    assert!(losses.iter().all(|loss| loss.is_finite()), "{stdout}");
    let (first, last) = (losses[0], losses[losses.len() - 1]);
    assert!(last < first, "loss {first} -> {last}:\n{stdout}");
    for (path, start) in [
        ("@Norm.gamma", "Tensor<f32, [3]> = [1, 1, 1]"),
        ("@Norm.beta", "Tensor<f32, [3]> = [0, 0, 0]"),
    ] {
        assert!(
            stdout.lines().any(|l| l == format!("init {path}: {start}")),
            "{path} starts at {start}:\n{stdout}"
        );
        let last = stdout
            .lines()
            .find(|l| l.starts_with(&format!("param {path}: ")))
            .unwrap_or_else(|| panic!("no final line for {path}:\n{stdout}"));
        assert_ne!(last, format!("param {path}: {start}"), "{path} was trained");
    }
}
