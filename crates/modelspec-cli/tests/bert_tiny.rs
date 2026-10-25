//! `examples/bert_tiny.mspec`, BERT-tiny as a checked example, and its hidden-32 twin
//! `crates/modelspec-cli/tests/fixtures/bert_cpu.mspec` (story:bert-tiny-cpu-model).
//!
//! The parameter count of an MLM-only BERT without token-type embedding and pooler is
//! `V H + S H + 2H + L (4 (H^2 + H) + 2 (H F) + F + H + 4H) + (H^2 + H) + 2H + V`: the token and
//! position tables, the embedding layer norm, per layer the four attention projections with their
//! biases, the two feed-forward matrices with their biases and two layer norms, and the MLM head's
//! dense layer, layer norm and decoder bias. The decoder is the token table read again, so it adds
//! no parameter. The example (`L = 2`, `H = 128`, `F = 512`, `S = 128`) has the 8,192-token
//! vocabulary of the repository's corpus tokenizer (`spec/domains/data.yaml` `TokenizerKind`,
//! coordinator decisions D19, D23 and D71), so it has 1,486,720; Turc et al.'s BERT-tiny, at the
//! 30,522 tokens of the original BERT release, has 4,367,290.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const EXAMPLE: &str = "examples/bert_tiny.mspec";
const TWIN: &str = "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs `modelspec <args>` from the repository root.
fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The count of the story's formula.
fn formula(l: u64, h: u64, f: u64, s: u64, v: u64) -> u64 {
    v * h
        + s * h
        + 2 * h
        + l * (4 * (h * h + h) + 2 * (h * f) + f + h + 4 * h)
        + (h * h + h)
        + 2 * h
        + v
}

/// Each `param @<path>: Tensor<f32, [d, ...]> = <init>` line of `modelspec ir <file> --stage
/// tensor`, as the path and its element count.
fn parameters(file: &str) -> Vec<(String, u64)> {
    let output = modelspec(&["ir", file, "--stage", "tensor"]);
    assert_exit(&output, 0);
    let stdout = String::from_utf8(output.stdout).expect("UTF-8");
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("param @"))
        .map(|rest| {
            let (path, ty) = rest.split_once(": ").expect("`param @<path>: <type>`");
            let shape = ty
                .split_once('[')
                .and_then(|(_, after)| after.split_once(']'))
                .map(|(dims, _)| dims)
                .unwrap_or_else(|| panic!("no shape in {line:?}", line = rest));
            let count = shape
                .split(", ")
                .map(|d| d.parse::<u64>().expect("a static dimension"))
                .product();
            (path.to_owned(), count)
        })
        .collect()
}

/// Scenario 2: the parameter types `modelspec ir examples/bert_tiny.mspec --stage tensor`
/// prints have 1,486,720 elements in total, the count of the formula at `V = 8192`, and every
/// path is unique; the twin has the formula's count at `H = 32`, `F = 64`, `S = 16`, `V = 300`.
/// The formula gives Turc et al.'s 4,367,290 at `V = 30522`.
#[test]
fn parameter_count() {
    assert_eq!(formula(2, 128, 512, 128, 30522), 4_367_290);
    assert_eq!(formula(2, 128, 512, 128, 8192), 1_486_720);
    for (file, want, params) in [
        (EXAMPLE, 1_486_720u64, 41usize),
        (TWIN, formula(2, 32, 64, 16, 300), 41),
    ] {
        let parameters = parameters(file);
        let total: u64 = parameters.iter().map(|(_, n)| n).sum();
        assert_eq!(total, want, "{file}: {parameters:?}");
        assert_eq!(parameters.len(), params, "{file}: {parameters:?}");
        let paths: BTreeSet<&str> = parameters.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(
            paths.len(),
            parameters.len(),
            "{file}: a parameter path repeats: {parameters:?}"
        );
    }
}

/// Scenario 1, as a test: `modelspec check` exits 0 on the example and on the twin.
#[test]
fn check_exits_0() {
    for file in [EXAMPLE, TWIN] {
        assert_exit(&modelspec(&["check", file]), 0);
    }
}
