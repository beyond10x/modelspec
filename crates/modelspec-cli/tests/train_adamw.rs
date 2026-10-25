//! `modelspec train --backend cpu` and `modelspec ir --stage training` on linear regression
//! trained with AdamW (DESIGN.md § 2.1, § 6, § 8; story:bert-adamw-warmup).
//!
//! The fixture `tests/fixtures/linear_regression_adamw.mspec` is
//! `examples/linear_regression.mspec` with `optimizer = adamw(...)` and a warmup of 10 steps. The
//! update is compiled into the training step: the `Optimizer` subgraph holds the internal ops
//! `Sqrt` and `Div`, reads each parameter's moments and the step's rank-0 inputs `lr_t`, `c1` and
//! `c2`, which the host computes every step (coordinator decision D50, 2026-10-25,
//! `.engineering/waves/bert-1.md`).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FIXTURE: &str = "crates/modelspec-cli/tests/fixtures/linear_regression_adamw.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

/// A successful `modelspec train <FIXTURE> --backend cpu`: its stdout.
fn train() -> String {
    let output = modelspec(&["train", FIXTURE, "--backend", "cpu"]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "stderr: {}", text(&output.stderr));
    text(&output.stdout)
}

/// The `(step, loss)` of every `step <k> loss <v>` line.
fn losses(stdout: &str) -> Vec<(u64, f32)> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (
                step.parse().expect("a step number"),
                loss.parse().expect("a loss value"),
            )
        })
        .collect()
}

/// Scenario 5. The run exits 0, its last loss is below its step-0 loss, and two runs with one
/// seed print the same bytes.
#[test]
fn train_adamw_lowers_the_loss_and_repeats() {
    let first = train();
    let losses = losses(&first);
    let (&(first_step, start), &(last_step, end)) =
        (losses.first().unwrap(), losses.last().unwrap());
    assert_eq!(first_step, 0, "{first}");
    assert_eq!(last_step, 200, "{first}");
    assert!(end < start, "loss {start} -> {end}:\n{first}");
    assert_eq!(train(), first, "two runs with one seed");
    let params: Vec<&str> = first.lines().filter(|l| l.starts_with("param @")).collect();
    assert_eq!(params.len(), 1, "the parameters only, no moments:\n{first}");
    assert!(params[0].starts_with("param @LinearReg.w: "), "{first}");
}

/// Scenario 5. `ir --stage training` prints the `adamw` setting and an `Optimizer` subgraph with
/// `Sqrt` and `Div` nodes, the moments as inputs and outputs of the step, each carrying its
/// effect, and the three per-step inputs.
#[test]
fn train_adamw_ir_prints_the_optimizer_subgraph() {
    let output = modelspec(&["ir", FIXTURE, "--stage", "training"]);
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let optimizer: Vec<&str> = stdout
        .lines()
        .filter(|l| l.starts_with("optimizer "))
        .collect();
    assert!(
        optimizer.iter().any(|l| l.contains("= Sqrt %")),
        "no Sqrt:\n{stdout}"
    );
    assert!(
        optimizer.iter().any(|l| l.contains("= Div %")),
        "no Div:\n{stdout}"
    );
    assert!(
        stdout.lines().any(|l| l
            == "optimizer adamw lr = 0.05 weight_decay = 0.01 beta1 = 0.9 beta2 = 0.999 \
                eps = 1e-8 warmup_steps = 10"),
        "{stdout}"
    );
    for input in ["lr_t", "c1", "c2"] {
        assert!(
            optimizer
                .iter()
                .any(|l| l.contains(": Tensor<f32, []> = Input \"optimizer.")
                    && l.ends_with(&format!("{input}\""))),
            "no `{input}` input:\n{stdout}"
        );
    }
    for moment in ["m", "v"] {
        assert!(
            optimizer.iter().any(|l| l
                .contains(&format!("= Input \"optimizer.{moment}@LinearReg.w\" {{"))
                && l.contains(&format!(" read {moment}@LinearReg.w v0}}"))),
            "no read of `{moment}`:\n{stdout}"
        );
        assert!(
            stdout
                .lines()
                .any(|l| l.ends_with(&format!(" {moment} @LinearReg.w"))
                    && l.starts_with("output %")),
            "no output of `{moment}`:\n{stdout}"
        );
    }
}
