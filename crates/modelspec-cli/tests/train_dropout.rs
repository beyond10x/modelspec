//! `modelspec train --backend cpu` on a model with dropout (DESIGN.md § 2.3, § 6, § 8;
//! story:bert-dropout).
//!
//! Dropout applies only in the training step: the loss printed for each step `k < steps` is the
//! training step's, dropout included, and the final loss line is the forward program, without
//! dropout, on the final parameters (coordinator decision D47, 2026-10-25,
//! `.engineering/waves/bert-1.md`). The mask of each step comes from the run's seed, the site and
//! the step alone, so two runs with one seed print the same stdout, and `p = 0.0` is no dropout
//! at all.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::compile::compile_training;
use modelspec_runtime::cpu::{HostTensor, evaluate};
use modelspec_runtime::train::data::synthetic;

const FIXTURES: &str = "crates/modelspec-cli/tests/fixtures";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("UTF-8")
}

/// The stdout of a successful `modelspec train <fixture> --backend cpu`.
fn train(fixture: &str) -> String {
    let output: Output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args([
            "train",
            &format!("{FIXTURES}/{fixture}"),
            "--backend",
            "cpu",
        ])
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{fixture}: {}",
        text(&output.stderr)
    );
    text(&output.stdout)
}

/// The `(step, printed loss)` of every `step <k> loss <v>` line.
fn losses(stdout: &str) -> Vec<(u64, String)> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (step.parse().expect("a step"), loss.to_owned())
        })
        .collect()
}

/// The shape-less values of the line `param @<path>: <type> = [v, ...]`.
fn final_values(stdout: &str, path: &str) -> Vec<f32> {
    let head = format!("param @{path}: ");
    let line = stdout
        .lines()
        .find(|line| line.starts_with(&head))
        .unwrap_or_else(|| panic!("no `{head}` line in:\n{stdout}"));
    let list = line
        .split_once(" = [")
        .and_then(|(_, rest)| rest.strip_suffix(']'))
        .unwrap_or_else(|| panic!("`{line}` is not `... = [v, ...]`"));
    list.split(", ")
        .map(|v| v.parse().expect("a value"))
        .collect()
}

/// Scenario 5. Two runs with one seed print identical stdout.
#[test]
fn train_dropout_is_reproducible_under_one_seed() {
    let first = train("dropout_train.mspec");
    let second = train("dropout_train.mspec");
    assert_eq!(first, second);
    assert!(losses(&first).len() >= 2, "{first}");
}

/// Scenario 5. `p = 0.0` prints the stdout of the same model without the call.
#[test]
fn train_dropout_with_p_0_is_the_model_without_the_call() {
    assert_eq!(train("dropout_p0.mspec"), train("dropout_none.mspec"));
}

/// Scenario 5. The final loss line equals the loss of the forward program, without dropout, on
/// the printed final parameters and the run's synthetic data; the per-step losses include
/// dropout, so the step-0 loss differs from the step-0 loss of the same model without the call
/// (same initial parameters, same data).
#[test]
fn train_dropout_final_loss_is_the_forward_program_without_dropout() {
    let stdout = train("dropout_train.mspec");
    let source = std::fs::read_to_string(repo_root().join(FIXTURES).join("dropout_train.mspec"))
        .expect("the fixture exists");
    let compiled = compile_training("dropout_train.mspec", &source).expect("it compiles");
    let forward = &compiled.forward;
    assert!(
        forward
            .program
            .nodes
            .iter()
            .all(|n| !n.op.name().starts_with("Dropout")),
        "the forward program has no dropout"
    );
    let data = synthetic(&compiled.config, forward).expect("synthetic data");
    let params: BTreeMap<_, HostTensor<f32>> = forward
        .program
        .params
        .iter()
        .map(|(&id, decl)| {
            let shape = decl
                .ty
                .shape
                .iter()
                .map(|d| d.to_string().parse::<u64>().unwrap())
                .collect();
            let values = final_values(&stdout, &decl.path);
            (id, HostTensor::new(shape, values).unwrap())
        })
        .collect();
    let loss = evaluate(&forward.program, &data.inputs, &params).unwrap()[0].data()[0];
    let printed = losses(&stdout);
    let (last_step, last) = printed.last().expect("a final loss line").clone();
    assert_eq!(last_step, compiled.config.steps);
    assert_eq!(last, loss.to_string(), "{stdout}");

    let without = train("dropout_none.mspec");
    assert_ne!(
        losses(&without)[0],
        printed[0],
        "the step-0 loss includes dropout:\n{stdout}\n{without}"
    );
}
