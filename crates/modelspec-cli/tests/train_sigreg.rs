//! `modelspec train --backend cpu` and `infer --backend cpu` on a `sigreg` loss (DESIGN.md § 2.3,
//! § 4, § 6; story:jepa-sigreg-cpu).
//!
//! The fixture's loss is `sigreg` of a `[256, 8]` `xavier_uniform` parameter alone, under `sgd`:
//! the forward returns the parameter (plus `0 x`, so the data interface is read), and the loss
//! adds `0 mse(x, y)` for the target. The projections of each step are drawn from the run's
//! seed, the site and the step alone, so two runs with one seed print the same stdout; the final
//! loss is the forward program on the final parameters with the projections of the last step,
//! and `infer` on the checkpoint of the last step prints that loss again.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::compile::compile_training;
use modelspec_core::ir::tensor::TensorType;
use modelspec_runtime::cpu::{HostTensor, evaluate};
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::train::init::sigreg_projection;

const FIXTURES: &str = "crates/modelspec-cli/tests/fixtures";
const FIXTURE: &str = "sigreg_train.mspec";
/// The site of the fixture's one `sigreg` call, written in the loss.
const SITE: &str = "Spread.train.sigreg#0";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("UTF-8")
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

/// The stdout of a successful `modelspec train <fixture> --backend cpu <more>`.
fn train(more: &[&str]) -> String {
    let path = format!("{FIXTURES}/{FIXTURE}");
    let mut args = vec!["train", path.as_str(), "--backend", "cpu"];
    args.extend_from_slice(more);
    let output = modelspec(&args);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
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

/// The values of the line `param @<path>: <type> = [v, ...]`.
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

/// Done when: two runs from one seed print identical lines, and the last loss is below a tenth
/// of the step-0 loss.
#[test]
fn train_sigreg_is_reproducible_and_lowers_the_statistic_tenfold() {
    let first = train(&[]);
    let second = train(&[]);
    assert_eq!(first, second, "two runs from one seed");
    let printed = losses(&first);
    assert!(printed.len() >= 2, "{first}");
    let start: f64 = printed[0].1.parse().unwrap();
    let last: f64 = printed.last().unwrap().1.parse().unwrap();
    eprintln!(
        "step 0 loss {start}, step {} loss {last}, ratio {}",
        printed.last().unwrap().0,
        last / start
    );
    assert!(
        last < start / 10.0,
        "last loss {last} is not below a tenth of the step-0 loss {start}:\n{}",
        printed
            .iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The final loss line is the forward program on the printed final parameters with the
/// projection of the last step, not of step 0: the projections are redrawn every step.
#[test]
fn train_sigreg_final_loss_uses_the_projection_of_the_last_step() {
    let stdout = train(&[]);
    let source = std::fs::read_to_string(repo_root().join(FIXTURES).join(FIXTURE))
        .expect("the fixture exists");
    let compiled = compile_training(FIXTURE, &source).expect("it compiles");
    let forward = &compiled.forward;
    let steps = compiled.config.steps;
    let seed = compiled.config.seed;
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
            (
                id,
                HostTensor::new(shape, final_values(&stdout, &decl.path)).unwrap(),
            )
        })
        .collect();
    let at = |step: u64| {
        let mut inputs = data.inputs.clone();
        let projection = sigreg_projection(seed, SITE, step, &TensorType::f32(&[8, 16])).unwrap();
        inputs.insert(SITE.to_owned(), projection);
        evaluate(&forward.program, &inputs, &params).unwrap()[0].data()[0]
    };
    let printed = losses(&stdout);
    let (last_step, last) = printed.last().expect("a final loss line").clone();
    assert_eq!(last_step, steps);
    assert_eq!(last, at(steps).to_string(), "{stdout}");
    assert_ne!(at(steps), at(0), "another step draws another projection");
}

/// `infer --backend cpu` on the checkpoint a run wrote at its last step prints that run's last
/// `step` loss: inference draws the projection of the checkpoint's step.
#[test]
fn infer_sigreg_uses_the_projection_of_the_checkpoint_step() {
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("train_sigreg");
    std::fs::create_dir_all(&out).unwrap();
    let checkpoint = out.join("trained.msckpt");
    let stdout = train(&["--checkpoint", checkpoint.to_str().unwrap()]);
    let (_, last) = losses(&stdout).last().expect("a final loss line").clone();
    let path = format!("{FIXTURES}/{FIXTURE}");
    let output = modelspec(&[
        "infer",
        &path,
        "--backend",
        "cpu",
        "--checkpoint",
        checkpoint.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let inferred = text(&output.stdout);
    let loss = inferred
        .lines()
        .find_map(|l| l.strip_prefix("loss "))
        .unwrap_or_else(|| panic!("no loss line:\n{inferred}"));
    assert_eq!(loss, last, "infer prints the last step loss");
}
