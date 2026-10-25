//! Adversary cases of story:bert-heldout-evaluation against `modelspec train --backend cpu
//! --data <packed-dir> --eval-every <n>` (coordinator decision D55).
//!
//! The packed directory is the one `train_eval.rs` makes: the original-text fixture
//! `crates/modelspec-runtime/tests/fixtures/corpus-tokenizer/`, a 300-token vocabulary and
//! sequences of 16, `mlm_bag.mspec` with `B = 4`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const CORPUS: &str = "crates/modelspec-runtime/tests/fixtures/corpus-tokenizer";
const MLM_BAG: &str = "crates/modelspec-cli/tests/fixtures/mlm_bag.mspec";
const MLM_BAG_DROPOUT: &str = "crates/modelspec-cli/tests/fixtures/mlm_bag_dropout.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adv_heldout_cli_{name}"));
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

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout: {}\nstderr: {}",
        stdout(output),
        stderr(output)
    );
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

fn packed(case: &Path) -> PathBuf {
    let dataset = case.join("dataset");
    assert_exit(
        &modelspec(&["data", "prepare", CORPUS, "--out", path(&dataset)]),
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

fn train(file: &str, data: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["train", file, "--backend", "cpu", "--data", path(data)];
    args.extend_from_slice(extra);
    modelspec(&args)
}

/// `source_file` with `edits` applied, written to `<case>/<name>.mspec`.
fn variant(case: &Path, source_file: &str, name: &str, edits: &[(&str, &str)]) -> String {
    let mut source = std::fs::read_to_string(repo_root().join(source_file)).expect("the fixture");
    for (from, to) in edits {
        assert!(source.contains(from), "the fixture holds {from:?}");
        source = source.replace(from, to);
    }
    let file = case.join(format!("{name}.mspec"));
    std::fs::write(&file, source).expect("the variant is written");
    path(&file).to_owned()
}

/// `(k, "loss <v> accuracy <a>")` of every `eval` line.
fn evals(out: &str) -> Vec<(u64, String)> {
    out.lines()
        .filter_map(|l| l.strip_prefix("eval "))
        .map(|rest| {
            let (k, values) = rest.split_once(' ').expect("eval <k> ...");
            (k.parse().expect("a step"), values.to_owned())
        })
        .collect()
}

fn eval_at(out: &str, k: u64) -> String {
    evals(out)
        .into_iter()
        .find(|(s, _)| *s == k)
        .unwrap_or_else(|| panic!("an eval {k} line:\n{out}"))
        .1
}

fn line(out: &str, prefix: &str) -> String {
    out.lines()
        .find(|l| l.starts_with(prefix))
        .unwrap_or_else(|| panic!("a {prefix:?} line:\n{out}"))
        .to_owned()
}

/// The steps of every `step <k> loss <v>` line, in order.
fn step_lines(out: &str) -> Vec<u64> {
    out.lines()
        .filter_map(|l| l.strip_prefix("step "))
        .map(|rest| rest.split(' ').next().unwrap().parse().unwrap())
        .collect()
}

/// H1: `eval 1` of a two-step run is the evaluation of the parameters after exactly one update:
/// it equals, as printed, `eval 1` of a run that resumes from the checkpoint written after one
/// step and runs no step; likewise `eval 2` against the checkpoint after two steps, and `eval 0`
/// against a run of no step.
#[test]
fn adv_ho_eval_k_is_the_parameters_after_k_updates() {
    let case = scratch("after_k");
    let packed = packed(&case);
    let whole = train(MLM_BAG, &packed, &["--steps", "2", "--eval-every", "1"]);
    assert_exit(&whole, 0);
    let whole = stdout(&whole);
    assert_eq!(
        evals(&whole).iter().map(|e| e.0).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    for k in [1u64, 2] {
        let checkpoint = case.join(format!("after_{k}.msckpt"));
        let k_text = k.to_string();
        assert_exit(
            &train(
                MLM_BAG,
                &packed,
                &["--steps", &k_text, "--checkpoint", path(&checkpoint)],
            ),
            0,
        );
        let fresh = train(
            MLM_BAG,
            &packed,
            &[
                "--steps",
                "0",
                "--resume",
                path(&checkpoint),
                "--eval-every",
                "1",
            ],
        );
        assert_exit(&fresh, 0);
        let fresh = stdout(&fresh);
        assert_eq!(evals(&fresh).len(), 1, "{fresh}");
        assert_eq!(
            eval_at(&whole, k),
            eval_at(&fresh, k),
            "eval {k} of the two-step run against the checkpoint after {k} steps"
        );
    }
    let none = train(MLM_BAG, &packed, &["--steps", "0", "--eval-every", "1"]);
    assert_exit(&none, 0);
    assert_eq!(eval_at(&whole, 0), eval_at(&stdout(&none), 0));
}

/// H2: in `mlm_bag_dropout.mspec`, `p = 0.1` and `p = 0.5` with everything else fixed print the
/// same `init` and `eval 0` lines (dropout off, same init) and different `step 0` lines.
#[test]
fn adv_ho_dropout_rate_does_not_change_eval_0() {
    let case = scratch("dropout_rate");
    let packed = packed(&case);
    let half = variant(&case, MLM_BAG_DROPOUT, "half", &[("p = 0.1", "p = 0.5")]);
    let a = train(
        MLM_BAG_DROPOUT,
        &packed,
        &["--steps", "1", "--eval-every", "1"],
    );
    let b = train(&half, &packed, &["--steps", "1", "--eval-every", "1"]);
    assert_exit(&a, 0);
    assert_exit(&b, 0);
    let (a, b) = (stdout(&a), stdout(&b));
    assert_eq!(line(&a, "init "), line(&b, "init "));
    assert_eq!(eval_at(&a, 0), eval_at(&b, 0), "eval 0 without dropout");
    assert_ne!(
        line(&a, "step 0 "),
        line(&b, "step 0 "),
        "dropout in training"
    );
}

/// H2 (observation): with the parameters fixed by `--init`, two seeds print different `eval 0`
/// lines, because the held-out masking is drawn with the run seed (D55); held-out values do not
/// compare across seeds.
#[test]
fn adv_ho_seed_changes_heldout_masking_at_fixed_parameters() {
    let case = scratch("seed_masking");
    let packed = packed(&case);
    let first = train(MLM_BAG, &packed, &["--steps", "0"]);
    assert_exit(&first, 0);
    let init_line = line(&stdout(&first), "init @MlmBag.table:");
    let list = init_line
        .split_once(" = [")
        .unwrap()
        .1
        .strip_suffix(']')
        .unwrap();
    let bytes: Vec<u8> = list
        .split(", ")
        .flat_map(|v| v.parse::<f32>().unwrap().to_le_bytes())
        .collect();
    let init = case.join("init");
    std::fs::create_dir_all(&init).unwrap();
    std::fs::write(init.join("table.f32"), bytes).unwrap();
    let other = variant(&case, MLM_BAG, "seed8", &[("seed = 7;", "seed = 8;")]);
    let a = train(
        MLM_BAG,
        &packed,
        &["--steps", "0", "--init", path(&init), "--eval-every", "1"],
    );
    let b = train(
        &other,
        &packed,
        &["--steps", "0", "--init", path(&init), "--eval-every", "1"],
    );
    assert_exit(&a, 0);
    assert_exit(&b, 0);
    let (a, b) = (stdout(&a), stdout(&b));
    assert_eq!(
        line(&a, "init "),
        line(&b, "init "),
        "one set of parameters"
    );
    assert_ne!(
        eval_at(&a, 0),
        eval_at(&b, 0),
        "the masking follows the seed"
    );
}

/// H5: a `--checkpoint-every 100` run of 200 steps resumed at 100 for the rest prints the same
/// `eval 100` and `eval 200` lines as the unbroken run, and two unbroken runs print one stdout.
#[test]
fn adv_ho_checkpoint_every_resume_matches_the_unbroken_run() {
    let case = scratch("ckpt_every");
    let packed = packed(&case);
    let checkpoint = case.join("run.msckpt");
    let whole = train(
        MLM_BAG,
        &packed,
        &[
            "--eval-every",
            "100",
            "--checkpoint",
            path(&checkpoint),
            "--checkpoint-every",
            "100",
        ],
    );
    assert_exit(&whole, 0);
    let again = train(MLM_BAG, &packed, &["--eval-every", "100"]);
    assert_exit(&again, 0);
    assert_eq!(stdout(&whole), stdout(&again), "one seed, one stdout");
    let resumed = train(
        MLM_BAG,
        &packed,
        &[
            "--steps",
            "100",
            "--resume",
            path(&case.join("run-100.msckpt")),
            "--eval-every",
            "100",
        ],
    );
    assert_exit(&resumed, 0);
    let (whole, resumed) = (stdout(&whole), stdout(&resumed));
    assert_eq!(eval_at(&whole, 100), eval_at(&resumed, 100));
    assert_eq!(eval_at(&whole, 200), eval_at(&resumed, 200));
}

/// H7: refusals exit 2 on one line without a panic; `--eval-every 7 --steps 5` evaluates at 0
/// and 5 only; `--eval-every 1 --steps 0` evaluates step 0 once; `--eval` beside a packed
/// `--data` stays the usage error of D56.
#[test]
fn adv_ho_refusals_and_boundaries() {
    let case = scratch("refusals");
    let packed = packed(&case);
    let dataset = case.join("dataset");
    for value in ["-1", "abc", "1.5", "18446744073709551616"] {
        let output = train(MLM_BAG, &packed, &["--eval-every", value]);
        assert_exit(&output, 2);
        assert!(!stderr(&output).contains("panicked"), "{}", stderr(&output));
        assert_eq!(stdout(&output), "");
    }
    let output = train(
        MLM_BAG,
        &packed,
        &["--eval-every", "1", "--eval", path(&dataset)],
    );
    assert_exit(&output, 2);
    assert_eq!(stderr(&output).lines().count(), 1, "{}", stderr(&output));

    let short = train(MLM_BAG, &packed, &["--steps", "5", "--eval-every", "7"]);
    assert_exit(&short, 0);
    assert_eq!(
        evals(&stdout(&short))
            .iter()
            .map(|e| e.0)
            .collect::<Vec<_>>(),
        [0, 5]
    );
    let zero = train(MLM_BAG, &packed, &["--steps", "0", "--eval-every", "1"]);
    assert_exit(&zero, 0);
    let zero = stdout(&zero);
    assert_eq!(evals(&zero).iter().map(|e| e.0).collect::<Vec<_>>(), [0]);
    assert_eq!(step_lines(&zero), [0], "{zero}");
}

/// H8: `--eval-every 7 --steps 50` prints each `step` line once, at 0, 7, ..., 49 and 50, and
/// each `eval` line directly after its `step` line.
#[test]
fn adv_ho_extra_step_lines_are_not_duplicated() {
    let case = scratch("step_lines");
    let packed = packed(&case);
    let output = train(MLM_BAG, &packed, &["--steps", "50", "--eval-every", "7"]);
    assert_exit(&output, 0);
    let out = stdout(&output);
    let mut expected: Vec<u64> = (0..=50).step_by(7).collect();
    expected.push(50);
    assert_eq!(step_lines(&out), expected, "{out}");
    assert_eq!(
        evals(&out).iter().map(|e| e.0).collect::<Vec<_>>(),
        expected
    );
    let lines: Vec<&str> = out.lines().collect();
    for (i, l) in lines.iter().enumerate() {
        if let Some(rest) = l.strip_prefix("eval ") {
            let k = rest.split(' ').next().unwrap();
            assert!(
                lines[i - 1].starts_with(&format!("step {k} loss ")),
                "{out}"
            );
        }
    }
}

/// H6: the `--record` file's `StepCompleted` of steps 0, 100 and 200 carry exactly the two
/// samples with the printed values, in the order loss then accuracy; every other one has an
/// empty `metrics`.
#[test]
fn adv_ho_recorded_events_carry_the_printed_values() {
    let case = scratch("record");
    let packed = packed(&case);
    let record = case.join("run.jsonl");
    let output = train(
        MLM_BAG,
        &packed,
        &["--eval-every", "100", "--record", path(&record)],
    );
    assert_exit(&output, 0);
    let out = stdout(&output);
    let printed: BTreeMap<u64, String> = evals(&out).into_iter().collect();
    let text = std::fs::read_to_string(&record).unwrap();
    let mut seen = 0;
    for l in text.lines() {
        let event: Value = serde_json::from_str(l).unwrap();
        if event["body"]["kind"] != "StepCompleted" {
            assert!(
                event["body"]["value"].get("metrics").is_none(),
                "only StepCompleted carries metrics: {l}"
            );
            continue;
        }
        let value = &event["body"]["value"];
        let k = value["step"].as_u64().unwrap();
        let metrics = value["metrics"].as_array().unwrap();
        seen += 1;
        match printed.get(&k) {
            None => assert!(metrics.is_empty(), "step {k}: {l}"),
            Some(values) => {
                let loss = metrics[0]["value"].as_f64().unwrap() as f32;
                let accuracy = metrics[1]["value"].as_f64().unwrap() as f32;
                assert_eq!(metrics.len(), 2);
                assert_eq!(metrics[0]["name"], "heldout_mlm_loss");
                assert_eq!(metrics[1]["name"], "heldout_mlm_accuracy");
                assert_eq!(
                    *values,
                    format!("loss {loss} accuracy {accuracy}"),
                    "step {k}"
                );
            }
        }
    }
    assert_eq!(seen, 201);
}

/// H4 / H7: a finite `--init` table of 1e20 everywhere overflows every logit to +inf, so the
/// step-0 loss is NaN; the run stops with E-TRAIN-001, exit 1, no panic, and still prints
/// `eval 0` after `step 0`.
#[test]
fn adv_ho_non_finite_init_stops_without_panic() {
    let case = scratch("non_finite");
    let packed = packed(&case);
    let init = case.join("init");
    std::fs::create_dir_all(&init).unwrap();
    let table = vec![1e20f32; 300 * 16];
    let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(init.join("table.f32"), bytes).unwrap();
    let output = train(
        MLM_BAG,
        &packed,
        &["--steps", "3", "--init", path(&init), "--eval-every", "1"],
    );
    assert_exit(&output, 1);
    let err = stderr(&output);
    assert!(!err.contains("panicked"), "{err}");
    assert!(err.contains("E-TRAIN-001"), "{err}");
    let out = stdout(&output);
    assert_eq!(evals(&out).first().map(|e| e.0), Some(0), "{out}\n{err}");
}

/// Deviation 5: a model whose `prediction` (the value `model.forward` returns) is `f32[300, 64]`,
/// transposed into the loss, trains, and with `--eval-every` is refused before the first step,
/// exit 1, on one line naming the expected rows and the prediction's type.
#[test]
fn adv_ho_prediction_of_another_shape_is_refused_before_step_0() {
    let case = scratch("prediction_shape");
    let packed = packed(&case);
    let file = variant(
        &case,
        MLM_BAG,
        "transposed",
        &[
            ("-> Tensor<f32, [64, 300]>", "-> Tensor<f32, [300, 64]>"),
            (
                "return matmul(embedding(table, ids), transpose(table));",
                "return transpose(matmul(embedding(table, ids), transpose(table)));",
            ),
            (
                "cross_entropy(model.forward(ids), labels, weights)",
                "cross_entropy(transpose(model.forward(ids)), labels, weights)",
            ),
        ],
    );
    let plain = train(&file, &packed, &["--steps", "1"]);
    assert_exit(&plain, 0);
    let output = train(&file, &packed, &["--steps", "1", "--eval-every", "1"]);
    assert_exit(&output, 1);
    let err = stderr(&output);
    assert_eq!(err.lines().count(), 1, "{err}");
    assert!(err.contains("[64, V]"), "the expected rows: {err}");
    assert!(err.contains("300, 64"), "the prediction's shape: {err}");
    assert!(
        step_lines(&stdout(&output)).is_empty(),
        "{}",
        stdout(&output)
    );
}
