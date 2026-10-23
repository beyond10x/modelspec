//! Adversary cases for `StepCompleted.parameters` of `modelspec train --backend cpu`
//! (story:observatory-parameter-statistics; `spec/domains/observatory.yaml`, `StepCompleted`).
//!
//! The suite of the story derives every expected gradient from the runtime's own `sgd_step` and
//! every parameter label from the same `forward.program.params` map the CLI reads. These cases
//! take the expected norms from an oracle that shares neither: an `f64` forward pass and
//! gradient written out here from the model's formula, checked against central finite
//! differences, with plain `f64` SGD from the run's initial values. They also cover dense data,
//! where each step reads another batch, sampling with N larger than the run, diverging runs
//! and the path order of the entries.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::train::runner;
use serde_json::Value;

const LINEAR: &str = "examples/linear_regression.mspec";
const TWO_LAYER: &str = "crates/modelspec-docs/models/two_layer_relu.mspec";
const SCHEMA: &str =
    "crates/modelspec-observatory/schema/entities/modelspec.observatory.RunEvent.schema.json";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("adversary_parameter_stats");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join(name);
    if path.is_dir() {
        std::fs::remove_dir_all(&path).expect("the old fixture is removed");
    } else {
        let _ = std::fs::remove_file(&path);
    }
    path
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .output()
        .expect("modelspec runs")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Runs `train <model> --backend cpu <extra> --record <file>` and returns its exit code and the
/// recorded events, each checked against the generated JSON Schema.
fn record(model: &str, name: &str, extra: &[&str]) -> (Option<i32>, Vec<Value>) {
    let path = scratch(name);
    let recording = path.display().to_string();
    let mut args = vec!["train", model, "--backend", "cpu"];
    args.extend_from_slice(extra);
    args.extend(["--record", recording.as_str()]);
    let run = modelspec(&args);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}: {}", stderr(&run)));
    let schema: Value = serde_json::from_str(
        &std::fs::read_to_string(repo_root().join(SCHEMA)).expect("the schema"),
    )
    .expect("the schema is JSON");
    let validator = jsonschema::validator_for(&schema).expect("the schema compiles");
    let events: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}")))
        .collect();
    for event in &events {
        let errors: Vec<String> = validator
            .iter_errors(event)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "{event}: {errors:?}");
    }
    (run.status.code(), events)
}

/// The `StepCompleted` values of `events`, by step.
fn steps(events: &[Value]) -> BTreeMap<u64, Value> {
    events
        .iter()
        .filter(|e| e["body"]["kind"] == "StepCompleted")
        .map(|e| {
            let v = e["body"]["value"].clone();
            (v["step"].as_u64().expect("a step"), v)
        })
        .collect()
}

/// The steps whose `parameters` list is not empty.
fn sampled(events: &[Value]) -> Vec<u64> {
    steps(events)
        .into_iter()
        .filter(|(_, v)| !v["parameters"].as_array().expect("a list").is_empty())
        .map(|(k, _)| k)
        .collect()
}

/// `(path, subject)` of every entry of `step`, in order.
fn entries(step: &Value) -> Vec<(String, String)> {
    step["parameters"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|e| {
            (
                e["path"].as_str().expect("a path").to_owned(),
                e["subject"].as_str().expect("a subject").to_owned(),
            )
        })
        .collect()
}

/// The norm of the entry `path`/`subject` of `step`.
fn norm_of(step: &Value, path: &str, subject: &str) -> f64 {
    let entry = step["parameters"]
        .as_array()
        .expect("a list")
        .iter()
        .find(|e| e["path"] == path && e["subject"] == subject)
        .unwrap_or_else(|| panic!("an entry {path} {subject}: {step}"));
    entry["norm"]
        .as_f64()
        .unwrap_or_else(|| panic!("a finite norm: {entry}"))
}

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-12)
}

fn compiled(model: &str, steps: u64) -> CompiledTraining {
    let source = std::fs::read_to_string(repo_root().join(model)).expect("the model");
    let mut c = compile_training(model, &source).unwrap_or_else(|d| panic!("{d:?}"));
    c.config.steps = steps;
    c
}

/// The initial value of each parameter by path, as `f64`.
fn initial(c: &CompiledTraining) -> BTreeMap<String, Vec<f64>> {
    let run = runner::run(&compiled_like(c, 0)).expect("a run of no step");
    c.forward
        .program
        .params
        .iter()
        .map(|(id, decl)| {
            let data = run.state.values[id].data();
            (
                decl.path.clone(),
                data.iter().map(|&v| f64::from(v)).collect(),
            )
        })
        .collect()
}

fn compiled_like(c: &CompiledTraining, steps: u64) -> CompiledTraining {
    let mut c = c.clone();
    c.config.steps = steps;
    c
}

/// The two-layer model in `f64`: `pred = relu(x · w1) · w2`, loss the mean of `(pred - y)²`.
/// `x` is 16×4, `w1` 4×8, `w2` 8×1, `y` 16×1, all row-major.
struct TwoLayer<'a> {
    x: &'a [f64],
    y: &'a [f64],
}

impl TwoLayer<'_> {
    fn loss(&self, w1: &[f64], w2: &[f64]) -> f64 {
        let mut sum = 0.0;
        for r in 0..16 {
            let mut pred = 0.0;
            for j in 0..8 {
                let pre: f64 = (0..4).map(|i| self.x[r * 4 + i] * w1[i * 8 + j]).sum();
                pred += pre.max(0.0) * w2[j];
            }
            sum += (pred - self.y[r]).powi(2);
        }
        sum / 16.0
    }

    /// The analytic gradients `(dL/dw1, dL/dw2)`.
    fn gradient(&self, w1: &[f64], w2: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut g1 = vec![0.0; 32];
        let mut g2 = vec![0.0; 8];
        for r in 0..16 {
            let pre: Vec<f64> = (0..8)
                .map(|j| (0..4).map(|i| self.x[r * 4 + i] * w1[i * 8 + j]).sum())
                .collect();
            let h: Vec<f64> = pre.iter().map(|p| p.max(0.0)).collect();
            let pred: f64 = (0..8).map(|j| h[j] * w2[j]).sum();
            let d = 2.0 * (pred - self.y[r]) / 16.0;
            for j in 0..8 {
                g2[j] += d * h[j];
                if pre[j] > 0.0 {
                    for i in 0..4 {
                        g1[i * 8 + j] += d * w2[j] * self.x[r * 4 + i];
                    }
                }
            }
        }
        (g1, g2)
    }
}

/// `observatory.stats.gradient-at-step` against an independent oracle: on two-layer-relu every
/// streamed `Gradient` norm of `TwoLayer.w1` and `TwoLayer.w2` at a sampled step is the norm of
/// the analytic `f64` gradient at the values `f64` SGD reaches by that step, and that gradient
/// agrees with central finite differences of the `f64` loss; the `Value` norms are those values'.
#[test]
fn two_layer_gradients_match_finite_differences_of_an_f64_oracle() {
    let c = compiled(TWO_LAYER, 6);
    let data = synthetic(&c.config, &c.forward).expect("synthetic data");
    let x: Vec<f64> = data.inputs["x"]
        .data()
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let y: Vec<f64> = data.inputs["y"]
        .data()
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let model = TwoLayer { x: &x, y: &y };
    let init = initial(&c);
    let (mut w1, mut w2) = (init["TwoLayer.w1"].clone(), init["TwoLayer.w2"].clone());

    let (code, events) = record(
        TWO_LAYER,
        "two-layer-fd.run.jsonl",
        &["--steps", "6", "--statistics-every", "2"],
    );
    assert_eq!(code, Some(0));
    assert_eq!(sampled(&events), vec![0, 2, 4, 6]);
    let by_step = steps(&events);
    let lr = 0.05;
    for k in 0..=6_u64 {
        let (g1, g2) = model.gradient(&w1, &w2);
        if k == 0 {
            let eps = 1e-6;
            for (which, len) in [(1, 32), (2, 8)] {
                for e in 0..len {
                    let (mut p1, mut p2) = (w1.clone(), w2.clone());
                    let (mut m1, mut m2) = (w1.clone(), w2.clone());
                    if which == 1 {
                        p1[e] += eps;
                        m1[e] -= eps;
                    } else {
                        p2[e] += eps;
                        m2[e] -= eps;
                    }
                    let fd = (model.loss(&p1, &p2) - model.loss(&m1, &m2)) / (2.0 * eps);
                    let analytic = if which == 1 { g1[e] } else { g2[e] };
                    assert!(
                        (fd - analytic).abs() < 1e-5,
                        "the oracle's gradient of w{which}[{e}]: {analytic} vs FD {fd}"
                    );
                }
            }
        }
        if k % 2 == 0 {
            let step = &by_step[&k];
            for (path, values) in [("TwoLayer.w1", &w1), ("TwoLayer.w2", &w2)] {
                let shown = norm_of(step, path, "Value");
                assert!(
                    close(shown, l2(values), 1e-4),
                    "step {k} {path} value norm {shown} vs {}",
                    l2(values)
                );
            }
            if k < 6 {
                for (path, gradient) in [("TwoLayer.w1", &g1), ("TwoLayer.w2", &g2)] {
                    let shown = norm_of(step, path, "Gradient");
                    assert!(
                        close(shown, l2(gradient), 1e-4),
                        "step {k} {path} gradient norm {shown} vs oracle {}",
                        l2(gradient)
                    );
                }
            }
        }
        for (w, g) in [(&mut w1, &g1), (&mut w2, &g2)] {
            for (p, d) in w.iter_mut().zip(g) {
                *p -= lr * d;
            }
        }
    }
}

const DENSE_MODEL: &str = "\
model Lin {
    param w: Tensor<f32, [2, 2]> = xavier_uniform();

    fn forward(x: Tensor<f32, [2, 2]>)
        -> Tensor<f32, [2, 2]> {
        return matmul(x, w);
    }
}

train Fit {
    model = Lin;
    input x: Tensor<f32, [2, 2]>;
    target y: Tensor<f32, [2, 2]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 3;
    seed = 42;
    device = cpu;
}
";

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// On dense data each step reads another batch: the `Gradient` norm at step `k` is the norm of
/// the gradient of batch `k`'s loss at the values after `k` updates, from `f64` arithmetic on the
/// four samples of `train_dense_data.rs` with `order.u32 = [2, 3, 0, 1, 3, 2]` and `w` the
/// identity.
#[test]
fn dense_data_gradients_are_of_the_batch_the_step_reads() {
    let root = scratch("dense");
    std::fs::create_dir_all(root.join("data")).unwrap();
    std::fs::create_dir_all(root.join("init")).unwrap();
    let x = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0];
    let y = [1.0_f32, 0.0, 1.0, 0.0, 0.0, 3.0, 1.0, 0.0];
    let order = [2_u32, 3, 0, 1, 3, 2];
    std::fs::write(root.join("model.mspec"), DENSE_MODEL).unwrap();
    std::fs::write(root.join("data/x.f32"), f32_bytes(&x)).unwrap();
    std::fs::write(root.join("data/y.f32"), f32_bytes(&y)).unwrap();
    std::fs::write(
        root.join("data/order.u32"),
        order
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    std::fs::write(root.join("init/w.f32"), f32_bytes(&[1.0, 0.0, 0.0, 1.0])).unwrap();
    let model = root.join("model.mspec").display().to_string();
    let data = root.join("data").display().to_string();
    let init = root.join("init").display().to_string();
    let (code, events) = record(
        &model,
        "dense.run.jsonl",
        &["--data", &data, "--init", &init, "--statistics-every", "1"],
    );
    assert_eq!(code, Some(0));
    assert_eq!(sampled(&events), vec![0, 1, 2, 3]);
    let by_step = steps(&events);
    let mut w = [1.0_f64, 0.0, 0.0, 1.0];
    for k in 0..3_usize {
        let rows = [order[2 * k] as usize, order[2 * k + 1] as usize];
        let mut g = [0.0_f64; 4];
        for &s in &rows {
            let xs = [f64::from(x[2 * s]), f64::from(x[2 * s + 1])];
            let ys = [f64::from(y[2 * s]), f64::from(y[2 * s + 1])];
            for col in 0..2 {
                let pred = xs[0] * w[col] + xs[1] * w[2 + col];
                let d = 2.0 * (pred - ys[col]) / 4.0;
                g[col] += d * xs[0];
                g[2 + col] += d * xs[1];
            }
        }
        let step = &by_step[&(k as u64)];
        let shown_value = norm_of(step, "Lin.w", "Value");
        assert!(
            close(shown_value, l2(&w), 1e-6),
            "step {k}: value norm {shown_value} vs {}",
            l2(&w)
        );
        let shown = norm_of(step, "Lin.w", "Gradient");
        assert!(
            close(shown, l2(&g), 1e-6),
            "step {k}: gradient norm {shown} vs the gradient of batch {rows:?}, {}",
            l2(&g)
        );
        for (p, d) in w.iter_mut().zip(g) {
            *p -= 0.1 * d;
        }
    }
    assert_eq!(
        entries(&by_step[&3]),
        vec![("Lin.w".to_owned(), "Value".to_owned())]
    );
}

/// N larger than the run samples only its first and last step, from 0 and from a resumed step;
/// N = 1 samples every step, the last with `Value` entries only.
#[test]
fn sampling_with_n_beyond_the_run_and_n_of_one() {
    let (code, events) = record(
        LINEAR,
        "n-beyond.run.jsonl",
        &["--steps", "3", "--statistics-every", "1000"],
    );
    assert_eq!(code, Some(0));
    assert_eq!(sampled(&events), vec![0, 3]);

    let (code, events) = record(
        LINEAR,
        "n-one.run.jsonl",
        &["--steps", "3", "--statistics-every", "1"],
    );
    assert_eq!(code, Some(0));
    assert_eq!(sampled(&events), vec![0, 1, 2, 3]);
    let by_step = steps(&events);
    for k in 0..3 {
        assert_eq!(
            entries(&by_step[&k]),
            vec![
                ("LinearReg.w".to_owned(), "Value".to_owned()),
                ("LinearReg.w".to_owned(), "Gradient".to_owned())
            ],
            "step {k}"
        );
    }
    assert_eq!(
        entries(&by_step[&3]),
        vec![("LinearReg.w".to_owned(), "Value".to_owned())]
    );

    let checkpoint = scratch("at-7.msckpt");
    let saved = modelspec(&[
        "train",
        LINEAR,
        "--backend",
        "cpu",
        "--steps",
        "7",
        "--checkpoint",
        &checkpoint.display().to_string(),
    ]);
    assert_eq!(saved.status.code(), Some(0), "{}", stderr(&saved));
    let (code, events) = record(
        LINEAR,
        "resumed-n-beyond.run.jsonl",
        &[
            "--resume",
            &checkpoint.display().to_string(),
            "--steps",
            "5",
            "--statistics-every",
            "1000",
        ],
    );
    assert_eq!(code, Some(0));
    assert_eq!(sampled(&events), vec![7, 12]);
    let (code, events) = record(
        LINEAR,
        "resumed-zero.run.jsonl",
        &[
            "--resume",
            &checkpoint.display().to_string(),
            "--steps",
            "0",
        ],
    );
    assert_eq!(code, Some(0));
    assert_eq!(sampled(&events), vec![7]);
    assert_eq!(
        entries(&steps(&events)[&7]),
        vec![("LinearReg.w".to_owned(), "Value".to_owned())],
        "a run of no step makes no update"
    );
}

fn with_lr(name: &str, lr: &str) -> String {
    let source = std::fs::read_to_string(repo_root().join(LINEAR)).expect("the example");
    let changed = source.replace("sgd(lr = 0.1)", &format!("sgd(lr = {lr})"));
    assert_ne!(changed, source);
    let path = scratch(name);
    std::fs::write(&path, changed).expect("the source is written");
    path.display().to_string()
}

/// A diverging run streams valid events up to its failure: every histogram holds the tensor's
/// four elements, in bins or beside them, and each step it reaches carries its entries.
#[test]
fn diverging_runs_stream_whole_valid_histograms() {
    for (lr, name) in [("100.0", "lr100"), ("1e38", "lr1e38")] {
        let model = with_lr(&format!("{name}.mspec"), lr);
        let (code, events) = record(
            &model,
            &format!("{name}.run.jsonl"),
            &["--statistics-every", "1"],
        );
        assert_eq!(code, Some(1), "lr {lr} diverges");
        let by_step = steps(&events);
        assert!(!by_step.is_empty());
        for (k, step) in &by_step {
            let list = step["parameters"].as_array().expect("a list");
            assert!(!list.is_empty(), "lr {lr} step {k} is sampled");
            for entry in list {
                let h = &entry["histogram"];
                let in_bins: u64 = h["counts"]
                    .as_array()
                    .expect("counts")
                    .iter()
                    .map(|c| c.as_u64().expect("a count"))
                    .sum();
                let beside = h["nan_count"].as_u64().unwrap()
                    + h["positive_infinity_count"].as_u64().unwrap()
                    + h["negative_infinity_count"].as_u64().unwrap();
                assert_eq!(in_bins + beside, 4, "lr {lr} step {k}: {entry}");
            }
        }
        assert_eq!(events.last().unwrap()["body"]["kind"], "RunFailed");
    }
}

/// A step whose loss overflows streams its gradient as it is: with `w = 3e38 · I` the batch of
/// samples 2 and 3 (`x = [1, 1]`, `[2, 0]`) predicts `6e38`, +inf in Binary32, so the loss of
/// step 0 is +inf and its gradient holds non-finite elements. The `Gradient` entry's norm is
/// null, its non-finite elements are counted beside the bins, and the event still validates.
#[test]
fn a_non_finite_gradient_streams_a_null_norm_and_its_counts() {
    let root = scratch("overflow");
    std::fs::create_dir_all(root.join("data")).unwrap();
    std::fs::create_dir_all(root.join("init")).unwrap();
    let x = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0];
    let y = [1.0_f32, 0.0, 1.0, 0.0, 0.0, 3.0, 1.0, 0.0];
    std::fs::write(root.join("model.mspec"), DENSE_MODEL).unwrap();
    std::fs::write(root.join("data/x.f32"), f32_bytes(&x)).unwrap();
    std::fs::write(root.join("data/y.f32"), f32_bytes(&y)).unwrap();
    std::fs::write(
        root.join("data/order.u32"),
        [2_u32, 3, 0, 1, 3, 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    std::fs::write(root.join("init/w.f32"), f32_bytes(&[3e38, 0.0, 0.0, 3e38])).unwrap();
    let model = root.join("model.mspec").display().to_string();
    let data = root.join("data").display().to_string();
    let init = root.join("init").display().to_string();
    let (code, events) = record(
        &model,
        "overflow.run.jsonl",
        &["--data", &data, "--init", &init, "--statistics-every", "1"],
    );
    assert_eq!(code, Some(1), "the loss of step 0 is not finite");
    let by_step = steps(&events);
    let step = &by_step[&0];
    assert!(step["loss"].is_null(), "{step}");
    let value = norm_of(step, "Lin.w", "Value");
    let expected = 3e38_f32 as f64 * 2.0_f64.sqrt();
    assert!(close(value, expected, 1e-12), "{value} vs {expected}");
    let gradient = step["parameters"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["subject"] == "Gradient")
        .unwrap_or_else(|| panic!("a Gradient entry at a failing step: {step}"));
    assert!(gradient["norm"].is_null(), "{gradient}");
    let h = &gradient["histogram"];
    let special = h["nan_count"].as_u64().unwrap()
        + h["positive_infinity_count"].as_u64().unwrap()
        + h["negative_infinity_count"].as_u64().unwrap();
    let in_bins: u64 = h["counts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_u64().unwrap())
        .sum();
    assert!(special >= 1, "{gradient}");
    assert_eq!(in_bins + special, 4, "{gradient}");
    assert!(
        h["bin_edges"]
            .as_array()
            .unwrap()
            .iter()
            .all(|e| e.is_f64()),
        "edges between finite min and max are finite: {gradient}"
    );
    assert_eq!(events.last().unwrap()["body"]["kind"], "RunFailed");
}

const ORDERED: &str = "\
model M {
    param zeta: Tensor<f32, [4, 2]> = xavier_uniform();
    param alpha: Tensor<f32, [2, 1]> = xavier_uniform();
    param mid: Tensor<f32, [3, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 1]> {
        return matmul(matmul(x, zeta), alpha);
    }
}

train Fit {
    model = M;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 3;
    device = cpu;
}
";

/// A sampled step lists every declared parameter in path order, not declaration order, `Value`
/// before `Gradient`, and an unread one with `Value` only.
#[test]
fn entries_are_in_path_order_whatever_the_declaration_order() {
    let model = scratch("ordered.mspec");
    std::fs::write(&model, ORDERED).unwrap();
    let (code, events) = record(
        &model.display().to_string(),
        "ordered.run.jsonl",
        &["--statistics-every", "1"],
    );
    assert_eq!(code, Some(0));
    let by_step = steps(&events);
    let pair = |p: &str, s: &str| (p.to_owned(), s.to_owned());
    assert_eq!(
        entries(&by_step[&0]),
        vec![
            pair("M.alpha", "Value"),
            pair("M.alpha", "Gradient"),
            pair("M.mid", "Value"),
            pair("M.zeta", "Value"),
            pair("M.zeta", "Gradient"),
        ]
    );
    assert_eq!(
        entries(&by_step[&2]),
        vec![
            pair("M.alpha", "Value"),
            pair("M.mid", "Value"),
            pair("M.zeta", "Value")
        ]
    );
}
