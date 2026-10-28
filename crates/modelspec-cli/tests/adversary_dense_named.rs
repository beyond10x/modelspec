//! Adversary of story:decider-dense-named-data: dense `--data` and `--eval` with named `f32` and
//! `i32` declarations (DESIGN.md § 2.2, § 8, § 8.1; `spec/domains/data.yaml`).
//!
//! Each case drives `modelspec train` on a directory it writes under this crate's build
//! directory. The device cases are ignored; run them with `--ignored` on a Vulkan device.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

/// Three inputs and two targets of leading dimension 4; `table` has 6 rows and the logits 3
/// classes, so an id must be below 6 and a label below 3.
const NAMED: &str = "\
model Named {
    param table: Tensor<f32, [6, 3]> = xavier_uniform();
    param w: Tensor<f32, [2, 3]> = xavier_uniform();

    fn forward(a: Tensor<i32, [4]>, b: Tensor<i32, [4]>, x: Tensor<f32, [4, 2]>,
        labels: Tensor<i32, [4]>, weights: Tensor<f32, [4]>) -> Tensor<f32, []> {
        let logits = add(add(embedding(table, a), embedding(table, b)), matmul(x, w));
        return cross_entropy(logits, labels, weights);
    }
}

train Fit {
    model = Named;
    input a: Tensor<i32, [4]>;
    input b: Tensor<i32, [4]>;
    input x: Tensor<f32, [4, 2]>;
    target labels: Tensor<i32, [4]>;
    target weights: Tensor<f32, [4]>;
    loss = model.forward(a, b, x, labels, weights);
    optimizer = sgd(lr = 0.5);
    steps = 10;
    seed = 3;
    device = cpu;
}
";

/// Every declaration of leading dimension 2, so `B = 2` and a sample is one row; the loss is
/// the `cross_entropy` of the input rows themselves, a known function of the batch, and the
/// parameter is read only through `mul(mean(p), 0.0)`.
const SHARED: &str = "\
model Shared {
    param p: Tensor<f32, [2]> = zeros();

    fn forward(x: Tensor<f32, [2, 3]>, labels: Tensor<i32, [2]>, weights: Tensor<f32, [2]>)
        -> Tensor<f32, []> {
        return add(cross_entropy(x, labels, weights), mul(mean(p), 0.0));
    }
}

train Shared {
    model = Shared;
    input x: Tensor<f32, [2, 3]>;
    target labels: Tensor<i32, [2]>;
    target weights: Tensor<f32, [2]>;
    loss = model.forward(x, labels, weights);
    optimizer = sgd(lr = 0.1);
    steps = 3;
    seed = 5;
    device = cpu;
}
";

/// A model output that is its input, `x + 0`, compared with one `f32` target by `mse`: the one
/// kind of program whose `--eval` line has an accuracy, with a loss and an accuracy a test can
/// compute by hand.
const IDENTITY: &str = "\
model Identity {
    param p: Tensor<f32, [2]> = zeros();

    fn forward(x: Tensor<f32, [2, 3]>) -> Tensor<f32, [2, 3]> {
        return add(x, mul(mean(p), 0.0));
    }
}

train Fit {
    model = Identity;
    input x: Tensor<f32, [2, 3]>;
    target y: Tensor<f32, [2, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 9;
    device = cpu;
}
";

/// A classifier: the model returns logits, the block's one target is the `i32` labels, the
/// weights are an input, and the loss is the `cross_entropy` of the model output with the
/// labels.
const CLASSIFIER: &str = "\
model Classifier {
    param w: Tensor<f32, [2, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [4, 2]>) -> Tensor<f32, [4, 3]> {
        return matmul(x, w);
    }
}

train Fit {
    model = Classifier;
    input x: Tensor<f32, [4, 2]>;
    input weights: Tensor<f32, [4]>;
    target labels: Tensor<i32, [4]>;
    loss = cross_entropy(model.forward(x), labels, weights);
    optimizer = sgd(lr = 0.1);
    steps = 4;
    seed = 7;
    device = cpu;
}
";

const LINEAR: &str = "../../examples/linear_regression.mspec";

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

/// A fresh, empty directory `name` under this crate's build directory.
fn fresh(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_dense_named")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old fixture is removed");
    }
    std::fs::create_dir_all(&dir).expect("the build directory is writable");
    dir
}

/// The case directory `name` with `model.mspec` holding `source`.
fn case(name: &str, source: &str) -> (PathBuf, PathBuf) {
    let root = fresh(name);
    let file = root.join("model.mspec");
    std::fs::write(&file, source).unwrap();
    (root, file)
}

fn write_f32(dir: &Path, name: &str, values: &[f32]) {
    std::fs::create_dir_all(dir).unwrap();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.f32")), bytes).unwrap();
}

fn write_i32(dir: &Path, name: &str, values: &[i32]) {
    std::fs::create_dir_all(dir).unwrap();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.i32")), bytes).unwrap();
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("modelspec runs")
}

fn train(file: &Path, backend: &str, data: &Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "train",
        path(file),
        "--backend",
        backend,
        "--data",
        path(data),
    ];
    args.extend_from_slice(extra);
    modelspec(&args)
}

fn ok(output: &Output) -> String {
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    text(&output.stdout)
}

/// The loss of every `StepCompleted` of the recorded run `file`, by step; empty when the file
/// does not exist.
fn recorded_losses(file: &Path) -> BTreeMap<u64, f64> {
    let Ok(content) = std::fs::read_to_string(file) else {
        return BTreeMap::new();
    };
    content
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("a JSON event"))
        .filter(|event| event["body"]["kind"] == "StepCompleted")
        .map(|event| {
            let value = &event["body"]["value"];
            (
                value["step"].as_u64().expect("a step"),
                value["loss"].as_f64().expect("a loss"),
            )
        })
        .collect()
}

/// The files of `dir`, by name; empty when it does not exist.
fn files_in(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn close(got: f64, expected: f64) -> bool {
    (got - expected).abs() <= 1e-5 * (1.0 + expected.abs())
}

fn eval_line(stdout: &str) -> &str {
    let evals: Vec<&str> = stdout.lines().filter(|l| l.starts_with("eval ")).collect();
    assert_eq!(evals.len(), 1, "{stdout}");
    evals[0]
}

/// Valid `NAMED` data of `samples` samples: ids below 6, labels below 3, positive weights.
fn write_named(dir: &Path, samples: usize, seed: u64) -> (Vec<i32>, Vec<i32>) {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    };
    let a: Vec<i32> = (0..samples).map(|_| (next() % 6) as i32).collect();
    let b: Vec<i32> = (0..samples).map(|_| (next() % 6) as i32).collect();
    let x: Vec<f32> = (0..2 * samples)
        .map(|_| (next() % 128) as f32 / 64.0 - 1.0)
        .collect();
    let labels: Vec<i32> = (0..samples).map(|_| (next() % 3) as i32).collect();
    let weights: Vec<f32> = (0..samples)
        .map(|_| 0.25 + (next() % 4) as f32 / 4.0)
        .collect();
    write_i32(dir, "a", &a);
    write_i32(dir, "b", &b);
    write_f32(dir, "x", &x);
    write_i32(dir, "labels", &labels);
    write_f32(dir, "weights", &weights);
    (a, labels)
}

/// Asserts that `output` is a run refused before its first step: exit 1, no step recorded in
/// `record`, no checkpoint in `checkpoints`, and stderr naming `name`.
fn refused_before_any_step(output: &Output, record: &Path, checkpoints: &Path, name: &str) {
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "stderr: {stderr}");
    let steps = recorded_losses(record);
    assert!(
        steps.is_empty(),
        "refused only after steps {:?} ran and checkpoints {:?} were written; stderr: {stderr}",
        steps.keys().collect::<Vec<_>>(),
        files_in(checkpoints)
    );
    assert_eq!(files_in(checkpoints), Vec::<String>::new(), "{stderr}");
    assert!(stderr.contains(name), "the refusal names {name}: {stderr}");
}

/// ADV-DN-1: an `i32` id outside the 6 rows of the `embedding` table that reads it, in block 2
/// of `a.i32`. Every file is read before the first step and an `f32` NaN is refused then
/// (E-DATA-007), and packed data refuses a table of the wrong size before the first step
/// (E-DATA-006); a dense id file the table cannot index must be refused then too, not after two
/// steps have updated the parameters and written their checkpoints.
#[test]
fn adv_dn_1_an_id_outside_the_table_in_a_later_block_is_refused_before_the_first_step() {
    let (root, file) = case("id_outside", NAMED);
    let data = root.join("data");
    let (mut a, _) = write_named(&data, 12, 11);
    a[8] = 6;
    write_i32(&data, "a", &a);
    let record = root.join("run.jsonl");
    let checkpoints = root.join("ck");
    std::fs::create_dir_all(&checkpoints).unwrap();
    let output = train(
        &file,
        "cpu",
        &data,
        &[
            "--record",
            path(&record),
            "--checkpoint",
            path(&checkpoints.join("run.msckpt")),
            "--checkpoint-every",
            "1",
        ],
    );
    refused_before_any_step(&output, &record, &checkpoints, "a.i32");
}

/// ADV-DN-1: a label outside the 3 classes of the `cross_entropy` that reads it, in block 1 of
/// `labels.i32`, refused before the first step as the id above.
#[test]
fn adv_dn_1_a_label_outside_the_classes_in_a_later_block_is_refused_before_the_first_step() {
    let (root, file) = case("label_outside", NAMED);
    let data = root.join("data");
    let (_, mut labels) = write_named(&data, 12, 11);
    labels[5] = 3;
    write_i32(&data, "labels", &labels);
    let record = root.join("run.jsonl");
    let checkpoints = root.join("ck");
    std::fs::create_dir_all(&checkpoints).unwrap();
    let output = train(
        &file,
        "cpu",
        &data,
        &[
            "--record",
            path(&record),
            "--checkpoint",
            path(&checkpoints.join("run.msckpt")),
            "--checkpoint-every",
            "1",
        ],
    );
    refused_before_any_step(&output, &record, &checkpoints, "labels.i32");
}

/// ADV-DN-1: a negative id in the `--eval` directory. DESIGN.md § 8: "Every file is read before
/// the first step"; the eval files are, but their ids are first looked at after the whole run,
/// so the run trains, writes its checkpoints and only then fails.
#[test]
fn adv_dn_1_an_eval_id_outside_the_table_is_refused_before_the_first_step() {
    let (root, file) = case("eval_id_outside", NAMED);
    let data = root.join("data");
    write_named(&data, 12, 11);
    let eval = root.join("eval");
    let (mut a, _) = write_named(&eval, 8, 12);
    a[5] = -1;
    write_i32(&eval, "a", &a);
    let record = root.join("run.jsonl");
    let checkpoints = root.join("ck");
    std::fs::create_dir_all(&checkpoints).unwrap();
    let output = train(
        &file,
        "cpu",
        &data,
        &[
            "--eval",
            path(&eval),
            "--record",
            path(&record),
            "--checkpoint",
            path(&checkpoints.join("run.msckpt")),
            "--checkpoint-every",
            "1",
        ],
    );
    refused_before_any_step(&output, &record, &checkpoints, "a.i32");
}

/// ADV-DN-2: the unit's `refused` helper (`train_dense_named.rs`) proves "before the first step"
/// by the absence of a `step` line on stdout. A run that fails at step 2, after two updates and
/// two checkpoints, prints no `step` line either, so that absence cannot tell a refusal before
/// the first step from a failure in the middle of the run. This case states the helper's
/// premise: a run that recorded steps prints them.
#[test]
fn adv_dn_2_a_run_that_fails_after_steps_ran_prints_the_steps_it_ran() {
    let (root, file) = case("silent_steps", NAMED);
    let data = root.join("data");
    let (mut a, _) = write_named(&data, 12, 11);
    a[8] = 6;
    write_i32(&data, "a", &a);
    let record = root.join("run.jsonl");
    let output = train(&file, "cpu", &data, &["--record", path(&record)]);
    let ran = recorded_losses(&record);
    if ran.is_empty() {
        return;
    }
    let stdout = text(&output.stdout);
    assert!(
        stdout.lines().any(|l| l.starts_with("step ")),
        "steps {:?} ran and were recorded, exit {:?}, yet stdout holds no `step` line, so the \
         unit's `refused` helper would read this run as refused before its first step; stdout: \
         {stdout:?}, stderr: {}",
        ran.keys().collect::<Vec<_>>(),
        output.status.code(),
        text(&output.stderr)
    );
}

/// ADV-DN-3: a classifier whose one target is the `i32` labels, compared with the model output
/// by `cross_entropy`. DESIGN.md § 8 prints the accuracy "only when the loss compares the model
/// output with the block's one `f32` target", and `read_dense` keeps a compared target for the
/// accuracy only when it is `f32` (`compared.filter(|t| t.ty.dtype == DType::F32)`): both
/// presuppose a block whose one compared target is `i32` trains. It does not: E-DATA-002 holds
/// the `i32[4]` labels to the `f32[4, 3]` output type first, so the filter is dead code and the
/// canonical classifier is refused while the same program with the weights declared a target
/// trains.
#[test]
fn adv_dn_3_a_classifier_with_one_i32_label_target_trains_and_evaluates_without_accuracy() {
    let (root, file) = case("classifier", CLASSIFIER);
    let data = root.join("data");
    let eval = root.join("eval");
    for (dir, samples) in [(&data, 8usize), (&eval, 4)] {
        let x: Vec<f32> = (0..2 * samples)
            .map(|i| (i % 5) as f32 / 4.0 - 0.5)
            .collect();
        let labels: Vec<i32> = (0..samples).map(|i| (i % 3) as i32).collect();
        write_f32(dir, "x", &x);
        write_f32(dir, "weights", &vec![1.0; samples]);
        write_i32(dir, "labels", &labels);
    }
    let stdout = ok(&train(&file, "cpu", &data, &["--eval", path(&eval)]));
    let line = eval_line(&stdout);
    assert!(line.starts_with("eval loss="), "{line}");
    assert!(
        !line.contains("accuracy"),
        "the one target is i32, not f32: {line}"
    );
}

/// Block `k` with a shared leading dimension: `B = 2` over `N = 3` samples, so step 1 reads
/// samples 2 and 0 across the end of the files, and every file must wrap at the same sample.
/// Each step's loss is computed here.
#[test]
fn adv_dn_shared_leading_dimension_wraps_every_file_at_the_same_sample() {
    let (root, file) = case("shared_wrap", SHARED);
    let data = root.join("data");
    let x: [f32; 9] = [0.5, -1.0, 0.25, 2.0, 0.0, -0.5, -1.5, 1.0, 0.75];
    let labels: [i32; 3] = [2, 0, 1];
    let weights: [f32; 3] = [1.0, 0.5, 0.25];
    write_f32(&data, "x", &x);
    write_i32(&data, "labels", &labels);
    write_f32(&data, "weights", &weights);
    let row_loss = |s: usize| -> f64 {
        let row: Vec<f64> = x[3 * s..3 * s + 3].iter().map(|&v| f64::from(v)).collect();
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + row.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
        lse - row[labels[s] as usize]
    };
    let block = |k: usize| -> f64 {
        let samples = [(2 * k) % 3, (2 * k + 1) % 3];
        let num: f64 = samples
            .iter()
            .map(|&s| f64::from(weights[s]) * row_loss(s))
            .sum();
        let den: f64 = samples.iter().map(|&s| f64::from(weights[s])).sum();
        num / den
    };
    let record = root.join("run.jsonl");
    ok(&train(&file, "cpu", &data, &["--record", path(&record)]));
    let losses = recorded_losses(&record);
    assert_eq!(losses.len(), 4, "{losses:?}");
    for (&step, &loss) in &losses {
        let expected = block(step as usize);
        assert!(close(loss, expected), "step {step}: {loss} {expected}");
    }
}

/// `--eval` against a hand computation: the model output is its input, so over two batches of
/// two rows the mean `mse` is (2.25 / 6 + 11.25 / 6) / 2 = 1.125, and two of the four target
/// rows have their first maximum where the output row has its first (a tie `[1, 1, 0]` counts
/// at index 0), so the accuracy is 0.5. The unit's test checks only that `accuracy=` appears.
#[test]
fn adv_dn_eval_loss_and_accuracy_match_a_hand_computation() {
    let (root, file) = case("eval_hand", IDENTITY);
    let data = root.join("data");
    write_f32(&data, "x", &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    write_f32(&data, "y", &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let eval = root.join("eval");
    write_f32(
        &eval,
        "x",
        &[1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0],
    );
    write_f32(
        &eval,
        "y",
        &[0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
    );
    let stdout = ok(&train(&file, "cpu", &data, &["--eval", path(&eval)]));
    assert_eq!(
        eval_line(&stdout),
        "eval loss=1.125 accuracy=0.5",
        "{stdout}"
    );
}

/// The claim of `train.rs`: the directory `--write-batch` writes "is also a dense `--data`
/// directory that holds one batch". Synthetic data feeds one batch at every step, so a run on
/// that directory prints the synthetic run's lines.
#[test]
fn adv_dn_the_written_batch_trains_as_the_synthetic_run_does() {
    let root = fresh("write_batch");
    let batch = root.join("batch");
    let synthetic = ok(&modelspec(&[
        "train",
        LINEAR,
        "--backend",
        "cpu",
        "--steps",
        "60",
        "--write-batch",
        path(&batch),
    ]));
    let dense = ok(&train(Path::new(LINEAR), "cpu", &batch, &["--steps", "60"]));
    assert_eq!(dense, synthetic);
}

/// An empty `.i32` file holds no sample: E-DATA-007 before the first step, as an empty `.f32`
/// file is, with nothing recorded.
#[test]
fn adv_dn_an_empty_i32_file_is_refused_before_the_first_step() {
    let (root, file) = case("empty_i32", NAMED);
    let data = root.join("data");
    write_named(&data, 12, 11);
    write_i32(&data, "b", &[]);
    let record = root.join("run.jsonl");
    let checkpoints = root.join("ck");
    std::fs::create_dir_all(&checkpoints).unwrap();
    let output = train(
        &file,
        "cpu",
        &data,
        &[
            "--record",
            path(&record),
            "--checkpoint",
            path(&checkpoints.join("run.msckpt")),
            "--checkpoint-every",
            "1",
        ],
    );
    refused_before_any_step(&output, &record, &checkpoints, "b.i32");
    assert!(text(&output.stderr).starts_with("E-DATA-007: "));
}

/// An `--eval` directory whose files differ in sample count is refused before the first step,
/// with nothing recorded and no checkpoint written.
#[test]
fn adv_dn_an_eval_directory_of_unequal_files_is_refused_before_the_first_step() {
    let (root, file) = case("eval_unequal", NAMED);
    let data = root.join("data");
    write_named(&data, 12, 11);
    let eval = root.join("eval");
    write_named(&eval, 8, 12);
    write_i32(&eval, "labels", &[0, 1, 2, 0]);
    let record = root.join("run.jsonl");
    let checkpoints = root.join("ck");
    std::fs::create_dir_all(&checkpoints).unwrap();
    let output = train(
        &file,
        "cpu",
        &data,
        &[
            "--eval",
            path(&eval),
            "--record",
            path(&record),
            "--checkpoint",
            path(&checkpoints.join("run.msckpt")),
            "--checkpoint-every",
            "1",
        ],
    );
    refused_before_any_step(&output, &record, &checkpoints, "labels");
    assert!(text(&output.stderr).starts_with("E-DATA-007: "));
}

#[cfg(feature = "vulkan")]
mod device {
    use super::*;

    /// ADV-DN-1 on the device: the stepwise runner uploads an `i32` tensor as its bit patterns,
    /// and nothing on the host looks at a dense id before it, so an id outside the table is an
    /// out-of-bounds read of the `embedding` kernel. It must be refused before the first step as
    /// on the CPU.
    #[test]
    #[ignore = "device reserved"]
    fn adv_dn_1_vulkan_an_id_outside_the_table_is_refused_before_the_first_step() {
        let (root, file) = case("id_outside_vulkan", NAMED);
        let data = root.join("data");
        let (mut a, _) = write_named(&data, 12, 11);
        a[8] = 6;
        write_i32(&data, "a", &a);
        let record = root.join("run.jsonl");
        let checkpoints = root.join("ck");
        std::fs::create_dir_all(&checkpoints).unwrap();
        let output = train(
            &file,
            "vulkan",
            &data,
            &[
                "--record",
                path(&record),
                "--checkpoint",
                path(&checkpoints.join("run.msckpt")),
                "--checkpoint-every",
                "1",
            ],
        );
        refused_before_any_step(&output, &record, &checkpoints, "a.i32");
    }

    /// The accuracy branch of the device's `--eval` (the prediction download and
    /// `correct_rows`), which no test of the unit reaches: its `NAMED` eval has no accuracy.
    #[test]
    #[ignore = "device reserved"]
    fn adv_dn_vulkan_eval_loss_and_accuracy_match_the_hand_computation() {
        let (root, file) = case("eval_hand_vulkan", IDENTITY);
        let data = root.join("data");
        write_f32(&data, "x", &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        write_f32(&data, "y", &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let eval = root.join("eval");
        write_f32(
            &eval,
            "x",
            &[1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0],
        );
        write_f32(
            &eval,
            "y",
            &[0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        );
        let stdout = ok(&train(&file, "vulkan", &data, &["--eval", path(&eval)]));
        let line = eval_line(&stdout);
        let rest = line.strip_prefix("eval loss=").expect("eval loss=");
        let (loss, accuracy) = rest.split_once(" accuracy=").expect("an accuracy");
        let loss: f64 = loss.parse().unwrap();
        assert!((loss - 1.125).abs() <= 1e-4 + 1e-3 * 1.125, "{line}");
        assert_eq!(accuracy, "0.5", "{line}");
    }

    /// Block `k` with a shared leading dimension on the device, wrapping inside a block.
    #[test]
    #[ignore = "device reserved"]
    fn adv_dn_vulkan_shared_leading_dimension_wraps_as_on_the_cpu() {
        let (root, file) = case("shared_wrap_vulkan", SHARED);
        let data = root.join("data");
        write_f32(
            &data,
            "x",
            &[0.5, -1.0, 0.25, 2.0, 0.0, -0.5, -1.5, 1.0, 0.75],
        );
        write_i32(&data, "labels", &[2, 0, 1]);
        write_f32(&data, "weights", &[1.0, 0.5, 0.25]);
        let cpu_record = root.join("cpu.jsonl");
        let gpu_record = root.join("gpu.jsonl");
        ok(&train(
            &file,
            "cpu",
            &data,
            &["--record", path(&cpu_record)],
        ));
        ok(&train(
            &file,
            "vulkan",
            &data,
            &["--record", path(&gpu_record)],
        ));
        let (c, g) = (recorded_losses(&cpu_record), recorded_losses(&gpu_record));
        assert_eq!(c.len(), 4, "{c:?}");
        assert_eq!(c.keys().collect::<Vec<_>>(), g.keys().collect::<Vec<_>>());
        for (step, &cl) in &c {
            let gl = g[step];
            assert!(
                (gl - cl).abs() <= 1e-4 + 1e-3 * cl.abs(),
                "step {step}: {gl} {cl}"
            );
        }
    }
}
