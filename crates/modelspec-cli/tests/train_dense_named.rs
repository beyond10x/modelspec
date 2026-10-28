//! `modelspec train --data <dir>` on a dense directory with any number of named `f32` and `i32`
//! inputs and targets (story:decider-dense-named-data; DESIGN.md § 8, `spec/domains/data.yaml`,
//! `DenseDataDirectory`).
//!
//! Every `input` and `target` declaration reads `<dir>/<name>.f32` when it is `f32` and
//! `<dir>/<name>.i32` when it is `i32`, row-major little-endian with no header. When every
//! declaration has the same leading dimension `B`, a sample is one row along it; otherwise
//! `B = 1` and a sample is the whole declared tensor. Step `k` feeds block `k` of every file, the
//! samples `(k·B + i) mod N`.
//!
//! The fixtures are written by the tests under this crate's build directory:
//!
//! - `NAMED`: three inputs (`a`, `b` as `i32`, `x` as `f32`) and two targets (`labels` as `i32`,
//!   `weights` as `f32`), all of leading dimension 4, whose loss is the `cross_entropy` the
//!   forward function returns, so no target is compared with a model output;
//! - `PROBE`: the same kinds of declarations of different leading dimensions, so each sample is
//!   a whole tensor, and a loss that is a known function of the batch: the parameter is read
//!   only through `mul(mean(p), 0.0)`, so its gradient is 0 and it never moves;
//! - `fixtures/decider_small.mspec`: the decider program of the 0.21.0 workaround form at test
//!   sizes, 19 named tensors a step, 12 of them `i32`.
//!
//! The device tests are ignored by default; run them with
//! `cargo test -p modelspec-cli --features vulkan --test train_dense_named -- --ignored`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

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

/// The samples of each `NAMED` file: 12, three blocks of `B = 4`.
const NAMED_SAMPLES: usize = 12;

const PROBE: &str = "\
model Probe {
    param p: Tensor<f32, [2]> = zeros();

    fn forward(x: Tensor<f32, [4, 3]>, a: Tensor<i32, [2]>, b: Tensor<i32, [2]>,
        labels: Tensor<i32, [2]>, weights: Tensor<f32, [2]>) -> Tensor<f32, []> {
        let logits = add(embedding(x, a), embedding(x, b));
        return add(cross_entropy(logits, labels, weights), mul(mean(p), 0.0));
    }
}

train Probe {
    model = Probe;
    input x: Tensor<f32, [4, 3]>;
    input a: Tensor<i32, [2]>;
    input b: Tensor<i32, [2]>;
    target labels: Tensor<i32, [2]>;
    target weights: Tensor<f32, [2]>;
    loss = model.forward(x, a, b, labels, weights);
    optimizer = sgd(lr = 0.1);
    steps = 4;
    seed = 5;
    device = cpu;
}
";

/// The blocks of each `PROBE` file: step 3 wraps to block 0 and step 4 reads block 1.
const PROBE_BLOCKS: usize = 3;

/// One `i32` input and one `f32` target of the model output type, the loss comparing them: the
/// one kind of program whose `--eval` line has an accuracy.
const COMPARED: &str = "\
model Compared {
    param table: Tensor<f32, [5, 3]> = xavier_uniform();

    fn forward(a: Tensor<i32, [2]>) -> Tensor<f32, [2, 3]> {
        return embedding(table, a);
    }
}

train Fit {
    model = Compared;
    input a: Tensor<i32, [2]>;
    target y: Tensor<f32, [2, 3]>;
    loss = mse(model.forward(a), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 9;
    device = cpu;
}
";

/// Exactly one target, the `f32[4]` weights, of a type other than the model output's (`f32[]`),
/// and a loss that is the scalar the forward function returns: no model output is compared
/// with the target, so E-DATA-002 does not apply.
const SCALAR: &str = "\
model Scalar {
    param w: Tensor<f32, [2, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [4, 2]>, labels: Tensor<i32, [4]>, weights: Tensor<f32, [4]>)
        -> Tensor<f32, []> {
        return cross_entropy(matmul(x, w), labels, weights);
    }
}

train Fit {
    model = Scalar;
    input x: Tensor<f32, [4, 2]>;
    input labels: Tensor<i32, [4]>;
    target weights: Tensor<f32, [4]>;
    loss = model.forward(x, labels, weights);
    optimizer = sgd(lr = 0.1);
    steps = 4;
    seed = 7;
    device = cpu;
}
";

/// The same one target, now compared by the loss with a model output of another type: the
/// forward function returns the `f32[4, 3]` logits and the loss reads them beside the `f32[4]`
/// weights, so E-DATA-002 holds the target to the output type.
const SCALAR_COMPARED: &str = "\
model Logits {
    param w: Tensor<f32, [2, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [4, 2]>) -> Tensor<f32, [4, 3]> {
        return matmul(x, w);
    }
}

train Fit {
    model = Logits;
    input x: Tensor<f32, [4, 2]>;
    input labels: Tensor<i32, [4]>;
    target weights: Tensor<f32, [4]>;
    loss = cross_entropy(model.forward(x), labels, weights);
    optimizer = sgd(lr = 0.1);
    steps = 4;
    seed = 7;
    device = cpu;
}
";

const DECIDER: &str = "tests/fixtures/decider_small.mspec";

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

/// A fresh, empty directory `name` under this crate's build directory.
fn fresh(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("train_dense_named")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old fixture is removed");
    }
    std::fs::create_dir_all(&dir).expect("the build directory is writable");
    dir
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A deterministic stream of values for fixture data: a 64-bit LCG.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// A value in `[-1, 1)` in steps of 1/64, exact in `f32`.
    fn f32(&mut self) -> f32 {
        (self.next() % 128) as f32 / 64.0 - 1.0
    }

    fn below(&mut self, n: u64) -> i32 {
        (self.next() % n) as i32
    }
}

/// The values of one dense file.
#[derive(Clone, Debug)]
enum Values {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl Values {
    fn file(&self, name: &str) -> (String, Vec<u8>) {
        match self {
            Values::F32(v) => (format!("{name}.f32"), f32_bytes(v)),
            Values::I32(v) => (format!("{name}.i32"), i32_bytes(v)),
        }
    }
}

/// Writes each `(name, values)` of `tensors` into `dir` as its dense file.
fn write_dir(dir: &Path, tensors: &BTreeMap<&str, Values>) {
    std::fs::create_dir_all(dir).unwrap();
    for (name, values) in tensors {
        let (file, bytes) = values.file(name);
        std::fs::write(dir.join(file), bytes).unwrap();
    }
}

/// The `NAMED` data: `samples` samples of `a`, `b` (ids below 6), `x`, `labels` (below 3) and
/// `weights` (positive).
fn named_data(seed: u64, samples: usize) -> BTreeMap<&'static str, Values> {
    let mut s = Stream(seed);
    let a = (0..samples).map(|_| s.below(6)).collect();
    let b = (0..samples).map(|_| s.below(6)).collect();
    let x = (0..2 * samples).map(|_| s.f32()).collect();
    let labels = (0..samples).map(|_| s.below(3)).collect();
    let weights = (0..samples)
        .map(|_| 0.25 + (s.next() % 4) as f32 / 4.0)
        .collect();
    BTreeMap::from([
        ("a", Values::I32(a)),
        ("b", Values::I32(b)),
        ("x", Values::F32(x)),
        ("labels", Values::I32(labels)),
        ("weights", Values::F32(weights)),
    ])
}

/// The `PROBE` data: `blocks` whole tensors of each declaration.
fn probe_data(seed: u64, blocks: usize) -> BTreeMap<&'static str, Values> {
    let mut s = Stream(seed);
    let x = (0..12 * blocks).map(|_| 2.0 * s.f32()).collect();
    let a = (0..2 * blocks).map(|_| s.below(4)).collect();
    let b = (0..2 * blocks).map(|_| s.below(4)).collect();
    let labels = (0..2 * blocks).map(|_| s.below(3)).collect();
    let weights = (0..2 * blocks)
        .map(|_| 0.5 + (s.next() % 4) as f32 / 4.0)
        .collect();
    BTreeMap::from([
        ("x", Values::F32(x)),
        ("a", Values::I32(a)),
        ("b", Values::I32(b)),
        ("labels", Values::I32(labels)),
        ("weights", Values::F32(weights)),
    ])
}

fn f32s<'v>(data: &'v BTreeMap<&str, Values>, name: &str) -> &'v [f32] {
    match &data[name] {
        Values::F32(v) => v,
        Values::I32(_) => panic!("{name} is i32"),
    }
}

fn i32s<'v>(data: &'v BTreeMap<&str, Values>, name: &str) -> &'v [i32] {
    match &data[name] {
        Values::I32(v) => v,
        Values::F32(_) => panic!("{name} is f32"),
    }
}

/// The `PROBE` loss of block `k`, computed here in `f64`: rows `r` of
/// `logits = x[a_r] + x[b_r]`, `Σ w_r (logsumexp(logits_r) - logits_r[label_r]) / Σ w_r`.
fn probe_loss(data: &BTreeMap<&str, Values>, k: usize) -> f64 {
    let x = &f32s(data, "x")[12 * k..12 * (k + 1)];
    let (a, b) = (&i32s(data, "a")[2 * k..], &i32s(data, "b")[2 * k..]);
    let labels = &i32s(data, "labels")[2 * k..];
    let weights = &f32s(data, "weights")[2 * k..];
    let (mut numerator, mut denominator) = (0.0f64, 0.0f64);
    for r in 0..2 {
        let row = |i: i32| &x[3 * i as usize..3 * i as usize + 3];
        let logits: Vec<f64> = row(a[r])
            .iter()
            .zip(row(b[r]))
            .map(|(&p, &q)| f64::from(p + q))
            .collect();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln();
        let w = f64::from(weights[r]);
        numerator += w * (lse - logits[labels[r] as usize]);
        denominator += w;
    }
    numerator / denominator
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("modelspec runs")
}

/// `modelspec train <file> --backend <backend> --data <data>` and `extra`.
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

/// The stdout of a run that exits 0.
fn ok(output: &Output) -> String {
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    text(&output.stdout)
}

/// The stderr of `modelspec train <file> --backend cpu --data <data>` with `extra`, refused with
/// `code` before its first step: exit 1, stderr opening with the code, a record that ends in the
/// `RunFailed` of `code` and holds no `StepCompleted`, and no checkpoint file. The run records
/// to `refused/run.jsonl` beside `data` and asks for a checkpoint after every step into
/// `refused/ck/`, a fresh directory each call. Stdout cannot show "before the first step": a run
/// that fails after steps ran prints no `step` line either (ADV-DN-2).
fn refused(file: &Path, data: &Path, extra: &[&str], code: &str) -> String {
    let probe = data.with_file_name("refused");
    if probe.exists() {
        std::fs::remove_dir_all(&probe).expect("the old probe is removed");
    }
    let checkpoints = probe.join("ck");
    std::fs::create_dir_all(&checkpoints).expect("the build directory is writable");
    let record = probe.join("run.jsonl");
    let checkpoint = checkpoints.join("run.msckpt");
    let mut args = vec![
        "--record",
        path(&record),
        "--checkpoint",
        path(&checkpoint),
        "--checkpoint-every",
        "1",
    ];
    args.extend_from_slice(extra);
    let output = train(file, "cpu", data, &args);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "stderr: {stderr}");
    assert!(
        stderr.starts_with(&format!("{code}: ")),
        "expected {code}, stderr: {stderr}"
    );
    let events = recorded_events(&record);
    let steps: Vec<&Value> = events
        .iter()
        .filter(|event| event["body"]["kind"] == "StepCompleted")
        .collect();
    assert!(steps.is_empty(), "steps ran: {steps:?}; stderr: {stderr}");
    let last = events.last().expect("the record holds the run's end");
    assert_eq!(last["body"]["kind"], "RunFailed", "{last}");
    assert_eq!(last["body"]["value"]["code"], code, "{last}");
    let written: Vec<_> = std::fs::read_dir(&checkpoints)
        .expect("the checkpoint directory")
        .map(|entry| entry.expect("an entry").file_name())
        .collect();
    assert!(
        written.is_empty(),
        "checkpoints {written:?}; stderr: {stderr}"
    );
    stderr
}

/// Every event of the recorded run `file`, in order.
fn recorded_events(file: &Path) -> Vec<Value> {
    std::fs::read_to_string(file)
        .expect("the recording")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("a JSON event"))
        .collect()
}

/// The `step <k> loss <v>` lines of `stdout`.
fn step_lines(stdout: &str) -> Vec<&str> {
    stdout.lines().filter(|l| l.starts_with("step ")).collect()
}

/// The loss of every `StepCompleted` of the recorded run `file`, by step.
fn recorded_losses(file: &Path) -> BTreeMap<u64, f64> {
    std::fs::read_to_string(file)
        .expect("the recording")
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

/// The one `eval loss=…` line of `stdout`, which is its last line.
fn eval_line(stdout: &str) -> &str {
    let evals: Vec<&str> = stdout.lines().filter(|l| l.starts_with("eval ")).collect();
    assert_eq!(evals.len(), 1, "{stdout}");
    assert_eq!(stdout.lines().last(), Some(evals[0]), "{stdout}");
    evals[0]
}

fn close(got: f64, expected: f64) -> bool {
    (got - expected).abs() <= 1e-5 * (1.0 + expected.abs())
}

/// The case directory `name` with `model.mspec` holding `source`.
fn case(name: &str, source: &str) -> (PathBuf, PathBuf) {
    let root = fresh(name);
    let file = root.join("model.mspec");
    std::fs::write(&file, source).unwrap();
    (root, file)
}

/// dense-named-cpu: three inputs (two `i32`, one `f32`) and two targets (one `i32`, one `f32`)
/// train 10 steps on the CPU from a directory of `.i32` and `.f32` files, and two runs from one
/// seed print the same `step` lines.
#[test]
fn dense_named_cpu_trains_three_inputs_and_two_targets_reproducibly() {
    let (root, file) = case("named_cpu", NAMED);
    let data = root.join("data");
    write_dir(&data, &named_data(11, NAMED_SAMPLES));
    let first = ok(&train(&file, "cpu", &data, &[]));
    let second = ok(&train(&file, "cpu", &data, &[]));
    let steps = step_lines(&first);
    assert_eq!(steps.len(), 2, "step 0 and step 10: {first}");
    assert!(steps[0].starts_with("step 0 loss "), "{first}");
    assert!(steps[1].starts_with("step 10 loss "), "{first}");
    assert_eq!(steps, step_lines(&second));
    assert_eq!(first, second);
    assert_ne!(steps[0]["step 0".len()..], steps[1]["step 10".len()..]);
}

/// dense-named-block-k: without `order.u32`, step `k` reads block `k` of every file. The
/// declarations do not share a leading dimension, so each block is one whole tensor of each
/// file, and the loss of every step is the function of its block computed here: steps 0, 1 and
/// 2 read blocks 0, 1 and 2, step 3 wraps to block 0, and step 4 reads block 1.
#[test]
fn dense_named_block_k_feeds_block_k_of_every_file_and_wraps() {
    let (root, file) = case("block_k", PROBE);
    let data = root.join("data");
    let tensors = probe_data(21, PROBE_BLOCKS);
    write_dir(&data, &tensors);
    let record = root.join("run.jsonl");
    ok(&train(&file, "cpu", &data, &["--record", path(&record)]));
    let losses = recorded_losses(&record);
    assert_eq!(losses.len(), 5, "{losses:?}");
    for (step, &loss) in &losses {
        let block = *step as usize % PROBE_BLOCKS;
        let expected = probe_loss(&tensors, block);
        assert!(
            close(loss, expected),
            "step {step} reads block {block}: {loss} {expected}"
        );
    }
    assert!(
        !close(probe_loss(&tensors, 0), probe_loss(&tensors, 1)),
        "the blocks differ"
    );
}

/// dense-named-i32-refusals: a missing `NAME.i32`, an `.i32` file that is not a whole number of
/// samples, and files of different sample counts are E-DATA-007 before the first step; a
/// declaration whose file exists only under the other dtype's extension is refused naming the
/// file it reads.
#[test]
fn dense_named_i32_refusals_come_before_the_first_step() {
    let (root, file) = case("refusals", NAMED);
    let data = root.join("data");
    let tensors = named_data(11, NAMED_SAMPLES);

    write_dir(&data, &tensors);
    std::fs::remove_file(data.join("labels.i32")).unwrap();
    let stderr = refused(&file, &data, &[], "E-DATA-007");
    assert!(stderr.contains("labels.i32"), "{stderr}");

    write_dir(&data, &tensors);
    std::fs::write(data.join("b.i32"), i32_bytes(&[1; 8])).unwrap();
    let stderr = refused(&file, &data, &[], "E-DATA-007");
    assert!(stderr.contains("samples"), "{stderr}");

    write_dir(&data, &tensors);
    std::fs::write(data.join("a.i32"), &i32_bytes(&[1; NAMED_SAMPLES])[..47]).unwrap();
    let stderr = refused(&file, &data, &[], "E-DATA-007");
    assert!(stderr.contains("a.i32"), "{stderr}");

    write_dir(&data, &tensors);
    std::fs::rename(data.join("x.f32"), data.join("x.i32")).unwrap();
    let stderr = refused(&file, &data, &[], "E-DATA-007");
    assert!(stderr.contains("x.f32"), "{stderr}");
    assert!(stderr.contains("x.i32"), "{stderr}");
    std::fs::rename(data.join("x.i32"), data.join("x.f32")).unwrap();

    std::fs::rename(data.join("a.i32"), data.join("a.f32")).unwrap();
    let stderr = refused(&file, &data, &[], "E-DATA-007");
    assert!(stderr.contains("a.i32"), "{stderr}");
    assert!(stderr.contains("a.f32"), "{stderr}");

    let (probe_root, probe) = case("refusals_probe", PROBE);
    let probe_data_dir = probe_root.join("data");
    write_dir(&probe_data_dir, &probe_data(21, PROBE_BLOCKS));
    std::fs::write(probe_data_dir.join("a.i32"), i32_bytes(&[0; 5])).unwrap();
    let stderr = refused(&probe, &probe_data_dir, &[], "E-DATA-007");
    assert!(stderr.contains("a.i32"), "{stderr}");
    assert!(stderr.contains("not a whole number of samples"), "{stderr}");
}

/// An `i32` value outside the bound of an op that reads it, in a later block of a `--data` file
/// or in an `--eval` file, is E-DATA-011 before the first step, naming the file, the element
/// position and the bound: an id at or past the 6 rows of the `embedding` table, a label at or
/// past the 3 classes of the `cross_entropy` logits, and a negative id (ADV-DN-1).
#[test]
fn dense_named_i32_values_outside_their_bound_are_refused_before_the_first_step() {
    let (root, file) = case("out_of_range", NAMED);
    let data = root.join("data");
    let tensors = named_data(11, NAMED_SAMPLES);
    let with = |name: &'static str, position: usize, value: i32| {
        let mut changed = tensors.clone();
        if let Some(Values::I32(values)) = changed.get_mut(name) {
            values[position] = value;
        }
        changed
    };

    write_dir(&data, &with("a", 8, 6));
    let stderr = refused(&file, &data, &[], "E-DATA-011");
    for expected in [
        "a.i32",
        "position: 8",
        "value: 6",
        "0 <= value < 6",
        "`embedding` table",
    ] {
        assert!(stderr.contains(expected), "{expected}: {stderr}");
    }

    write_dir(&data, &with("labels", 5, 3));
    let stderr = refused(&file, &data, &[], "E-DATA-011");
    for expected in [
        "labels.i32",
        "position: 5",
        "value: 3",
        "0 <= value < 3",
        "`cross_entropy` logits",
    ] {
        assert!(stderr.contains(expected), "{expected}: {stderr}");
    }

    write_dir(&data, &tensors);
    let eval = root.join("eval");
    let mut held = named_data(12, 8);
    if let Some(Values::I32(b)) = held.get_mut("b") {
        b[7] = -1;
    }
    write_dir(&eval, &held);
    let stderr = refused(&file, &data, &["--eval", path(&eval)], "E-DATA-011");
    let eval_file = eval.join("b.i32");
    for expected in [
        path(&eval_file),
        "position: 7",
        "value: -1",
        "0 <= value < 6",
    ] {
        assert!(stderr.contains(expected), "{expected}: {stderr}");
    }
}

/// dense-named-eval-cpu: `--eval DIR` reads the same layout and prints `eval loss=<mean>` over
/// every block, without `accuracy=` for a loss that is the scalar the forward function returns;
/// a program whose one target is of the model output type gets `accuracy=` as before.
#[test]
fn dense_named_eval_cpu_prints_the_mean_block_loss() {
    let (root, file) = case("eval_cpu", PROBE);
    let data = root.join("data");
    write_dir(&data, &probe_data(21, PROBE_BLOCKS));
    let eval = root.join("eval");
    let held = probe_data(33, 2);
    write_dir(&eval, &held);
    let stdout = ok(&train(&file, "cpu", &data, &["--eval", path(&eval)]));
    let line = eval_line(&stdout);
    let loss: f64 = line
        .strip_prefix("eval loss=")
        .unwrap_or_else(|| panic!("`eval loss=<mean>`: {line}"))
        .parse()
        .unwrap_or_else(|_| panic!("only the loss: {line}"));
    let expected = (probe_loss(&held, 0) + probe_loss(&held, 1)) / 2.0;
    assert!(close(loss, expected), "{loss} {expected}");

    let (root, file) = case("eval_compared", COMPARED);
    let mut s = Stream(5);
    let ids: Vec<i32> = (0..4).map(|_| s.below(5)).collect();
    let y: Vec<f32> = (0..12).map(|_| s.f32()).collect();
    let tensors = BTreeMap::from([("a", Values::I32(ids)), ("y", Values::F32(y))]);
    for dir in ["data", "eval"] {
        write_dir(&root.join(dir), &tensors);
    }
    let stdout = ok(&train(
        &file,
        "cpu",
        &root.join("data"),
        &["--eval", path(&root.join("eval"))],
    ));
    let line = eval_line(&stdout);
    assert!(line.starts_with("eval loss="), "{line}");
    assert!(line.contains(" accuracy="), "{line}");
}

/// dense-named-scalar-loss: a block with exactly one target, of a type other than the model
/// output's, whose loss is the scalar the forward function returns, trains on the CPU without
/// E-DATA-002; the same target compared by the loss with a model output of another type is
/// refused with E-DATA-002 before the first step.
#[test]
fn dense_named_scalar_loss_holds_no_target_to_the_model_output_type() {
    let mut s = Stream(13);
    let tensors = BTreeMap::from([
        ("x", Values::F32((0..8 * 2).map(|_| s.f32()).collect())),
        ("labels", Values::I32((0..8).map(|_| s.below(3)).collect())),
        (
            "weights",
            Values::F32((0..8).map(|_| 0.5 + (s.next() % 4) as f32 / 4.0).collect()),
        ),
    ]);

    let (root, file) = case("scalar_loss", SCALAR);
    let data = root.join("data");
    write_dir(&data, &tensors);
    let record = root.join("run.jsonl");
    let stdout = ok(&train(&file, "cpu", &data, &["--record", path(&record)]));
    let steps = step_lines(&stdout);
    assert!(steps[0].starts_with("step 0 loss "), "{stdout}");
    assert!(
        steps.last().unwrap().starts_with("step 4 loss "),
        "{stdout}"
    );
    let losses = recorded_losses(&record);
    assert_eq!(losses.len(), 5, "{losses:?}");
    assert!(losses.values().all(|l| l.is_finite()), "{losses:?}");

    let (root, file) = case("scalar_loss_compared", SCALAR_COMPARED);
    let data = root.join("data");
    write_dir(&data, &tensors);
    let stderr = refused(&file, &data, &[], "E-DATA-002");
    assert!(stderr.contains("Tensor<f32, [4]>"), "{stderr}");
    assert!(stderr.contains("Tensor<f32, [4, 3]>"), "{stderr}");
}

/// The 19 tensors of `decider_small.mspec` for `blocks` steps: a pool of 4 events of 4 tokens,
/// 2 decisions over windows of 2 events, 3 templates and the abstain row.
fn decider_data(blocks: usize) -> BTreeMap<&'static str, Values> {
    let mut s = Stream(77);
    let mut i32s: BTreeMap<&str, Vec<i32>> = BTreeMap::new();
    let mut f32s: BTreeMap<&str, Vec<f32>> = BTreeMap::new();
    for k in 0..blocks {
        let mut push_i = |name, values: Vec<i32>| i32s.entry(name).or_default().extend(values);
        push_i("tok_ids", (0..16).map(|_| s.below(32)).collect());
        push_i("tok_pos", (0..16).map(|t| t % 4).collect());
        push_i("evt_row", vec![0, 4, 8, 12]);
        push_i(
            "ctx_idx",
            if k % 2 == 0 {
                vec![0, 1, 1, 2]
            } else {
                vec![1, 2, 2, 3]
            },
        );
        push_i("ctx_slot", vec![0, 1, 0, 1]);
        push_i("ctx_sess", vec![0, 1, 1, 2]);
        push_i("last_row", vec![1, 3]);
        push_i("next_evt", if k % 2 == 0 { vec![2, 3] } else { vec![3, 0] });
        push_i("pair_ctx", (0..8).map(|p| p / 4).collect());
        push_i("pair_act", (0..8).map(|p| p % 4).collect());
        let taken = [s.below(3), s.below(3)];
        push_i("taken_pair", vec![taken[0], 4 + taken[1]]);
        push_i(
            "labels",
            vec![taken[0], if k % 2 == 0 { 3 } else { taken[1] }],
        );
        let mut push_f = |name, values: Vec<f32>| f32s.entry(name).or_default().extend(values);
        let mut mask = vec![1.0; 16];
        mask[4 * (k % 4) + 3] = 0.0;
        push_f("tok_mask", mask);
        push_f("ctx_mask", vec![1.0; 4]);
        push_f(
            "fold_rows",
            (0..16)
                .map(|i| f32::from(u8::from(i % 8 / 4 == i / 8)))
                .collect(),
        );
        push_f(
            "fold_cols",
            (0..32)
                .map(|i| f32::from(u8::from((i / 4) % 4 == i % 4)))
                .collect(),
        );
        push_f("fold_bcast", vec![1.0; 4]);
        push_f("cand_bias", vec![0.0, 0.0, -30.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        push_f("label_weights", vec![1.0, 0.5]);
    }
    i32s.into_iter()
        .map(|(n, v)| (n, Values::I32(v)))
        .chain(f32s.into_iter().map(|(n, v)| (n, Values::F32(v))))
        .collect()
}

/// decider-small-cpu: the decider program at test sizes trains 10 steps on the CPU from a
/// directory of its 19 tensors, 12 of them `i32`.
#[test]
fn decider_small_cpu_trains_on_its_19_tensors() {
    let root = fresh("decider_cpu");
    let data = root.join("data");
    let tensors = decider_data(2);
    assert_eq!(tensors.len(), 19);
    assert_eq!(
        tensors
            .values()
            .filter(|v| matches!(v, Values::I32(_)))
            .count(),
        12
    );
    write_dir(&data, &tensors);
    let record = root.join("run.jsonl");
    let stdout = ok(&train(
        Path::new(DECIDER),
        "cpu",
        &data,
        &["--record", path(&record)],
    ));
    let steps = step_lines(&stdout);
    assert!(steps[0].starts_with("step 0 loss "), "{stdout}");
    assert!(
        steps.last().unwrap().starts_with("step 10 loss "),
        "{stdout}"
    );
    let losses = recorded_losses(&record);
    assert_eq!(losses.len(), 11, "{losses:?}");
    assert!(losses.values().all(|l| l.is_finite()), "{losses:?}");
}

/// A JEPA-shaped program on dense data: an `i32` token input read by `embedding`, an `f32`
/// target of the model output type, and the loss
/// `mse(predict, stop_gradient(target_embedding)) + 0.1 * sigreg(embedding output, slices = 4)`.
/// The target branch shares the encoder with the context branch and passes it no gradient
/// (story:jepa-stop-gradient, story:jepa-sigreg-cpu, story:decider-dense-named-data).
const JEPA: &str = "\
model Jepa {
    param table: Tensor<f32, [16, 8]> = xavier_uniform();
    param encoder: Tensor<f32, [8, 8]> = xavier_uniform();
    param predictor: Tensor<f32, [8, 8]> = xavier_uniform();

    fn embed(tokens: Tensor<i32, [32]>) -> Tensor<f32, [32, 8]> {
        return embedding(table, tokens);
    }

    fn target_embedding(y: Tensor<f32, [32, 8]>) -> Tensor<f32, [32, 8]> {
        return matmul(y, encoder);
    }

    fn forward(tokens: Tensor<i32, [32]>) -> Tensor<f32, [32, 8]> {
        return matmul(matmul(embedding(table, tokens), encoder), predictor);
    }
}

train Fit {
    model = Jepa;
    input tokens: Tensor<i32, [32]>;
    target y: Tensor<f32, [32, 8]>;
    loss = add(mse(model.forward(tokens), stop_gradient(model.target_embedding(y))), mul(sigreg(model.embed(tokens), slices = 4), 0.1));
    optimizer = sgd(lr = 0.1);
    steps = 5;
    seed = 13;
    device = cpu;
}
";

/// The `JEPA` data: `samples` token ids below 16 and `samples` rows of 8 target values.
fn jepa_data(seed: u64, samples: usize) -> BTreeMap<&'static str, Values> {
    let mut s = Stream(seed);
    let tokens = (0..samples).map(|_| s.below(16)).collect();
    let y = (0..8 * samples).map(|_| s.f32()).collect();
    BTreeMap::from([("tokens", Values::I32(tokens)), ("y", Values::F32(y))])
}

/// The loss of `eval loss=<v>` or `eval loss=<v> accuracy=<a>`.
fn eval_loss(stdout: &str) -> f64 {
    eval_line(stdout)
        .strip_prefix("eval loss=")
        .and_then(|rest| rest.split(' ').next())
        .expect("`eval loss=<mean>`")
        .parse()
        .expect("a loss")
}

/// dense-named-jepa-cpu: the JEPA-shaped program trains 5 steps on the CPU from a dense
/// directory of `tokens.i32` and `y.f32`; every recorded loss and the `--eval` loss are finite,
/// and two runs from one seed record bit-identical losses and print the same lines.
#[test]
fn dense_named_jepa_sigreg_stop_gradient_cpu_trains_reproducibly() {
    let (root, file) = case("jepa_cpu", JEPA);
    let data = root.join("data");
    write_dir(&data, &jepa_data(31, 64));
    let eval = root.join("eval");
    write_dir(&eval, &jepa_data(32, 32));
    let run = |name: &str| {
        let record = root.join(format!("{name}.jsonl"));
        let stdout = ok(&train(
            &file,
            "cpu",
            &data,
            &["--record", path(&record), "--eval", path(&eval)],
        ));
        (stdout, recorded_losses(&record))
    };
    let (first, first_losses) = run("first");
    let (second, second_losses) = run("second");
    assert_eq!(
        first_losses.keys().copied().collect::<Vec<_>>(),
        (0..=5).collect::<Vec<u64>>(),
        "{first}"
    );
    assert!(
        first_losses.values().all(|l| l.is_finite()),
        "{first_losses:?}"
    );
    let bits = |losses: &BTreeMap<u64, f64>| -> Vec<u64> {
        losses.values().map(|l| l.to_bits()).collect()
    };
    assert_eq!(bits(&first_losses), bits(&second_losses));
    assert!(eval_loss(&first).is_finite(), "{first}");
    assert_eq!(first, second);
}

/// The values of every `param @<path>: <type> = [v, ...]` line of `stdout`, by path.
fn final_params(stdout: &str) -> BTreeMap<String, Vec<f32>> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("param @"))
        .map(|rest| {
            let (name, rest) = rest.split_once(": ").expect("`@<path>: <type> = [..]`");
            let (_, list) = rest.split_once(" = [").expect("a value list");
            let list = list.strip_suffix(']').expect("a closed list");
            let values = list
                .split(", ")
                .map(|v| v.parse::<f32>().expect("an f32"))
                .collect();
            (name.to_owned(), values)
        })
        .collect()
}

/// The `JEPA` program with one more model function, `spread`, that calls `sigreg`: every
/// embedding row scaled by the `sigreg` statistic of the embedding, one row per token.
fn jepa_with_spread() -> String {
    JEPA.replace(
        "    fn forward(",
        "    fn spread(tokens: Tensor<i32, [32]>) -> Tensor<f32, [32, 8]> {\n        \
         return mul(embedding(table, tokens), sigreg(embedding(table, tokens), slices = 4));\n    \
         }\n\n    fn forward(",
    )
}

/// The little-endian `f32` values of `file`.
fn read_f32(file: &Path) -> Vec<f32> {
    let bytes = std::fs::read(file).expect("the output file");
    let (words, rest) = bytes.as_chunks::<4>();
    assert!(
        rest.is_empty(),
        "{} is not whole f32 values",
        file.display()
    );
    words.iter().map(|&w| f32::from_le_bytes(w)).collect()
}

/// `modelspec infer <file> --backend <backend> --checkpoint <ckpt> --data <data> --function
/// <function> --out <out>`.
fn infer(
    file: &Path,
    backend: &str,
    ckpt: &Path,
    data: &Path,
    function: &str,
    out: &Path,
) -> Output {
    modelspec(&[
        "infer",
        path(file),
        "--backend",
        backend,
        "--checkpoint",
        path(ckpt),
        "--data",
        path(data),
        "--function",
        function,
        "--out",
        path(out),
    ])
}

/// A JEPA checkpoint: the case `name` of `source` trained 3 steps on the CPU from the dense
/// directory of `jepa_data(31, 64)` (`tokens.i32`, `y.f32`), with `--checkpoint` after the last
/// step. Returns the case root, the model file, the data directory, the checkpoint and the
/// stdout of the run.
fn jepa_checkpoint(name: &str, source: &str) -> (PathBuf, PathBuf, PathBuf, PathBuf, String) {
    let (root, file) = case(name, source);
    let data = root.join("data");
    write_dir(&data, &jepa_data(31, 64));
    let ckpt = root.join("run.msckpt");
    let stdout = ok(&train(
        &file,
        "cpu",
        &data,
        &["--steps", "3", "--checkpoint", path(&ckpt)],
    ));
    assert!(ckpt.is_file(), "{stdout}");
    (root, file, data, ckpt, stdout)
}

/// infer-function-jepa (CPU part; the device part is `device::` of the same name): a non-loss
/// function of a JEPA checkpoint trained 3 steps from a dense directory with `i32` tokens runs
/// over that directory on the CPU (story:decider-infer-on-data, story:jepa-sigreg-cpu,
/// story:decider-dense-named-data). `embed` writes, for every one of the 64 tokens, the row of
/// the checkpoint's `table` it names, bit for bit; `forward` writes 64 finite rows of 8.
#[test]
fn infer_function_on_a_jepa_checkpoint_matches_on_cpu_and_vulkan() {
    let (root, file, data, ckpt, trained) = jepa_checkpoint("infer_jepa_cpu", JEPA);
    let table = final_params(&trained)
        .remove("Jepa.table")
        .unwrap_or_else(|| panic!("no `param @Jepa.table` line: {trained}"));
    assert_eq!(table.len(), 16 * 8);
    let out = root.join("embed.f32");
    let stdout = ok(&infer(&file, "cpu", &ckpt, &data, "embed", &out));
    assert_eq!(
        stdout,
        format!("function embed blocks 2 rows 64 out {}\n", out.display())
    );
    let embedded = read_f32(&out);
    assert_eq!(embedded.len(), 64 * 8);
    let tokens = jepa_data(31, 64);
    for (row, &token) in i32s(&tokens, "tokens").iter().enumerate() {
        let token = usize::try_from(token).expect("a token id");
        let expected: Vec<u32> = table[8 * token..8 * token + 8]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let got: Vec<u32> = embedded[8 * row..8 * row + 8]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got, expected, "row {row}, token {token}");
    }
    let out = root.join("forward.f32");
    ok(&infer(&file, "cpu", &ckpt, &data, "forward", &out));
    let predicted = read_f32(&out);
    assert_eq!(predicted.len(), 64 * 8);
    assert!(predicted.iter().all(|v| v.is_finite()), "{predicted:?}");
}

/// infer-function-sigreg (CPU part; the device part is `device::` of the same name): a model
/// function that calls `sigreg` runs over the dense directory on the CPU with the projection of
/// the checkpoint's step, as `infer` without `--data` reads it, and writes 64 finite rows of 8.
#[test]
fn infer_function_calling_sigreg_matches_on_cpu_and_vulkan() {
    let (root, file, data, ckpt, _) = jepa_checkpoint("infer_sigreg_cpu", &jepa_with_spread());
    let out = root.join("spread.f32");
    let stdout = ok(&infer(&file, "cpu", &ckpt, &data, "spread", &out));
    assert_eq!(
        stdout,
        format!("function spread blocks 2 rows 64 out {}\n", out.display())
    );
    let spread = read_f32(&out);
    assert_eq!(spread.len(), 64 * 8);
    assert!(spread.iter().all(|v| v.is_finite()), "{spread:?}");
}

/// infer-function-e-data-011: an `i32` argument value that is not below the rows of the
/// `embedding` table of the function that reads it is E-DATA-011, as for `train --data`, naming
/// the file, the element position and the bound, before anything is written.
#[test]
fn infer_function_refuses_an_i32_argument_outside_its_bound_with_e_data_011() {
    let (root, file, _, ckpt, _) = jepa_checkpoint("infer_out_of_range", JEPA);
    let mut tokens = jepa_data(31, 64);
    if let Some(Values::I32(ids)) = tokens.get_mut("tokens") {
        ids[37] = 16;
    }
    let bad = root.join("bad");
    write_dir(&bad, &tokens);
    let out = root.join("forward.f32");
    let output = infer(&file, "cpu", &ckpt, &bad, "forward", &out);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "stderr: {stderr}");
    assert!(stderr.starts_with("E-DATA-011: "), "stderr: {stderr}");
    assert!(stderr.contains("tokens.i32"), "stderr: {stderr}");
    assert!(stderr.contains("37"), "stderr: {stderr}");
    assert!(stderr.contains("0 <= value < 16"), "stderr: {stderr}");
    assert!(!out.exists(), "stderr: {stderr}");
    assert!(output.stdout.is_empty());
}

#[cfg(feature = "vulkan")]
#[allow(dead_code)]
#[path = "../../modelspec-runtime/tests/common/mod.rs"]
mod common;

#[cfg(feature = "vulkan")]
mod device {
    use super::common::{GPU_RUN_ATOL, GPU_RUN_RTOL};
    use super::*;

    fn bound(c: f64) -> f64 {
        f64::from(GPU_RUN_ATOL) + f64::from(GPU_RUN_RTOL) * c.abs()
    }

    /// Every recorded loss of `--backend vulkan` within the run tolerance of the CPU run's.
    fn agree(file: &Path, data: &Path, root: &Path, extra: &[&str]) -> (String, String) {
        let cpu_record = root.join("cpu.jsonl");
        let gpu_record = root.join("gpu.jsonl");
        let mut cpu_flags = vec!["--record", path(&cpu_record)];
        cpu_flags.extend_from_slice(extra);
        let mut gpu_flags = vec!["--record", path(&gpu_record)];
        gpu_flags.extend_from_slice(extra);
        let cpu = ok(&train(file, "cpu", data, &cpu_flags));
        let gpu = ok(&train(file, "vulkan", data, &gpu_flags));
        let (c, g) = (recorded_losses(&cpu_record), recorded_losses(&gpu_record));
        assert_eq!(c.keys().collect::<Vec<_>>(), g.keys().collect::<Vec<_>>());
        for (step, &cl) in &c {
            let gl = g[step];
            assert!(
                (gl - cl).abs() <= bound(cl),
                "step {step}: vulkan {gl}, cpu {cl}"
            );
        }
        (cpu, gpu)
    }

    /// dense-named-vulkan: the `NAMED` fixture on the device, every loss within the run
    /// tolerance of the CPU run's.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn dense_named_vulkan_agrees_with_the_cpu_run() {
        let (root, file) = case("named_vulkan", NAMED);
        let data = root.join("data");
        write_dir(&data, &named_data(11, NAMED_SAMPLES));
        agree(&file, &data, &root, &[]);
    }

    /// decider-small-vulkan: the decider program at test sizes on the device, every loss within
    /// the run tolerance of the CPU run's.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn decider_small_vulkan_agrees_with_the_cpu_run() {
        let root = fresh("decider_vulkan");
        let data = root.join("data");
        write_dir(&data, &decider_data(2));
        agree(Path::new(DECIDER), &data, &root, &[]);
    }

    /// dense-named-eval-vulkan: `--eval` on the device prints the eval line within the run
    /// tolerance of the CPU run's.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn dense_named_eval_vulkan_agrees_with_the_cpu_run() {
        let (root, file) = case("named_eval_vulkan", NAMED);
        let data = root.join("data");
        write_dir(&data, &named_data(11, NAMED_SAMPLES));
        let eval = root.join("eval");
        write_dir(&eval, &named_data(12, 8));
        let (cpu, gpu) = agree(&file, &data, &root, &["--eval", path(&eval)]);
        let loss = |out: &str| -> f64 {
            eval_line(out)
                .strip_prefix("eval loss=")
                .expect("`eval loss=<mean>`")
                .parse()
                .expect("only the loss")
        };
        let (c, g) = (loss(&cpu), loss(&gpu));
        assert!((g - c).abs() <= bound(c), "eval: vulkan {g}, cpu {c}");
    }

    /// `infer --data --function <function>` of the checkpoint `ckpt` of `file` on the CPU and on
    /// Vulkan: the same stdout line but for the file, the same row count of `row` values, and
    /// every value of the device's within the run tolerance of the CPU's.
    fn infer_agrees(
        file: &Path,
        root: &Path,
        data: &Path,
        ckpt: &Path,
        function: &str,
        row: usize,
    ) {
        let (cpu_out, gpu_out) = (
            root.join(format!("{function}-cpu.f32")),
            root.join(format!("{function}-vulkan.f32")),
        );
        let cpu = ok(&infer(file, "cpu", ckpt, data, function, &cpu_out));
        let gpu = ok(&infer(file, "vulkan", ckpt, data, function, &gpu_out));
        assert_eq!(
            cpu.replace(path(&cpu_out), "OUT"),
            gpu.replace(path(&gpu_out), "OUT")
        );
        let (c, g) = (read_f32(&cpu_out), read_f32(&gpu_out));
        assert_eq!(c.len(), 64 * row, "{cpu}");
        assert_eq!(g.len(), c.len(), "{gpu}");
        let outside: Vec<String> = c
            .iter()
            .zip(&g)
            .enumerate()
            .filter(|&(_, (&c, &g))| (f64::from(g) - f64::from(c)).abs() > bound(f64::from(c)))
            .map(|(i, (c, g))| format!("{function}[{i}]: vulkan {g}, cpu {c}"))
            .collect();
        assert!(outside.is_empty(), "{outside:#?}");
    }

    /// infer-function-jepa (device part): `embed` and `forward` of a JEPA checkpoint trained 3
    /// steps from a dense directory with `i32` tokens, on Vulkan, print the CPU run's line and
    /// write as many rows, every value within the run tolerance of the CPU's.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn infer_function_on_a_jepa_checkpoint_matches_on_cpu_and_vulkan() {
        let (root, file, data, ckpt, _) = jepa_checkpoint("infer_jepa_vulkan", JEPA);
        infer_agrees(&file, &root, &data, &ckpt, "embed", 8);
        infer_agrees(&file, &root, &data, &ckpt, "forward", 8);
    }

    /// infer-function-sigreg (device part): a function that calls `sigreg`, on Vulkan, reads
    /// the projection of the checkpoint's step as the CPU run does, so every value is within the
    /// run tolerance of the CPU's.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn infer_function_calling_sigreg_matches_on_cpu_and_vulkan() {
        let (root, file, data, ckpt, _) =
            jepa_checkpoint("infer_sigreg_vulkan", &jepa_with_spread());
        infer_agrees(&file, &root, &data, &ckpt, "spread", 8);
    }

    /// dense-named-jepa-vulkan: the JEPA-shaped program trains on the device from the dense
    /// directory of `tokens.i32` and `y.f32` with `--eval` (story:jepa-sigreg-vulkan,
    /// story:decider-dense-named-data): every recorded loss, every element of the three
    /// parameters after the run and the `--eval` loss, whose batches read the `sigreg`
    /// projection of the last step as the CPU evaluation does, are within the run tolerance of
    /// the CPU run's.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn dense_named_jepa_vulkan_agrees_with_the_cpu_run() {
        let (root, file) = case("jepa_vulkan", JEPA);
        let data = root.join("data");
        write_dir(&data, &jepa_data(31, 64));
        let eval = root.join("eval");
        write_dir(&eval, &jepa_data(32, 32));
        let (cpu, gpu) = agree(&file, &data, &root, &["--eval", path(&eval)]);
        assert_eq!(
            recorded_losses(&root.join("gpu.jsonl"))
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            (0..=5).collect::<Vec<u64>>(),
            "{gpu}"
        );
        let (c, g) = (eval_loss(&cpu), eval_loss(&gpu));
        assert!((g - c).abs() <= bound(c), "eval: vulkan {g}, cpu {c}");
        let (cpu_params, gpu_params) = (final_params(&cpu), final_params(&gpu));
        assert_eq!(cpu_params.len(), 3, "{cpu}");
        assert_eq!(
            cpu_params.keys().collect::<Vec<_>>(),
            gpu_params.keys().collect::<Vec<_>>(),
            "{gpu}"
        );
        let mut elements = 0usize;
        let mut outside = Vec::new();
        for (name, c) in &cpu_params {
            let g = &gpu_params[name];
            assert_eq!(g.len(), c.len(), "{name}");
            for (i, (&g, &c)) in g.iter().zip(c).enumerate() {
                elements += 1;
                if (f64::from(g) - f64::from(c)).abs() > bound(f64::from(c)) {
                    outside.push(format!("{name}[{i}]: vulkan {g}, cpu {c}"));
                }
            }
        }
        assert_eq!(elements, 16 * 8 + 8 * 8 + 8 * 8);
        assert!(outside.is_empty(), "{outside:#?}");
    }
}
