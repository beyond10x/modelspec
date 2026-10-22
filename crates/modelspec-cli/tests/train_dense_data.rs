//! `modelspec train --backend cpu --data <dir> --init <dir> --eval <dir>`: dense `f32` tensors
//! from files (DESIGN.md § 8, § 8.1; `spec/domains/data.yaml`, `DenseDataDirectory`).
//!
//! The fixture is four samples of a 2-feature, 2-class linear model, `B = 2`, written by the test
//! under its temp dir:
//!
//! | sample | x | y |
//! |---|---|---|
//! | 0 | [1, 0] | [1, 0] |
//! | 1 | [0, 1] | [1, 0] |
//! | 2 | [1, 1] | [0, 3] |
//! | 3 | [2, 0] | [1, 0] |
//!
//! With `w` the identity (`--init`), the prediction of a sample is its `x`. Step 0 without
//! `order.u32` feeds samples 0 and 1: squared errors 0 and 1 + 1, mean over 4 elements 0.5. With
//! `order.u32 = [2, 3, …]` it feeds samples 2 and 3: squared errors 1 + 4 and 1, mean 1.5.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const MODEL: &str = "\
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

/// The model with a second parameter `v`, so `--init` can name one and leave the other.
const TWO_PARAMS: &str = "\
model Two {
    param w: Tensor<f32, [2, 2]> = xavier_uniform();
    param v: Tensor<f32, [2, 2]> = xavier_uniform();

    fn forward(x: Tensor<f32, [2, 2]>)
        -> Tensor<f32, [2, 2]> {
        return matmul(matmul(x, w), v);
    }
}

train Fit {
    model = Two;
    input x: Tensor<f32, [2, 2]>;
    target y: Tensor<f32, [2, 2]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 3;
    seed = 42;
    device = cpu;
}
";

const X: [f32; 8] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0];
const Y: [f32; 8] = [1.0, 0.0, 1.0, 0.0, 0.0, 3.0, 1.0, 0.0];
const IDENTITY: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const ORDER: [u32; 6] = [2, 3, 0, 1, 3, 2];
const B: usize = 2;
const STEPS: usize = 3;
const LR: f32 = 0.1;

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

/// A fresh, empty directory `name` under the test temp dir.
fn fresh(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("train_dense_data")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old fixture is removed");
    }
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    dir
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A fixture directory `name` holding `model.mspec`, `data/` with `x.f32` and `y.f32` (and
/// `order.u32` when `order` is given), `init/` and `eval/` with the same four samples.
struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new(name: &str, source: &str, order: Option<&[u32]>) -> Self {
        let root = fresh(name);
        std::fs::write(root.join("model.mspec"), source).unwrap();
        for dir in ["data", "eval", "init"] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        for dir in ["data", "eval"] {
            std::fs::write(root.join(dir).join("x.f32"), f32_bytes(&X)).unwrap();
            std::fs::write(root.join(dir).join("y.f32"), f32_bytes(&Y)).unwrap();
        }
        if let Some(order) = order {
            std::fs::write(root.join("data/order.u32"), u32_bytes(order)).unwrap();
        }
        std::fs::write(root.join("init/w.f32"), f32_bytes(&IDENTITY)).unwrap();
        Fixture { root }
    }

    fn path(&self, rel: &str) -> String {
        self.root.join(rel).display().to_string()
    }

    fn write(&self, rel: &str, bytes: &[u8]) {
        std::fs::write(self.root.join(rel), bytes).unwrap();
    }

    fn remove(&self, rel: &str) {
        std::fs::remove_file(self.root.join(rel)).unwrap();
    }

    /// `modelspec train <root>/model.mspec --backend cpu` plus `flags`, each `--flag` taking the
    /// fixture subdirectory of its own name.
    fn train(&self, flags: &[&str]) -> Output {
        let mut args = vec![
            "train".to_string(),
            self.path("model.mspec"),
            "--backend".into(),
            "cpu".into(),
        ];
        for flag in flags {
            args.push(format!("--{flag}"));
            args.push(self.path(flag));
        }
        Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(&args)
            .output()
            .expect("modelspec runs")
    }
}

/// The stdout of a run that exits 0 with nothing on stderr.
fn ok(output: Output) -> String {
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "stderr: {}", text(&output.stderr));
    text(&output.stdout)
}

/// The stderr of a run refused with `code`: exit 1 and no `step` line.
fn refused(output: Output, code: &str) -> String {
    let stderr = text(&output.stderr);
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(1), "stderr: {stderr}");
    assert!(
        stderr.starts_with(&format!("{code}: ")),
        "expected {code}, stderr: {stderr}"
    );
    assert!(
        !stdout.lines().any(|line| line.starts_with("step ")),
        "a step ran: {stdout}"
    );
    stderr
}

/// The `(step, loss)` of every `step <k> loss <v>` line.
fn losses(stdout: &str) -> Vec<(usize, f32)> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (step.parse().unwrap(), loss.parse().unwrap())
        })
        .collect()
}

/// The values of the line `<prefix> @<path>: <type> = [...]`.
fn values(stdout: &str, prefix: &str, path: &str) -> Vec<f32> {
    let head = format!("{prefix} @{path}: ");
    let line = stdout
        .lines()
        .find(|line| line.starts_with(&head))
        .unwrap_or_else(|| panic!("no `{head}` line in {stdout}"));
    let list = line.rsplit_once(" = [").expect("a value list").1;
    list.trim_end_matches(']')
        .split(", ")
        .map(|v| v.parse().unwrap())
        .collect()
}

/// `(loss, accuracy)` of the one `eval loss=<l> accuracy=<a>` line.
fn eval_line(stdout: &str) -> (f64, f64) {
    let evals: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with("eval "))
        .collect();
    assert_eq!(evals.len(), 1, "{stdout}");
    let rest = evals[0].strip_prefix("eval loss=").expect("`eval loss=`");
    let (loss, accuracy) = rest.split_once(" accuracy=").expect("` accuracy=`");
    (loss.parse().unwrap(), accuracy.parse().unwrap())
}

/// `[B, 2] @ [2, 2]` in `f32`, row-major.
fn matmul(x: &[f32], w: &[f32]) -> Vec<f32> {
    x.chunks(2)
        .flat_map(|row| (0..2).map(move |j| row[0] * w[j] + row[1] * w[2 + j]))
        .collect()
}

/// The rows of `samples` gathered from `data`, 2 values per sample.
fn gather(data: &[f32], samples: &[usize]) -> Vec<f32> {
    samples
        .iter()
        .flat_map(|&s| data[2 * s..2 * s + 2].iter().copied())
        .collect()
}

/// Mean squared error over all elements.
fn mse(p: &[f32], y: &[f32]) -> f32 {
    p.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / p.len() as f32
}

/// The samples of step `k`: `order[k·B + i]` (positions wrapping) or `(k·B + i) mod 4`.
fn batch(order: Option<&[u32]>, k: usize) -> Vec<usize> {
    (0..B)
        .map(|i| match order {
            Some(order) => order[(k * B + i) % order.len()] as usize,
            None => (k * B + i) % 4,
        })
        .collect()
}

/// The losses after 0..=STEPS SGD updates of `mse(x @ w, y)` from `w`, computed here by hand.
fn reference(mut w: Vec<f32>, order: Option<&[u32]>) -> (Vec<f32>, Vec<f32>) {
    let mut out = Vec::new();
    for k in 0..=STEPS {
        let samples = batch(order, k);
        let (x, y) = (gather(&X, &samples), gather(&Y, &samples));
        let p = matmul(&x, &w);
        out.push(mse(&p, &y));
        if k == STEPS {
            break;
        }
        let scale = 2.0 / p.len() as f32;
        let mut grad = [0.0f32; 4];
        for (r, row) in x.chunks(2).enumerate() {
            for i in 0..2 {
                for j in 0..2 {
                    grad[2 * i + j] += row[i] * (p[2 * r + j] - y[2 * r + j]) * scale;
                }
            }
        }
        for (wi, g) in w.iter_mut().zip(grad) {
            *wi -= LR * g;
        }
    }
    (out, w)
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-5 * (1.0 + b.abs())
}

#[test]
fn step_0_feeds_samples_0_and_1_without_an_order_file() {
    let f = Fixture::new("no_order", MODEL, None);
    let stdout = ok(f.train(&["data", "init"]));
    assert_eq!(values(&stdout, "init", "Lin.w"), IDENTITY);
    let got = losses(&stdout);
    assert_eq!(got[0], (0, 0.5), "{stdout}");
    let (expected, w) = reference(IDENTITY.to_vec(), None);
    assert_eq!(got.last().unwrap().0, STEPS);
    assert!(
        close(got.last().unwrap().1, expected[STEPS]),
        "{got:?} {expected:?}"
    );
    for (g, e) in values(&stdout, "param", "Lin.w").iter().zip(&w) {
        assert!(close(*g, *e), "{g} {e}");
    }
}

#[test]
fn step_k_feeds_the_order_file_indices_k_b_to_k_plus_1_b() {
    let f = Fixture::new("order", MODEL, Some(&ORDER));
    let stdout = ok(f.train(&["data", "init"]));
    let got = losses(&stdout);
    assert_eq!(got[0], (0, 1.5), "{stdout}");
    let (expected, w) = reference(IDENTITY.to_vec(), Some(&ORDER));
    assert!(
        close(got.last().unwrap().1, expected[STEPS]),
        "{got:?} {expected:?}"
    );
    for (g, e) in values(&stdout, "param", "Lin.w").iter().zip(&w) {
        assert!(close(*g, *e), "{g} {e}");
    }
    let (no_order, _) = reference(IDENTITY.to_vec(), None);
    assert_ne!(
        expected[STEPS], no_order[STEPS],
        "the fixture tells the orders apart"
    );
}

#[test]
fn without_init_the_step_0_loss_is_the_fixture_loss_under_the_drawn_parameters() {
    let f = Fixture::new("drawn", MODEL, None);
    let stdout = ok(f.train(&["data"]));
    let w = values(&stdout, "init", "Lin.w");
    assert_ne!(w, IDENTITY);
    let (expected, _) = reference(w, None);
    let got = losses(&stdout);
    assert!(close(got[0].1, expected[0]), "{got:?} {expected:?}");
    assert!(
        close(got.last().unwrap().1, expected[STEPS]),
        "{got:?} {expected:?}"
    );
}

#[test]
fn init_replaces_exactly_the_params_with_a_file() {
    let f = Fixture::new("two_params", TWO_PARAMS, None);
    let drawn = ok(f.train(&["data"]));
    let from_file = ok(f.train(&["data", "init"]));
    assert_ne!(values(&drawn, "init", "Two.w"), IDENTITY);
    assert_eq!(values(&from_file, "init", "Two.w"), IDENTITY);
    assert_eq!(
        values(&from_file, "init", "Two.v"),
        values(&drawn, "init", "Two.v")
    );
}

#[test]
fn eval_prints_the_mean_loss_and_argmax_accuracy_over_the_eval_samples() {
    let f = Fixture::new("eval", MODEL, None);
    let stdout = ok(f.train(&["data", "init", "eval"]));
    let w = values(&stdout, "param", "Lin.w");
    let mut loss = 0.0f64;
    let mut correct = 0u32;
    for first in (0..4).step_by(B) {
        let samples: Vec<usize> = (first..first + B).collect();
        let (x, y) = (gather(&X, &samples), gather(&Y, &samples));
        let p = matmul(&x, &w);
        loss += f64::from(mse(&p, &y));
        for (pr, yr) in p.chunks(2).zip(y.chunks(2)) {
            let argmax = |r: &[f32]| usize::from(r[1] > r[0]);
            correct += u32::from(argmax(pr) == argmax(yr));
        }
    }
    let (got_loss, got_accuracy) = eval_line(&stdout);
    assert!(
        (got_loss - loss / 2.0).abs() <= 1e-6 * (1.0 + loss),
        "{got_loss} {}",
        loss / 2.0
    );
    assert_eq!(got_accuracy, f64::from(correct) / 4.0);
    assert!(stdout.ends_with('\n'));
    assert!(
        stdout.lines().last().unwrap().starts_with("eval "),
        "{stdout}"
    );
}

/// Three classes, so that the first maximum is told apart from the first minimum and from the
/// last maximum. The training samples are fitted exactly by the identity, so their gradient is 0
/// and `w` stays the identity: the eval predictions are the eval inputs.
const THREE: &str = "\
model Lin3 {
    param w: Tensor<f32, [3, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [2, 3]>)
        -> Tensor<f32, [2, 3]> {
        return matmul(x, w);
    }
}

train Fit {
    model = Lin3;
    input x: Tensor<f32, [2, 3]>;
    target y: Tensor<f32, [2, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 42;
    device = cpu;
}
";

#[test]
fn eval_accuracy_counts_the_first_maximum_of_each_row() {
    let f = Fixture::new("eval_three", THREE, None);
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    f.write("init/w.f32", &f32_bytes(&identity));
    let fitted = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    f.write("data/x.f32", &f32_bytes(&fitted));
    f.write("data/y.f32", &f32_bytes(&fitted));
    let x = [3.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0];
    let y = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    f.write("eval/x.f32", &f32_bytes(&x));
    f.write("eval/y.f32", &f32_bytes(&y));
    let stdout = ok(f.train(&["data", "init", "eval"]));
    assert_eq!(values(&stdout, "param", "Lin3.w"), identity, "{stdout}");
    let (loss, accuracy) = eval_line(&stdout);
    assert_eq!(accuracy, 0.75, "{stdout}");
    let expected = (10.0 / 6.0 + 25.0 / 6.0) / 2.0;
    assert!((loss - expected).abs() <= 1e-6, "{loss} {expected}");
}

#[test]
fn without_eval_no_eval_line_is_printed() {
    let f = Fixture::new("no_eval", MODEL, None);
    let stdout = ok(f.train(&["data"]));
    assert!(!stdout.contains("eval"), "{stdout}");
}

#[test]
fn a_missing_data_file_is_e_data_007() {
    let f = Fixture::new("missing", MODEL, None);
    f.remove("data/y.f32");
    let stderr = refused(f.train(&["data"]), "E-DATA-007");
    assert!(stderr.contains("y.f32"), "{stderr}");
}

#[test]
fn a_data_file_of_part_of_a_sample_is_e_data_007() {
    let f = Fixture::new("partial", MODEL, None);
    f.write("data/x.f32", &f32_bytes(&X[..7]));
    refused(f.train(&["data"]), "E-DATA-007");
    f.write("data/x.f32", &f32_bytes(&X)[..30]);
    refused(f.train(&["data"]), "E-DATA-007");
}

#[test]
fn data_files_of_different_sample_counts_are_e_data_007() {
    let f = Fixture::new("counts", MODEL, None);
    f.write("data/y.f32", &f32_bytes(&Y[..6]));
    refused(f.train(&["data"]), "E-DATA-007");
}

#[test]
fn an_empty_data_file_is_e_data_007() {
    let f = Fixture::new("empty", MODEL, None);
    f.write("data/x.f32", &[]);
    f.write("data/y.f32", &[]);
    refused(f.train(&["data"]), "E-DATA-007");
}

#[test]
fn an_init_file_not_of_the_param_size_is_e_data_007() {
    let f = Fixture::new("init_size", MODEL, None);
    f.write("init/w.f32", &f32_bytes(&IDENTITY[..3]));
    let stderr = refused(f.train(&["data", "init"]), "E-DATA-007");
    assert!(stderr.contains("w.f32"), "{stderr}");
}

/// A missing `--init` directory is reported as unreadable, not as a directory without
/// parameter files; one with only stray files lists the file names it expected.
#[test]
fn an_init_directory_that_is_missing_or_names_no_parameter_says_which() {
    let f = Fixture::new("init_dir", TWO_PARAMS, None);
    std::fs::remove_dir_all(f.root.join("init")).unwrap();
    let stderr = refused(f.train(&["data", "init"]), "E-DATA-007");
    assert!(
        stderr.starts_with("E-DATA-007: cannot read the init directory"),
        "{stderr}"
    );
    std::fs::create_dir_all(f.root.join("init")).unwrap();
    f.write("init/Two.w.f32", &f32_bytes(&IDENTITY));
    let stderr = refused(f.train(&["data", "init"]), "E-DATA-007");
    assert!(stderr.contains("no file of any parameter"), "{stderr}");
    assert!(
        stderr.contains("w.f32, v.f32") || stderr.contains("v.f32, w.f32"),
        "{stderr}"
    );
}

#[test]
fn a_missing_eval_file_is_e_data_007() {
    let f = Fixture::new("eval_missing", MODEL, None);
    f.remove("eval/x.f32");
    refused(f.train(&["data", "eval"]), "E-DATA-007");
}

#[test]
fn an_order_file_shorter_than_steps_times_b_is_e_data_008() {
    let f = Fixture::new("short_order", MODEL, Some(&ORDER[..5]));
    refused(f.train(&["data"]), "E-DATA-008");
}

#[test]
fn an_order_index_past_the_sample_count_is_e_data_008() {
    let f = Fixture::new("far_order", MODEL, Some(&[2, 3, 0, 1, 4, 2]));
    refused(f.train(&["data"]), "E-DATA-008");
}

#[test]
fn an_order_file_of_part_of_an_index_is_e_data_007() {
    let f = Fixture::new("partial_order", MODEL, None);
    f.write("data/order.u32", &u32_bytes(&ORDER)[..23]);
    refused(f.train(&["data"]), "E-DATA-007");
}

#[test]
fn eval_samples_not_a_multiple_of_b_are_e_data_009() {
    let f = Fixture::new("eval_odd", MODEL, None);
    f.write("eval/x.f32", &f32_bytes(&X[..6]));
    f.write("eval/y.f32", &f32_bytes(&Y[..6]));
    refused(f.train(&["data", "eval"]), "E-DATA-009");
}

#[test]
fn the_refusals_come_before_the_first_step_even_with_init_and_eval_last() {
    let f = Fixture::new("late", MODEL, None);
    f.write("eval/x.f32", &f32_bytes(&X[..6]));
    f.write("eval/y.f32", &f32_bytes(&Y[..6]));
    let output = f.train(&["data", "init", "eval"]);
    assert!(output.stdout.is_empty(), "{}", text(&output.stdout));
    refused(output, "E-DATA-009");
}

/// A non-finite value in each kind of file the run reads: refused with E-DATA-007 before step 0,
/// naming the file.
#[test]
fn a_non_finite_value_in_a_data_eval_or_init_file_is_e_data_007() {
    let cases: [(&str, &str, f32); 4] = [
        ("nan_input", "data/x.f32", f32::NAN),
        ("inf_target", "data/y.f32", f32::INFINITY),
        ("neg_inf_eval", "eval/x.f32", f32::NEG_INFINITY),
        ("nan_eval_target", "eval/y.f32", f32::NAN),
    ];
    for (name, file, bad) in cases {
        let f = Fixture::new(name, MODEL, None);
        let source = if file.ends_with("x.f32") { X } else { Y };
        let mut values = source;
        values[5] = bad;
        f.write(file, &f32_bytes(&values));
        let output = f.train(&["data", "init", "eval"]);
        assert!(output.stdout.is_empty(), "{name}: {}", text(&output.stdout));
        let stderr = refused(output, "E-DATA-007");
        assert!(stderr.contains(file), "{name}: {stderr}");
    }
    let f = Fixture::new("nan_init", MODEL, None);
    f.write("init/w.f32", &f32_bytes(&[1.0, f32::NAN, 0.0, 1.0]));
    let output = f.train(&["data", "init", "eval"]);
    assert!(
        output.stdout.is_empty(),
        "nan_init: {}",
        text(&output.stdout)
    );
    let stderr = refused(output, "E-DATA-007");
    assert!(stderr.contains("init/w.f32"), "nan_init: {stderr}");
}

/// Two layers of three classes. With `w = 2·I` and `v = I`, an eval input of `f32::MAX` in the
/// first column overflows `x · w` to `+Inf`, and `Inf · 0` in `· v` makes the rest of that
/// prediction row NaN, from finite files. The training samples are fitted exactly (`y = 2x`), so
/// the parameters do not move.
const DEEP_THREE: &str = "\
model Deep3 {
    param w: Tensor<f32, [3, 3]> = xavier_uniform();
    param v: Tensor<f32, [3, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [2, 3]>)
        -> Tensor<f32, [2, 3]> {
        return matmul(matmul(x, w), v);
    }
}

train Fit {
    model = Deep3;
    input x: Tensor<f32, [2, 3]>;
    target y: Tensor<f32, [2, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 42;
    device = cpu;
}
";

/// The NaN row `[Inf, NaN, NaN]` against target class 0 is incorrect, though a strict `>` scan
/// would keep index 0; the finite row `[2, 0, 0]` is correct. Accuracy 1/2.
#[test]
fn an_eval_row_whose_prediction_has_a_nan_counts_as_incorrect() {
    let f = Fixture::new("eval_nan_row", DEEP_THREE, None);
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let double = identity.map(|v| 2.0 * v);
    f.write("init/w.f32", &f32_bytes(&double));
    f.write("init/v.f32", &f32_bytes(&identity));
    f.write("data/x.f32", &f32_bytes(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
    f.write("data/y.f32", &f32_bytes(&[2.0, 0.0, 0.0, 0.0, 2.0, 0.0]));
    f.write(
        "eval/x.f32",
        &f32_bytes(&[f32::MAX, 0.0, 0.0, 1.0, 0.0, 0.0]),
    );
    f.write("eval/y.f32", &f32_bytes(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
    let stdout = ok(f.train(&["data", "init", "eval"]));
    assert_eq!(values(&stdout, "param", "Deep3.w"), double, "{stdout}");
    let (_, accuracy) = eval_line(&stdout);
    assert_eq!(accuracy, 0.5, "{stdout}");
}
