//! Adversary pass 1 on `modelspec train --data/--init/--eval` (story train-data-dense-tensors).
//!
//! Fixture: four samples of a 2-feature model, `B = 2`, `steps = 3`, as in `train_dense_data.rs`.

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

const X: [f32; 8] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0];
const Y: [f32; 8] = [1.0, 0.0, 1.0, 0.0, 0.0, 3.0, 1.0, 0.0];
const IDENTITY: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("adversary_dense_1")
            .join(name);
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        for dir in ["data", "eval", "init"] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        std::fs::write(root.join("model.mspec"), MODEL).unwrap();
        for dir in ["data", "eval"] {
            std::fs::write(root.join(dir).join("x.f32"), f32_bytes(&X)).unwrap();
            std::fs::write(root.join(dir).join("y.f32"), f32_bytes(&Y)).unwrap();
        }
        std::fs::write(root.join("init/w.f32"), f32_bytes(&IDENTITY)).unwrap();
        Fixture { root }
    }

    fn write(&self, rel: &str, bytes: &[u8]) {
        std::fs::write(self.root.join(rel), bytes).unwrap();
    }

    fn remove(&self, rel: &str) {
        std::fs::remove_file(self.root.join(rel)).unwrap();
    }

    /// `train` with each `(flag, subdirectory)` pair.
    fn train(&self, flags: &[(&str, &str)]) -> Output {
        let mut args = vec![
            "train".to_string(),
            self.root.join("model.mspec").display().to_string(),
            "--backend".into(),
            "cpu".into(),
        ];
        for (flag, rel) in flags {
            args.push(format!("--{flag}"));
            args.push(self.root.join(rel).display().to_string());
        }
        Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(&args)
            .output()
            .expect("modelspec runs")
    }
}

/// Exit 1, stderr opening with `code`, and nothing at all on stdout.
fn refused_silently(output: Output, code: &str) {
    let (stdout, stderr) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(
        output.status.code(),
        Some(1),
        "exit {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status.code()
    );
    assert!(stderr.starts_with(&format!("{code}: ")), "stderr: {stderr}");
    assert!(stdout.is_empty(), "stdout: {stdout}");
}

fn init_w(stdout: &str) -> String {
    stdout
        .lines()
        .find(|l| l.starts_with("init @Lin.w: "))
        .unwrap_or_else(|| panic!("no init line: {stdout}"))
        .to_string()
}

// --- Red candidates: `--init` that silently starts from the drawn initializer. ---

/// `--init` naming a directory that does not exist: every `p.f32` is NotFound, so every
/// parameter keeps its initializer and the run exits 0. E-DATA-007 covers "a file is missing";
/// a mistyped `--init` path then trains from random weights with no diagnostic.
#[test]
fn an_init_directory_that_does_not_exist_is_refused() {
    let f = Fixture::new("init_dir_missing");
    let output = f.train(&[("data", "data"), ("init", "no-such-dir")]);
    refused_silently(output, "E-DATA-007");
}

/// `--init` whose only file names the full canonical path (`Lin.w.f32`) instead of the stem
/// (`w.f32`) the contract asks for: no parameter is replaced, nothing says so, exit 0.
#[test]
fn an_init_directory_whose_files_name_no_parameter_is_refused() {
    let f = Fixture::new("init_no_match");
    f.remove("init/w.f32");
    f.write("init/Lin.w.f32", &f32_bytes(&IDENTITY));
    let output = f.train(&[("data", "data"), ("init", "init")]);
    refused_silently(output, "E-DATA-007");
}

// --- Pins (expected green): boundaries of the binary layout and the order. ---

#[test]
fn one_trailing_byte_on_a_data_file_is_e_data_007_with_empty_stdout() {
    let f = Fixture::new("trailing_byte");
    let mut bytes = f32_bytes(&Y);
    bytes.push(0);
    f.write("data/y.f32", &bytes);
    refused_silently(f.train(&[("data", "data")]), "E-DATA-007");
}

#[test]
fn a_zero_length_input_file_is_e_data_007_with_empty_stdout() {
    let f = Fixture::new("zero_input");
    f.write("data/x.f32", &[]);
    refused_silently(f.train(&[("data", "data")]), "E-DATA-007");
}

#[test]
fn an_empty_order_file_is_e_data_008_with_empty_stdout() {
    let f = Fixture::new("empty_order");
    f.write("data/order.u32", &[]);
    refused_silently(f.train(&[("data", "data")]), "E-DATA-008");
}

/// One index short of `steps · B = 6`.
#[test]
fn an_order_of_steps_times_b_minus_one_is_e_data_008_with_empty_stdout() {
    let f = Fixture::new("order_short_by_one");
    f.write("data/order.u32", &u32_bytes(&[0, 1, 2, 3, 0]));
    refused_silently(f.train(&[("data", "data")]), "E-DATA-008");
}

/// An index equal to N at a position past `steps · B`, which no step reads: still refused.
#[test]
fn an_order_index_equal_to_n_past_the_run_is_e_data_008() {
    let f = Fixture::new("order_n_late");
    f.write("data/order.u32", &u32_bytes(&[0, 1, 2, 3, 0, 1, 2, 4]));
    refused_silently(f.train(&[("data", "data")]), "E-DATA-008");
}

#[test]
fn an_order_of_index_n_minus_one_everywhere_is_accepted() {
    let f = Fixture::new("order_last");
    f.write("data/order.u32", &u32_bytes(&[3; 9]));
    let output = f.train(&[("data", "data"), ("init", "init")]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    // sample 3: x = [2, 0], y = [1, 0], identity prediction [2, 0]: (1 + 0) per row, mean 0.5.
    assert!(
        text(&output.stdout).contains("\nstep 0 loss 0.5\n"),
        "{}",
        text(&output.stdout)
    );
}

#[test]
fn an_init_file_one_value_too_long_is_e_data_007_with_empty_stdout() {
    let f = Fixture::new("init_long");
    f.write("init/w.f32", &f32_bytes(&[1.0, 0.0, 0.0, 1.0, 0.0]));
    refused_silently(f.train(&[("data", "data"), ("init", "init")]), "E-DATA-007");
}

#[test]
fn a_stray_file_beside_the_init_files_is_ignored() {
    let f = Fixture::new("init_stray");
    f.write("init/notes.txt", b"hello");
    f.write("init/order.u32", &u32_bytes(&[9, 9]));
    let output = f.train(&[("data", "data"), ("init", "init")]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert_eq!(
        init_w(&text(&output.stdout)),
        "init @Lin.w: Tensor<f32, [2, 2]> = [1, 0, 0, 1]"
    );
}

#[test]
fn the_eval_directory_order_file_is_not_read() {
    let f = Fixture::new("eval_order");
    f.write("eval/order.u32", &[1, 2, 3]);
    let output = f.train(&[("data", "data"), ("eval", "eval")]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
}

/// Eval with only `--eval`: synthetic training, then the eval line; and eval data of different
/// sample counts is E-DATA-007 before any step.
#[test]
fn eval_without_data_runs_and_mismatched_eval_counts_are_e_data_007() {
    let f = Fixture::new("eval_only");
    let output = f.train(&[("eval", "eval")]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(
        text(&output.stdout)
            .lines()
            .last()
            .unwrap()
            .starts_with("eval loss="),
        "{}",
        text(&output.stdout)
    );
    f.write("eval/y.f32", &f32_bytes(&Y[..4]));
    refused_silently(f.train(&[("eval", "eval")]), "E-DATA-007");
}

/// All-equal prediction and target rows: both first maxima are index 0, so every sample counts
/// as correct, and the denominator is N = 4, not the batch count.
#[test]
fn eval_ties_take_the_first_maximum_and_divide_by_the_sample_count() {
    let f = Fixture::new("eval_ties");
    f.write("init/w.f32", &f32_bytes(&IDENTITY));
    let fitted = [1.0, 0.0, 0.0, 1.0];
    f.write("data/x.f32", &f32_bytes(&fitted));
    f.write("data/y.f32", &f32_bytes(&fitted));
    let tied = [1.0f32, 1.0, 0.0, 0.0, 2.0, 2.0, 5.0, 5.0];
    f.write("eval/x.f32", &f32_bytes(&tied));
    f.write(
        "eval/y.f32",
        &f32_bytes(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
    );
    let output = f.train(&[("data", "data"), ("init", "init"), ("eval", "eval")]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let stdout = text(&output.stdout);
    let last = stdout.lines().last().unwrap();
    // rows 0,1,2 have first max 0 on both sides; row 3 is [5,5] vs [0,1]: 0 != 1.
    assert!(last.ends_with(" accuracy=0.75"), "{stdout}");
}
