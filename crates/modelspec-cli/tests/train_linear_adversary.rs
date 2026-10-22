//! Adversary cases for `modelspec train --backend cpu` and `modelspec ir --stage training`
//! (story m0-cpu-sgd-training, pass 2, phase B).
//!
//! The oracles here are independent of the runner: an `f64` re-implementation of SGD on
//! `mse(matmul(x, w), y)`, central finite differences of the forward loss, and the CLI contract
//! the module docs of `src/train.rs` state. Sources are inline.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use modelspec_core::compile::{compile_hir, compile_tensor, compile_training};
use modelspec_core::ir::tensor::{ParamId, TensorType};
use modelspec_runtime::cpu::{HostTensor, evaluate, evaluate_all};
use modelspec_runtime::train::data::{DataError, synthetic};
use modelspec_runtime::train::init::{Stream, initialize, standard_normal};
use modelspec_runtime::train::runner::{self, RunError};

const EXAMPLE: &str = "examples/linear_regression.mspec";
const LINEAR_REGRESSION: &str = include_str!("../../../examples/linear_regression.mspec");

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
    String::from_utf8_lossy(bytes).into_owned()
}

fn source_file(name: &str, source: &str) -> String {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("train_linear_adversary");
    std::fs::create_dir_all(&dir).expect("scratch directory");
    let path = dir.join(name);
    std::fs::write(&path, source).expect("source written");
    path.display().to_string()
}

fn train_ok(path: &str) -> String {
    let output = modelspec(&["train", path, "--backend", "cpu"]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "stderr: {}", text(&output.stderr));
    text(&output.stdout)
}

fn losses(stdout: &str) -> Vec<(u64, f32)> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (step.parse().unwrap(), loss.parse().unwrap())
        })
        .collect()
}

fn values(stdout: &str, prefix: &str, path: &str) -> Vec<f32> {
    let head = format!("{prefix} @{path}: ");
    let line = stdout
        .lines()
        .find(|line| line.starts_with(&head))
        .unwrap_or_else(|| panic!("no `{head}` line in:\n{stdout}"));
    let (_, rest) = line.split_once(" = [").expect("a value list");
    rest.strip_suffix(']')
        .expect("a closing bracket")
        .split(", ")
        .map(|v| v.parse().expect("a value"))
        .collect()
}

/// `x`, an `[8, 4]` standard normal draw on the data stream, and `y = x @ teacher` in `f64`.
fn linear_data(source: &str, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let x = standard_normal(seed, Stream::Data, "x", &TensorType::f32(&[8, 4])).unwrap();
    let lowered = compile_tensor("fit.mspec", source).expect("the source lowers");
    let teacher = initialize(&lowered.program, seed, Stream::Teacher).unwrap();
    let t: Vec<f64> = teacher[&ParamId(0)]
        .data()
        .iter()
        .map(|&v| v as f64)
        .collect();
    let x: Vec<f64> = x.data().iter().map(|&v| v as f64).collect();
    let y = (0..8)
        .map(|i| (0..4).map(|k| x[i * 4 + k] * t[k]).sum())
        .collect();
    (x, y)
}

fn residual(x: &[f64], y: &[f64], w: &[f64]) -> Vec<f64> {
    (0..8)
        .map(|i| (0..4).map(|k| x[i * 4 + k] * w[k]).sum::<f64>() - y[i])
        .collect()
}

fn mse(r: &[f64]) -> f64 {
    r.iter().map(|v| v * v).sum::<f64>() / r.len() as f64
}

/// `w - lr * (2 / 8) x^T (x w - y)`.
fn sgd(x: &[f64], y: &[f64], w: &[f64], lr: f64) -> Vec<f64> {
    let r = residual(x, y, w);
    (0..4)
        .map(|k| {
            let g: f64 = (0..8).map(|i| x[i * 4 + k] * r[i]).sum::<f64>() * 2.0 / 8.0;
            w[k] - lr * g
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The training loop against an independent f64 SGD.
// ---------------------------------------------------------------------------------------------

/// DESIGN.md § 9 gate 4 says "loss decreases and parameters approach a known solution"; the
/// acceptance asserts only the end points, which a doubled learning rate, a wrong mse scale or a
/// step counter off by one still meet. Every printed loss, and the final `w`, is checked against
/// SGD re-done in `f64` from the printed initial `w`.
#[test]
fn every_printed_loss_and_the_final_w_follow_f64_sgd_from_the_printed_init() {
    let stdout = train_ok(EXAMPLE);
    let (x, y) = linear_data(LINEAR_REGRESSION, 42);
    let mut w: Vec<f64> = values(&stdout, "init", "LinearReg.w")
        .iter()
        .map(|&v| v as f64)
        .collect();
    let mut expected = BTreeMap::new();
    for step in 0..=500u64 {
        expected.insert(step, mse(&residual(&x, &y, &w)));
        if step < 500 {
            w = sgd(&x, &y, &w, 0.1);
        }
    }
    let printed = losses(&stdout);
    assert_eq!(printed.len(), 11, "{stdout}");
    for (step, got) in printed {
        let want = expected[&step];
        assert!(
            (got as f64 - want).abs() <= 1e-3 * want + 1e-9,
            "step {step}: printed loss {got}, f64 SGD {want}"
        );
    }
    let learned = values(&stdout, "param", "LinearReg.w");
    for (k, (&got, &want)) in learned.iter().zip(&w).enumerate() {
        assert!(
            (got as f64 - want).abs() <= 1e-5,
            "w[{k}]: printed {got}, f64 SGD {want}"
        );
    }
}

/// `lr`, `seed` and `steps` come from the source: with `lr = 0.05`, `seed = 7` and `steps = 1`
/// the run prints exactly `step 0` and `step 1`, its `init` is the seed-7 draw and its `param`
/// is one SGD update at 0.05.
#[test]
fn one_step_at_lr_0_05_and_seed_7_is_one_update_at_0_05() {
    let source = LINEAR_REGRESSION
        .replace("lr = 0.1", "lr = 0.05")
        .replace("seed = 42;", "seed = 7;")
        .replace("steps = 500;", "steps = 1;");
    assert_ne!(source, LINEAR_REGRESSION);
    let path = source_file("one_step.mspec", &source);
    let stdout = train_ok(&path);
    let steps: Vec<u64> = losses(&stdout).iter().map(|&(s, _)| s).collect();
    assert_eq!(steps, vec![0, 1], "{stdout}");
    let lowered = compile_tensor("fit.mspec", &source).unwrap();
    let init = initialize(&lowered.program, 7, Stream::Init).unwrap();
    let w0 = values(&stdout, "init", "LinearReg.w");
    assert_eq!(w0, init[&ParamId(0)].data());
    let (x, y) = linear_data(&source, 7);
    let w0: Vec<f64> = w0.iter().map(|&v| v as f64).collect();
    let want = sgd(&x, &y, &w0, 0.05);
    let got = values(&stdout, "param", "LinearReg.w");
    for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g as f64 - w).abs() <= 1e-6,
            "w[{k}]: printed {g}, want {w}"
        );
    }
}

/// Loss lines are step 0, every 50 steps and the last step, once each.
#[test]
fn the_loss_lines_follow_the_source_steps() {
    for (steps, want) in [
        (120u64, vec![0u64, 50, 100, 120]),
        (50, vec![0, 50]),
        (49, vec![0, 49]),
    ] {
        let source = LINEAR_REGRESSION.replace("steps = 500;", &format!("steps = {steps};"));
        let path = source_file(&format!("steps_{steps}.mspec"), &source);
        let got: Vec<u64> = losses(&train_ok(&path)).iter().map(|&(s, _)| s).collect();
        assert_eq!(got, want, "steps = {steps}");
    }
}

// ---------------------------------------------------------------------------------------------
// The compiled step against the interpreter's forward and finite differences.
// ---------------------------------------------------------------------------------------------

const TWO_LAYER: &str = "model Two {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();
    param v: Tensor<f32, [3, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        let h = sub(mul(2.0, matmul(x, w)), 0.5);
        return matmul(h, v);
    }
}

train Fit {
    model = Two;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.25);
    steps = 3;
    seed = 11;
    device = cpu;
}
";

/// For every element of every parameter: the gradient output of the compiled step equals the
/// central difference of the forward loss (exact for a loss quadratic in each element), the
/// step's loss output equals the forward loss, and the update output is `p - 0.25 * grad`.
#[test]
fn the_compiled_step_matches_the_forward_and_central_differences() {
    let compiled = compile_training("two.mspec", TWO_LAYER).expect("it compiles");
    let data = synthetic(&compiled.config, &compiled.forward).expect("data");
    let forward = &compiled.forward.program;
    let mut params = initialize(forward, 11, Stream::Init).unwrap();
    for (i, tensor) in params.values_mut().enumerate() {
        let shifted: Vec<f32> = tensor.data().iter().map(|v| v + 0.1 * i as f32).collect();
        *tensor = HostTensor::new(tensor.shape().to_vec(), shifted).unwrap();
    }
    let loss_at = |p: &BTreeMap<ParamId, HostTensor<f32>>| -> f64 {
        evaluate(forward, &data.inputs, p).unwrap()[0].data()[0] as f64
    };
    let step = &compiled.training;
    let all = evaluate_all(&step.training.program, &data.inputs, &params).unwrap();
    assert_eq!(
        all.values[&step.training.loss].data()[0] as f64,
        loss_at(&params)
    );
    assert_eq!(step.training.gradients.len(), 2);
    for (&param, &grad) in &step.training.gradients {
        let g = all.values[&grad].data().to_vec();
        let update = all.values[&step.optimizer.updates[&param]].data().to_vec();
        let p = params[&param].data().to_vec();
        for e in 0..p.len() {
            let mut plus = params.clone();
            let mut minus = params.clone();
            let h = 1e-2f32;
            let bump = |t: &mut HostTensor<f32>, d: f32| {
                let mut v = t.data().to_vec();
                v[e] += d;
                *t = HostTensor::new(t.shape().to_vec(), v).unwrap();
            };
            bump(plus.get_mut(&param).unwrap(), h);
            bump(minus.get_mut(&param).unwrap(), -h);
            let fd = (loss_at(&plus) - loss_at(&minus)) / (2.0 * h as f64);
            assert!(
                (fd - g[e] as f64).abs() <= 2e-3 * (g[e] as f64).abs().max(1.0),
                "param {param} element {e}: gradient {}, central difference {fd}",
                g[e]
            );
            let want = p[e] - 0.25 * g[e];
            assert!(
                (update[e] - want).abs() <= 1e-6 * want.abs().max(1.0),
                "param {param} element {e}: update {}, p - lr * g {want}",
                update[e]
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The CLI contract.
// ---------------------------------------------------------------------------------------------

/// `src/train.rs`: built without the `vulkan` feature, "`--backend vulkan` exits 1 with
/// E-TARGET-001 before the file is read". With a file that does not exist, or one that does not
/// check, the refusal is still E-TARGET-001.
#[cfg(not(feature = "vulkan"))]
#[test]
fn vulkan_is_refused_before_the_file_is_read() {
    let want = "E-TARGET-001: the vulkan backend is not available in this build\n  \
                backend: vulkan\n  \
                available: cpu\n";
    let broken = source_file(
        "broken_vulkan.mspec",
        &LINEAR_REGRESSION.replace("matmul(x, w)", "matmul(w, x)"),
    );
    for path in ["no-such-model.mspec", broken.as_str()] {
        let output = modelspec(&["train", path, "--backend", "vulkan"]);
        assert_eq!(output.status.code(), Some(1), "{path}");
        assert!(output.stdout.is_empty(), "{path}");
        assert_eq!(text(&output.stderr), want, "{path}");
    }
}

/// With the `vulkan` feature, a file that does not exist or does not check is reported as on
/// the CPU, before any Vulkan loader is opened: the run names the file or prints the
/// diagnostic, never E-TARGET-002, with a loader path that does not exist.
#[cfg(feature = "vulkan")]
#[test]
fn vulkan_reports_the_file_before_the_loader() {
    let broken = source_file(
        "broken_vulkan.mspec",
        &LINEAR_REGRESSION.replace("matmul(x, w)", "matmul(w, x)"),
    );
    for path in ["no-such-model.mspec", broken.as_str()] {
        let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(["train", path, "--backend", "vulkan"])
            .env(
                "MODELSPEC_VULKAN_LIBRARY",
                "/nonexistent/modelspec-test/libvulkan-missing.so.1",
            )
            .current_dir(repo_root())
            .output()
            .expect("modelspec runs");
        let cpu = modelspec(&["train", path, "--backend", "cpu"]);
        assert_eq!(output.status.code(), Some(1), "{path}");
        assert!(output.stdout.is_empty(), "{path}");
        assert_eq!(text(&output.stderr), text(&cpu.stderr), "{path}");
    }
}

/// "the backend comes from `--backend`, not from the block's `device`": a file naming a Vulkan
/// device trains on the CPU and prints what the example prints.
#[test]
fn the_device_line_is_ignored() {
    let source = LINEAR_REGRESSION.replace("device = cpu;", "device = vulkan(0);");
    assert_ne!(source, LINEAR_REGRESSION);
    let path = source_file("vulkan_device.mspec", &source);
    assert_eq!(train_ok(&path), train_ok(EXAMPLE));
}

/// The data are keyed by name, not by declaration position: a `target` declared before the
/// `input` trains to the same bytes.
#[test]
fn a_target_declared_before_the_input_trains_the_same() {
    let source = LINEAR_REGRESSION.replace(
        "    input x: Tensor<f32, [8, 4]>;\n    target y: Tensor<f32, [8, 1]>;\n",
        "    target y: Tensor<f32, [8, 1]>;\n    input x: Tensor<f32, [8, 4]>;\n",
    );
    assert_ne!(source, LINEAR_REGRESSION);
    let path = source_file("target_first.mspec", &source);
    assert_eq!(train_ok(&path), train_ok(EXAMPLE));
}

#[test]
fn train_with_a_closed_stdout_pipe_exits_1_without_a_panic() {
    let (reader, writer) = std::io::pipe().expect("a pipe");
    drop(reader);
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["train", EXAMPLE, "--backend", "cpu"])
        .current_dir(repo_root())
        .stdout(writer)
        .stderr(Stdio::piped())
        .output()
        .expect("modelspec runs");
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(stderr.contains("cannot write the run"), "{stderr}");
}

#[cfg(target_os = "linux")]
#[test]
fn train_with_a_full_stdout_exits_1_without_a_panic() {
    let full = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/full")
        .expect("/dev/full opens");
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["train", EXAMPLE, "--backend", "cpu"])
        .current_dir(repo_root())
        .stdout(Stdio::from(full))
        .stderr(Stdio::piped())
        .output()
        .expect("modelspec runs");
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("cannot write the run"), "{stderr}");
}

// ---------------------------------------------------------------------------------------------
// E-DATA-003 from a real source.
// ---------------------------------------------------------------------------------------------

const UNREAD_Y: &str = "model Scale {
    param w: Tensor<f32, [1, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 1]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

train Fit {
    model = Scale;
    input x: Tensor<f32, [8, 1]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), x);
    optimizer = sgd(lr = 0.1);
    steps = 5;
    seed = 42;
    device = cpu;
}
";

#[test]
fn a_target_the_loss_does_not_read_is_refused_at_its_declaration() {
    let path = source_file("unread_y.mspec", UNREAD_Y);
    let output = modelspec(&["train", &path, "--backend", "cpu"]);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output.stderr));
    assert!(output.stdout.is_empty(), "{}", text(&output.stdout));
    assert_eq!(
        text(&output.stderr),
        format!(
            "E-DATA-003: the target `y` is not read by the loss\n  \
             {path}:12:5\n  \
             target: y\n"
        )
    );
}

// ---------------------------------------------------------------------------------------------
// Sources check accepts: no compiler bug, and a run or a data refusal.
// ---------------------------------------------------------------------------------------------

const UNUSED_PARAM: &str = "model Id {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 1]>) -> Tensor<f32, [8, 1]> {
        return x;
    }
}

train Fit {
    model = Id;
    input x: Tensor<f32, [8, 1]>;
    target y: Tensor<f32, [8, 1]>;
    loss = add(mse(model.forward(x), y), 1.0);
    optimizer = sgd(lr = 0.1);
    steps = 3;
    seed = 42;
    device = cpu;
}
";

/// Unit 6's decision: a parameter the loss does not read keeps its value. Here no parameter is
/// read at all; the step has no update and the file is not a compiler bug.
#[test]
fn a_loss_that_reads_no_parameter_trains_and_keeps_the_parameter() {
    let path = source_file("unused_param.mspec", UNUSED_PARAM);
    let stdout = train_ok(&path);
    assert_eq!(
        values(&stdout, "init", "Id.w"),
        values(&stdout, "param", "Id.w")
    );
    let ir = modelspec(&["ir", &path, "--stage", "training"]);
    assert_eq!(ir.status.code(), Some(0), "{}", text(&ir.stderr));
}

const SHARED: &str = "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return add(matmul(x, w), matmul(x, w));
    }
}

train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.02);
    steps = 2000;
    seed = 42;
    device = cpu;
}
";

/// A parameter used twice: its gradient accumulates both uses, and the run still reaches the
/// teacher (the teacher's prediction uses it twice too).
#[test]
fn a_parameter_read_twice_trains_to_the_teacher() {
    let path = source_file("shared.mspec", SHARED);
    let stdout = train_ok(&path);
    let lowered = compile_tensor("shared.mspec", SHARED).unwrap();
    let teacher = initialize(&lowered.program, 42, Stream::Teacher).unwrap();
    let learned = values(&stdout, "param", "LinearReg.w");
    for (k, (&got, &want)) in learned.iter().zip(teacher[&ParamId(0)].data()).enumerate() {
        assert!((got - want).abs() <= 1e-2, "w[{k}] = {got}, teacher {want}");
    }
}

const SEMANTICS: &str = "model Semantics {
    param w: Tensor<f32, [2, 2]> = xavier_uniform();
    param v: Tensor<f32, [2, 1]> = xavier_uniform();

    fn forward(a: Tensor<f32, [2, 2]>, b: Tensor<f32, [2, 2]>) -> Tensor<f32, [2, 1]> {
        let d = sub(a, b);
        let dead = relu(d);
        let s = mul(3.0, d);
        let m = matmul(s, w);
        let r = relu(sub(m, 10.0));
        return matmul(r, v);
    }
}

train Fit {
    model = Semantics;
    input p: Tensor<f32, [2, 2]>;
    input q: Tensor<f32, [2, 2]>;
    target y: Tensor<f32, [2, 1]>;
    loss = add(mse(model.forward(p, q), y), mul(0.5, mean(transpose(p))));
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}
";

fn tokens(source: &str) -> Vec<(usize, usize)> {
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.')
            {
                i += 1;
            }
            out.push((start, i));
        } else if c == b'-' && bytes.get(i + 1) == Some(&b'>') {
            out.push((i, i + 2));
            i += 2;
        } else if c == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else {
            let len = source[i..].chars().next().unwrap().len_utf8();
            out.push((i, i + len));
            i += len;
        }
    }
    out
}

/// What `modelspec train` would do with a source `check` accepts, short of printing: `None` when
/// it compiles for training and runs one step or is refused by the data loader with a user code.
fn training_violation(source: &str) -> Option<String> {
    compile_hir("fuzz.mspec", source).ok()?;
    let mut compiled = match compile_training("fuzz.mspec", source) {
        Ok(compiled) => compiled,
        Err(d) if d.iter().all(|d| !d.code.as_str().starts_with("E-INTERNAL")) => return None,
        Err(d) => return Some(format!("compile_training: {}", d[0])),
    };
    compiled.config.steps = 1;
    match runner::run(&compiled) {
        Ok(run) if run.losses.len() == 2 => None,
        Ok(run) => Some(format!("{} losses for one step", run.losses.len())),
        Err(RunError::Data(DataError::Refused(_))) => None,
        Err(e) => Some(format!("run: {e}")),
    }
}

/// Every one-token edit of three sources that `check` accepts either trains one step or is
/// refused with a user code: E-INTERNAL "No source file is meant to reach it" (compile.rs).
#[test]
fn every_checked_edit_trains_one_step_or_is_refused_with_a_user_code() {
    let mut sources = Vec::new();
    for base in [LINEAR_REGRESSION, SEMANTICS, SHARED] {
        let toks = tokens(base);
        let mut vocab: Vec<String> = toks.iter().map(|&(a, b)| base[a..b].to_owned()).collect();
        for extra in [
            "add",
            "sub",
            "mul",
            "relu",
            "transpose",
            "sum",
            "mean",
            "mse",
            "matmul",
            "0.5",
            "1.0",
            "w",
            "v",
            "x",
            "y",
            "p",
            "q",
            "(",
            ")",
            ",",
            ";",
            "model.forward(x)",
            "model.forward(p, q)",
        ] {
            vocab.push(extra.to_owned());
        }
        vocab.sort();
        vocab.dedup();
        for &(a, b) in &toks {
            sources.push(format!("{}{}", &base[..a], &base[b..]));
            for word in &vocab {
                sources.push(format!("{} {word} {}", &base[..a], &base[b..]));
                sources.push(format!("{} {word} {}", &base[..a], &base[a..]));
            }
        }
    }
    let mut accepted = 0usize;
    let mut trained = 0usize;
    let mut out = Vec::new();
    for source in &sources {
        if compile_hir("fuzz.mspec", source).is_ok() {
            accepted += 1;
            if compile_training("fuzz.mspec", source).is_ok() {
                trained += 1;
            }
        }
        match std::panic::catch_unwind(|| training_violation(source)) {
            Ok(None) => {}
            Ok(Some(problem)) => out.push(format!("{problem}\n--- source ---\n{source}")),
            Err(_) => out.push(format!("panicked\n--- source ---\n{source}")),
        }
    }
    assert!(accepted > 100, "only {accepted} edits checked clean");
    assert!(trained > 100, "only {trained} edits compiled for training");
    assert!(
        out.is_empty(),
        "{} of {} edits ({accepted} checked clean) fail training; first:\n{}",
        out.len(),
        sources.len(),
        out[0]
    );
}
