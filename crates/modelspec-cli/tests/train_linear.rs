//! `modelspec train --backend cpu` on linear regression, and `modelspec ir --stage training`
//! (DESIGN.md § 3.1, § 6, § 8, § 9 gate 4).
//!
//! The acceptance runs `modelspec train examples/linear_regression.mspec --backend cpu` (seed 42,
//! 500 steps, as in the file) and asserts exit 0, a final loss at most 1e-3 times the step-0 loss,
//! and every element of the learned `LinearReg.w` within 1e-2 of the teacher weights. The teacher
//! is drawn here, independently of the CLI, from the file's Tensor IR on the teacher stream.
//!
//! The golden `tests/golden/linear_regression.training.ir` is written by hand from DESIGN.md § 3.1
//! and the printer's line format (`ir/print.rs`): the Tensor IR golden's five nodes, the reverse
//! pass of `mse` and `matmul` (the seed `1.0`, `d = h - y`, the scale `2 / 8`, `x^T @ (d * s)`),
//! then the SGD subgraph `w - 0.1 * grad` with its write token.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::compile;
use modelspec_core::ir::tensor::ParamId;
use modelspec_runtime::train::init::{Stream, initialize};

const EXAMPLE: &str = "examples/linear_regression.mspec";
const GOLDEN: &str = include_str!("golden/linear_regression.training.ir");

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs `modelspec <args>` from the repository root.
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

fn example_source() -> String {
    std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example exists")
}

/// Writes `source` to a fresh file named `name` under the test temp dir and returns its path.
fn source_file(name: &str, source: &str) -> String {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("train_linear");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join(name);
    std::fs::write(&path, source).expect("the source is written");
    path.display().to_string()
}

/// A successful `modelspec train <path> --backend cpu`.
fn train(path: &str) -> String {
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

/// The values of the line `<prefix> @<path>: <type> = [v, ...]`.
fn values(stdout: &str, prefix: &str, path: &str) -> Vec<f32> {
    let head = format!("{prefix} @{path}: ");
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

/// The teacher `LinearReg.w` of the example under `seed`.
fn teacher_w(seed: u64) -> Vec<f32> {
    let lowered = compile::compile_tensor(EXAMPLE, &example_source()).expect("the example lowers");
    let teacher = initialize(&lowered.program, seed, Stream::Teacher).expect("teacher draws");
    teacher[&ParamId(0)].data().to_vec()
}

#[test]
fn training_the_example_reaches_the_teacher() {
    let stdout = train(EXAMPLE);
    let losses = losses(&stdout);
    let steps: Vec<u64> = losses.iter().map(|&(step, _)| step).collect();
    assert_eq!(
        steps,
        (0..=500).step_by(50).collect::<Vec<u64>>(),
        "{stdout}"
    );
    let (_, first) = losses[0];
    let (_, last) = losses[losses.len() - 1];
    assert!(first > 0.0, "step-0 loss {first}");
    assert!(
        last <= 1e-3 * first,
        "final loss {last} is above 1e-3 x step-0 loss {first}"
    );
    let learned = values(&stdout, "param", "LinearReg.w");
    let teacher = teacher_w(42);
    assert_eq!(learned.len(), 4);
    for (i, (&got, &want)) in learned.iter().zip(&teacher).enumerate() {
        assert!(
            (got - want).abs() <= 1e-2,
            "w[{i}] = {got}, teacher {want}\n{stdout}"
        );
    }
    let largest = learned
        .iter()
        .zip(&teacher)
        .map(|(&got, &want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    println!("step-0 loss {first}, final loss {last}, largest weight error {largest}");
    println!("teacher {teacher:?}");
}

#[test]
fn the_output_is_the_initial_parameters_the_losses_and_the_learned_parameters() {
    let stdout = train(EXAMPLE);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 1 + 11 + 1, "{stdout}");
    assert!(
        lines[0].starts_with("init @LinearReg.w: Tensor<f32, [4, 1]> = ["),
        "{stdout}"
    );
    assert!(lines[1].starts_with("step 0 loss "), "{stdout}");
    assert!(lines[11].starts_with("step 500 loss "), "{stdout}");
    assert!(
        lines[12].starts_with("param @LinearReg.w: Tensor<f32, [4, 1]> = ["),
        "{stdout}"
    );
}

#[test]
fn the_initial_parameters_are_the_init_stream_draw() {
    let stdout = train(EXAMPLE);
    let lowered = compile::compile_tensor(EXAMPLE, &example_source()).expect("the example lowers");
    let init = initialize(&lowered.program, 42, Stream::Init).expect("init draws");
    assert_eq!(
        values(&stdout, "init", "LinearReg.w"),
        init[&ParamId(0)].data()
    );
}

#[test]
fn two_runs_with_seed_42_print_identical_bytes() {
    let a = modelspec(&["train", EXAMPLE, "--backend", "cpu"]);
    let b = modelspec(&["train", EXAMPLE, "--backend", "cpu"]);
    assert_eq!(a.status.code(), Some(0));
    assert_eq!(a.stdout, b.stdout);
    assert_eq!(a.stderr, b.stderr);
}

#[test]
fn seeds_42_and_43_print_different_initial_parameters() {
    let source = example_source();
    assert!(source.contains("seed = 42;"));
    let seed_43 = source_file("seed_43.mspec", &source.replace("seed = 42;", "seed = 43;"));
    let a = values(&train(EXAMPLE), "init", "LinearReg.w");
    let b = values(&train(&seed_43), "init", "LinearReg.w");
    assert_ne!(a, b);
}

/// Built without the `vulkan` feature (`cargo test -p modelspec-cli --no-default-features`),
/// `--backend vulkan` is refused with E-TARGET-001 before the file is read.
#[cfg(not(feature = "vulkan"))]
#[test]
fn a_build_without_the_vulkan_feature_refuses_the_vulkan_backend() {
    for path in [EXAMPLE, "no-such-model.mspec"] {
        let output = modelspec(&["train", path, "--backend", "vulkan"]);
        assert_eq!(output.status.code(), Some(1), "{path}");
        assert!(output.stdout.is_empty(), "stdout: {}", text(&output.stdout));
        assert_eq!(
            text(&output.stderr),
            "E-TARGET-001: the vulkan backend is not available in this build\n  \
             backend: vulkan\n  \
             available: cpu\n",
            "{path}"
        );
    }
}

/// With the `vulkan` feature and no Vulkan loader, `--backend vulkan` compiles the file and then
/// exits 1 with E-TARGET-002; no instance is created.
#[cfg(feature = "vulkan")]
#[test]
fn a_missing_vulkan_loader_refuses_the_vulkan_backend_with_e_target_002() {
    let loader = "/nonexistent/modelspec-test/libvulkan-missing.so.1";
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["train", EXAMPLE, "--backend", "vulkan"])
        .env("MODELSPEC_VULKAN_LIBRARY", loader)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("E-TARGET-002: "), "{stderr}");
    assert!(stderr.contains(loader), "{stderr}");
    assert!(output.stdout.is_empty(), "stdout: {}", text(&output.stdout));
}

#[test]
fn train_needs_a_backend() {
    let output = modelspec(&["train", EXAMPLE]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output.stderr));
}

#[test]
fn train_on_a_missing_file_exits_1_naming_it() {
    let output = modelspec(&["train", "no-such-model.mspec", "--backend", "cpu"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(
        text(&output.stderr).starts_with("modelspec train no-such-model.mspec: "),
        "{}",
        text(&output.stderr)
    );
}

#[test]
fn a_file_that_does_not_check_prints_its_diagnostics_and_exits_1() {
    let broken = source_file(
        "broken.mspec",
        &example_source().replace("matmul(x, w)", "matmul(w, x)"),
    );
    let output = modelspec(&["train", &broken, "--backend", "cpu"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty(), "stdout: {}", text(&output.stdout));
    assert!(
        text(&output.stderr).starts_with("E-TYPE-"),
        "{}",
        text(&output.stderr)
    );
}

#[test]
fn ir_stage_training_prints_the_golden() {
    let output = modelspec(&["ir", EXAMPLE, "--stage", "training"]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "stderr: {}", text(&output.stderr));
    assert_eq!(text(&output.stdout), GOLDEN);
}

#[test]
fn ir_stage_training_matches_compile_training() {
    let compiled =
        compile::compile_training(EXAMPLE, &example_source()).expect("the example compiles");
    assert_eq!(
        modelspec_core::ir::print::print_step(&compiled.training),
        GOLDEN
    );
    assert_eq!(compiled.config.steps, 500);
    assert_eq!(compiled.config.seed, 42);
    assert_eq!(compiled.config.lr, 0.1);
}

const UNREAD_X: &str = "model Const {
    param b: Tensor<f32, [8, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 1]> {
        return b;
    }
}

train Fit {
    model = Const;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
";

#[test]
fn an_input_the_loss_does_not_read_is_refused_at_its_declaration() {
    let path = source_file("unread_x.mspec", UNREAD_X);
    let output = modelspec(&["train", &path, "--backend", "cpu"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty(), "stdout: {}", text(&output.stdout));
    let stderr = text(&output.stderr);
    assert_eq!(
        stderr,
        format!(
            "E-DATA-003: the input `x` is not read by the loss\n  \
             {path}:12:5\n  \
             input: x\n"
        )
    );
    assert!(!stderr.contains("not an input of the program"), "{stderr}");
}

#[test]
fn a_second_input_is_refused_by_the_data_loader() {
    let source = example_source().replace(
        "    target y:",
        "    input z: Tensor<f32, [8, 4]>;\n    target y:",
    );
    assert!(source.contains("input z"));
    let path = source_file("two_inputs.mspec", &source);
    let output = modelspec(&["train", &path, "--backend", "cpu"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty(), "stdout: {}", text(&output.stdout));
    let stderr = text(&output.stderr);
    assert!(
        stderr
            .starts_with("E-DATA-001: M0 synthetic data needs exactly one input and one target\n"),
        "{stderr}"
    );
    assert!(stderr.ends_with("  inputs: 2\n  targets: 1\n"), "{stderr}");
}
