//! `examples/mnist_mlp.mspec`: the shapes of the MNIST benchmark (784 -> 64 relu -> 10, no bias,
//! `f32`, MSE, SGD lr 0.1, batch 100, 1,800 steps) in the M0 subset, trained on M0 synthetic data
//! (DESIGN.md § 2.3, § 9 gate 4).
//!
//! The golden `tests/golden/mnist_mlp.training.ir` is the output of `modelspec ir
//! examples/mnist_mlp.mspec --stage training`, checked line by line against the linear-regression
//! golden's rules: the forward pass, the reverse pass of `mse`, `matmul`, `relu` and `matmul`
//! (seed `1.0`, `d = o - y`, the scale `2 / 1000`, `a^T @ dO` for `w2`, `dA @ w2^T` through
//! `ReluGrad` to `x^T @ dH` for `w1`), then `w - 0.1 * grad` for each parameter with its write
//! token. `trains_on_cpu` is ignored: run it by hand with `--release`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::compile;
use modelspec_core::ir::tensor::TensorType;

const EXAMPLE: &str = "examples/mnist_mlp.mspec";
const GOLDEN: &str = include_str!("golden/mnist_mlp.training.ir");

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

#[test]
fn check_accepts_the_example() {
    let output = modelspec(&["check", EXAMPLE]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "stderr: {}", text(&output.stderr));
}

/// The parameters, data, learning rate, steps and device the story names.
#[test]
fn the_example_has_the_benchmark_shapes() {
    let compiled =
        compile::compile_training(EXAMPLE, &example_source()).expect("the example compiles");
    let program = &compiled.forward.program;
    let params: Vec<(&str, &TensorType)> = program
        .params
        .values()
        .map(|decl| (decl.path.as_str(), &decl.ty))
        .collect();
    assert_eq!(
        params,
        vec![
            ("MnistMlp.w1", &TensorType::f32(&[784, 64])),
            ("MnistMlp.w2", &TensorType::f32(&[64, 10])),
        ]
    );
    let inputs: Vec<String> = program
        .inputs
        .iter()
        .map(|&v| {
            let node = program.node(v).unwrap();
            format!("{:?} {}", node.op, node.ty)
        })
        .collect();
    assert_eq!(
        inputs,
        vec![
            "Input { name: \"x\" } Tensor<f32, [100, 784]>",
            "Input { name: \"y\" } Tensor<f32, [100, 10]>",
        ]
    );
    assert_eq!(compiled.config.lr, 0.1);
    assert_eq!(compiled.config.steps, 1800);
    let source = example_source();
    assert!(source.contains("seed = 42;"), "{source}");
    assert!(source.contains("device = vulkan(0);"), "{source}");
}

#[test]
fn training_ir() {
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

/// The golden holds a gradient and an SGD update for each parameter, each of its type.
#[test]
fn training_ir_golden_has_a_gradient_and_an_update_per_parameter() {
    for (path, ty) in [
        ("MnistMlp.w1", "Tensor<f32, [784, 64]>"),
        ("MnistMlp.w2", "Tensor<f32, [64, 10]>"),
    ] {
        for kind in ["gradient", "update"] {
            let output = GOLDEN
                .lines()
                .find(|l| l.starts_with("output %") && l.ends_with(&format!(" {kind} @{path}")))
                .unwrap_or_else(|| panic!("no `{kind} @{path}` output in:\n{GOLDEN}"));
            let value = output
                .strip_prefix("output ")
                .and_then(|rest| rest.split_once(' '))
                .map(|(v, _)| v)
                .unwrap();
            let tag = if kind == "gradient" {
                "gradient"
            } else {
                "optimizer"
            };
            let node = format!("{tag} {value}: {ty} = ");
            assert!(
                GOLDEN.lines().any(|l| l.starts_with(&node)),
                "no `{node}` line in:\n{GOLDEN}"
            );
        }
        assert!(
            GOLDEN.lines().any(
                |l| l.starts_with("optimizer %") && l.contains(&format!(" write @{path} v1}}"))
            ),
            "no write of @{path} in:\n{GOLDEN}"
        );
    }
    assert!(GOLDEN.contains("optimizer sgd lr = 0.1\n"), "{GOLDEN}");
}

/// `cargo test --release -p modelspec-cli --test mnist_mlp -- --ignored trains_on_cpu`.
#[test]
#[ignore = "1,800 steps of the benchmark shape; run by hand with --release"]
fn trains_on_cpu() {
    let output = modelspec(&["train", EXAMPLE, "--backend", "cpu"]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "stderr: {}", text(&output.stderr));
    let stdout = text(&output.stdout);
    let count = |prefix: &str| stdout.lines().filter(|l| l.starts_with(prefix)).count();
    assert_eq!(count("init @"), 2, "init lines");
    assert_eq!(count("param @"), 2, "param lines");
    let losses: Vec<(u64, f32)> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (step.parse().unwrap(), loss.parse().unwrap())
        })
        .collect();
    assert_eq!(
        losses.iter().map(|&(k, _)| k).collect::<Vec<_>>(),
        (0..=1800).step_by(50).collect::<Vec<u64>>(),
        "37 step lines"
    );
    assert_eq!(stdout.lines().count(), 2 + 37 + 2, "{stdout}");
    let (_, first) = losses[0];
    let (_, last) = losses[losses.len() - 1];
    assert!(
        last < first * 1e-6,
        "final loss {last} is not below 1e-6 of the step-0 loss {first}"
    );
    println!("step-0 loss {first}, final loss {last}");
}
