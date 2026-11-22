//! Adversary cases for `modelspec` on a file trained with `adamw(...)` (story:bert-adamw-warmup,
//! DESIGN.md § 2.1, § 8, coordinator decision D50).
//!
//! The commands besides `train --backend cpu` that take such a file: `compile`, `ir --stage
//! kernel`, `train --backend vulkan`, the checkpoint flags, and `--steps` at the warm-up's edge.
//! A case named `adv_adamw_f<N>_...` is a finding of the adversary's report, ignored until its
//! fix landed in fix pass 1 (coordinator decision D64).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FIXTURE: &str = "crates/modelspec-cli/tests/fixtures/linear_regression_adamw.mspec";

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
    String::from_utf8(bytes.to_vec()).expect("UTF-8")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("adversary_adamw_cli");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join(name);
    let _ = std::fs::remove_dir_all(&path);
    let _ = std::fs::remove_file(&path);
    path
}

/// Finding F1. `compile --backend cpu` on an AdamW file reaches the user as `E-INTERNAL-002`, a
/// compiler bug ("No source file is meant to reach it", `compile::codes::COMPILER_BUG`): the
/// scheduler takes the step's host inputs `optimizer.lr_t`, `c1`, `c2` and the moments for train
/// block data it does not declare. A CPU artifact keeps the views and needs no kernel
/// (`compile_artifact_for`), so the file should compile.
#[test]
fn adv_adamw_f1_compile_backend_cpu_writes_an_artifact() {
    let out = scratch("artifact");
    let output = modelspec(&[
        "compile",
        FIXTURE,
        "--backend",
        "cpu",
        "--output",
        out.to_str().unwrap(),
    ]);
    let stderr = text(&output.stderr);
    assert!(!stderr.contains("E-INTERNAL-002"), "{stderr}");
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    assert!(out.join("artifact.json").exists());
}

/// `ir --stage kernel` prints the AdamW update as one `sgd_update AdamW` kernel, which computes
/// its `Sqrt` and its `Div` (coordinator decision D73,
/// story:vulkan-adamw-one-dispatch-per-parameter); `ir --stage training` prints the moments.
#[test]
fn adamw_vulkan_paths_print_the_update_as_one_kernel() {
    let output = modelspec(&["ir", FIXTURE, "--stage", "kernel"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    let stdout = text(&output.stdout);
    let kernels = |op: &str| {
        stdout
            .lines()
            .filter(|line| line.starts_with("kernel ") && line.contains(op))
            .count()
    };
    assert_eq!(kernels(" sgd_update AdamW f32 "), 1, "{stdout}");
    for op in [" elementwise Sqrt f32 ", " elementwise Div f32 "] {
        assert_eq!(kernels(op), 0, "{op}: {stdout}");
    }
    let output = modelspec(&["ir", FIXTURE, "--stage", "training"]);
    assert_eq!(output.status.code(), Some(0));
    let stdout = text(&output.stdout);
    for line in [
        "output %25 moment m @LinearReg.w",
        "output %29 moment v @LinearReg.w",
    ] {
        assert!(stdout.contains(line), "{stdout}");
    }
}

/// Each stepwise readback includes the post-update moments of exactly the checkpointed step.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires a Vulkan compute device"]
fn adamw_vulkan_checkpoint_resumes_bit_exactly() {
    let full = scratch("gpu.msckpt");
    let intermediate = scratch("gpu-7.msckpt");
    let resumed = scratch("gpu-resumed.msckpt");
    let run = |args: &[&str]| {
        let mut command = vec!["train", FIXTURE, "--backend", "vulkan"];
        command.extend_from_slice(args);
        let output = modelspec(&command);
        assert!(output.status.success(), "{}", text(&output.stderr));
    };
    run(&[
        "--steps",
        "18",
        "--checkpoint",
        full.to_str().unwrap(),
        "--checkpoint-every",
        "7",
    ]);
    run(&[
        "--steps",
        "11",
        "--resume",
        intermediate.to_str().unwrap(),
        "--checkpoint",
        resumed.to_str().unwrap(),
    ]);
    assert_eq!(
        std::fs::read(full).unwrap(),
        std::fs::read(resumed).unwrap()
    );
    let cpu = scratch("gpu-state-on-cpu.msckpt");
    let restored = modelspec(&[
        "train",
        FIXTURE,
        "--backend",
        "cpu",
        "--steps",
        "0",
        "--resume",
        intermediate.to_str().unwrap(),
        "--checkpoint",
        cpu.to_str().unwrap(),
    ]);
    assert!(restored.status.success(), "{}", text(&restored.stderr));
    assert_eq!(
        std::fs::read(cpu).unwrap(),
        std::fs::read(intermediate).unwrap()
    );
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires a Vulkan compute device"]
fn vulkan_runs_a_parameter_free_training_graph() {
    let dir = scratch("stateless");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("multiply.mspec");
    std::fs::write(&source, "
        model Transform {
            fn forward(x: Tensor<f32, [1, 1]>) -> Tensor<f32, [1, 1]> { return mul(x, 2.0); }
        }
        train Apply {
            model = Transform;
            input x: Tensor<f32, [1, 1]>;
            loss = sum(model.forward(x));
            optimizer = adamw(lr = 0.1, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, warmup_steps = 0);
            steps = 1;
            seed = 42;
            device = cpu;
        }
    ").unwrap();
    std::fs::write(dir.join("x.f32"), 3f32.to_le_bytes()).unwrap();
    let run = modelspec(&[
        "train",
        source.to_str().unwrap(),
        "--data",
        dir.to_str().unwrap(),
        "--backend",
        "vulkan",
    ]);
    assert!(run.status.success(), "{}", text(&run.stderr));
    assert!(text(&run.stdout).contains("step 1 loss 6"));
}

/// `--steps` equal to the warm-up stops the run before its first step, exit 1, one line naming
/// both numbers and no code; a step more runs to the end at a rate that never goes negative.
#[test]
fn adamw_steps_at_the_warmup_edge() {
    let output = modelspec(&["train", FIXTURE, "--backend", "cpu", "--steps", "10"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(output.stdout.is_empty(), "{}", text(&output.stdout));
    assert_eq!(stderr.lines().count(), 1, "{stderr}");
    assert!(stderr.contains("10 steps"), "{stderr}");
    assert!(!stderr.contains("E-"), "{stderr}");

    let output = modelspec(&["train", FIXTURE, "--backend", "cpu", "--steps", "11"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let output = modelspec(&["train", FIXTURE, "--backend", "cpu", "--steps", "400"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let stdout = text(&output.stdout);
    assert!(stdout.contains("step 400 loss "), "{stdout}");
}
