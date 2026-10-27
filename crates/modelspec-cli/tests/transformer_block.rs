//! `examples/transformer_block.mspec`, the two BERT-tiny encoder layers of the transformer-block
//! benchmark on synthetic data (story:transformer-block-benchmark).
//!
//! The CPU tests check the example, train it one step on the CPU, and export what the PyTorch
//! opponent's first-step loss check reads: the initial `.msckpt` of `train --steps 0
//! --checkpoint` and the synthetic batch of `train --write-batch`, `<name>.f32` per input and
//! target, which must be the batch the run trains on. The device test trains it ten steps on
//! `--backend vulkan` and on `--backend cpu` and holds the eleven recorded losses and the final
//! parameters within the run tolerance of DESIGN.md § 9 gate 5, `|g - c| <= GPU_RUN_ATOL +
//! GPU_RUN_RTOL * |c|`, the attention key biases, whose exact gradient is zero by the model's
//! structure, within `zero_gradient_bound` instead. It is ignored by default; run it with
//! `cargo test --release -p modelspec-cli --features vulkan --test transformer_block -- --ignored`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::compile::compile_training;
use modelspec_runtime::train::checkpoint;
use modelspec_runtime::train::data::synthetic;

const EXAMPLE: &str = "examples/transformer_block.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh, empty directory under this crate's build directory, named for the test.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("transformer_block_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// Runs `modelspec <args>` from the repository root.
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

fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout: {}\nstderr: {}",
        stdout(output),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

/// The `(step, loss)` of every `step` line.
fn step_lines(out: &str) -> Vec<(u64, f32)> {
    out.lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (step.parse().unwrap(), loss.parse().unwrap())
        })
        .collect()
}

/// The values of every `<label> @<path>: <type> = [v, ...]` line, by path.
fn tensors(out: &str, label: &str) -> BTreeMap<String, Vec<f32>> {
    let prefix = format!("{label} @");
    out.lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        .map(|rest| {
            let (path, rest) = rest.split_once(": ").expect("`@<path>: <type> = [..]`");
            let (_, list) = rest.split_once(" = [").expect("a value list");
            let list = list.strip_suffix(']').expect("a closed list");
            let values = list
                .split(", ")
                .map(|v| v.parse::<f32>().expect("an f32"))
                .collect();
            (path.to_owned(), values)
        })
        .collect()
}

/// The little-endian `f32` values of `file`.
fn read_f32(file: &Path) -> Vec<f32> {
    let bytes = std::fs::read(file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    assert_eq!(bytes.len() % 4, 0, "{}", file.display());
    let (words, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "{}", file.display());
    words.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

#[test]
fn the_example_checks() {
    let output = modelspec(&["check", EXAMPLE]);
    assert_exit(&output, 0);
}

/// The two encoder layers of `bert_tiny.mspec`: 32 parameters, 16 per layer, and
/// `2 (4 (H^2 + H) + 2 H F + F + H + 4 H) = 396,544` values at `H = 128`, `F = 512`.
#[test]
fn the_example_trains_one_cpu_step_and_its_loss_falls() {
    let output = modelspec(&["train", EXAMPLE, "--backend", "cpu", "--steps", "1"]);
    assert_exit(&output, 0);
    let out = stdout(&output);
    let steps = step_lines(&out);
    assert_eq!(steps.iter().map(|(k, _)| *k).collect::<Vec<_>>(), [0, 1]);
    let (first, second) = (steps[0].1, steps[1].1);
    assert!(first.is_finite() && second.is_finite(), "{steps:?}");
    assert!(second < first, "{steps:?}");
    let params = tensors(&out, "param");
    assert_eq!(params.len(), 32);
    assert_eq!(params.values().map(Vec::len).sum::<usize>(), 396_544);
}

/// What the opponent's loss check reads: `--steps 0 --checkpoint` writes the initial parameters
/// the `init` lines print, and `--write-batch` writes `x.f32`, `key_mask.f32` (all ones) and
/// `y.f32`, row-major little-endian `f32`, equal to the synthetic batch the run trains on.
#[test]
fn the_initial_checkpoint_and_the_batch_are_exported_for_the_loss_check() {
    let case = scratch("export");
    let ckpt = case.join("initial.msckpt");
    let batch = case.join("batch");
    let output = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--steps",
        "0",
        "--checkpoint",
        path(&ckpt),
        "--write-batch",
        path(&batch),
    ]);
    assert_exit(&output, 0);
    let out = stdout(&output);
    assert_eq!(step_lines(&out).len(), 1, "{out}");

    let read = checkpoint::read(&ckpt).expect("a readable checkpoint");
    assert_eq!(read.step, 0);
    let init = tensors(&out, "init");
    let saved: BTreeMap<String, Vec<f32>> = read
        .parameters
        .iter()
        .map(|p| (p.path.clone(), p.value.data().to_vec()))
        .collect();
    assert_eq!(saved, init);

    let source = std::fs::read_to_string(repo_root().join(EXAMPLE)).unwrap();
    let compiled = compile_training(EXAMPLE, &source).expect("the example compiles");
    let data = synthetic(&compiled.config, &compiled.forward).expect("synthetic data");
    let mut names: Vec<String> = std::fs::read_dir(&batch)
        .expect("the batch directory")
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(names, ["key_mask.f32", "x.f32", "y.f32"]);
    for (name, tensor) in &data.inputs {
        assert_eq!(
            read_f32(&batch.join(format!("{name}.f32"))),
            tensor.data(),
            "{name}"
        );
    }
    assert_eq!(data.inputs["x"].shape(), &[1024, 128]);
    assert!(data.inputs["key_mask"].data().iter().all(|&m| m == 1.0));
}

/// `--write-batch` writes the synthetic batch, so it is a usage error beside `--data`.
#[test]
fn write_batch_beside_data_is_a_usage_error() {
    let case = scratch("usage");
    let output = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--data",
        path(&case),
        "--write-batch",
        path(&case.join("batch")),
    ]);
    assert_exit(&output, 2);
    assert!(!case.join("batch").exists());
}

#[cfg(feature = "vulkan")]
#[allow(dead_code)]
#[path = "../../modelspec-runtime/tests/common/mod.rs"]
mod common;

#[cfg(feature = "vulkan")]
mod device {
    use super::*;

    use super::common::{
        GPU_RUN_ATOL, GPU_RUN_RTOL, attention_key_biases, within, zero_gradient_bound,
    };
    use serde_json::Value;

    const STEPS: u64 = 10;
    /// The `lr` of the example's `sgd(...)`.
    const LR: f64 = 0.01;

    /// The loss of every `StepCompleted` of a recorded run file, by step.
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

    fn bound(c: f32) -> f64 {
        f64::from(GPU_RUN_ATOL) + f64::from(GPU_RUN_RTOL) * f64::from(c).abs()
    }

    /// Ten steps on the GPU and on the CPU: the same `init` values, the eleven recorded losses
    /// and the final parameters within the run tolerance.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn ten_vulkan_steps_agree_with_the_cpu_run() {
        assert!(
            std::fs::read_to_string(repo_root().join(EXAMPLE))
                .unwrap()
                .contains("optimizer = sgd(lr = 0.01);"),
            "LR is the example's"
        );
        let case = scratch("steps");
        let steps = STEPS.to_string();
        let run = |backend: &str, record: &Path| {
            modelspec(&[
                "train",
                EXAMPLE,
                "--backend",
                backend,
                "--steps",
                &steps,
                "--record",
                path(record),
            ])
        };
        let (cpu_record, gpu_record) = (case.join("cpu.run.jsonl"), case.join("gpu.run.jsonl"));
        let cpu = run("cpu", &cpu_record);
        assert_exit(&cpu, 0);
        let gpu = run("vulkan", &gpu_record);
        assert_exit(&gpu, 0);
        let (cpu_out, gpu_out) = (stdout(&cpu), stdout(&gpu));
        assert_eq!(tensors(&cpu_out, "init"), tensors(&gpu_out, "init"));

        let mut outside = Vec::new();
        let (cpu_losses, gpu_losses) = (recorded_losses(&cpu_record), recorded_losses(&gpu_record));
        assert_eq!(cpu_losses.len() as u64, STEPS + 1);
        assert_eq!(
            cpu_losses.keys().collect::<Vec<_>>(),
            gpu_losses.keys().collect::<Vec<_>>()
        );
        eprintln!("step | CPU loss | GPU loss | |diff| | bound");
        for (k, &c) in &cpu_losses {
            let (c, g) = (c as f32, gpu_losses[k] as f32);
            let diff = (f64::from(g) - f64::from(c)).abs();
            eprintln!("{k} | {c} | {g} | {diff:.3e} | {:.3e}", bound(c));
            if !within(g, c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) {
                outside.push(format!("recorded step {k}: gpu {g} cpu {c}"));
            }
        }
        assert!(
            cpu_losses[&STEPS] < cpu_losses[&0],
            "the loss falls: {cpu_losses:?}"
        );

        let (cpu_params, gpu_params) = (tensors(&cpu_out, "param"), tensors(&gpu_out, "param"));
        assert_eq!(cpu_params.len(), 32);
        assert_eq!(
            cpu_params.keys().collect::<Vec<_>>(),
            gpu_params.keys().collect::<Vec<_>>()
        );
        let tensor_ir = modelspec(&["ir", EXAMPLE, "--stage", "tensor"]);
        assert_exit(&tensor_ir, 0);
        let biases = attention_key_biases(&stdout(&tensor_ir));
        assert_eq!(biases.len(), 2, "one key bias per layer: {biases:?}");
        let biased = zero_gradient_bound(STEPS, LR);
        let (mut elements, mut worst) = (0usize, 0.0f64);
        for (name, c) in &cpu_params {
            let g = &gpu_params[name];
            assert_eq!(g.len(), c.len(), "{name}");
            for (i, (&g, &c)) in g.iter().zip(c).enumerate() {
                let diff = (f64::from(g) - f64::from(c)).abs();
                elements += 1;
                let passes = if biases.contains(name) {
                    diff <= biased
                } else {
                    worst = worst.max(diff / bound(c));
                    within(g, c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL)
                };
                if !passes && outside.len() < 16 {
                    outside.push(format!("param @{name} [{i}]: gpu {g} cpu {c}"));
                }
            }
        }
        eprintln!("final parameters: {elements} elements, worst ratio to the bound {worst:.3}");
        assert_eq!(elements, 396_544);
        assert!(outside.is_empty(), "{outside:#?}");
    }
}
