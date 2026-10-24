//! `story:fast-math-mode`, acceptances `fast-math-tolerance` and the run-record half of
//! `numerics-class-follows-fast-math` (DESIGN.md § 5.4, § 9 gate 5, § 12 ADR-008): `train
//! --backend vulkan --fast-math` on `examples/mnist_mlp.mspec` and on the `mlp_sweep` source at
//! batch 8192, hidden 4096 passes against `train --backend cpu` within ADR-008's tolerance, two
//! runs of it print the same bytes, and its `--stats` line names the class `tolerance`.
//!
//! Every printed loss `g` passes against the CPU loss `c` when `|g - c| <= GPU_ATOL + GPU_RTOL *
//! c`, the `mse` reference `r` being the loss itself (every term is a square); every parameter
//! after the last step passes `||g - c||_2 <= 1e-4 * ||c||_2`. Values print in their shortest
//! round-trip form, so equal stdout is equal bits. Each test prints the largest share of its
//! tolerance any loss uses and the largest L2 ratio of any parameter.
//!
//! The tests are ignored by default because they create a Vulkan instance, which `task ci` never
//! does; run them with
//! `cargo test --release -p modelspec-cli --test train_fast_math -- --ignored --nocapture`.
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const GPU_ATOL: f64 = 1e-5;
const GPU_RTOL: f64 = 1e-4;
const RUN_L2_RTOL: f64 = 1e-4;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(root())
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The `step <k> loss <v>` lines of `stdout`, by step.
fn losses(stdout: &str) -> BTreeMap<u64, f32> {
    stdout
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("step ")?;
            let (k, v) = rest.split_once(" loss ")?;
            Some((k.parse().ok()?, v.parse().ok()?))
        })
        .collect()
}

/// The `param @<path>: <type> = [v, ...]` lines of `stdout`, by path.
fn params(stdout: &str) -> BTreeMap<String, Vec<f32>> {
    stdout
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("param ")?;
            let (head, values) = rest.split_once(" = [")?;
            let values = values.strip_suffix(']')?;
            let values = values
                .split(", ")
                .map(|v| v.parse().expect("an f32"))
                .collect();
            Some((head.to_owned(), values))
        })
        .collect()
}

/// The two `--fast-math` runs and the CPU run of `file` with `extra` arguments agree as the module
/// documentation says; `name` labels the printed figures.
fn assert_fast_math_run(name: &str, file: &str, extra: &[&str]) {
    let mut gpu_args = vec![
        "train",
        file,
        "--backend",
        "vulkan",
        "--fast-math",
        "--stats",
    ];
    gpu_args.extend_from_slice(extra);
    let first = modelspec(&gpu_args);
    assert_eq!(first.status.code(), Some(0), "{}", text(&first.stderr));
    let second = modelspec(&gpu_args);
    assert_eq!(second.status.code(), Some(0), "{}", text(&second.stderr));
    let mut cpu_args = vec!["train", file, "--backend", "cpu"];
    cpu_args.extend_from_slice(extra);
    let cpu = modelspec(&cpu_args);
    assert_eq!(cpu.status.code(), Some(0), "{}", text(&cpu.stderr));

    let stats = text(&first.stderr);
    assert!(
        stats.lines().any(|l| l == "numerics class: tolerance"),
        "{name}: the run record names the class tolerance:\n{stats}"
    );
    let (gpu_out, cpu_out) = (text(&first.stdout), text(&cpu.stdout));
    let same_bits = first.stdout == second.stdout;
    eprintln!("{name}: two fast_math runs print the same bytes: {same_bits}");
    assert!(
        same_bits,
        "{name}: two runs of one fast_math artifact give the same bits"
    );

    let (g_losses, c_losses) = (losses(&gpu_out), losses(&cpu_out));
    assert!(!c_losses.is_empty(), "{name}: the CPU run prints losses");
    assert_eq!(
        g_losses.keys().collect::<Vec<_>>(),
        c_losses.keys().collect::<Vec<_>>(),
        "{name}: the same step lines"
    );
    let mut worst_loss = (0.0f64, 0u64);
    let mut outside = Vec::new();
    let mut differ = 0;
    for (k, &c) in &c_losses {
        let g = g_losses[k];
        if g.to_bits() != c.to_bits() {
            differ += 1;
        }
        let bound = GPU_ATOL + GPU_RTOL * f64::from(c).abs();
        let share = (f64::from(g) - f64::from(c)).abs() / bound;
        if share > worst_loss.0 {
            worst_loss = (share, *k);
        }
        if share > 1.0 {
            outside.push((*k, g, c));
        }
    }
    eprintln!(
        "{name}: {} printed losses, {differ} differ from the CPU in bits; largest share of the \
         loss tolerance {:.6} at step {}",
        c_losses.len(),
        worst_loss.0,
        worst_loss.1
    );
    assert!(
        outside.is_empty(),
        "{name}: losses outside |g - c| <= {GPU_ATOL:e} + {GPU_RTOL:e} * c: {outside:?}"
    );

    let (g_params, c_params) = (params(&gpu_out), params(&cpu_out));
    assert!(
        !c_params.is_empty(),
        "{name}: the CPU run prints parameters"
    );
    assert_eq!(
        g_params.keys().collect::<Vec<_>>(),
        c_params.keys().collect::<Vec<_>>(),
        "{name}: the same parameters"
    );
    for (path, c) in &c_params {
        let g = &g_params[path];
        assert_eq!(g.len(), c.len(), "{name}: {path}");
        let difference: f64 = g
            .iter()
            .zip(c)
            .map(|(&g, &c)| (f64::from(g) - f64::from(c)).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm: f64 = c.iter().map(|&c| f64::from(c).powi(2)).sum::<f64>().sqrt();
        let differ = g
            .iter()
            .zip(c)
            .filter(|(g, c)| g.to_bits() != c.to_bits())
            .count();
        let ratio = difference / norm;
        eprintln!(
            "{name}: {path}: ||g - c||_2 / ||c||_2 = {ratio:.3e} (bound {RUN_L2_RTOL:e}), {differ} \
             of {} values differ from the CPU in bits",
            c.len()
        );
        assert!(
            ratio <= RUN_L2_RTOL,
            "{name}: {path} is outside the whole-run L2 bound: {ratio:e}"
        );
    }
}

#[test]
#[ignore = "needs a Vulkan device"]
fn mnist_mlp_with_fast_math_passes_against_the_cpu_and_repeats_its_bits() {
    assert_fast_math_run("mnist_mlp", "examples/mnist_mlp.mspec", &[]);
}

/// The steps of the batch 8192, hidden 4096 run: the CPU oracle takes seconds per step there.
const SWEEP_STEPS: &str = "10";

/// The source `benches/mlp_sweep/sweep.rs` (`Point::source`) writes at batch 8192, hidden 4096.
const SWEEP_B8192_H4096: &str =
    "// The MLP size sweep at batch 8192, hidden 4096 (benches/mlp_sweep.rs).
model SweepMlp {
    param w1: Tensor<f32, [784, 4096]> = xavier_uniform();
    param w2: Tensor<f32, [4096, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8192, 784]>)
        -> Tensor<f32, [8192, 10]> {
        return matmul(relu(matmul(x, w1)), w2);
    }
}

train Fit {
    model = SweepMlp;
    input x: Tensor<f32, [8192, 784]>;
    target y: Tensor<f32, [8192, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 220;
    seed = 42;
    device = vulkan(0);
}
";

#[test]
#[ignore = "needs a Vulkan device"]
fn the_sweep_at_batch_8192_hidden_4096_with_fast_math_passes_against_the_cpu() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("train_fast_math");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let file = dir.join("mlp_sweep_b8192_h4096.mspec");
    std::fs::write(&file, SWEEP_B8192_H4096).expect("the source is written");
    let shown = file.display().to_string();
    assert_fast_math_run("sweep b8192 h4096", &shown, &["--steps", SWEEP_STEPS]);
}
