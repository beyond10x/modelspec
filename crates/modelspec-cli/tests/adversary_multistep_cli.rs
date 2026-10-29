//! Adversary pass 1 on story:vulkan-stepwise-multi-step-submit, at the command line: the hidden
//! `--submit-steps <N>` at its bounds. 0 and a negative N are usage errors; the largest N is a
//! refusal or a run, never a panic.
//!
//! `cargo test -p modelspec-cli --test adversary_multistep_cli -- --include-ignored`
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A linear regression trained with `adamw`, which runs stepwise on Vulkan.
const ADAMW: &str = "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = adamw(lr = 0.01, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, warmup_steps = 2);
    steps = 5;
    seed = 42;
    device = cpu;
}
";

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adv_multistep_cli_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

fn source(dir: &Path) -> String {
    let file = dir.join("adamw.mspec");
    std::fs::write(&file, ADAMW).expect("written");
    file.to_str().expect("UTF-8").to_owned()
}

fn train(file: &str, submit: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args([
            "train",
            file,
            "--backend",
            "vulkan",
            "--submit-steps",
            submit,
        ])
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// `--submit-steps 0` and `--submit-steps -1` are usage errors, exit 2, before any device.
#[test]
fn ms_cli_submit_steps_zero_and_negative_are_usage_errors() {
    let dir = scratch("usage");
    let file = source(&dir);
    for submit in ["0", "-1"] {
        let output = train(&file, submit);
        assert_eq!(
            output.status.code(),
            Some(2),
            "--submit-steps {submit}: stdout {} stderr {}",
            text(&output.stdout),
            text(&output.stderr)
        );
    }
}

/// MS-3. `--submit-steps 18446744073709551615` (u64::MAX, which the flag accepts) is a refusal
/// or a run, exit 0 or 1, not a panic (exit 101).
#[test]
#[ignore = "device: needs a Vulkan device"]
fn ms_cli_the_largest_submit_steps_is_not_a_panic() {
    let dir = scratch("largest");
    let file = source(&dir);
    let output = train(&file, &u64::MAX.to_string());
    let code = output.status.code();
    assert!(
        matches!(code, Some(0) | Some(1)),
        "exit {code:?}: stdout {} stderr {}",
        text(&output.stdout),
        text(&output.stderr)
    );
}
