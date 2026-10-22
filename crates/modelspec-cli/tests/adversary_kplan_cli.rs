//! Adversary cases for `modelspec ir --stage kernel` (story m1-kernel-plan): a file `check` and
//! `train --backend cpu` accept is not reported as a compiler bug by the kernel stage.

use std::path::PathBuf;
use std::process::{Command, Output};

const LITERAL_FORWARD: &str = "model Constant {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, []> {
        return 1.0;
    }
}

train Fit {
    model = Constant;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, []>;
    loss = mean(mul(model.forward(x), y));
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 42;
    device = cpu;
}
";

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

#[test]
fn a_checked_file_with_a_literal_forward_is_not_a_compiler_bug_at_the_kernel_stage() {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("adversary_kplan_literal.mspec");
    std::fs::write(&path, LITERAL_FORWARD).expect("the fixture is written");
    let file = path.to_str().expect("a UTF-8 path");

    let check = modelspec(&["check", file]);
    assert_eq!(check.status.code(), Some(0), "{}", text(&check.stderr));
    let training = modelspec(&["ir", file, "--stage", "training"]);
    assert_eq!(
        training.status.code(),
        Some(0),
        "{}",
        text(&training.stderr)
    );

    let kernel = modelspec(&["ir", file, "--stage", "kernel"]);
    let stderr = text(&kernel.stderr);
    assert!(!stderr.contains("E-INTERNAL-002"), "{stderr}");
    assert_eq!(kernel.status.code(), Some(0), "{stderr}");
}
