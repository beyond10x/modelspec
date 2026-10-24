//! `modelspec train --backend cpu` on a file with an `i32` input (DESIGN.md § 8.1;
//! story:bert-embedding-lookup): M0 synthetic inputs are standard normal `f32` draws, so the run
//! is refused with `E-DATA-004` at the declaration before anything runs.

use std::path::{Path, PathBuf};
use std::process::Command;

const FIXTURE: &str = "crates/modelspec-cli/tests/fixtures/embedding_ids.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn train_on_synthetic_data_refuses_an_i32_input_with_e_data_004() {
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["train", FIXTURE, "--backend", "cpu"])
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    let stderr = String::from_utf8(output.stderr).expect("stderr is UTF-8");
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(
        output.stdout.is_empty(),
        "nothing runs: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(
        stderr,
        format!(
            "E-DATA-004: M0 synthetic data cannot supply the i32 input `ids`\n  \
             {FIXTURE}:11:5\n  \
             input: Tensor<i32, [4]>\n"
        )
    );
}
