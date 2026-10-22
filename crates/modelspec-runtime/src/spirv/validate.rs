//! `spirv-val` on one module (DESIGN.md § 9 gate 6).
//!
//! The validator is the external `spirv-val` of SPIRV-Tools, run with `--target-env`
//! [`TARGET_ENV`] on the module's bytes. A validator that cannot be started is an error, never a
//! pass.

use std::ffi::OsStr;
use std::io::Write;
use std::process::{Command, Stdio};

/// The environment every module is validated for: the Vulkan 1.2 the backend requires.
pub const TARGET_ENV: &str = "vulkan1.2";

/// The validator [`validate`] runs, found on `PATH`.
pub const SPIRV_VAL: &str = "spirv-val";

/// Why a module did not pass.
#[derive(Debug, thiserror::Error)]
pub enum ValidateError {
    /// The validator could not be started or did not run to an exit status.
    #[error("{program} could not run: {error}")]
    Unavailable {
        program: String,
        error: std::io::Error,
    },
    /// The validator ran and refused the module.
    #[error("spirv-val --target-env {TARGET_ENV} refused the module: {output}")]
    Invalid { output: String },
}

/// Runs [`SPIRV_VAL`] on `words`.
pub fn validate(words: &[u32]) -> Result<(), ValidateError> {
    validate_with(SPIRV_VAL.as_ref(), words)
}

/// Runs `program` as `spirv-val --target-env vulkan1.2 -` with the module's bytes on stdin.
pub fn validate_with(program: &OsStr, words: &[u32]) -> Result<(), ValidateError> {
    let unavailable = |error| ValidateError::Unavailable {
        program: program.to_string_lossy().into_owned(),
        error,
    };
    let mut child = Command::new(program)
        .args(["--target-env", TARGET_ENV, "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(unavailable)?;
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut stdin = child.stdin.take().expect("stdin is piped");
    let written = stdin.write_all(&bytes);
    drop(stdin);
    let output = child.wait_with_output().map_err(unavailable)?;
    if output.status.success() {
        written.map_err(unavailable)?;
        return Ok(());
    }
    let mut text = String::from_utf8_lossy(&output.stderr).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    if text.trim().is_empty() {
        text = format!("exit status {}", output.status);
    }
    Err(ValidateError::Invalid { output: text })
}
