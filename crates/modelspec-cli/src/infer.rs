//! `modelspec infer <file> --backend cpu --checkpoint <path>`: the forward program of a compiled
//! file with a checkpoint's parameters (DESIGN.md § 6.2, § 8).
//!
//! The checkpoint is read and matched against the compiled file (paths, dtypes, shapes and schema
//! hash; not the optimizer) and refused with its `E-CKPT` code otherwise. The forward program runs
//! on the CPU interpreter on the synthetic batch of the file's `train` block; M1 takes no
//! `--input` (DESIGN.md § 12, "Data interface"). Stdout is two lines:
//!
//! - `prediction: <type> = [v, ...]`, the model output in row-major order;
//! - `loss <v>`, the loss of that batch, which for the checkpoint a run wrote at its last step is
//!   that run's last `step` loss.
//!
//! `--backend vulkan` exits 1 with [`E_TARGET_001`] before the file is read. A file that does not
//! compile, or data the file's declarations cannot have, prints its diagnostics and exits 1.

use std::io::Write;
use std::path::PathBuf;

use modelspec_core::compile;
use modelspec_core::diag::Diagnostic;
use modelspec_runtime::cpu::evaluate_all;
use modelspec_runtime::train::checkpoint;
use modelspec_runtime::train::data::{self, DataError};

use crate::train::{Backend, E_TARGET_001};

/// Arguments of `modelspec infer`.
#[derive(clap::Args)]
pub struct Args {
    /// The .mspec file whose forward program runs.
    pub path: PathBuf,
    /// The backend that runs the forward program.
    #[arg(long, value_enum)]
    pub backend: Backend,
    /// The .msckpt file whose parameters the forward program reads.
    #[arg(long, value_name = "PATH")]
    pub checkpoint: PathBuf,
}

/// Runs `modelspec infer` and returns the process exit code.
pub fn run(args: Args, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    if args.backend == Backend::Vulkan {
        let _ = writeln!(
            stderr,
            "{}",
            Diagnostic::new(
                E_TARGET_001,
                "the vulkan backend is not available for infer"
            )
            .with("backend", "vulkan")
            .with("available", "cpu")
        );
        return crate::EXIT_FAILURE;
    }
    let file = args.path.display().to_string();
    let source = match std::fs::read_to_string(&args.path) {
        Ok(source) => source,
        Err(error) => {
            let _ = writeln!(stderr, "modelspec infer {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    let compiled = match compile::compile_training(&file, &source) {
        Ok(compiled) => compiled,
        Err(diagnostics) => {
            for diagnostic in &diagnostics {
                let _ = writeln!(stderr, "{diagnostic}");
            }
            return crate::EXIT_FAILURE;
        }
    };
    let params = match checkpoint::read(&args.checkpoint).and_then(|c| c.parameters_for(&compiled))
    {
        Ok(params) => params,
        Err(error) => {
            let _ = writeln!(stderr, "{error}");
            return crate::EXIT_FAILURE;
        }
    };
    let batch = match data::synthetic(&compiled.config, &compiled.forward) {
        Ok(batch) => batch,
        Err(DataError::Refused(diagnostic)) => {
            let _ = writeln!(stderr, "{diagnostic}");
            return crate::EXIT_FAILURE;
        }
        Err(error) => {
            let _ = writeln!(stderr, "modelspec infer {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    let forward = &compiled.forward;
    let mut values = match evaluate_all(&forward.program, &batch.inputs, &params) {
        Ok(values) => values,
        Err(error) => {
            let _ = writeln!(stderr, "modelspec infer {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    let (Some(prediction), Some(node)) = (
        values.values.remove(&forward.prediction),
        forward.program.node(forward.prediction),
    ) else {
        let _ = writeln!(
            stderr,
            "modelspec infer {file}: the forward program defines no prediction value"
        );
        return crate::EXIT_FAILURE;
    };
    let loss = values.outputs[0].data()[0];
    let list: Vec<String> = prediction.data().iter().map(f32::to_string).collect();
    let printed = format!(
        "prediction: {} = [{}]\nloss {loss}\n",
        node.ty,
        list.join(", ")
    );
    match stdout
        .write_all(printed.as_bytes())
        .and_then(|()| stdout.flush())
    {
        Ok(()) => crate::EXIT_SUCCESS,
        Err(error) => {
            let _ = writeln!(
                stderr,
                "modelspec infer {file}: cannot write the result: {error}"
            );
            crate::EXIT_FAILURE
        }
    }
}
