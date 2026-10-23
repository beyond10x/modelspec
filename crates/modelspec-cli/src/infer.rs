//! `modelspec infer <file> --backend cpu|vulkan --checkpoint <path>`: the forward program of a
//! compiled file with a checkpoint's parameters (DESIGN.md § 6.2, § 8).
//!
//! The checkpoint is read and matched against the compiled file (paths, dtypes, shapes and schema
//! hash; not the optimizer) and refused with its `E-CKPT` code otherwise. The forward program runs
//! on the synthetic batch of the file's `train` block; M1 takes no `--input` (DESIGN.md § 12,
//! "Data interface"). Stdout is two lines:
//!
//! - `prediction: <type> = [v, ...]`, the model output in row-major order;
//! - `loss <v>`, the loss of that batch, which for the checkpoint a run wrote at its last step is
//!   that run's last `step` loss.
//!
//! `--backend cpu` runs it on the CPU interpreter. `--backend vulkan` compiles the file for
//! Vulkan, reads the checkpoint, then opens the loader and runs the steps of the step plan up to
//! the loss once on device 0, or on the device `--device <index>` names, with one host wait
//! ([`modelspec_runtime::vulkan::runner::infer`]); a checkpoint that cannot be used is therefore
//! refused with its `E-CKPT` code before the loader is opened, and then a missing loader exits 1
//! with `E-TARGET-002`, no device or a `--device` that names none with `E-TARGET-003`, and a
//! device below the requirements with `E-TARGET-004`. A build without the `vulkan` feature
//! refuses `--backend vulkan` with [`E_TARGET_001`] before the file is read. `--device` with
//! `--backend cpu` is a usage error (exit 2). A file that does not compile, or data the file's
//! declarations cannot have, prints its diagnostics and exits 1.

use std::io::Write;
use std::path::PathBuf;

use modelspec_core::compile;
#[cfg(not(feature = "vulkan"))]
use modelspec_core::diag::Diagnostic;
use modelspec_runtime::cpu::evaluate_all;
use modelspec_runtime::train::checkpoint;
use modelspec_runtime::train::data::{self, DataError};

use crate::train::Backend;
#[cfg(not(feature = "vulkan"))]
use crate::train::E_TARGET_001;

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
    /// With --backend vulkan, run on the physical device INDEX, in the order the Vulkan loader
    /// lists them, instead of device 0.
    #[arg(long, value_name = "INDEX")]
    pub device: Option<usize>,
}

/// Runs `modelspec infer` and returns the process exit code.
pub fn run(args: Args, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    if args.backend == Backend::Cpu && args.device.is_some() {
        let _ = writeln!(
            stderr,
            "error: the argument '--device' cannot be used with '--backend cpu'; it is \
             available only with '--backend vulkan'"
        );
        return crate::EXIT_USAGE;
    }
    #[cfg(not(feature = "vulkan"))]
    if args.backend == Backend::Vulkan {
        let _ = writeln!(
            stderr,
            "{}",
            Diagnostic::new(
                E_TARGET_001,
                "the vulkan backend is not available in this build"
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
    #[cfg(feature = "vulkan")]
    let artifact = match args.backend {
        Backend::Cpu => None,
        Backend::Vulkan => {
            use modelspec_core::artifact::Backend as Target;
            match compile::compile_artifact(&file, &source, Target::Vulkan) {
                Ok(artifact) => Some(artifact),
                Err(diagnostics) => {
                    for diagnostic in &diagnostics {
                        let _ = writeln!(stderr, "{diagnostic}");
                    }
                    return crate::EXIT_FAILURE;
                }
            }
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
    let Some(node) = forward.program.node(forward.prediction) else {
        let _ = writeln!(
            stderr,
            "modelspec infer {file}: the forward program defines no prediction value"
        );
        return crate::EXIT_FAILURE;
    };
    #[cfg(feature = "vulkan")]
    if let Some(artifact) = &artifact {
        use modelspec_runtime::vulkan::runner::{DeviceRunError, infer_on_device};
        return match infer_on_device(args.device, &compiled, artifact, &params) {
            Ok((inference, _)) => print(
                &file,
                &node.ty.to_string(),
                inference.prediction.data(),
                inference.loss,
                stdout,
                stderr,
            ),
            Err(DeviceRunError::Refused(diagnostic)) => {
                let _ = writeln!(stderr, "{diagnostic}");
                crate::EXIT_FAILURE
            }
            Err(DeviceRunError::Failed(error)) => {
                let _ = writeln!(stderr, "modelspec infer {file}: {error}");
                crate::EXIT_FAILURE
            }
        };
    }
    let mut values = match evaluate_all(&forward.program, &batch.inputs, &params) {
        Ok(values) => values,
        Err(error) => {
            let _ = writeln!(stderr, "modelspec infer {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    let Some(prediction) = values.values.remove(&forward.prediction) else {
        let _ = writeln!(
            stderr,
            "modelspec infer {file}: the forward program defines no prediction value"
        );
        return crate::EXIT_FAILURE;
    };
    let loss = values.outputs[0].data()[0];
    print(
        &file,
        &node.ty.to_string(),
        prediction.data(),
        loss,
        stdout,
        stderr,
    )
}

/// Prints the `prediction` and `loss` lines and returns the exit code.
fn print(
    file: &str,
    ty: &str,
    prediction: &[f32],
    loss: f32,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> u8 {
    let list: Vec<String> = prediction.iter().map(f32::to_string).collect();
    let printed = format!("prediction: {ty} = [{}]\nloss {loss}\n", list.join(", "));
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
