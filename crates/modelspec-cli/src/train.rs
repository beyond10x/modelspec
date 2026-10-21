//! `modelspec train <file> --backend cpu`: compile a `.mspec` file for training and run its SGD
//! steps on the CPU interpreter with M0 synthetic data (DESIGN.md § 6, § 8, § 9 gate 4).
//!
//! The run takes its steps, learning rate and seed from the file's `train` block; the backend
//! comes from `--backend`, not from the block's `device`. Stdout is, one line each:
//!
//! - `init @<path>: <type> = [v, ...]` for every parameter before the first step;
//! - `step <k> loss <v>` at step 0, every 50 steps and the last step, where step `k` is the loss
//!   after `k` updates;
//! - `param @<path>: <type> = [v, ...]` for every parameter after the last step.
//!
//! Parameters are listed by canonical path, values in row-major order, each `f32` in its shortest
//! round-trip decimal form. `--backend vulkan` exits 1 with [`E_TARGET_001`] before the file is
//! read: M0 has no Vulkan backend. A file that does not compile, or data the file's declarations
//! cannot have, prints its diagnostics to stderr and exits 1; a file that cannot be read, a run
//! that fails, or a run that cannot be written to stdout (a closed pipe, a full device) is
//! reported on one line and exits 1.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write;
use std::path::PathBuf;

use modelspec_core::compile::{self, CompiledTraining};
use modelspec_core::diag::{Code, Diagnostic};
use modelspec_core::ir::tensor::ParamId;
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::data::DataError;
use modelspec_runtime::train::runner::{self, Run, RunError};

/// The selected backend is not available in this build.
pub const E_TARGET_001: Code = Code::new("E-TARGET-001");

/// A loss line is printed every this many steps, besides the last.
const LOSS_EVERY: u64 = 50;

/// Arguments of `modelspec train`.
#[derive(clap::Args)]
pub struct Args {
    /// The .mspec file to train.
    pub path: PathBuf,
    /// The backend that runs the training step.
    #[arg(long, value_enum)]
    pub backend: Backend,
}

/// A backend `modelspec train` can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    /// The CPU interpreter.
    Cpu,
    /// Vulkan compute: not available in M0.
    Vulkan,
}

/// Runs `modelspec train` and returns the process exit code.
pub fn run(args: Args, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    if args.backend == Backend::Vulkan {
        let _ = writeln!(
            stderr,
            "{}",
            Diagnostic::new(E_TARGET_001, "the vulkan backend is not available in M0")
                .with("backend", "vulkan")
                .with("available", "cpu")
        );
        return crate::EXIT_FAILURE;
    }
    let file = args.path.display().to_string();
    let source = match std::fs::read_to_string(&args.path) {
        Ok(source) => source,
        Err(error) => {
            let _ = writeln!(stderr, "modelspec train {file}: {error}");
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
    let run = match runner::run(&compiled) {
        Ok(run) => run,
        Err(RunError::Data(DataError::Refused(diagnostic))) => {
            let _ = writeln!(stderr, "{diagnostic}");
            return crate::EXIT_FAILURE;
        }
        Err(error) => {
            let _ = writeln!(stderr, "modelspec train {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    match stdout
        .write_all(report(&compiled, &run).as_bytes())
        .and_then(|()| stdout.flush())
    {
        Ok(()) => crate::EXIT_SUCCESS,
        Err(error) => {
            let _ = writeln!(
                stderr,
                "modelspec train {file}: cannot write the run: {error}"
            );
            crate::EXIT_FAILURE
        }
    }
}

/// The stdout of a finished run.
fn report(compiled: &CompiledTraining, run: &Run) -> String {
    let params: BTreeMap<&str, ParamId> = compiled
        .forward
        .program
        .params
        .iter()
        .map(|(&id, decl)| (decl.path.as_str(), id))
        .collect();
    let mut out = String::new();
    let tensors = |out: &mut String, label: &str, values: &BTreeMap<ParamId, HostTensor<f32>>| {
        for (&path, id) in &params {
            let ty = &compiled.forward.program.params[id].ty;
            let _ = writeln!(out, "{label} @{path}: {ty} = [{}]", list(values[id].data()));
        }
    };
    tensors(&mut out, "init", &run.initial);
    let last = run.losses.len() - 1;
    for (k, loss) in run.losses.iter().enumerate() {
        if (k as u64).is_multiple_of(LOSS_EVERY) || k == last {
            let _ = writeln!(out, "step {k} loss {loss}");
        }
    }
    tensors(&mut out, "param", &run.state.values);
    out
}

fn list(values: &[f32]) -> String {
    values
        .iter()
        .map(f32::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
