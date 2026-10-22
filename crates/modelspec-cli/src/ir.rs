//! `modelspec ir`: print the IR of a `.mspec` file at a stage (DESIGN.md § 8).
//!
//! `--stage tensor` prints the verified Tensor IR with `ir/print.rs` to stdout and exits 0;
//! `--stage training` prints the verified training step (forward, loss, backward and the SGD
//! subgraph, with subgraph tags, effect tokens and state versions) the same way; `--stage kernel`
//! prints the Kernel IR of the step's Vulkan artifact, its views, kernels, step plan and forward
//! plan (`print_kernel`). A file that does not compile prints its diagnostics to stderr as
//! `modelspec check` does and exits 1; a file that cannot be read, or IR that cannot be written to
//! stdout (a closed pipe, a full device), is reported on one line and exits 1. Any other stage is
//! a usage error (exit 2): `tensor`, `training` and `kernel` are the stages this build has.

use std::io::Write;
use std::path::PathBuf;

use modelspec_core::artifact::Backend;
use modelspec_core::compile;
use modelspec_core::diag::Diagnostic;
use modelspec_core::ir::print::{print, print_kernel, print_step};

/// Arguments of `modelspec ir`.
#[derive(clap::Args)]
pub struct Args {
    /// The .mspec file to lower.
    pub path: PathBuf,
    /// The IR level to print.
    #[arg(long, value_enum)]
    pub stage: Stage,
}

/// An IR level `modelspec ir` prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Stage {
    /// Tensor IR after lowering and dead-code elimination.
    Tensor,
    /// Training IR: the Tensor IR with its reverse pass and SGD step attached.
    Training,
    /// Kernel IR: the kernels of the training step, its step plan and its forward plan.
    Kernel,
}

/// Runs `modelspec ir` and returns the process exit code.
pub fn run(args: Args, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    let file = args.path.display().to_string();
    let source = match std::fs::read_to_string(&args.path) {
        Ok(source) => source,
        Err(error) => {
            let _ = writeln!(stderr, "modelspec ir {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    let printed: Result<String, Vec<Diagnostic>> = match args.stage {
        Stage::Tensor => {
            compile::compile_tensor(&file, &source).map(|lowered| print(&lowered.program))
        }
        Stage::Training => {
            compile::compile_training(&file, &source).map(|compiled| print_step(&compiled.training))
        }
        Stage::Kernel => compile::compile_artifact(&file, &source, Backend::Vulkan)
            .map(|artifact| print_kernel(&artifact.execution)),
    };
    match printed {
        Ok(ir) => {
            match stdout
                .write_all(ir.as_bytes())
                .and_then(|()| stdout.flush())
            {
                Ok(()) => crate::EXIT_SUCCESS,
                Err(error) => {
                    let _ = writeln!(stderr, "modelspec ir {file}: cannot write the IR: {error}");
                    crate::EXIT_FAILURE
                }
            }
        }
        Err(diagnostics) => {
            for diagnostic in &diagnostics {
                let _ = writeln!(stderr, "{diagnostic}");
            }
            crate::EXIT_FAILURE
        }
    }
}
