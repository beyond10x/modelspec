//! `modelspec ir`: print the IR of a `.mspec` file at a stage (DESIGN.md § 8).
//!
//! `--stage tensor` prints the verified Tensor IR with `ir/print.rs` to stdout and exits 0. A file
//! that does not compile prints its diagnostics to stderr as `modelspec check` does and exits 1;
//! a file that cannot be read, or IR that cannot be written to stdout (a closed pipe, a full
//! device), is reported on one line and exits 1. Any other stage is a usage
//! error (exit 2): `tensor` is the only stage this build has.

use std::io::Write;
use std::path::PathBuf;

use modelspec_core::compile;
use modelspec_core::ir::print::print;

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
}

/// Runs `modelspec ir` and returns the process exit code.
pub fn run(args: Args) -> u8 {
    let file = args.path.display().to_string();
    let source = match std::fs::read_to_string(&args.path) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("modelspec ir {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    match args.stage {
        Stage::Tensor => match compile::compile_tensor(&file, &source) {
            Ok(lowered) => {
                let mut stdout = std::io::stdout().lock();
                match stdout
                    .write_all(print(&lowered.program).as_bytes())
                    .and_then(|()| stdout.flush())
                {
                    Ok(()) => crate::EXIT_SUCCESS,
                    Err(error) => {
                        let _ = writeln!(
                            std::io::stderr(),
                            "modelspec ir {file}: cannot write the IR: {error}"
                        );
                        crate::EXIT_FAILURE
                    }
                }
            }
            Err(diagnostics) => {
                for diagnostic in &diagnostics {
                    eprintln!("{diagnostic}");
                }
                crate::EXIT_FAILURE
            }
        },
    }
}
