//! The `modelspec` command-line interface (DESIGN.md § 8).
//!
//! Exit codes: 0 success; 1 diagnostics or a runtime error; 2 usage error (clap's own exit code
//! for a usage error).

mod check;
mod ir;
mod train;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

/// The process exits 0 on success.
pub const EXIT_SUCCESS: u8 = 0;
/// The process exits 1 when it reports diagnostics or a runtime error.
pub const EXIT_FAILURE: u8 = 1;
/// The process exits 2 on a usage error.
pub const EXIT_USAGE: u8 = 2;

#[derive(Parser)]
#[command(
    name = "modelspec",
    version,
    about = "Compile, check and train .mspec models"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parse and type-check a `.mspec` file.
    Check(check::Args),
    /// Print the IR of a `.mspec` file at a stage.
    Ir(ir::Args),
    /// Train a `.mspec` model.
    Train(train::Args),
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let _ = error.print();
            return ExitCode::from(if error.use_stderr() {
                EXIT_USAGE
            } else {
                EXIT_SUCCESS
            });
        }
    };
    let code = match cli.command {
        Command::Check(args) => check::run(args),
        Command::Ir(args) => ir::run(args),
        Command::Train(args) => train::run(args),
    };
    ExitCode::from(code)
}
