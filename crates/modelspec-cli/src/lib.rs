//! The `modelspec` command-line interface (DESIGN.md § 8).
//!
//! Exit codes: 0 success; 1 diagnostics or a runtime error; 2 usage error (clap's own exit code
//! for a usage error). The clap definition, [`Cli`], is public so that `modelspec-docs` can
//! generate the CLI reference from it.

pub mod check;
pub mod inspect_device;
pub mod ir;
pub mod train;

use std::io::Write;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

/// The process exits 0 on success.
pub const EXIT_SUCCESS: u8 = 0;
/// The process exits 1 when it reports diagnostics or a runtime error.
pub const EXIT_FAILURE: u8 = 1;
/// The process exits 2 on a usage error.
pub const EXIT_USAGE: u8 = 2;

/// The `modelspec` command line.
#[derive(Parser)]
#[command(
    name = "modelspec",
    version,
    about = "Compile, check and train .mspec models"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// A `modelspec` subcommand.
#[derive(Subcommand)]
pub enum Command {
    /// Parse and type-check a `.mspec` file.
    Check(check::Args),
    /// Print the profile and negotiation verdict of each Vulkan physical device.
    InspectDevice(inspect_device::Args),
    /// Print the IR of a `.mspec` file at a stage.
    Ir(ir::Args),
    /// Train a `.mspec` model.
    Train(train::Args),
}

/// Parses the process arguments, runs the subcommand and returns the process exit code.
pub fn main() -> ExitCode {
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
    ExitCode::from(dispatch(
        cli.command,
        &mut std::io::stdout().lock(),
        &mut std::io::stderr().lock(),
    ))
}

/// Runs a parsed subcommand, writing what it prints to `stdout` and `stderr`, and returns the
/// process exit code. `modelspec-docs` records the example pages through it.
pub fn dispatch(command: Command, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    match command {
        Command::Check(args) => check::run(args, stderr),
        Command::InspectDevice(args) => inspect_device::run(args, stdout, stderr),
        Command::Ir(args) => ir::run(args, stdout, stderr),
        Command::Train(args) => train::run(args, stdout, stderr),
    }
}
