//! `modelspec check`: parse and type-check a `.mspec` file.

use std::path::PathBuf;

/// Arguments of `modelspec check`.
#[derive(clap::Args)]
pub struct Args {
    /// The .mspec file to check.
    pub path: PathBuf,
}

/// Runs `modelspec check` and returns the process exit code.
pub fn run(args: Args) -> u8 {
    eprintln!(
        "modelspec check {}: {}",
        args.path.display(),
        crate::NOT_IMPLEMENTED
    );
    crate::EXIT_FAILURE
}
