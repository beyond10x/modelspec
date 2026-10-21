//! `modelspec ir`: print the IR of a `.mspec` file.

use std::path::PathBuf;

/// Arguments of `modelspec ir`.
#[derive(clap::Args)]
pub struct Args {
    /// The .mspec file to lower.
    pub path: PathBuf,
}

/// Runs `modelspec ir` and returns the process exit code.
pub fn run(args: Args) -> u8 {
    eprintln!(
        "modelspec ir {}: {}",
        args.path.display(),
        crate::NOT_IMPLEMENTED
    );
    crate::EXIT_FAILURE
}
