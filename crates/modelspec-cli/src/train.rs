//! `modelspec train`: train a `.mspec` model.

use std::path::PathBuf;

/// Arguments of `modelspec train`.
#[derive(clap::Args)]
pub struct Args {
    /// The .mspec file to train.
    pub path: PathBuf,
}

/// Runs `modelspec train` and returns the process exit code.
pub fn run(args: Args) -> u8 {
    eprintln!(
        "modelspec train {}: {}",
        args.path.display(),
        crate::NOT_IMPLEMENTED
    );
    crate::EXIT_FAILURE
}
