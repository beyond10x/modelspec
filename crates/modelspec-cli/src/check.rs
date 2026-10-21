//! `modelspec check`: parse and type-check a `.mspec` file.
//!
//! A clean file prints nothing and exits 0. Otherwise every diagnostic is printed to stderr in
//! the layout of DESIGN.md § 8.1, in source order, and the exit code is 1. A file that cannot be
//! read is reported on one line and exits 1.

use std::io::Write;
use std::path::PathBuf;

use modelspec_core::types;

/// Arguments of `modelspec check`.
#[derive(clap::Args)]
pub struct Args {
    /// The .mspec file to check.
    pub path: PathBuf,
}

/// Runs `modelspec check` and returns the process exit code.
pub fn run(args: Args, stderr: &mut dyn Write) -> u8 {
    let file = args.path.display().to_string();
    let source = match std::fs::read_to_string(&args.path) {
        Ok(source) => source,
        Err(error) => {
            let _ = writeln!(stderr, "modelspec check {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    match types::check_source(&file, &source) {
        Ok(_) => crate::EXIT_SUCCESS,
        Err(diagnostics) => {
            for diagnostic in &diagnostics {
                let _ = writeln!(stderr, "{diagnostic}");
            }
            crate::EXIT_FAILURE
        }
    }
}
