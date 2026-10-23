//! `modelspec-ui-types`: writes `ui/src/generated/observatory.ts`, the TypeScript types of the
//! observatory UI, from the Rust types of the stream (`task ui-types`); with `--check` it compares
//! the file instead and exits 1 when it differs or is missing (`task ui-types-check`, in
//! `task ci`).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use modelspec_observatory::{UI_TYPES_PATH, typescript};

/// Write or check the TypeScript types of the observatory UI.
#[derive(Parser)]
#[command(name = "modelspec-ui-types")]
struct Args {
    /// Compare the file with what the Rust types generate and exit 1 when it differs, instead of
    /// writing it.
    #[arg(long)]
    check: bool,
    /// The repository root the file is under.
    #[arg(long, value_name = "DIR", default_value = ".")]
    root: PathBuf,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let path = args.root.join(UI_TYPES_PATH);
    let generated = typescript();
    if args.check {
        return match std::fs::read_to_string(&path) {
            Ok(current) if current == generated => ExitCode::SUCCESS,
            Ok(_) => {
                eprintln!(
                    "modelspec-ui-types: {UI_TYPES_PATH} is not what the Rust types generate; \
                     run `task ui-types`"
                );
                ExitCode::FAILURE
            }
            Err(error) => {
                eprintln!(
                    "modelspec-ui-types: cannot read {UI_TYPES_PATH} under {}: {error}; \
                     run `task ui-types`",
                    args.root.display()
                );
                ExitCode::FAILURE
            }
        };
    }
    let written = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(&path, generated));
    match written {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!(
                "modelspec-ui-types: cannot write {}: {error}",
                path.display()
            );
            ExitCode::FAILURE
        }
    }
}
