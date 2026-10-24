//! `modelspec data`: the training data of a masked-language-model run (story:bert-corpus-splits).
//!
//! `data prepare <corpus-dir> --out <dir>` reads the WikiText-103 raw release's `wiki.train.raw`
//! and `wiki.valid.raw` from local files and writes the `Train` and `HeldOut` splits and their
//! record `dataset.json` into `<dir>` (`spec/domains/data.yaml`, `modelspec.data.Dataset`). It
//! prints the dataset id and one line per split. A path that cannot be read or written prints
//! `modelspec data prepare <path>: <error>` and exits 1.

use std::io::Write;
use std::path::PathBuf;

use clap::Subcommand;
use modelspec_runtime::dataset;

/// Arguments of `modelspec data`.
#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub command: DataCommand,
}

/// A `modelspec data` subcommand.
#[derive(Subcommand)]
pub enum DataCommand {
    /// Copy a WikiText-103 raw release's wiki.train.raw and wiki.valid.raw into the Train and
    /// HeldOut splits and write their record, dataset.json.
    Prepare(PrepareArgs),
}

/// Arguments of `modelspec data prepare`.
#[derive(clap::Args)]
pub struct PrepareArgs {
    /// The directory holding the release's wiki.train.raw and wiki.valid.raw; nothing is
    /// downloaded.
    pub corpus: PathBuf,
    /// The directory to write train.raw, heldout.raw and dataset.json into; created when missing.
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
}

/// Runs `modelspec data` and returns the process exit code.
pub fn run(args: Args, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    match args.command {
        DataCommand::Prepare(args) => prepare(args, stdout, stderr),
    }
}

fn prepare(args: PrepareArgs, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    match dataset::prepare(&args.corpus, &args.out) {
        Ok(record) => {
            let _ = writeln!(stdout, "dataset {} ({})", record.dataset_id, record.name);
            for split in &record.splits {
                let _ = writeln!(
                    stdout,
                    "{:?} {} documents, {} bytes, blake3 {}",
                    split.role, split.document_count, split.byte_count, split.content_hash
                );
            }
            crate::EXIT_SUCCESS
        }
        Err(error) => {
            let _ = writeln!(stderr, "modelspec data prepare {error}");
            crate::EXIT_FAILURE
        }
    }
}
