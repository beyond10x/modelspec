//! `modelspec data`: the training data of a masked-language-model run (story:bert-corpus-splits).
//!
//! `data prepare <corpus-dir> --out <dir>` reads the WikiText-103 raw release's `wiki.train.raw`
//! and `wiki.valid.raw` from local files and writes the `Train` and `HeldOut` splits and their
//! record `dataset.json` into `<dir>` (`spec/domains/data.yaml`, `modelspec.data.Dataset`). It
//! prints the dataset id and one line per split. A path that cannot be read or written prints
//! `modelspec data prepare <path>: <error>` and exits 1.
//!
//! `data tokenizer --dataset <dir> --vocab-size <n> --out <file>` trains the WordPiece tokenizer
//! on the `Train` split of a dataset `data prepare` wrote, to exactly `n` tokens, and writes its
//! record to `<file>` (`modelspec.data.Tokenizer`; story:bert-tokenizer). It prints the tokenizer
//! id and the token count. A path that cannot be read or written, or a split too small to yield
//! `n` tokens, prints `modelspec data tokenizer <path>: <error>` and exits 1.

use std::io::Write;
use std::path::PathBuf;

use clap::Subcommand;
use modelspec_runtime::dataset::{self, tokenizer};

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
    /// Train the WordPiece tokenizer on the Train split of a prepared dataset to an exact
    /// vocabulary size and write its record.
    Tokenizer(TokenizerArgs),
}

/// Arguments of `modelspec data tokenizer`.
#[derive(clap::Args)]
pub struct TokenizerArgs {
    /// The dataset directory `data prepare` wrote; only its train.raw is read.
    #[arg(long, value_name = "DIR")]
    pub dataset: PathBuf,
    /// The number of tokens of the vocabulary, the five special tokens included.
    #[arg(long, value_name = "N")]
    pub vocab_size: usize,
    /// The file to write the tokenizer record to, as JSON.
    #[arg(long, value_name = "FILE")]
    pub out: PathBuf,
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
        DataCommand::Tokenizer(args) => train_tokenizer(args, stdout, stderr),
    }
}

fn train_tokenizer(args: TokenizerArgs, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    let result = tokenizer::train(&args.dataset, args.vocab_size)
        .and_then(|record| tokenizer::write(&record, &args.out).map(|()| record));
    match result {
        Ok(record) => {
            let _ = writeln!(
                stdout,
                "tokenizer {} ({} tokens)",
                record.tokenizer_id,
                record.vocabulary.tokens.len()
            );
            crate::EXIT_SUCCESS
        }
        Err(error) => {
            let _ = writeln!(stderr, "modelspec data tokenizer {error}");
            crate::EXIT_FAILURE
        }
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
