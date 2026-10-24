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
//!
//! `data pack --dataset <dir> --tokenizer <file> --seq-len <S> --out <dir>` encodes both splits
//! of such a dataset with such a tokenizer, joins the documents with `[SEP]`, cuts them into
//! sequences of exactly `S` tokens and writes `train.tokens`, `heldout.tokens`, `binding.json`
//! and `packed.json` into `<dir>` (`modelspec.data.PackedDataset` and `TrainingDataBinding`;
//! story:bert-mlm-batches). It prints the packed id, one line per split, the binding id and the
//! held-out unigram baseline. A path that cannot be read or written, a split whose hash differs
//! from its record, or a split shorter than one sequence prints
//! `modelspec data pack <path>: <error>` and exits 1.

use std::io::Write;
use std::path::PathBuf;

use clap::Subcommand;
use modelspec_runtime::dataset::{self, mlm, tokenizer};

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
    /// Encode both splits of a prepared dataset with a tokenizer into sequences of an exact
    /// length and write them, their binding and their record with the held-out unigram baseline.
    Pack(PackArgs),
}

/// Arguments of `modelspec data pack`.
#[derive(clap::Args)]
pub struct PackArgs {
    /// The dataset directory `data prepare` wrote.
    #[arg(long, value_name = "DIR")]
    pub dataset: PathBuf,
    /// The tokenizer record `data tokenizer` wrote.
    #[arg(long, value_name = "FILE")]
    pub tokenizer: PathBuf,
    /// The number of tokens of every sequence; the partial tail of each split is dropped.
    #[arg(long, value_name = "S", value_parser = clap::value_parser!(u64).range(1..))]
    pub seq_len: u64,
    /// The directory to write train.tokens, heldout.tokens, binding.json and packed.json into;
    /// created when missing.
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
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
        DataCommand::Pack(args) => pack(args, stdout, stderr),
    }
}

fn pack(args: PackArgs, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    match mlm::pack(&args.dataset, &args.tokenizer, args.seq_len, &args.out) {
        Ok((binding, packed)) => {
            let _ = writeln!(
                stdout,
                "packed {} (seq_len {})",
                packed.packed_id, packed.seq_len
            );
            for split in &packed.splits {
                let _ = writeln!(
                    stdout,
                    "{:?} {} sequences, blake3 {}",
                    split.role, split.sequence_count, split.content_hash
                );
            }
            let _ = writeln!(
                stdout,
                "binding {} (dataset {}, tokenizer {})",
                binding.binding_id, binding.dataset_id, binding.tokenizer_id
            );
            let _ = writeln!(
                stdout,
                "heldout unigram baseline {} nats",
                packed.heldout_unigram_baseline_nats
            );
            crate::EXIT_SUCCESS
        }
        Err(error) => {
            let _ = writeln!(stderr, "modelspec data pack {error}");
            crate::EXIT_FAILURE
        }
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
