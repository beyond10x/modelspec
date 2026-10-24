//! Packed token sequences and masked-language-model batches (`spec/domains/data.yaml`,
//! `modelspec.data.PackedDataset`, `TrainingDataBinding` and `MaskingPolicy`;
//! story:bert-mlm-batches, coordinator decision D26).
//!
//! [`pack`] encodes the documents of both splits of a prepared dataset with a tokenizer, joins
//! them with `[SEP]` between two documents, cuts each stream into sequences of exactly `S` ids and
//! drops the partial tail, so no sequence needs padding. It writes the packed files, the
//! [`TrainingDataBinding`] and the [`PackedDataset`] record with the held-out unigram baseline.
//!
//! [`mlm_batch`] is a pure function of the packed data, the [`MaskingPolicy`], the run seed, the
//! step and the batch size: the epoch order of the `Train` sequences is drawn on
//! [`Stream::DataOrder`] (tag 5) and the 80/10/10 masking on [`Stream::Masking`] (tag 3), both
//! keyed by the run seed (DESIGN.md § 2.3).

use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::tokenizer::{Codec, MASK_TOKEN_ID, SEP_TOKEN_ID, SPECIAL_TOKENS, Tokenizer};
use super::{
    Dataset, RECORD_FILE, SplitRole, is_article_heading, missing_directories, partial_path,
    remove_directories, uuid_of,
};
use crate::train::init::{Stream, block};

/// The packed file of the `Train` split.
pub const TRAIN_TOKENS: &str = "train.tokens";
/// The packed file of the `HeldOut` split.
pub const HELDOUT_TOKENS: &str = "heldout.tokens";
/// The file [`pack`] writes the [`TrainingDataBinding`] to.
pub const BINDING_FILE: &str = "binding.json";
/// The file [`pack`] writes the [`PackedDataset`] record to.
pub const PACKED_FILE: &str = "packed.json";

/// The number of special tokens at the head of every vocabulary; ids below it are never
/// selected, and a random substitution draws from the ids at or above it.
const SPECIALS: u32 = SPECIAL_TOKENS.len() as u32;

/// The packed file of a split.
pub fn packed_file(role: SplitRole) -> &'static str {
    match role {
        SplitRole::Train => TRAIN_TOKENS,
        SplitRole::HeldOut => HELDOUT_TOKENS,
    }
}

/// The masked-language-modelling policy (`modelspec.data.MaskingPolicy`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaskingPolicy {
    /// The share of the tokens that are not special selected for prediction.
    pub selection_rate: f64,
    /// The share of the selected tokens replaced by `[MASK]`.
    pub mask_fraction: f64,
    /// The share of the selected tokens replaced by a random token.
    pub random_fraction: f64,
    /// The share of the selected tokens kept.
    pub keep_fraction: f64,
}

impl MaskingPolicy {
    /// BERT's policy (Devlin et al. 2018, § 3.1; coordinator decisions D19 and D26).
    pub const BERT: MaskingPolicy = MaskingPolicy {
        selection_rate: 0.15,
        mask_fraction: 0.8,
        random_fraction: 0.1,
        keep_fraction: 0.1,
    };
}

/// The lifecycle state of a [`TrainingDataBinding`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BindingState {
    Recorded,
}

/// What a run trains on (`modelspec.data.TrainingDataBinding`), written as [`BINDING_FILE`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingDataBinding {
    /// The leading 16 bytes, as a version-8 UUID, of blake3 over the dataset id, the tokenizer
    /// id, `seq_len` and the policy's four rates ([`binding_id`]).
    pub binding_id: String,
    pub dataset_id: String,
    pub tokenizer_id: String,
    pub seq_len: u64,
    pub masking_policy: MaskingPolicy,
    pub state: BindingState,
}

/// One split of a [`PackedDataset`] (`modelspec.data.PackedSplit`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedSplit {
    pub role: SplitRole,
    pub sequence_count: u64,
    /// blake3 of the packed file's bytes, lowercase hex.
    pub content_hash: String,
}

/// The lifecycle state of a [`PackedDataset`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackedState {
    Recorded,
}

/// The record of a packed dataset (`modelspec.data.PackedDataset`), written as [`PACKED_FILE`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PackedDataset {
    /// The leading 16 bytes, as a version-8 UUID, of blake3 over the packed files' digests,
    /// `Train` then `HeldOut`.
    pub packed_id: String,
    pub binding_id: String,
    pub seq_len: u64,
    pub vocabulary_size: u64,
    pub splits: Vec<PackedSplit>,
    pub heldout_unigram_baseline_nats: f64,
    pub state: PackedState,
}

/// A refused [`pack`] or [`load`]: the path it concerns and why.
#[derive(Debug)]
pub struct PackError {
    pub path: PathBuf,
    pub reason: String,
}

impl PackError {
    fn new(path: &Path, reason: impl fmt::Display) -> Self {
        PackError {
            path: path.to_path_buf(),
            reason: reason.to_string(),
        }
    }
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.reason)
    }
}

impl std::error::Error for PackError {}

/// The 16 bytes of a UUID in its hyphenated text form.
fn uuid_bytes(text: &str) -> Option<[u8; 16]> {
    let hex: Vec<u8> = text.bytes().filter(|&b| b != b'-').collect();
    if hex.len() != 32 || text.len() != 36 {
        return None;
    }
    let mut bytes = [0u8; 16];
    for (i, pair) in hex.as_chunks::<2>().0.iter().enumerate() {
        let pair = std::str::from_utf8(pair).ok()?;
        bytes[i] = u8::from_str_radix(pair, 16).ok()?;
    }
    Some(bytes)
}

/// The binding id of a dataset, a tokenizer, a sequence length and a policy: the leading 16
/// bytes, as a version-8 UUID, of blake3 over the dataset id's 16 bytes, the tokenizer id's 16
/// bytes, `seq_len` as a little-endian `u64` and the four rates as little-endian IEEE binary64,
/// in that order (coordinator decision D26). `None` when an id is not a UUID.
pub fn binding_id(
    dataset_id: &str,
    tokenizer_id: &str,
    seq_len: u64,
    policy: &MaskingPolicy,
) -> Option<String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&uuid_bytes(dataset_id)?);
    hasher.update(&uuid_bytes(tokenizer_id)?);
    hasher.update(&seq_len.to_le_bytes());
    for rate in [
        policy.selection_rate,
        policy.mask_fraction,
        policy.random_fraction,
        policy.keep_fraction,
    ] {
        hasher.update(&rate.to_le_bytes());
    }
    Some(uuid_of(hasher.finalize().as_bytes()))
}

/// `line` without its line terminator, `\n` or `\r\n`.
fn content(line: &str) -> &str {
    let line = line.strip_suffix('\n').unwrap_or(line);
    line.strip_suffix('\r').unwrap_or(line)
}

/// Encodes the lines of a split into one token stream, the documents joined with `[SEP]`
/// between two of them. A document starts as `DatasetSplit.document_count` defines it (the rule
/// of `corpus::DocumentCounter`, coordinator decision D22): at a line of the shape
/// [`is_article_heading`] with a blank line directly before and after it, and non-blank text
/// before the first such line is a document too. Lines are encoded one at a time, which gives the
/// ids of the whole text, since a word never spans a line.
struct Joiner<'c, W: Write> {
    codec: &'c Codec<'c>,
    out: W,
    /// The ids written.
    count: u64,
    /// Whether a document has started.
    started: bool,
    previous_blank: bool,
    /// A heading-shaped line after a blank one, a heading if the next line is blank.
    candidate: Option<String>,
}

impl<'c, W: Write> Joiner<'c, W> {
    fn new(codec: &'c Codec<'c>, out: W) -> Self {
        Joiner {
            codec,
            out,
            count: 0,
            started: false,
            previous_blank: false,
            candidate: None,
        }
    }

    fn line(&mut self, line: &str) -> io::Result<()> {
        let blank = content(line) == " ";
        if let Some(heading) = self.candidate.take() {
            if blank {
                self.start_document()?;
            }
            self.text(&heading)?;
        }
        if self.previous_blank && is_article_heading(line) {
            self.candidate = Some(line.to_owned());
        } else if !line.trim().is_empty() {
            self.text(line)?;
        }
        self.previous_blank = blank;
        Ok(())
    }

    fn start_document(&mut self) -> io::Result<()> {
        if self.started {
            self.emit(SEP_TOKEN_ID)?;
        }
        self.started = true;
        Ok(())
    }

    /// A line of the current document; text before any heading starts the first document.
    fn text(&mut self, line: &str) -> io::Result<()> {
        self.started = true;
        for id in self.codec.encode(line) {
            self.emit(id)?;
        }
        Ok(())
    }

    fn emit(&mut self, id: u32) -> io::Result<()> {
        self.count += 1;
        self.out.write_all(&(id as i32).to_le_bytes())
    }

    /// The writer and the ids written; a heading-shaped last line is text.
    fn finish(mut self) -> io::Result<(W, u64)> {
        if let Some(heading) = self.candidate.take() {
            self.text(&heading)?;
        }
        self.out.flush()?;
        Ok((self.out, self.count))
    }
}

/// Encodes the split file `source` into `target` as little-endian `i32` ids, checking that its
/// blake3 hash is `expected_hash`; returns the number of ids written.
fn encode_split(
    source: &Path,
    target: &Path,
    codec: &Codec<'_>,
    expected_hash: &str,
) -> Result<u64, PackError> {
    let input = File::open(source).map_err(|e| PackError::new(source, e))?;
    let output = File::create(target).map_err(|e| PackError::new(target, e))?;
    let mut reader = BufReader::new(input);
    let mut joiner = Joiner::new(codec, BufWriter::new(output));
    let mut hasher = blake3::Hasher::new();
    let mut line = Vec::new();
    let mut offset: u64 = 0;
    loop {
        line.clear();
        let read = reader
            .read_until(b'\n', &mut line)
            .map_err(|e| PackError::new(source, e))?;
        if read == 0 {
            break;
        }
        hasher.update(&line);
        let text = std::str::from_utf8(&line).map_err(|e| {
            let at = offset + e.valid_up_to() as u64;
            PackError::new(source, format!("not UTF-8 at byte {at}"))
        })?;
        joiner.line(text).map_err(|e| PackError::new(target, e))?;
        offset += read as u64;
    }
    if hasher.finalize().to_hex().as_str() != expected_hash {
        return Err(PackError::new(
            source,
            format!("blake3 differs from the content_hash {expected_hash} of {RECORD_FILE}"),
        ));
    }
    let (writer, count) = joiner.finish().map_err(|e| PackError::new(target, e))?;
    writer
        .into_inner()
        .map_err(|e| PackError::new(target, e.into_error()))?
        .sync_all()
        .map_err(|e| PackError::new(target, e))?;
    Ok(count)
}

/// Calls `f` with each id of the packed file `path` and returns its blake3 hash.
fn scan_ids(path: &Path, mut f: impl FnMut(u32)) -> Result<blake3::Hash, PackError> {
    let file = File::open(path).map_err(|e| PackError::new(path, e))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = read_full(&mut reader, &mut buffer).map_err(|e| PackError::new(path, e))?;
        if read == 0 {
            return Ok(hasher.finalize());
        }
        hasher.update(&buffer[..read]);
        for b in buffer[..read].as_chunks::<4>().0 {
            f(i32::from_le_bytes(*b) as u32);
        }
    }
}

/// Fills `buffer` as far as the reader allows, so a chunk boundary never splits an id.
fn read_full(reader: &mut impl Read, buffer: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, PackError> {
    let text = fs::read_to_string(path).map_err(|e| PackError::new(path, e))?;
    serde_json::from_str(&text).map_err(|e| PackError::new(path, e))
}

fn write_json(value: &impl Serialize, path: &Path) -> Result<(), PackError> {
    let mut json = serde_json::to_string_pretty(value).expect("JSON of a record");
    json.push('\n');
    fs::write(path, json).map_err(|e| PackError::new(path, e))
}

/// Reads the tokenizer record `path`, refusing one whose id is not the hash of its tokens or
/// whose vocabulary does not start with the special tokens.
fn read_tokenizer(path: &Path) -> Result<Tokenizer, PackError> {
    let tokenizer: Tokenizer = read_json(path)?;
    let tokens = &tokenizer.vocabulary.tokens;
    let mut id = blake3::Hasher::new();
    for token in tokens {
        id.update(token.as_bytes());
        id.update(b"\n");
    }
    if uuid_of(id.finalize().as_bytes()) != tokenizer.tokenizer_id {
        return Err(PackError::new(
            path,
            format!(
                "tokenizer_id {} is not the id of its tokens",
                tokenizer.tokenizer_id
            ),
        ));
    }
    if tokens.len() <= SPECIAL_TOKENS.len()
        || tokens.len() > i32::MAX as usize
        || tokens[..SPECIAL_TOKENS.len()] != SPECIAL_TOKENS
    {
        return Err(PackError::new(
            path,
            "the vocabulary does not start with the five special tokens and one more",
        ));
    }
    Ok(tokenizer)
}

/// Packs the dataset `dataset_dir`, a directory `prepare` wrote, with the tokenizer record
/// `tokenizer_file` into sequences of `seq_len` ids, and writes [`TRAIN_TOKENS`],
/// [`HELDOUT_TOKENS`], [`BINDING_FILE`] (with [`MaskingPolicy::BERT`]) and [`PACKED_FILE`] into
/// `out_dir`, created when missing. Every file is written as `<name>.partial` first and renamed
/// only once all of them are complete, so a refused run leaves what `out_dir` held before as it
/// was and removes the directories it created.
pub fn pack(
    dataset_dir: &Path,
    tokenizer_file: &Path,
    seq_len: u64,
    out_dir: &Path,
) -> Result<(TrainingDataBinding, PackedDataset), PackError> {
    let record_path = dataset_dir.join(RECORD_FILE);
    let dataset: Dataset = read_json(&record_path)?;
    let tokenizer = read_tokenizer(tokenizer_file)?;
    if seq_len == 0 {
        return Err(PackError::new(out_dir, "a sequence length of 0"));
    }
    let policy = MaskingPolicy::BERT;
    let binding_id = binding_id(
        &dataset.dataset_id,
        &tokenizer.tokenizer_id,
        seq_len,
        &policy,
    )
    .ok_or_else(|| PackError::new(&record_path, "dataset_id is not a UUID"))?;
    let mut hashes = Vec::new();
    for role in SplitRole::ALL {
        let split = dataset.splits.iter().find(|s| s.role == role);
        let split =
            split.ok_or_else(|| PackError::new(&record_path, format!("no {role:?} split")))?;
        hashes.push(split.content_hash.clone());
    }

    let created = missing_directories(out_dir);
    if let Err(e) = fs::create_dir_all(out_dir) {
        remove_directories(&created);
        return Err(PackError::new(out_dir, e));
    }
    let mut pending: Vec<(PathBuf, PathBuf)> = Vec::new();
    let result = (|| {
        let codec = tokenizer.codec();
        let vocabulary_size = tokenizer.vocabulary.tokens.len() as u64;
        let mut splits = Vec::new();
        let mut train_counts = vec![0u64; vocabulary_size as usize];
        let mut train_total: u64 = 0;
        let (mut log_sum, mut measured) = (0.0f64, 0u64);
        let mut id = blake3::Hasher::new();
        for (role, expected) in SplitRole::ALL.into_iter().zip(&hashes) {
            let source = dataset_dir.join(role.split_file());
            let target = out_dir.join(packed_file(role));
            let partial = partial_path(&target);
            pending.push((partial.clone(), target));
            let count = encode_split(&source, &partial, &codec, expected)?;
            let sequence_count = count / seq_len;
            if sequence_count == 0 {
                return Err(PackError::new(
                    &source,
                    format!(
                        "the split encodes to {count} tokens, fewer than one sequence of {seq_len}"
                    ),
                ));
            }
            let kept = sequence_count
                .checked_mul(seq_len)
                .and_then(|ids| ids.checked_mul(4))
                .expect("the kept ids are fewer than the written ones");
            File::options()
                .write(true)
                .open(&partial)
                .and_then(|f| f.set_len(kept).and_then(|()| f.sync_all()))
                .map_err(|e| PackError::new(&partial, e))?;
            let hash = match role {
                SplitRole::Train => scan_ids(&partial, |t| {
                    if t >= SPECIALS {
                        train_counts[t as usize] += 1;
                        train_total += 1;
                    }
                })?,
                SplitRole::HeldOut => {
                    let types = (vocabulary_size - u64::from(SPECIALS)) as f64;
                    let denominator = train_total as f64 + types;
                    scan_ids(&partial, |t| {
                        if t >= SPECIALS {
                            log_sum += ((train_counts[t as usize] as f64 + 1.0) / denominator).ln();
                            measured += 1;
                        }
                    })?
                }
            };
            id.update(hash.as_bytes());
            splits.push(PackedSplit {
                role,
                sequence_count,
                content_hash: hash.to_hex().to_string(),
            });
        }
        if measured == 0 {
            let source = dataset_dir.join(SplitRole::HeldOut.split_file());
            return Err(PackError::new(
                &source,
                "the packed split holds no token that is not special",
            ));
        }
        let binding = TrainingDataBinding {
            binding_id: binding_id.clone(),
            dataset_id: dataset.dataset_id.clone(),
            tokenizer_id: tokenizer.tokenizer_id.clone(),
            seq_len,
            masking_policy: policy,
            state: BindingState::Recorded,
        };
        let packed = PackedDataset {
            packed_id: uuid_of(id.finalize().as_bytes()),
            binding_id: binding_id.clone(),
            seq_len,
            vocabulary_size,
            splits,
            heldout_unigram_baseline_nats: -log_sum / measured as f64,
            state: PackedState::Recorded,
        };
        let target = out_dir.join(BINDING_FILE);
        let partial = partial_path(&target);
        pending.push((partial.clone(), target));
        write_json(&binding, &partial)?;
        let target = out_dir.join(PACKED_FILE);
        let partial = partial_path(&target);
        pending.push((partial.clone(), target));
        write_json(&packed, &partial)?;
        Ok((binding, packed))
    })();
    match result {
        Ok(records) => {
            for (partial, target) in &pending {
                fs::rename(partial, target).map_err(|e| PackError::new(target, e))?;
            }
            Ok(records)
        }
        Err(error) => {
            for (partial, _) in &pending {
                let _ = fs::remove_file(partial);
            }
            remove_directories(&created);
            Err(error)
        }
    }
}

/// The epoch order of the epoch's last computed draw.
struct OrderCache {
    seed: u64,
    epoch: u64,
    order: Arc<Vec<u32>>,
}

/// A packed dataset read into memory by [`load`], the input of [`mlm_batch`].
pub struct Packed {
    pub record: PackedDataset,
    train: Vec<i32>,
    heldout: Vec<i32>,
    order: Mutex<Option<OrderCache>>,
}

impl Packed {
    /// The ids of a split, `sequence_count * seq_len` of them, sequence after sequence.
    pub fn ids(&self, role: SplitRole) -> &[i32] {
        match role {
            SplitRole::Train => &self.train,
            SplitRole::HeldOut => &self.heldout,
        }
    }

    /// The number of sequences of a split.
    pub fn sequence_count(&self, role: SplitRole) -> usize {
        self.ids(role).len() / self.record.seq_len as usize
    }

    /// [`epoch_order`] of the `Train` split, the last one kept.
    fn train_order(&self, seed: u64, epoch: u64) -> Arc<Vec<u32>> {
        let mut cache = self.order.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = cache
            .as_ref()
            .filter(|c| c.seed == seed && c.epoch == epoch)
        {
            return Arc::clone(&c.order);
        }
        let n = self.sequence_count(SplitRole::Train) as u32;
        let order = Arc::new(epoch_order(seed, epoch, n));
        *cache = Some(OrderCache {
            seed,
            epoch,
            order: Arc::clone(&order),
        });
        order
    }
}

/// Reads the packed directory `dir` that [`pack`] wrote, refusing a packed file whose length,
/// blake3 hash or ids disagree with its record.
pub fn load(dir: &Path) -> Result<Packed, PackError> {
    let record_path = dir.join(PACKED_FILE);
    let record: PackedDataset = read_json(&record_path)?;
    let roles: Vec<SplitRole> = record.splits.iter().map(|s| s.role).collect();
    if roles != SplitRole::ALL || record.seq_len == 0 {
        return Err(PackError::new(
            &record_path,
            "not a record of a Train and a HeldOut split of a positive sequence length",
        ));
    }
    let mut ids = Vec::new();
    for split in &record.splits {
        let path = dir.join(packed_file(split.role));
        let bytes = fs::read(&path).map_err(|e| PackError::new(&path, e))?;
        let expected = Some(split.sequence_count)
            .filter(|&n| n <= u64::from(u32::MAX))
            .and_then(|n| n.checked_mul(record.seq_len))
            .and_then(|n| n.checked_mul(4));
        if expected != Some(bytes.len() as u64) {
            return Err(PackError::new(
                &path,
                format!(
                    "{} bytes, not {} sequences of {} ids",
                    bytes.len(),
                    split.sequence_count,
                    record.seq_len
                ),
            ));
        }
        if blake3::hash(&bytes).to_hex().as_str() != split.content_hash {
            return Err(PackError::new(
                &path,
                format!("blake3 differs from the content_hash of {PACKED_FILE}"),
            ));
        }
        let split_ids: Vec<i32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        if let Some(bad) = split_ids
            .iter()
            .find(|&&t| t < 0 || t as u64 >= record.vocabulary_size)
        {
            return Err(PackError::new(
                &path,
                format!(
                    "id {bad} outside a vocabulary of {}",
                    record.vocabulary_size
                ),
            ));
        }
        ids.push(split_ids);
    }
    let heldout = ids.pop().expect("two splits");
    let train = ids.pop().expect("two splits");
    Ok(Packed {
        record,
        train,
        heldout,
        order: Mutex::new(None),
    })
}

/// The order in which epoch `epoch` of a run with seed `seed` visits `n` sequences: ascending by
/// the 64-bit key `w1 * 2^32 + w0` of the Philox block of sequence `i` on [`Stream::DataOrder`]
/// with id `epoch`, ties by `i` (DESIGN.md § 2.3).
pub fn epoch_order(seed: u64, epoch: u64, n: u32) -> Vec<u32> {
    let mut keyed: Vec<(u64, u32)> = (0..n)
        .map(|i| {
            let w = block(seed, Stream::DataOrder, epoch, i);
            ((u64::from(w[1]) << 32) | u64::from(w[0]), i)
        })
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, i)| i).collect()
}

/// The tensors of one masked-language-model step, flattened to `N = B * S` rows.
#[derive(Clone, Debug, PartialEq)]
pub struct MlmBatch {
    /// The ids with the 80/10/10 substitution applied.
    pub ids: Vec<i32>,
    /// `0` to `S - 1` per sequence.
    pub positions: Vec<i32>,
    /// `[B, S]`, all ones: a packed sequence has no padding.
    pub key_mask: Vec<f32>,
    /// The original ids.
    pub labels: Vec<i32>,
    /// `1` exactly at the selected rows, `0` elsewhere.
    pub weights: Vec<f32>,
}

/// Why [`mlm_batch`] gave no batch.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BatchError {
    #[error("a batch of 0 sequences")]
    Empty,
    #[error("a batch of {batch_size} sequences, more than the {sequences} of the split")]
    TooLarge { batch_size: usize, sequences: usize },
    #[error("a batch of {rows} rows; a draw addresses at most 2^32")]
    TooManyRows { rows: u128 },
}

/// A word as a uniform draw in `(0, 1)` (DESIGN.md § 2.3).
fn uniform(word: u32) -> f64 {
    (f64::from(word) + 0.5) / 4_294_967_296.0
}

/// The batch of step `step` of `split` for a run with seed `seed` and `batch_size` sequences
/// per step (`spec/domains/data.yaml`, `PackedDataset` and `MaskingPolicy`). `Train` step `k` is
/// batch `k mod ⌊n / B⌋` of epoch `⌊k / ⌊n / B⌋⌋` in [`epoch_order`], so an incomplete last
/// batch is never used; `HeldOut` step `k` is batch `k mod ⌊n / B⌋` in stored order. Row `r`
/// takes the block of index `r` and id `k` (`HeldOut`: `k + 2^63`) on [`Stream::Masking`]: a row
/// that is not a special token is selected when word 0 is below `selection_rate`, and becomes
/// `[MASK]`, a random token `5 + ⌊w2 * (V - 5) / 2^32⌋` or stays itself by word 1.
pub fn mlm_batch(
    packed: &Packed,
    split: SplitRole,
    policy: &MaskingPolicy,
    seed: u64,
    step: u64,
    batch_size: usize,
) -> Result<MlmBatch, BatchError> {
    let seq_len = packed.record.seq_len as usize;
    let sequences = packed.sequence_count(split);
    if batch_size == 0 {
        return Err(BatchError::Empty);
    }
    if batch_size > sequences {
        return Err(BatchError::TooLarge {
            batch_size,
            sequences,
        });
    }
    let rows = batch_size as u128 * seq_len as u128;
    if rows > 1 << 32 {
        return Err(BatchError::TooManyRows { rows });
    }
    let per_epoch = (sequences / batch_size) as u64;
    let batch = (step % per_epoch) as usize;
    let chosen: Vec<usize> = match split {
        SplitRole::Train => {
            let order = packed.train_order(seed, step / per_epoch);
            order[batch * batch_size..(batch + 1) * batch_size]
                .iter()
                .map(|&i| i as usize)
                .collect()
        }
        SplitRole::HeldOut => (batch * batch_size..(batch + 1) * batch_size).collect(),
    };
    let draw_id = match split {
        SplitRole::Train => step,
        SplitRole::HeldOut => step.wrapping_add(1 << 63),
    };
    let ids = packed.ids(split);
    let random_tokens = packed.record.vocabulary_size - u64::from(SPECIALS);
    let rows = rows as usize;
    let mut out = MlmBatch {
        ids: Vec::with_capacity(rows),
        positions: Vec::with_capacity(rows),
        key_mask: vec![1.0; rows],
        labels: Vec::with_capacity(rows),
        weights: Vec::with_capacity(rows),
    };
    for (b, &sequence) in chosen.iter().enumerate() {
        let start = sequence * seq_len;
        for (p, &label) in ids[start..start + seq_len].iter().enumerate() {
            let row = (b * seq_len + p) as u32;
            let [w0, w1, w2, _] = block(seed, Stream::Masking, draw_id, row);
            let selected = label >= SPECIALS as i32 && uniform(w0) < policy.selection_rate;
            let id = if !selected {
                label
            } else if uniform(w1) < policy.mask_fraction {
                MASK_TOKEN_ID as i32
            } else if uniform(w1) < policy.mask_fraction + policy.random_fraction {
                (u64::from(SPECIALS) + ((u64::from(w2) * random_tokens) >> 32)) as i32
            } else {
                label
            };
            out.ids.push(id);
            out.positions.push(p as i32);
            out.labels.push(label);
            out.weights.push(if selected { 1.0 } else { 0.0 });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::tokenizer::{TokenizerKind, TokenizerState, Vocabulary};

    fn codec_tokens() -> Tokenizer {
        let mut tokens: Vec<String> = SPECIAL_TOKENS.iter().map(|t| t.to_string()).collect();
        tokens.extend(["a", "b", "=", "t"].map(String::from));
        Tokenizer {
            tokenizer_id: String::new(),
            kind: TokenizerKind::WordPiece,
            vocabulary: Vocabulary {
                tokens,
                mask_token_id: 4,
                pad_token_id: 0,
                unk_token_id: 1,
                cls_token_id: 2,
                sep_token_id: 3,
            },
            state: TokenizerState::Recorded,
        }
    }

    fn stream(text: &str) -> Vec<u32> {
        let tokenizer = codec_tokens();
        let codec = tokenizer.codec();
        let mut joiner = Joiner::new(&codec, Vec::new());
        for line in text.split_inclusive('\n') {
            joiner.line(line).unwrap();
        }
        let (bytes, count) = joiner.finish().unwrap();
        assert_eq!(bytes.len() as u64, count * 4);
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b) as u32)
            .collect()
    }

    #[test]
    fn documents_are_joined_with_sep_between_two_of_them() {
        // `a` 5, `b` 6, `=` 7, `t` 8.
        assert_eq!(
            stream(" \n = T = \n \n a \n \n = T = \n \n b \n"),
            [7, 8, 7, 5, 3, 7, 8, 7, 6]
        );
        assert_eq!(stream(" a \n \n = T = \n \n b \n"), [5, 3, 7, 8, 7, 6]);
        assert_eq!(
            stream(" \n = T = \n \n a \n = T = \n \n b \n"),
            [7, 8, 7, 5, 7, 8, 7, 6],
            "a heading needs a blank line before it"
        );
        assert_eq!(
            stream(" \n = T = \n a \n"),
            [7, 8, 7, 5],
            "and one after it"
        );
        assert_eq!(stream(" \n = T = \n"), [7, 8, 7], "a last line is text");
    }

    #[test]
    fn binding_id_hashes_the_ids_the_length_and_the_rates_in_order() {
        let d = "00112233-4455-8677-8899-aabbccddeeff";
        let t = "ffeeddcc-bbaa-8988-b766-554433221100";
        let id = binding_id(d, t, 128, &MaskingPolicy::BERT).unwrap();
        let mut h = blake3::Hasher::new();
        h.update(&uuid_bytes(d).unwrap());
        h.update(&uuid_bytes(t).unwrap());
        h.update(&128u64.to_le_bytes());
        for r in [0.15f64, 0.8, 0.1, 0.1] {
            h.update(&r.to_le_bytes());
        }
        assert_eq!(id, uuid_of(h.finalize().as_bytes()));
        assert_ne!(id, binding_id(d, t, 64, &MaskingPolicy::BERT).unwrap());
        assert_eq!(uuid_bytes(d).unwrap()[0..2], [0x00, 0x11]);
        assert!(binding_id("not a uuid", t, 1, &MaskingPolicy::BERT).is_none());
    }
}
