//! The pre-training corpus of a masked-language-model run (`spec/domains/data.yaml`,
//! `modelspec.data.Dataset`; story:bert-corpus-splits).
//!
//! [`prepare`] turns a directory holding the WikiText-103 raw release's files into the `Train`
//! and `HeldOut` splits and a [`Dataset`] record of them, `dataset.json`. Everything is derived
//! from the bytes read: two runs on the same files write byte-identical output.
//!
//! [`tokenizer`] trains the WordPiece tokenizer of a run on the `Train` split of a prepared
//! dataset and encodes and decodes text with it (story:bert-tokenizer).
//!
//! [`mlm`] packs both splits of a prepared dataset into token sequences with a tokenizer and
//! draws the masked-language-model batch of each step from them (story:bert-mlm-batches).

mod corpus;
pub mod mlm;
pub mod tokenizer;

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub use corpus::is_article_heading;

/// The file name of the record [`prepare`] writes.
pub const RECORD_FILE: &str = "dataset.json";

/// What a split is used for (`modelspec.data.SplitRole`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitRole {
    /// The split a run trains on.
    Train,
    /// The split MLM loss and accuracy are evaluated on.
    HeldOut,
}

impl SplitRole {
    /// Both roles, in the order the record lists them and the dataset id hashes them.
    pub const ALL: [SplitRole; 2] = [SplitRole::Train, SplitRole::HeldOut];

    /// The release file the split is made from.
    pub fn release_file(self) -> &'static str {
        match self {
            SplitRole::Train => "wiki.train.raw",
            SplitRole::HeldOut => "wiki.valid.raw",
        }
    }

    /// The file [`prepare`] copies the release file to in its output directory.
    pub fn split_file(self) -> &'static str {
        match self {
            SplitRole::Train => "train.raw",
            SplitRole::HeldOut => "heldout.raw",
        }
    }
}

/// One split of a corpus (`modelspec.data.DatasetSplit`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetSplit {
    pub role: SplitRole,
    /// blake3 of the split file's bytes, lowercase hex.
    pub content_hash: String,
    pub document_count: u64,
    pub byte_count: u64,
}

/// The lifecycle state of a [`Dataset`] (`modelspec.data.Dataset.State`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetState {
    Recorded,
}

/// The record of a prepared corpus (`modelspec.data.Dataset`), written as [`RECORD_FILE`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dataset {
    /// The leading 16 bytes, as a version-8 UUID, of blake3 over the splits' digests in
    /// [`SplitRole::ALL`] order.
    pub dataset_id: String,
    /// The file name of the corpus directory.
    pub name: String,
    pub splits: Vec<DatasetSplit>,
    pub state: DatasetState,
}

/// A refused [`prepare`]: the path it concerns and why.
#[derive(Debug)]
pub struct PrepareError {
    pub path: PathBuf,
    pub reason: String,
}

impl PrepareError {
    fn new(path: &Path, reason: impl fmt::Display) -> Self {
        PrepareError {
            path: path.to_path_buf(),
            reason: reason.to_string(),
        }
    }
}

impl fmt::Display for PrepareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.reason)
    }
}

impl std::error::Error for PrepareError {}

/// Reads the release files of `corpus_dir` and writes each split and [`RECORD_FILE`] into
/// `out_dir`, which is created when missing. The files are written as `<name>.partial` first
/// and renamed only once both splits have been read whole, so a refused run writes no split or
/// record and leaves what `out_dir` held before as it was; the directories it created for
/// `out_dir` are removed again.
pub fn prepare(corpus_dir: &Path, out_dir: &Path) -> Result<Dataset, PrepareError> {
    let meta = fs::metadata(corpus_dir).map_err(|e| PrepareError::new(corpus_dir, e))?;
    if !meta.is_dir() {
        return Err(PrepareError::new(corpus_dir, "not a directory"));
    }
    let name = fs::canonicalize(corpus_dir)
        .map_err(|e| PrepareError::new(corpus_dir, e))?
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    for role in SplitRole::ALL {
        let source = corpus_dir.join(role.release_file());
        fs::metadata(&source).map_err(|e| PrepareError::new(&source, e))?;
    }
    let created = missing_directories(out_dir);
    if let Err(e) = fs::create_dir_all(out_dir) {
        remove_directories(&created);
        return Err(PrepareError::new(out_dir, e));
    }

    let mut pending: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut splits = Vec::new();
    let result = (|| {
        for role in SplitRole::ALL {
            let source = corpus_dir.join(role.release_file());
            let target = out_dir.join(role.split_file());
            let partial = partial_path(&target);
            pending.push((partial.clone(), target));
            let summary = corpus::copy_split(&source, &partial)?;
            splits.push(DatasetSplit {
                role,
                content_hash: summary.hash.to_hex().to_string(),
                document_count: summary.document_count,
                byte_count: summary.byte_count,
            });
        }
        let mut id = blake3::Hasher::new();
        for split in &splits {
            let digest = blake3::Hash::from_hex(&split.content_hash).expect("hex of a digest");
            id.update(digest.as_bytes());
        }
        let dataset = Dataset {
            dataset_id: uuid_of(id.finalize().as_bytes()),
            name,
            splits,
            state: DatasetState::Recorded,
        };
        let record = out_dir.join(RECORD_FILE);
        let partial = partial_path(&record);
        pending.push((partial.clone(), record));
        let mut json = serde_json::to_string_pretty(&dataset).expect("JSON of a dataset record");
        json.push('\n');
        fs::write(&partial, json).map_err(|e| PrepareError::new(&partial, e))?;
        Ok(dataset)
    })();
    match result {
        Ok(dataset) => {
            for (partial, target) in &pending {
                fs::rename(partial, target).map_err(|e| PrepareError::new(target, e))?;
            }
            Ok(dataset)
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

/// `dir` and those of its ancestors that do not exist, deepest first: what `create_dir_all`
/// would create.
fn missing_directories(dir: &Path) -> Vec<PathBuf> {
    dir.ancestors()
        .filter(|d| !d.as_os_str().is_empty())
        .take_while(|d| fs::symlink_metadata(d).is_err())
        .map(Path::to_path_buf)
        .collect()
}

/// Removes the directories of [`missing_directories`], deepest first; each only if it is empty.
fn remove_directories(created: &[PathBuf]) {
    for dir in created {
        let _ = fs::remove_dir(dir);
    }
}

/// `<file>.partial` beside `target`.
fn partial_path(target: &Path) -> PathBuf {
    let mut name = target.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    target.with_file_name(name)
}

/// The first 16 bytes of `hash` as a version-8 UUID, in its hyphenated lowercase text form.
fn uuid_of(hash: &[u8; 32]) -> String {
    let mut b = [0u8; 16];
    b.copy_from_slice(&hash[..16]);
    b[6] = (b[6] & 0x0f) | 0x80;
    b[8] = (b[8] & 0x3f) | 0x80;
    let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_of_marks_version_8_and_the_rfc_variant() {
        let id = uuid_of(&[0xff; 32]);
        assert_eq!(id, "ffffffff-ffff-8fff-bfff-ffffffffffff");
        let id = uuid_of(&[0; 32]);
        assert_eq!(id, "00000000-0000-8000-8000-000000000000");
    }

    #[test]
    fn record_serializes_fields_in_schema_order() {
        let dataset = Dataset {
            dataset_id: uuid_of(&[0; 32]),
            name: "corpus".into(),
            splits: vec![DatasetSplit {
                role: SplitRole::HeldOut,
                content_hash: "ab".repeat(32),
                document_count: 2,
                byte_count: 9,
            }],
            state: DatasetState::Recorded,
        };
        let json = serde_json::to_string(&dataset).unwrap();
        assert_eq!(
            json,
            format!(
                "{{\"dataset_id\":\"00000000-0000-8000-8000-000000000000\",\"name\":\"corpus\",\
                 \"splits\":[{{\"role\":\"HeldOut\",\"content_hash\":\"{}\",\"document_count\":2,\
                 \"byte_count\":9}}],\"state\":\"Recorded\"}}",
                "ab".repeat(32)
            )
        );
    }
}
