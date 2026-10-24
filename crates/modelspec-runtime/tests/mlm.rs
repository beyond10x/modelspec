//! `dataset::mlm` (story:bert-mlm-batches; `spec/domains/data.yaml`, `modelspec.data.PackedDataset`,
//! `modelspec.data.MaskingPolicy` and coordinator decision D26): packed token sequences and the
//! 80/10/10 masked-language-model batches drawn from them on Philox stream tag 3, in an epoch
//! order drawn on stream tag 5.
//!
//! The corpus is the fixture `tests/fixtures/corpus-tokenizer/`, original text in the release's
//! layout, prepared, tokenized at 300 tokens and packed at 16 tokens through the library here, so
//! no case needs the corpus or the network.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use modelspec_runtime::dataset::mlm::{self, MaskingPolicy, MlmBatch, Packed};
use modelspec_runtime::dataset::{self, SplitRole, tokenizer};
use modelspec_runtime::train::init::{Stream, philox4x32_10};

/// The sequence length of the cases.
const SEQ_LEN: usize = 16;
/// The vocabulary size of the cases.
const VOCAB: usize = 300;
/// Ids below this are the special tokens (coordinator decision D19).
const SPECIALS: i32 = 5;
/// `[MASK]`'s id.
const MASK: i32 = 4;
/// The fixture positions scenario 3 covers at least.
const POSITIONS: usize = 20_000;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus-tokenizer")
}

/// A fresh, empty directory under this crate's build directory, named for the case.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("mlm_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// The fixture prepared, its tokenizer trained and written, and the dataset packed, for `case`.
fn packed(case: &str) -> (PathBuf, Packed) {
    let dir = scratch(case);
    let data = dir.join("dataset");
    dataset::prepare(&fixture(), &data).expect("the fixture prepares");
    let record = tokenizer::train(&data, VOCAB).expect("the tokenizer trains");
    let tokenizer_file = dir.join("tokenizer.json");
    tokenizer::write(&record, &tokenizer_file).expect("the tokenizer is written");
    let out = dir.join("packed");
    mlm::pack(&data, &tokenizer_file, SEQ_LEN as u64, &out).expect("the dataset packs");
    let packed = mlm::load(&out).expect("the packed dataset loads");
    (out, packed)
}

/// The sequences of a packed file, read here from its bytes: little-endian `i32`, `SEQ_LEN` per
/// sequence.
fn sequences(dir: &Path, file: &str) -> Vec<Vec<i32>> {
    let bytes = fs::read(dir.join(file)).expect("the packed file is written");
    assert_eq!(
        bytes.len() % (4 * SEQ_LEN),
        0,
        "{file} holds whole sequences"
    );
    let ids: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    ids.as_chunks::<SEQ_LEN>()
        .0
        .iter()
        .map(|s| s.to_vec())
        .collect()
}

/// The index of each label row of `batch` among `train`, whose rows are distinct.
fn rows_of(batch: &MlmBatch, train: &[Vec<i32>]) -> Vec<usize> {
    let index: HashMap<&[i32], usize> = train
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_slice(), i))
        .collect();
    assert_eq!(
        index.len(),
        train.len(),
        "the fixture's sequences are distinct"
    );
    batch
        .labels
        .as_chunks::<SEQ_LEN>()
        .0
        .iter()
        .map(|row| {
            *index
                .get(row.as_slice())
                .unwrap_or_else(|| panic!("labels {row:?} are a packed Train sequence"))
        })
        .collect()
}

fn uniform(word: u32) -> f64 {
    (f64::from(word) + 0.5) / 4_294_967_296.0
}

/// Scenario 3: over the batches of enough steps to cover 20 000 fixture positions, the selected
/// share lies within 0.01 of the settled selection rate, and the mask, random and keep shares of
/// the selected positions within 0.02 of 0.8, 0.1 and 0.1; no special token is selected, the
/// labels are the original ids and the weights are 1 exactly at the selected positions.
#[test]
fn proportions() {
    let (dir, packed) = packed("proportions");
    let train = sequences(&dir, mlm::TRAIN_TOKENS);
    let policy = MaskingPolicy::BERT;
    assert_eq!(
        policy.selection_rate, 0.15,
        "the settled selection rate (D19, D26)"
    );
    assert_eq!(
        (
            policy.mask_fraction,
            policy.random_fraction,
            policy.keep_fraction
        ),
        (0.8, 0.1, 0.1)
    );
    let batch_size = 4;
    let rows = batch_size * SEQ_LEN;
    let steps = POSITIONS.div_ceil(rows) as u64;

    let (mut eligible, mut selected) = (0usize, 0usize);
    let (mut masked, mut random, mut kept) = (0usize, 0usize, 0usize);
    for step in 0..steps {
        let batch = mlm::mlm_batch(&packed, SplitRole::Train, &policy, 42, step, batch_size)
            .expect("a batch");
        assert_eq!(batch.ids.len(), rows);
        assert_eq!(batch.labels.len(), rows);
        assert_eq!(batch.weights.len(), rows);
        assert_eq!(batch.positions.len(), rows);
        assert_eq!(batch.key_mask, vec![1.0; rows], "every key is unmasked");
        let expected: Vec<i32> = (0..rows).map(|n| (n % SEQ_LEN) as i32).collect();
        assert_eq!(
            batch.positions, expected,
            "positions 0 to S - 1 per sequence"
        );
        for (r, row) in rows_of(&batch, &train).into_iter().enumerate() {
            assert_eq!(
                batch.labels[r * SEQ_LEN..(r + 1) * SEQ_LEN],
                train[row][..],
                "labels are the original ids"
            );
        }
        for n in 0..rows {
            let (id, label, weight) = (batch.ids[n], batch.labels[n], batch.weights[n]);
            assert!(weight == 0.0 || weight == 1.0, "weight {weight}");
            assert!((0..VOCAB as i32).contains(&id), "id {id} in the vocabulary");
            if label < SPECIALS {
                assert_eq!(weight, 0.0, "special token {label} selected at step {step}");
                assert_eq!(id, label);
                continue;
            }
            eligible += 1;
            if weight == 0.0 {
                assert_eq!(id, label, "an unselected position keeps its token");
                continue;
            }
            selected += 1;
            if id == MASK {
                masked += 1;
            } else if id == label {
                kept += 1;
            } else {
                assert!(id >= SPECIALS, "a random token is not special: {id}");
                random += 1;
            }
        }
    }
    assert!(steps as usize * rows >= POSITIONS);
    let share = selected as f64 / eligible as f64;
    assert!((share - 0.15).abs() <= 0.01, "selected share {share}");
    let of = |count: usize| count as f64 / selected as f64;
    assert!(
        (of(masked) - 0.8).abs() <= 0.02,
        "mask share {}",
        of(masked)
    );
    assert!(
        (of(random) - 0.1).abs() <= 0.02,
        "random share {}",
        of(random)
    );
    assert!((of(kept) - 0.1).abs() <= 0.02, "keep share {}", of(kept));
}

/// Scenario 4: one `(seed, step)` gives an identical batch twice; the selection of element 0 is
/// the value derived here from `philox4x32_10` with the settled tag 3; another step gives
/// another selection.
#[test]
fn derivation() {
    let (_, packed) = packed("derivation");
    assert_eq!(Stream::Masking.tag(), 3, "the masking tag (D26)");
    assert_eq!(Stream::DataOrder.tag(), 5, "the data-order tag (D26)");
    let policy = MaskingPolicy::BERT;
    let batch = |seed: u64, step: u64| {
        mlm::mlm_batch(&packed, SplitRole::Train, &policy, seed, step, 2).expect("a batch")
    };
    let seed: u64 = 0x0123_4567_89ab_cdef;
    assert_eq!(
        batch(seed, 7),
        batch(seed, 7),
        "one (seed, step), one batch"
    );

    let (mut chosen, mut passed_over) = (0, 0);
    for step in [0u64, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 1 << 33] {
        let b = batch(seed, step);
        let label = b.labels[0];
        let word = philox4x32_10(
            [0, 3, step as u32, (step >> 32) as u32],
            [seed as u32, (seed >> 32) as u32],
        );
        let expected = label >= SPECIALS && uniform(word[0]) < policy.selection_rate;
        assert_eq!(
            b.weights[0] == 1.0,
            expected,
            "element 0 at step {step}: word 0 {:#x}",
            word[0]
        );
        if expected {
            chosen += 1;
        } else if label >= SPECIALS {
            passed_over += 1;
        }
    }
    assert!(
        chosen > 0 && passed_over > 0,
        "{chosen} chosen, {passed_over} not"
    );

    let many = |seed: u64| -> Vec<Vec<f32>> { (0..32).map(|k| batch(seed, k).weights).collect() };
    let selections = many(seed);
    let distinct: HashSet<Vec<u32>> = selections
        .iter()
        .map(|w| w.iter().map(|x| x.to_bits()).collect())
        .collect();
    assert!(distinct.len() > 1, "another step gives another selection");
    assert_ne!(
        selections,
        many(seed + 1),
        "another seed, another selection"
    );

    // The held-out split draws on the id k + 2^63.
    for step in 0..64u64 {
        let b = mlm::mlm_batch(&packed, SplitRole::HeldOut, &policy, seed, step, 1).unwrap();
        for n in 0..SEQ_LEN {
            let id = step + (1 << 63);
            let word = philox4x32_10(
                [n as u32, 3, id as u32, (id >> 32) as u32],
                [seed as u32, (seed >> 32) as u32],
            );
            let expected = b.labels[n] >= SPECIALS && uniform(word[0]) < policy.selection_rate;
            assert_eq!(
                b.weights[n] == 1.0,
                expected,
                "held-out step {step} row {n}"
            );
        }
    }
}

/// Scenario 5: across one epoch every Train sequence appears in exactly one batch, at most once;
/// the order is the one derived here from stream tag 5; two seeds give two orders.
#[test]
fn epoch_order() {
    let (dir, packed) = packed("epoch_order");
    let train = sequences(&dir, mlm::TRAIN_TOKENS);
    let n = train.len();
    assert!(n >= 8, "the fixture packs {n} Train sequences");
    let policy = MaskingPolicy::BERT;
    let order = |seed: u64, batch_size: usize, epoch: u64| -> Vec<usize> {
        let per_epoch = (n / batch_size) as u64;
        (epoch * per_epoch..(epoch + 1) * per_epoch)
            .flat_map(|step| {
                let batch =
                    mlm::mlm_batch(&packed, SplitRole::Train, &policy, seed, step, batch_size)
                        .expect("a batch");
                rows_of(&batch, &train)
            })
            .collect()
    };

    for batch_size in 1..=5 {
        let visited = order(42, batch_size, 0);
        let distinct: HashSet<usize> = visited.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            visited.len(),
            "B {batch_size}: at most once"
        );
        assert_eq!(
            visited.len(),
            n / batch_size * batch_size,
            "B {batch_size}: the incomplete last batch is not used"
        );
        if n.is_multiple_of(batch_size) {
            assert_eq!(distinct.len(), n, "B {batch_size}: every sequence once");
        }
    }

    let seed: u64 = 42;
    let derived = |epoch: u64| -> Vec<usize> {
        let mut keyed: Vec<(u64, usize)> = (0..n)
            .map(|i| {
                let w = philox4x32_10(
                    [i as u32, 5, epoch as u32, (epoch >> 32) as u32],
                    [seed as u32, (seed >> 32) as u32],
                );
                ((u64::from(w[1]) << 32) | u64::from(w[0]), i)
            })
            .collect();
        keyed.sort_unstable();
        keyed.into_iter().map(|(_, i)| i).collect()
    };
    assert_eq!(order(seed, 1, 0), derived(0), "epoch 0 is the tag-5 order");
    assert_eq!(order(seed, 1, 1), derived(1), "epoch 1 is the tag-5 order");
    assert_ne!(derived(0), derived(1), "each epoch has its own order");
    assert_eq!(
        mlm::epoch_order(seed, 0, n as u32),
        derived(0).iter().map(|&i| i as u32).collect::<Vec<_>>()
    );
    assert_ne!(order(42, 1, 0), order(43, 1, 0), "two seeds, two orders");

    let heldout = sequences(&dir, mlm::HELDOUT_TOKENS);
    let batch = mlm::mlm_batch(&packed, SplitRole::HeldOut, &policy, seed, 0, 1).unwrap();
    assert_eq!(
        batch.labels, heldout[0],
        "the held-out split in stored order"
    );
    let last = mlm::mlm_batch(
        &packed,
        SplitRole::HeldOut,
        &policy,
        seed,
        heldout.len() as u64,
        1,
    )
    .unwrap();
    assert_eq!(last.labels, heldout[0], "the held-out order wraps");
}

/// The packed record names files whose blake3 hashes and lengths it states, and a refused
/// batch names why.
#[test]
fn record_hashes_and_refusals() {
    let (dir, packed) = packed("record_hashes");
    let record = &packed.record;
    assert_eq!(record.seq_len, SEQ_LEN as u64);
    assert_eq!(record.vocabulary_size, VOCAB as u64);
    let mut digests = blake3::Hasher::new();
    for (split, file) in record
        .splits
        .iter()
        .zip([mlm::TRAIN_TOKENS, mlm::HELDOUT_TOKENS])
    {
        let bytes = fs::read(dir.join(file)).unwrap();
        let hash = blake3::hash(&bytes);
        assert_eq!(split.content_hash, hash.to_hex().to_string(), "{file}");
        assert_eq!(
            bytes.len() as u64,
            split.sequence_count * 4 * SEQ_LEN as u64
        );
        digests.update(hash.as_bytes());
    }
    assert_eq!(record.splits[0].role, SplitRole::Train);
    assert_eq!(record.splits[1].role, SplitRole::HeldOut);
    let id = &record.packed_id;
    assert_eq!(&id[14..15], "8", "{id} is a version-8 UUID");
    let hex: String = id.chars().filter(|c| *c != '-').collect();
    let leading = &digests.finalize().to_hex()[..32];
    assert_eq!(hex[..12], leading[..12], "packed_id over the split digests");

    let policy = MaskingPolicy::BERT;
    let train = record.splits[0].sequence_count as usize;
    assert!(mlm::mlm_batch(&packed, SplitRole::Train, &policy, 1, 0, 0).is_err());
    assert!(mlm::mlm_batch(&packed, SplitRole::Train, &policy, 1, 0, train + 1).is_err());
    assert!(mlm::mlm_batch(&packed, SplitRole::Train, &policy, 1, 0, train).is_ok());

    fs::write(dir.join(mlm::TRAIN_TOKENS), [0u8; 4 * SEQ_LEN]).unwrap();
    let error = mlm::load(&dir)
        .err()
        .expect("an altered packed file is refused");
    assert!(
        error.to_string().contains(mlm::TRAIN_TOKENS),
        "names the file: {error}"
    );
}
