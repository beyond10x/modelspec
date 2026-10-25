//! Adversary pass 1 on story:bert-mlm-batches: `dataset::mlm` driven against DESIGN.md § 2.3
//! (stream tags 3 and 5) and `spec/domains/data.yaml` (`MaskingPolicy`, `PackedDataset`,
//! `TrainingDataBinding`) rather than against the unit's own tests.
//!
//! Corpora are the fixture `tests/fixtures/corpus-tokenizer/` or original text written here; the
//! packed directories of the boundary cases are written here byte by byte with their hashes, so
//! `mlm::load` accepts them exactly as it accepts what `pack` writes.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use modelspec_runtime::dataset::mlm::{
    self, BatchError, MaskingPolicy, MlmBatch, Packed, PackedDataset, PackedSplit, PackedState,
};
use modelspec_runtime::dataset::tokenizer::{
    self, SPECIAL_TOKENS, Tokenizer, TokenizerKind, TokenizerState, Vocabulary,
};
use modelspec_runtime::dataset::{self, SplitRole};
use modelspec_runtime::train::init::philox4x32_10;

const SEQ_LEN: usize = 16;
const VOCAB: usize = 300;
const SPECIALS: i32 = 5;
const MASK: i32 = 4;
const SEP: i32 = 3;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus-tokenizer")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adversary_mlm_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// The fixture prepared, tokenized at 300 tokens and packed at 16, as the unit's own cases do.
fn fixture_packed(case: &str) -> (PathBuf, Packed) {
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

/// A version-8 UUID of the leading 16 bytes of a digest, as `spec/domains/data.yaml` defines the
/// ids of `Dataset` and `Tokenizer`.
fn uuid_v8(hash: &[u8; 32]) -> String {
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

/// A tokenizer record over the special tokens and `extra`, its id the hash of its tokens.
fn hand_tokenizer(extra: &[&str]) -> Tokenizer {
    let mut tokens: Vec<String> = SPECIAL_TOKENS.iter().map(|t| t.to_string()).collect();
    tokens.extend(extra.iter().map(|t| t.to_string()));
    let mut id = blake3::Hasher::new();
    for token in &tokens {
        id.update(token.as_bytes());
        id.update(b"\n");
    }
    Tokenizer {
        tokenizer_id: uuid_v8(id.finalize().as_bytes()),
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

fn le_bytes(ids: &[i32]) -> Vec<u8> {
    ids.iter().flat_map(|t| t.to_le_bytes()).collect()
}

/// A packed directory written here: the two files of `ids` and a record whose lengths and hashes
/// match them, so `load` accepts it.
fn hand_packed(case: &str, seq_len: usize, vocab: u64, train: &[i32], heldout: &[i32]) -> Packed {
    let dir = scratch(case);
    let mut splits = Vec::new();
    for (role, file, ids) in [
        (SplitRole::Train, mlm::TRAIN_TOKENS, train),
        (SplitRole::HeldOut, mlm::HELDOUT_TOKENS, heldout),
    ] {
        assert_eq!(ids.len() % seq_len, 0);
        let bytes = le_bytes(ids);
        fs::write(dir.join(file), &bytes).unwrap();
        splits.push(PackedSplit {
            role,
            sequence_count: (ids.len() / seq_len) as u64,
            content_hash: blake3::hash(&bytes).to_hex().to_string(),
        });
    }
    let record = PackedDataset {
        packed_id: uuid_v8(&[7; 32]),
        binding_id: uuid_v8(&[9; 32]),
        seq_len: seq_len as u64,
        vocabulary_size: vocab,
        splits,
        heldout_unigram_baseline_nats: 1.0,
        state: PackedState::Recorded,
    };
    let json = serde_json::to_string_pretty(&record).unwrap();
    fs::write(dir.join(mlm::PACKED_FILE), json).unwrap();
    mlm::load(&dir).expect("the hand-written packed directory loads")
}

fn uniform(word: u32) -> f64 {
    (f64::from(word) + 0.5) / 4_294_967_296.0
}

/// The id DESIGN.md § 2.3 (stream tag 3) gives flattened row `n` of step `step`, written from
/// the text: counter `[n, 3, id lo, id hi]`, word 0 selects, word 1 picks, word 2 is the random
/// token `5 + floor(w2 * (V - 5) / 2^32)`. Returns `(id, selected)`.
fn derived(seed: u64, draw_id: u64, n: u32, label: i32, vocab: u64) -> (i32, bool) {
    let policy = MaskingPolicy::BERT;
    let w = philox4x32_10(
        [n, 3, draw_id as u32, (draw_id >> 32) as u32],
        [seed as u32, (seed >> 32) as u32],
    );
    let selected = label >= SPECIALS && uniform(w[0]) < policy.selection_rate;
    if !selected {
        return (label, false);
    }
    let id = if uniform(w[1]) < policy.mask_fraction {
        MASK
    } else if uniform(w[1]) < policy.mask_fraction + policy.random_fraction {
        let r = (u128::from(w[2]) * u128::from(vocab - 5)) >> 32;
        5 + r as i32
    } else {
        label
    };
    (id, true)
}

/// Attack 2: every row of a batch of several sequences follows the tag-3 derivation with `n` the
/// flattened row index, and every selected row's substituted id (the random token included)
/// is the one § 2.3 states. The unit's `derivation` case checks element 0 of `Train` and the rows
/// of a one-sequence `HeldOut` batch only, where the flattened index equals the position.
#[test]
fn adv_mlm_every_row_follows_the_tag_3_derivation() {
    let (_, packed) = fixture_packed("every_row");
    let policy = MaskingPolicy::BERT;
    let seed: u64 = 0xdead_beef_0bad_f00d;
    let mut beyond_first_sequence = 0;
    let mut random_checked = 0;
    let steps: Vec<u64> = (0..48).chain([1 << 33, u64::MAX >> 2]).collect();
    for (split, batch_size) in [(SplitRole::Train, 4usize), (SplitRole::HeldOut, 2)] {
        for &step in &steps {
            let batch = mlm::mlm_batch(&packed, split, &policy, seed, step, batch_size).unwrap();
            let draw_id = match split {
                SplitRole::Train => step,
                SplitRole::HeldOut => step.wrapping_add(1 << 63),
            };
            for n in 0..batch_size * SEQ_LEN {
                let (id, selected) =
                    derived(seed, draw_id, n as u32, batch.labels[n], VOCAB as u64);
                assert_eq!(
                    (batch.ids[n], batch.weights[n] == 1.0),
                    (id, selected),
                    "{split:?} step {step} row {n} (label {})",
                    batch.labels[n]
                );
                if selected && n >= SEQ_LEN {
                    beyond_first_sequence += 1;
                }
                if selected && id != MASK && id != batch.labels[n] {
                    random_checked += 1;
                }
            }
        }
    }
    assert!(beyond_first_sequence > 50, "{beyond_first_sequence}");
    assert!(random_checked > 5, "{random_checked}");
}

/// Attack 2: two rows holding the same ids at the same positions of two sequences of one batch
/// draw independently, because the counter's index is the flattened row and not the position.
#[test]
fn adv_mlm_identical_sequences_at_two_rows_draw_differently() {
    let seq_len = 8;
    let sequence: Vec<i32> = (0..seq_len as i32).map(|p| 5 + p).collect();
    let train: Vec<i32> = sequence.iter().copied().cycle().take(seq_len * 4).collect();
    let packed = hand_packed("identical_rows", seq_len, 64, &train, &sequence);
    let policy = MaskingPolicy::BERT;
    let mut differing_steps = 0;
    for step in 0..200 {
        let batch = mlm::mlm_batch(&packed, SplitRole::Train, &policy, 3, step, 2).unwrap();
        assert_eq!(batch.labels[..seq_len], batch.labels[seq_len..]);
        if batch.weights[..seq_len] != batch.weights[seq_len..] {
            differing_steps += 1;
        }
    }
    assert!(
        differing_steps > 100,
        "the two rows' selections differ at {differing_steps} of 200 steps"
    );
}

/// Attack 1: a sequence of special tokens only has nothing selected and keeps every id.
#[test]
fn adv_mlm_all_special_sequences_select_nothing() {
    let seq_len = 8;
    let specials = [SEP, 0, 1, 2, MASK, SEP, SEP, SEP];
    let train: Vec<i32> = specials.iter().copied().cycle().take(seq_len * 3).collect();
    let packed = hand_packed("all_special", seq_len, 300, &train, &specials);
    let policy = MaskingPolicy::BERT;
    for step in 0..500 {
        for (split, b) in [(SplitRole::Train, 3), (SplitRole::HeldOut, 1)] {
            let batch = mlm::mlm_batch(&packed, split, &policy, 11, step, b).unwrap();
            assert!(batch.weights.iter().all(|&w| w == 0.0), "step {step}");
            assert_eq!(batch.ids, batch.labels, "step {step}");
        }
    }
}

/// Attack 1: with a vocabulary of six the random range `[5, 6)` holds one token, so a random
/// substitution of token 5 gives token 5 back; selected rows are `[MASK]` or 5, about 80/20.
#[test]
fn adv_mlm_vocabulary_of_six() {
    let seq_len = 8;
    let train = vec![5; seq_len * 4];
    let packed = hand_packed("vocab_six", seq_len, 6, &train, &train[..seq_len]);
    let policy = MaskingPolicy::BERT;
    let (mut selected, mut masked) = (0usize, 0usize);
    for step in 0..2000 {
        let batch = mlm::mlm_batch(&packed, SplitRole::Train, &policy, 5, step, 4).unwrap();
        for n in 0..batch.ids.len() {
            if batch.weights[n] == 1.0 {
                selected += 1;
                assert!(
                    batch.ids[n] == MASK || batch.ids[n] == 5,
                    "{}",
                    batch.ids[n]
                );
                masked += usize::from(batch.ids[n] == MASK);
            } else {
                assert_eq!(batch.ids[n], 5);
            }
        }
    }
    let share = masked as f64 / selected as f64;
    assert!(
        (share - 0.8).abs() < 0.02,
        "mask share {share} of {selected}"
    );
}

/// Attack 1: `spec/domains/data.yaml` states `vocabulary_size > 5` for a `PackedDataset`; `load`
/// should refuse a record that breaks it rather than hand it to `mlm_batch`, which computes
/// `vocabulary_size - 5` on every call.
#[test]
fn adv_load_refuses_a_vocabulary_size_the_spec_forbids() {
    let seq_len = 4;
    let train = vec![SEP; seq_len * 2];
    let dir_case = "vocab_four";
    let result = std::panic::catch_unwind(|| {
        let packed = hand_packed(dir_case, seq_len, 4, &train, &train[..seq_len]);
        mlm::mlm_batch(&packed, SplitRole::Train, &MaskingPolicy::BERT, 1, 0, 1)
    });
    match result {
        Ok(batch) => panic!("a record of vocabulary_size 4 was loaded and batched: {batch:?}"),
        Err(e) => {
            let message = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            assert!(
                message.contains("loads"),
                "load should refuse; the panic was elsewhere: {message}"
            );
        }
    }
}

/// Attack 3: the epoch order is a bijection for any `n`, the empty one included, and depends on
/// the seed and on both halves of the epoch.
#[test]
fn adv_epoch_order_is_a_bijection_for_any_n() {
    for n in [0u32, 1, 2, 3, 5, 7, 31, 33, 100, 257, 1000] {
        for (seed, epoch) in [(0u64, 0u64), (1, 0), (42, 7), (u64::MAX, 1 << 32)] {
            let mut order = mlm::epoch_order(seed, epoch, n);
            order.sort_unstable();
            assert_eq!(order, (0..n).collect::<Vec<_>>(), "n {n} seed {seed}");
        }
        if n >= 3 {
            let o = |seed, epoch| mlm::epoch_order(seed, epoch, n);
            assert_ne!(o(0, 0), o(1, 0), "n {n}: seed 0 and 1");
            for e in [0u64, 1, u64::from(u32::MAX)] {
                assert_ne!(o(9, e), o(9, e + 1), "n {n}: epoch {e} and {}", e + 1);
            }
            assert_ne!(o(9, 0), o(9, 1 << 32), "n {n}: the high word of the epoch");
        }
    }
}

/// Attack 3: fewer sequences than a batch is refused, naming both numbers; exactly one batch per
/// epoch makes every step a new epoch.
#[test]
fn adv_mlm_fewer_sequences_than_a_batch() {
    let seq_len = 4;
    let train: Vec<i32> = (0..seq_len as i32 * 3).map(|i| 5 + i).collect();
    let packed = hand_packed("fewer_than_b", seq_len, 64, &train, &train[..seq_len]);
    let policy = MaskingPolicy::BERT;
    assert_eq!(
        mlm::mlm_batch(&packed, SplitRole::Train, &policy, 1, 0, 4),
        Err(BatchError::TooLarge {
            batch_size: 4,
            sequences: 3
        })
    );
    let first = |step| {
        mlm::mlm_batch(&packed, SplitRole::Train, &policy, 1, step, 3)
            .unwrap()
            .labels
    };
    let orders: HashSet<Vec<i32>> = (0..20).map(first).collect();
    assert!(orders.len() > 1, "each step is its own epoch order");
}

/// Attack 6: a packed file changed in place, its length kept and every id still in the
/// vocabulary, is refused by its hash, naming the file. The unit's case shortens the file, so
/// only the length check runs there.
#[test]
fn adv_load_refuses_a_flipped_byte_naming_the_file() {
    let (dir, packed) = fixture_packed("flipped_byte");
    drop(packed);
    let path = dir.join(mlm::HELDOUT_TOKENS);
    let mut bytes = fs::read(&path).unwrap();
    let at = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .position(|b| (6..VOCAB as i32 - 1).contains(&i32::from_le_bytes(*b)))
        .expect("a held-out id that stays in the vocabulary when its low bit flips");
    bytes[at * 4] ^= 1;
    fs::write(&path, &bytes).unwrap();
    let error = mlm::load(&dir)
        .err()
        .expect("a changed packed file is refused");
    let message = error.to_string();
    assert!(message.contains(mlm::HELDOUT_TOKENS), "{message}");
    assert!(message.contains("blake3"), "refused by its hash: {message}");
}

/// Attack 6: the `binding_id` changes when any rate moves by one ULP or `seq_len` by one.
#[test]
fn adv_binding_id_moves_with_one_ulp_of_any_rate() {
    let d = "0ddcfa78-c328-8cc9-ae3b-490c096d55a5";
    let t = "dfd98df4-cbd0-8076-b713-99ab202efa85";
    let base = mlm::binding_id(d, t, 128, &MaskingPolicy::BERT).unwrap();
    let ulp = |x: f64| f64::from_bits(x.to_bits() + 1);
    let mut ids = HashSet::from([base.clone()]);
    for k in 0..4 {
        let mut p = MaskingPolicy::BERT;
        match k {
            0 => p.selection_rate = ulp(p.selection_rate),
            1 => p.mask_fraction = ulp(p.mask_fraction),
            2 => p.random_fraction = ulp(p.random_fraction),
            _ => p.keep_fraction = ulp(p.keep_fraction),
        }
        assert!(
            ids.insert(mlm::binding_id(d, t, 128, &p).unwrap()),
            "rate {k}"
        );
    }
    for s in [127, 129] {
        assert!(ids.insert(mlm::binding_id(d, t, s, &MaskingPolicy::BERT).unwrap()));
    }
}

/// A corpus written here, prepared, and a tokenizer over its two words.
fn tiny_inputs(case: &str, train: &str, valid: &str) -> (PathBuf, PathBuf, PathBuf) {
    let dir = scratch(case);
    let corpus = dir.join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    fs::write(corpus.join("wiki.train.raw"), train).unwrap();
    fs::write(corpus.join("wiki.valid.raw"), valid).unwrap();
    let data = dir.join("dataset");
    dataset::prepare(&corpus, &data).expect("the corpus prepares");
    let tokenizer_file = dir.join("tokenizer.json");
    tokenizer::write(&hand_tokenizer(&["a", "b"]), &tokenizer_file).unwrap();
    (dir, data, tokenizer_file)
}

/// Attack 6, the implementor's self-reported case: two sequence lengths that both divide both
/// streams give byte-identical packed files, and so the same `packed_id`, for two records that
/// differ in `seq_len`, `sequence_count` and `binding_id`. An identity derived from content must
/// change with what changes the content's meaning.
#[test]
fn adv_packed_id_changes_with_seq_len() {
    let (dir, data, tokenizer_file) =
        tiny_inputs("packed_id_seq_len", " a b a b a b a b \n", " b a b a \n");
    let (_, two) = mlm::pack(&data, &tokenizer_file, 2, &dir.join("s2")).unwrap();
    let (_, four) = mlm::pack(&data, &tokenizer_file, 4, &dir.join("s4")).unwrap();
    for file in [mlm::TRAIN_TOKENS, mlm::HELDOUT_TOKENS] {
        assert_eq!(
            fs::read(dir.join("s2").join(file)).unwrap(),
            fs::read(dir.join("s4").join(file)).unwrap(),
            "{file}: the same ids"
        );
    }
    assert_ne!(two.binding_id, four.binding_id);
    assert_ne!(two.splits[0].sequence_count, four.splits[0].sequence_count);
    assert_ne!(
        two.packed_id, four.packed_id,
        "two PackedDataset records with seq_len 2 and 4 share packed_id"
    );
}

fn batch_digest(batches: &[MlmBatch]) -> String {
    let mut h = blake3::Hasher::new();
    for b in batches {
        h.update(&le_bytes(&b.ids));
        h.update(&le_bytes(&b.labels));
        h.update(&le_bytes(&b.positions));
        for w in b.weights.iter().chain(&b.key_mask) {
            h.update(&w.to_le_bytes());
        }
    }
    h.finalize().to_hex().to_string()
}

fn digest_of_this_process(case: &str) -> String {
    let (_, packed) = fixture_packed(case);
    let policy = MaskingPolicy::BERT;
    let batches: Vec<MlmBatch> = (0..12)
        .flat_map(|step| {
            [
                mlm::mlm_batch(&packed, SplitRole::Train, &policy, 77, step, 3).unwrap(),
                mlm::mlm_batch(&packed, SplitRole::HeldOut, &policy, 77, step, 1).unwrap(),
            ]
        })
        .collect();
    batch_digest(&batches)
}

/// Attack 2: one `(seed, step)` gives the same batches in two other processes as in this one.
#[test]
fn adv_mlm_batches_are_identical_across_processes() {
    const CHILD: &str = "ADV_MLM_CHILD_OUT";
    if let Ok(out) = std::env::var(CHILD) {
        let name = Path::new(&out).file_name().unwrap().to_string_lossy();
        let digest = digest_of_this_process(&format!("process_{name}"));
        fs::write(&out, digest).unwrap();
        return;
    }
    let here = digest_of_this_process("process_parent");
    let dir = scratch("process_outputs");
    for k in 0..2 {
        let out = dir.join(format!("child{k}"));
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "adv_mlm_batches_are_identical_across_processes",
                "--exact",
                "--test-threads=1",
            ])
            .env(CHILD, &out)
            .stdout(Stdio::null())
            .status()
            .expect("the test binary runs");
        assert!(status.success());
        assert_eq!(fs::read_to_string(&out).unwrap(), here, "process {k}");
    }
}

/// Attacks 4 and 5 on the real release: the `[SEP]` counts of the packed files against the
/// documents of `dataset.json`, and the held-out unigram baseline recomputed here from the
/// packed ids. Set `ADV_MLM_REAL_PACKED` to the packed directory of the release.
#[test]
#[ignore = "needs the packed real release in ADV_MLM_REAL_PACKED"]
fn adv_real_release_separators_and_baseline() {
    let dir = PathBuf::from(std::env::var("ADV_MLM_REAL_PACKED").expect("ADV_MLM_REAL_PACKED"));
    let packed = mlm::load(&dir).expect("the real release loads");
    let v = packed.record.vocabulary_size as usize;
    let train = packed.ids(SplitRole::Train);
    let heldout = packed.ids(SplitRole::HeldOut);
    let seps = |ids: &[i32]| ids.iter().filter(|&&t| t == SEP).count();
    let specials = |ids: &[i32]| ids.iter().filter(|&&t| t < SPECIALS && t != SEP).count();
    println!(
        "train ids {} seps {} other specials {}; heldout ids {} seps {} other specials {}",
        train.len(),
        seps(train),
        specials(train),
        heldout.len(),
        seps(heldout),
        specials(heldout)
    );
    assert!((28_470..=28_471).contains(&seps(train)), "{}", seps(train));
    assert!((58..=59).contains(&seps(heldout)), "{}", seps(heldout));

    let mut counts = vec![0u64; v];
    for &t in train.iter().filter(|&&t| t >= SPECIALS) {
        counts[t as usize] += 1;
    }
    let total: u64 = counts.iter().sum();
    let types = (v - 5) as f64;
    let mut sum = 0.0f64;
    let mut m = 0u64;
    for &t in heldout.iter().filter(|&&t| t >= SPECIALS) {
        sum += ((counts[t as usize] as f64 + 1.0) / (total as f64 + types)).ln();
        m += 1;
    }
    let baseline = -sum / m as f64;
    let entropy: f64 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total as f64;
            -p * p.ln()
        })
        .sum();
    let unseen = (5..v).filter(|&t| counts[t] == 0).count();
    println!(
        "baseline recomputed {baseline}, recorded {}, uniform ln(V-5) {}, train unigram entropy {entropy}, perplexity {}, unseen train types {unseen}",
        packed.record.heldout_unigram_baseline_nats,
        types.ln(),
        baseline.exp()
    );
    assert!((baseline - packed.record.heldout_unigram_baseline_nats).abs() <= 1e-9);
    assert!(baseline < types.ln());

    let policy = MaskingPolicy::BERT;
    let (mut eligible, mut selected, mut empty) = (0usize, 0usize, 0usize);
    for step in 0..400 {
        let b = mlm::mlm_batch(&packed, SplitRole::Train, &policy, 1234, step, 32).unwrap();
        eligible += b.labels.iter().filter(|&&t| t >= SPECIALS).count();
        let s = b.weights.iter().filter(|&&w| w == 1.0).count();
        selected += s;
        empty += usize::from(s == 0);
    }
    println!(
        "selected share over 400 steps of 32 x 128: {}; batches with nothing selected: {empty}",
        selected as f64 / eligible as f64
    );
}

/// Attack 1, the caller's mean: on the fixture at `S = 16`, a one-sequence batch with no
/// selected row, so `sum(weights) = 0`, is reachable (probability about `0.85^14`).
#[test]
fn adv_mlm_batches_with_nothing_selected_occur() {
    let (_, packed) = fixture_packed("nothing_selected");
    let policy = MaskingPolicy::BERT;
    let empty = (0..1000)
        .filter(|&step| {
            let b = mlm::mlm_batch(&packed, SplitRole::Train, &policy, 99, step, 1).unwrap();
            b.weights.iter().all(|&w| w == 0.0)
        })
        .count();
    println!("{empty} of 1000 one-sequence batches select nothing");
    assert!(empty > 30, "{empty}");
}
