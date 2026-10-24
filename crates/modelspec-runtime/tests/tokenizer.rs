//! `dataset::tokenizer` (story:bert-tokenizer; `spec/domains/data.yaml`, `modelspec.data.Tokenizer`
//! and coordinator decision D23): a WordPiece tokenizer trained on the `Train` split of a prepared
//! dataset, whose `decode(encode(text))` gives back the text as the settled normalization defines
//! it.
//!
//! The corpora are the fixtures under `tests/fixtures/`, original text in the release's layout,
//! prepared through `dataset::prepare` here, so no case needs the corpus or the network.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use modelspec_runtime::dataset::tokenizer::{self, Tokenizer};
use modelspec_runtime::dataset::{self, SplitRole};

/// The vocabulary size scenario 3 names.
const SIZE: usize = 300;
/// `[UNK]`'s settled id (coordinator decision D19).
const UNK: u32 = 1;

/// The corpus of story:bert-corpus-splits, pinned by its tests.
const CORPUS: &str = "corpus";
/// This story's corpus: `zqjx` six times in the `Train` split only and `xjqz` six times in
/// the `HeldOut` split only; neither of their four letters is in any other word of the `Train`
/// split.
const CORPUS_TOKENIZER: &str = "corpus-tokenizer";
const TRAIN_WORD: &str = "zqjx";
const HELD_OUT_WORD: &str = "xjqz";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// A fresh, empty directory under this crate's build directory, named for the case.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("tokenizer_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// The fixture `name` prepared into a dataset directory for `case`.
fn prepared(name: &str, case: &str) -> PathBuf {
    let out = scratch(&format!("{case}_{name}"));
    dataset::prepare(&fixture(name), &out).expect("the fixture prepares");
    out
}

fn split_text(dataset: &Path, role: SplitRole) -> String {
    fs::read_to_string(dataset.join(role.split_file())).expect("the split is written")
}

/// The words of `text` under the settled normalization (coordinator decision D23), written here
/// from the settlement and not from the code under test: Unicode lowercase, control characters
/// dropped, whitespace separates words, and every character that is neither alphanumeric nor
/// whitespace is a word of its own.
fn normalized_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    for c in text.to_lowercase().chars() {
        if c.is_whitespace() || !(c.is_alphanumeric() || c.is_control()) {
            if !word.is_empty() {
                words.push(std::mem::take(&mut word));
            }
            if !c.is_whitespace() {
                words.push(c.to_string());
            }
        } else if c.is_alphanumeric() {
            word.push(c);
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    words
}

/// The documents of a split file: a document starts at an article heading, a line ` = <title> = `
/// with a blank line before and after it (story:bert-corpus-splits); text before the first one is
/// a document too.
fn documents(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let blank = |i: usize| {
        lines
            .get(i)
            .is_some_and(|l| l.trim_end_matches('\n') == " ")
    };
    let mut docs = vec![String::new()];
    for (i, line) in lines.iter().enumerate() {
        if i > 0 && blank(i - 1) && blank(i + 1) && dataset::is_article_heading(line) {
            docs.push(String::new());
        }
        docs.last_mut().unwrap().push_str(line);
    }
    docs.retain(|d| !d.trim().is_empty());
    docs
}

/// How often each normalized word occurs in `text`.
fn word_counts(text: &str) -> HashMap<String, u64> {
    let mut counts = HashMap::new();
    for word in normalized_words(text) {
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}

fn trained(dataset: &Path, size: usize) -> Tokenizer {
    tokenizer::train(dataset, size).unwrap_or_else(|e| panic!("training to {size} tokens: {e}"))
}

/// Scenario 3: every fixture document, both splits, survives `decode(encode(text))` as the
/// settlement defines it, and no id is outside `[0, 300)`. A word encodes to `[UNK]` only when it
/// holds a character the `Train` split does not (every character of the `Train` split is in the
/// alphabet, which has room for 147 characters at 300 tokens) or is over 100 characters long; in
/// the `Train` split no word does.
#[test]
fn round_trip() {
    for name in [CORPUS, CORPUS_TOKENIZER] {
        let dataset = prepared(name, "round_trip");
        let tokenizer = trained(&dataset, SIZE);
        assert_eq!(tokenizer.vocabulary.tokens.len(), SIZE, "{name}");
        let codec = tokenizer.codec();
        let alphabet: HashSet<char> = normalized_words(&split_text(&dataset, SplitRole::Train))
            .iter()
            .flat_map(|w| w.chars())
            .collect();
        let mut checked = 0;
        for role in SplitRole::ALL {
            let docs = documents(&split_text(&dataset, role));
            assert!(!docs.is_empty(), "{name} {role:?} holds documents");
            for doc in docs {
                let ids = codec.encode(&doc);
                assert!(!ids.is_empty(), "{name} {role:?}: {doc:?} encodes to ids");
                for id in &ids {
                    assert!((*id as usize) < SIZE, "{name} {role:?}: id {id} of {doc:?}");
                }
                let expected: Vec<String> = normalized_words(&doc)
                    .into_iter()
                    .map(|w| {
                        let unknown =
                            w.chars().count() > 100 || w.chars().any(|c| !alphabet.contains(&c));
                        if unknown {
                            assert_eq!(role, SplitRole::HeldOut, "{name}: {w:?} is unknown");
                            "[UNK]".to_string()
                        } else {
                            w
                        }
                    })
                    .collect();
                let unknown = expected.iter().filter(|w| *w == "[UNK]").count();
                assert_eq!(
                    ids.iter().filter(|id| **id == UNK).count(),
                    unknown,
                    "{name} {role:?}: one [UNK] per unknown word of {doc:?}"
                );
                assert_eq!(
                    codec.decode(&ids).as_deref(),
                    Some(expected.join(" ").as_str()),
                    "{name} {role:?}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 3, "{name}: {checked} documents checked");
    }
}

/// Scenario 4: `xjqz`, six times in the `HeldOut` split and never in `Train`, is not a single
/// vocabulary token, while `zqjx`, six times in `Train`, is. Trained on both splits together the
/// same algorithm does make `xjqz` a token, so the case tells the two apart.
#[test]
fn train_split_only() {
    let dataset = prepared(CORPUS_TOKENIZER, "train_split_only");
    let train = word_counts(&split_text(&dataset, SplitRole::Train));
    let held_out = word_counts(&split_text(&dataset, SplitRole::HeldOut));
    assert_eq!(train.get(TRAIN_WORD), Some(&6));
    assert_eq!(held_out.get(HELD_OUT_WORD), Some(&6), "equal frequency");
    assert_eq!(
        train.get(HELD_OUT_WORD),
        None,
        "{HELD_OUT_WORD} is held out only"
    );
    assert_eq!(
        held_out.get(TRAIN_WORD),
        None,
        "{TRAIN_WORD} is in Train only"
    );

    let tokenizer = trained(&dataset, SIZE);
    let tokens = &tokenizer.vocabulary.tokens;
    assert!(
        tokens.iter().any(|t| t == TRAIN_WORD),
        "{TRAIN_WORD} is a token"
    );
    assert!(
        !tokens.iter().any(|t| t == HELD_OUT_WORD),
        "{HELD_OUT_WORD} is no token"
    );
    let pieces = tokenizer.codec().encode(HELD_OUT_WORD);
    assert!(pieces.len() > 1 && !pieces.contains(&UNK), "{pieces:?}");

    let mut both = tokenizer::WordCounts::new();
    for role in SplitRole::ALL {
        let file = File::open(dataset.join(role.split_file())).unwrap();
        for (word, count) in tokenizer::count_words(BufReader::new(file)).unwrap() {
            *both.entry(word).or_insert(0) += count;
        }
    }
    let control = tokenizer::train_counts(&both, SIZE).expect("both splits reach 300 tokens");
    assert!(
        control.vocabulary.tokens.iter().any(|t| t == HELD_OUT_WORD),
        "trained on both splits, {HELD_OUT_WORD} would be a token"
    );
}

/// The record's id is the leading 16 bytes, as a version-8 UUID, of blake3 over the token list,
/// each token followed by a newline (coordinator decision D23), computed here from the tokens.
#[test]
fn record_id_is_derived_from_the_token_list() {
    let dataset = prepared(CORPUS, "record_id");
    let tokenizer = trained(&dataset, SIZE);
    let mut hasher = blake3::Hasher::new();
    for token in &tokenizer.vocabulary.tokens {
        hasher.update(token.as_bytes());
        hasher.update(b"\n");
    }
    let digest = hasher.finalize();
    let mut b = [0u8; 16];
    b.copy_from_slice(&digest.as_bytes()[..16]);
    b[6] = (b[6] & 0x0f) | 0x80;
    b[8] = (b[8] & 0x3f) | 0x80;
    let hex: String = b.iter().map(|v| format!("{v:02x}")).collect();
    let expected = format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    );
    assert_eq!(tokenizer.tokenizer_id, expected);
}

/// The settled encoding rules on a hand-built vocabulary: greedy longest match first, a word with
/// an unmatchable remainder or over 100 characters is `[UNK]` whole, no `[CLS]`/`[SEP]` frame.
#[test]
fn encode_is_greedy_longest_match_first() {
    let mut counts = tokenizer::WordCounts::new();
    counts.insert("abab".into(), 5);
    let tokenizer = tokenizer::train_counts(&counts, 5 + 4 + 1).unwrap();
    let tokens = &tokenizer.vocabulary.tokens;
    assert_eq!(
        tokens[..5],
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"].map(String::from)
    );
    let codec = tokenizer.codec();
    let id = |t: &str| tokens.iter().position(|x| x == t).unwrap() as u32;
    // (a, ##b), (##b, ##a) and (##a, ##b) all occur 5 times; the byte order of the merged token
    // breaks the tie, and `#` sorts before `a`.
    assert_eq!(tokens[9], "##ab");
    // Longest match first: after `a ##b` the remainder `ab` is `##ab`, not `##a ##b`, which a
    // shortest-match encoder gives and which decodes to the same text.
    assert_eq!(codec.encode("abab"), [id("a"), id("##b"), id("##ab")]);
    assert_eq!(codec.encode("b"), [id("b")]);
    assert_eq!(codec.encode("AB, c"), [id("a"), id("##b"), UNK, UNK]);
    assert_eq!(codec.encode(&"a".repeat(100)).len(), 100);
    assert_eq!(codec.encode(&"a".repeat(101)), [UNK]);
    assert_eq!(
        codec.decode(&[id("a"), id("##b"), id("b")]).unwrap(),
        "ab b"
    );
    assert_eq!(codec.decode(&[SIZE as u32]), None);
}

/// Merges are ranked by pair frequency (coordinator decision D25), not by the likelihood score
/// `count(ab) / (count(a) * count(b))`: `ab` occurs 10 times and `cd` twice, so `ab` is merged
/// first, although `cd` scores 2 / (2 * 2) against 10 / (110 * 10) because `a` is frequent on
/// its own.
#[test]
fn merge_ranks_pairs_by_frequency_not_by_score() {
    let mut counts = tokenizer::WordCounts::new();
    counts.insert("ab".into(), 10);
    counts.insert("a".into(), 100);
    counts.insert("cd".into(), 2);
    let tokenizer = tokenizer::train_counts(&counts, 5 + 8 + 1).unwrap();
    let tokens = &tokenizer.vocabulary.tokens;
    assert_eq!(
        tokens[5..13],
        ["a", "##a", "b", "##b", "c", "##c", "d", "##d"].map(String::from),
        "the alphabet by frequency, ties by byte order, each as c and ##c"
    );
    assert_eq!(tokens[13], "ab");
}

/// A tie between pairs breaks by the byte order of the merged token, not of its first part.
/// `xac` and `yab` once each: all four pairs occur once. By merged token `##ab` wins; by first
/// part `##a` ties between `(##a, ##c)` and `(##a, ##b)`, and `(##a, ##c)` is met first, in
/// `xac`.
#[test]
fn ties_break_by_the_merged_token_not_its_first_part() {
    let mut counts = tokenizer::WordCounts::new();
    counts.insert("xac".into(), 1);
    counts.insert("yab".into(), 1);
    let tokenizer = tokenizer::train_counts(&counts, 5 + 10 + 1).unwrap();
    let tokens = &tokenizer.vocabulary.tokens;
    assert_eq!(
        tokens[5..15],
        ["a", "##a", "b", "##b", "c", "##c", "x", "##x", "y", "##y"].map(String::from)
    );
    assert_eq!(tokens[15], "##ab");
}

/// A word of exactly 100 characters takes part in merging; one of 101 does not, so with no
/// other word no pair is left.
#[test]
fn trainer_word_cap_is_100_characters() {
    let mut hundred = tokenizer::WordCounts::new();
    hundred.insert("a".repeat(100), 1);
    let tokenizer = tokenizer::train_counts(&hundred, 5 + 2 + 1).unwrap();
    // (##a, ##a) occurs 98 times, (a, ##a) once.
    assert_eq!(tokenizer.vocabulary.tokens[7], "##aa");

    let mut long = tokenizer::WordCounts::new();
    long.insert("a".repeat(101), 1);
    assert_eq!(
        tokenizer::train_counts(&long, 5 + 2 + 1).unwrap_err(),
        tokenizer::TrainError::TooSmall {
            reached: 7,
            requested: 8
        }
    );
}

/// A size too small for the whole alphabet that still needs a merge is refused naming the size
/// reached and the tokens the alphabet alone needs, not as a split too small: `abc` needs
/// 5 + 2 * 3 tokens before any merge.
#[test]
fn too_small_for_the_alphabet_says_what_the_alphabet_needs() {
    let mut counts = tokenizer::WordCounts::new();
    counts.insert("abc".into(), 1);
    for (requested, reached) in [(6, 5), (8, 7), (10, 9)] {
        let error = tokenizer::train_counts(&counts, requested).unwrap_err();
        assert_eq!(
            error,
            tokenizer::TrainError::AlphabetTooLarge {
                reached,
                requested,
                alphabet_tokens: 11
            }
        );
        let message = error.to_string();
        assert!(
            message.contains(&format!(" {reached} "))
                && message.contains("the alphabet alone needs 11 tokens"),
            "{message}"
        );
    }
    assert_eq!(
        tokenizer::train_counts(&counts, 11)
            .unwrap()
            .vocabulary
            .tokens
            .len(),
        11
    );
    assert_eq!(
        tokenizer::train_counts(&counts, 14).unwrap_err(),
        tokenizer::TrainError::TooSmall {
            reached: 13,
            requested: 14
        }
    );
}
