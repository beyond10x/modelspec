//! Adversary pass 1 on story:bert-tokenizer: `dataset::tokenizer` driven against the rules the
//! unit wrote into `spec/domains/data.yaml` (`modelspec.data.Tokenizer`, coordinator decision
//! D23). Every expected value here is written from those sentences, not read back from the code.
//!
//! The corpora are hand-built word tables or the original-text fixture
//! `tests/fixtures/corpus-tokenizer/`, so no case needs the corpus or the network.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use modelspec_runtime::dataset::tokenizer::{self, TrainError, WordCounts};
use modelspec_runtime::dataset::{self, SplitRole};

const UNK: u32 = 1;
const SPECIALS: [&str; 5] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adv_tokenizer_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

fn fixture_train_counts(case: &str) -> WordCounts {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus-tokenizer");
    let out = scratch(case);
    dataset::prepare(&fixture, &out).expect("the fixture prepares");
    let file = File::open(out.join(SplitRole::Train.split_file())).unwrap();
    tokenizer::count_words(BufReader::new(file)).unwrap()
}

fn counts_of(text: &str) -> WordCounts {
    tokenizer::count_words(text.as_bytes()).unwrap()
}

/// The vocabulary size that holds the specials and the whole alphabet of `words`, no merge.
fn alphabet_size(words: &[&str]) -> usize {
    let chars: HashSet<char> = words.iter().flat_map(|w| w.chars()).collect();
    SPECIALS.len() + 2 * chars.len()
}

/// Attack 6 and 10: greedy longest match first, pinned to literal ids. On the vocabulary
/// `[specials] a ##a b ##b ##ab` the word `abab` is `a`, then `##b` (no `##bab`, `##ba`), then
/// `##ab`. A shortest-match or character-by-character encoder gives `a ##b ##a ##b`, which
/// decodes to the same text, so only a literal sequence tells them apart.
#[test]
fn adv_encode_pins_longest_match_segmentation() {
    let mut counts = WordCounts::new();
    counts.insert("abab".into(), 5);
    let tokenizer = tokenizer::train_counts(&counts, 10).unwrap();
    assert_eq!(
        tokenizer.vocabulary.tokens[5..],
        ["a", "##a", "b", "##b", "##ab"].map(String::from)
    );
    let codec = tokenizer.codec();
    assert_eq!(codec.encode("abab"), [5, 8, 9]);
    assert_eq!(codec.encode("ABAB bab"), [5, 8, 9, 7, 9]);
    assert_eq!(codec.decode(&[5, 8, 9, 7, 9]).unwrap(), "abab bab");
}

/// Attack 2, mutant 2: the alphabet's ties break by byte order, whatever the hash order of the
/// word table. Twenty-six letters, each exactly once, must enter in `a`..`z` order; a sort by
/// frequency alone keeps the `HashMap`'s order, which matches this one in 1 of 26! runs.
#[test]
fn adv_alphabet_ties_break_by_byte_order() {
    let letters: Vec<char> = ('a'..='z').collect();
    let mut counts = WordCounts::new();
    for pair in letters.chunks(2) {
        counts.insert(pair.iter().collect(), 1);
    }
    let tokenizer = tokenizer::train_counts(&counts, 5 + 52).unwrap();
    let expected: Vec<String> = letters
        .iter()
        .flat_map(|c| [c.to_string(), format!("##{c}")])
        .collect();
    assert_eq!(tokenizer.vocabulary.tokens[5..], expected[..]);
}

/// Attack 2 and 10: a tie breaks by the byte order of the merged token, not of its first part.
/// `xac` and `yab` once each: all four pairs occur once (D25 ranks by frequency, so they tie).
/// By merged token `##ab` wins; by first
/// part `##a` ties between `(##a, ##c)` and `(##a, ##b)`, and the one met first, from `xac`,
/// would give `##ac`. The unit's own tie case cannot tell the two rules apart.
#[test]
fn adv_tie_breaks_by_merged_token_not_first_part() {
    let mut counts = WordCounts::new();
    counts.insert("xac".into(), 1);
    counts.insert("yab".into(), 1);
    let tokenizer = tokenizer::train_counts(&counts, 5 + 10 + 1).unwrap();
    assert_eq!(
        tokenizer.vocabulary.tokens[5..15],
        ["a", "##a", "b", "##b", "c", "##c", "x", "##x", "y", "##y"].map(String::from)
    );
    assert_eq!(tokenizer.vocabulary.tokens[15], "##ab");
}

/// Attack 3, adapted to coordinator decision D25: merges rank by pair frequency, not by the
/// likelihood score. `cd` occurs 4 times and `ab` 3 times, but `c` is frequent on its own
/// (`c` 60 times), so `ab` scores 3 / (3 * 3) = 1 / 3 against `cd` at 4 / (64 * 4) = 1 / 64. By
/// frequency `cd` is merged; a trainer still ranking by score merges `ab`. Counts at a real
/// release's scale (3e9) keep the case meaningful for an arithmetic that would overflow.
#[test]
fn adv_merges_rank_by_pair_frequency_not_score() {
    for scale in [1u64, 3_000_000_000] {
        let mut counts = WordCounts::new();
        counts.insert("ab".into(), 3 * scale);
        counts.insert("cd".into(), 4 * scale);
        counts.insert("c".into(), 60 * scale);
        let tokenizer = tokenizer::train_counts(&counts, 5 + 8 + 1).unwrap();
        assert_eq!(
            tokenizer.vocabulary.tokens[5..13],
            ["c", "##c", "d", "##d", "a", "##a", "b", "##b"].map(String::from)
        );
        assert_eq!(tokenizer.vocabulary.tokens[13], "cd", "scale {scale}");
    }
}

/// Attack 4: normalization edge cases, read from the D23 sentence "lowercases ... drops control
/// characters, separates words at whitespace and makes every character that is neither
/// alphanumeric nor whitespace a word of its own". `İ` lowercases to `i` and U+0307, a combining
/// mark that is not alphanumeric; U+00A0 is whitespace; CR, LF and tab are whitespace; BEL is a
/// control character and dropped; the ZWJ of an emoji sequence and a soft hyphen are format
/// characters, neither control nor whitespace, so each is a word.
#[test]
fn adv_normalization_edge_cases_round_trip() {
    let text = "İstanbul\u{a0}2016\r\nCafe\u{301} \u{1F468}\u{200D}\u{1F469}\tco\u{ad}op B\u{7}ell";
    let expected = [
        "i",
        "\u{307}",
        "stanbul",
        "2016",
        "cafe",
        "\u{301}",
        "\u{1F468}",
        "\u{200D}",
        "\u{1F469}",
        "co",
        "\u{ad}",
        "op",
        "bell",
    ];
    let mut words = Vec::new();
    tokenizer::for_each_word(text, |w| words.push(w.to_string()));
    assert_eq!(words, expected);

    let tokenizer = tokenizer::train_counts(&counts_of(text), alphabet_size(&expected)).unwrap();
    let codec = tokenizer.codec();
    let ids = codec.encode(text);
    assert!(!ids.contains(&UNK), "{ids:?}");
    assert_eq!(codec.decode(&ids).unwrap(), expected.join(" "));
}

/// Attack 6: a special token's text is not its id. `[MASK]` lowercases to `[mask]`, and `[` and
/// `]` are words of their own, so it encodes as the pieces of `[`, `mask` and `]`, never as id 4.
#[test]
fn adv_special_token_text_encodes_as_pieces() {
    let text = "[MASK] mask [CLS]";
    let words = ["[", "mask", "]", "cls"];
    let tokenizer = tokenizer::train_counts(&counts_of(text), alphabet_size(&words)).unwrap();
    let codec = tokenizer.codec();
    let ids = codec.encode("[MASK]");
    assert!(!ids.contains(&4) && !ids.contains(&UNK), "{ids:?}");
    assert_eq!(codec.decode(&ids).unwrap(), "[ mask ]");
    assert_eq!(codec.decode(&codec.encode("[CLS]")).unwrap(), "[ cls ]");
}

/// Attack 1: the story says "a corpus too small to yield `n` tokens exits 1 naming the count it
/// reached". The fixture's `Train` split yields 408 tokens, so every size from 5 to 408 should
/// train to exactly that size and no size in between is "too small".
#[test]
#[ignore = "ADV-TOK-1, kept by design (coordinator, 2026-10-24): a size below 5 + 2 * alphabet that needs a merge is refused, naming what the alphabet alone needs"]
fn adv_every_size_up_to_the_maximum_trains() {
    let counts = fixture_train_counts("every_size");
    let mut refused = Vec::new();
    for n in 5..=408 {
        match tokenizer::train_counts(&counts, n) {
            Ok(t) => assert_eq!(t.vocabulary.tokens.len(), n),
            Err(e) => refused.push(format!("{n}: {e}")),
        }
    }
    assert!(
        refused.is_empty(),
        "{} sizes refused: {refused:?}",
        refused.len()
    );
}

/// Attack 1: the fixture's maximum, 408, is reached exactly with distinct tokens, and 409 is
/// refused naming 408 (408 under coordinator decision D25 on the extended fixture; 314 under D23).
#[test]
fn adv_fixture_maximum_is_exact() {
    let counts = fixture_train_counts("maximum");
    let tokenizer = tokenizer::train_counts(&counts, 408).unwrap();
    let tokens = &tokenizer.vocabulary.tokens;
    assert_eq!(tokens.len(), 408);
    assert_eq!(tokens.iter().collect::<HashSet<_>>().len(), 408);
    assert_eq!(tokens[..5], SPECIALS.map(String::from));
    assert_eq!(
        tokenizer::train_counts(&counts, 409).unwrap_err(),
        TrainError::TooSmall {
            reached: 408,
            requested: 409
        }
    );
}

/// Attack 1: the smallest sizes. 5 holds the specials only and every word encodes to `[UNK]`;
/// 0 and 4 are refused; 6 leaves room for no character (the alphabet takes two tokens per
/// character), so it is refused at 5, naming the 9 tokens the alphabet alone needs (ADV-TOK-1).
#[test]
fn adv_smallest_sizes() {
    let mut counts = WordCounts::new();
    counts.insert("ab".into(), 3);
    assert_eq!(
        tokenizer::train_counts(&counts, 0).unwrap_err(),
        TrainError::BelowSpecialTokens { requested: 0 }
    );
    let five = tokenizer::train_counts(&counts, 5).unwrap();
    assert_eq!(five.vocabulary.tokens, SPECIALS.map(String::from));
    assert_eq!(five.codec().encode("ab ab"), [UNK, UNK]);
    assert_eq!(
        tokenizer::train_counts(&counts, 6).unwrap_err(),
        TrainError::AlphabetTooLarge {
            reached: 5,
            requested: 6,
            alphabet_tokens: 9
        }
    );
    assert_eq!(
        tokenizer::train_counts(&counts, 7)
            .unwrap()
            .vocabulary
            .tokens
            .len(),
        7
    );
}

/// Attack 6: a word of 100 characters takes part in merging, one of 101 does not.
#[test]
fn adv_trainer_word_cap_is_100_characters() {
    let mut hundred = WordCounts::new();
    hundred.insert("a".repeat(100), 5);
    let tokenizer = tokenizer::train_counts(&hundred, 8).unwrap();
    // (##a, ##a) occurs 490 times, (a, ##a) 5 times (D25 ranks by frequency).
    assert_eq!(tokenizer.vocabulary.tokens[7], "##aa");

    let mut long = WordCounts::new();
    long.insert("a".repeat(101), 5);
    assert_eq!(
        tokenizer::train_counts(&long, 8).unwrap_err(),
        TrainError::TooSmall {
            reached: 7,
            requested: 8
        }
    );
}

/// Attack 7 and 10: the id is a function of the token list, not of its length. Two vocabularies
/// of ten tokens that differ in their last tokens differ in id.
#[test]
fn adv_record_id_differs_for_equal_sized_vocabularies() {
    let mut ab = WordCounts::new();
    ab.insert("ab".into(), 1);
    let mut cd = WordCounts::new();
    cd.insert("cd".into(), 1);
    let x = tokenizer::train_counts(&ab, 10).unwrap();
    let y = tokenizer::train_counts(&cd, 10).unwrap();
    assert_eq!(x.vocabulary.tokens.len(), y.vocabulary.tokens.len());
    assert_ne!(x.tokenizer_id, y.tokenizer_id);
    let variant = x.tokenizer_id.as_bytes()[19];
    assert!(b"89ab".contains(&variant), "{}", x.tokenizer_id);
    assert_eq!(&x.tokenizer_id[14..15], "8");
}
