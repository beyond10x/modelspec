//! The WordPiece tokenizer of a masked-language-model run (`spec/domains/data.yaml`,
//! `modelspec.data.Tokenizer`; story:bert-tokenizer, coordinator decisions D19, D23 and D25).
//!
//! [`train`] counts the words of a prepared dataset's `Train` split once, then merges on that
//! word table, never on the text, until the vocabulary has exactly the requested size. Every
//! choice is a total order on counts and bytes, so two runs on the same split give the same
//! [`Tokenizer`], and [`write`] writes it byte for byte the same. [`Codec::encode`] and
//! [`Codec::decode`] map text to token ids and back.

use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{RECORD_FILE, SplitRole, partial_path, uuid_of};

/// The special tokens, at the head of every vocabulary in this order (coordinator decision D19).
pub const SPECIAL_TOKENS: [&str; 5] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
/// The id of `[PAD]`.
pub const PAD_TOKEN_ID: u32 = 0;
/// The id of `[UNK]`.
pub const UNK_TOKEN_ID: u32 = 1;
/// The id of `[CLS]`.
pub const CLS_TOKEN_ID: u32 = 2;
/// The id of `[SEP]`.
pub const SEP_TOKEN_ID: u32 = 3;
/// The id of `[MASK]`.
pub const MASK_TOKEN_ID: u32 = 4;
/// The prefix of a token that continues a word.
pub const CONTINUATION: &str = "##";
/// The most characters the alphabet holds.
pub const MAX_ALPHABET: usize = 1000;
/// A word of more characters encodes to `[UNK]`.
pub const MAX_WORD_CHARS: usize = 100;

/// The tokenization algorithm (`modelspec.data.TokenizerKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerKind {
    WordPiece,
}

/// The tokens a tokenizer maps text to (`modelspec.data.Vocabulary`); a token's id is its
/// position in `tokens`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vocabulary {
    pub tokens: Vec<String>,
    pub mask_token_id: u32,
    pub pad_token_id: u32,
    pub unk_token_id: u32,
    pub cls_token_id: u32,
    pub sep_token_id: u32,
}

/// The lifecycle state of a [`Tokenizer`] (`modelspec.data.Tokenizer.State`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerState {
    Recorded,
}

/// The record of a trained tokenizer (`modelspec.data.Tokenizer`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tokenizer {
    /// The leading 16 bytes, as a version-8 UUID, of blake3 over the tokens in id order, each
    /// followed by a newline.
    pub tokenizer_id: String,
    pub kind: TokenizerKind,
    pub vocabulary: Vocabulary,
    pub state: TokenizerState,
}

/// How often each word of a text occurs, words as [`for_each_word`] splits them.
pub type WordCounts = HashMap<String, u64>;

/// Why [`train_counts`] could not reach the requested size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainError {
    /// The size leaves no room for the special tokens.
    BelowSpecialTokens { requested: usize },
    /// No pair was left to merge at `reached` tokens.
    TooSmall { reached: usize, requested: usize },
    /// No pair was left to merge at `reached` tokens because the requested size left room for
    /// only part of the alphabet, which alone needs `alphabet_tokens` (the special tokens and
    /// two per character).
    AlphabetTooLarge {
        reached: usize,
        requested: usize,
        alphabet_tokens: usize,
    },
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainError::BelowSpecialTokens { requested } => write!(
                f,
                "a vocabulary of {requested} tokens has no room for the {} special tokens",
                SPECIAL_TOKENS.len()
            ),
            TrainError::TooSmall { reached, requested } => write!(
                f,
                "the split yields {reached} tokens, fewer than the {requested} requested"
            ),
            TrainError::AlphabetTooLarge {
                reached,
                requested,
                alphabet_tokens,
            } => write!(
                f,
                "training stopped at {reached} tokens, fewer than the {requested} requested: \
                 the alphabet alone needs {alphabet_tokens} tokens"
            ),
        }
    }
}

impl std::error::Error for TrainError {}

/// A refused [`train`] or [`write`]: the path it concerns and why.
#[derive(Debug)]
pub struct TokenizerError {
    pub path: PathBuf,
    pub reason: String,
}

impl TokenizerError {
    fn new(path: &Path, reason: impl fmt::Display) -> Self {
        TokenizerError {
            path: path.to_path_buf(),
            reason: reason.to_string(),
        }
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.reason)
    }
}

impl std::error::Error for TokenizerError {}

/// Calls `f` with each word of `text` under the settled normalization: Unicode lowercase,
/// control characters dropped, whitespace separating words, and every character that is
/// neither alphanumeric nor whitespace a word of its own.
pub fn for_each_word(text: &str, mut f: impl FnMut(&str)) {
    let mut word = String::new();
    let mut single = [0u8; 4];
    for c in text.to_lowercase().chars() {
        if c.is_alphanumeric() {
            word.push(c);
        } else if c.is_whitespace() || !c.is_control() {
            if !word.is_empty() {
                f(&word);
                word.clear();
            }
            if !c.is_whitespace() {
                f(c.encode_utf8(&mut single));
            }
        }
    }
    if !word.is_empty() {
        f(&word);
    }
}

/// Counts the words of `reader`'s text in one pass, line by line.
pub fn count_words(mut reader: impl BufRead) -> io::Result<WordCounts> {
    let mut counts = WordCounts::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(counts);
        }
        for_each_word(&line, |word| match counts.get_mut(word) {
            Some(count) => *count += 1,
            None => {
                counts.insert(word.to_owned(), 1);
            }
        });
    }
}

/// Trains a tokenizer on the `Train` split of the dataset `dataset_dir`, a directory
/// `prepare` wrote, to exactly `vocab_size` tokens. The `HeldOut` split is never read.
pub fn train(dataset_dir: &Path, vocab_size: usize) -> Result<Tokenizer, TokenizerError> {
    let record = dataset_dir.join(RECORD_FILE);
    fs::metadata(&record).map_err(|e| TokenizerError::new(&record, e))?;
    let split = dataset_dir.join(SplitRole::Train.split_file());
    let file = File::open(&split).map_err(|e| TokenizerError::new(&split, e))?;
    let counts = count_words(BufReader::new(file)).map_err(|e| TokenizerError::new(&split, e))?;
    train_counts(&counts, vocab_size).map_err(|e| TokenizerError::new(&split, e))
}

/// Writes `tokenizer` as pretty JSON with a final newline to `out`, through `<out>.partial`, so
/// a failed write leaves no record behind.
pub fn write(tokenizer: &Tokenizer, out: &Path) -> Result<(), TokenizerError> {
    let partial = partial_path(out);
    let mut json = serde_json::to_string_pretty(tokenizer).expect("JSON of a tokenizer record");
    json.push('\n');
    if let Err(e) = fs::write(&partial, json) {
        let _ = fs::remove_file(&partial);
        return Err(TokenizerError::new(out, e));
    }
    fs::rename(&partial, out).map_err(|e| {
        let _ = fs::remove_file(&partial);
        TokenizerError::new(out, e)
    })
}

/// The alphabet of `counts`: its characters by descending frequency, ties by byte order, at
/// most [`MAX_ALPHABET`] and at most `room` / 2; and how many characters it holds without the
/// `room` limit.
fn alphabet(counts: &WordCounts, room: usize) -> (Vec<char>, usize) {
    let mut frequency: HashMap<char, u64> = HashMap::new();
    for (word, count) in counts {
        for c in word.chars() {
            *frequency.entry(c).or_insert(0) += count;
        }
    }
    let mut chars: Vec<(char, u64)> = frequency.into_iter().collect();
    chars.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let unlimited = chars.len().min(MAX_ALPHABET);
    chars.truncate(unlimited.min(room / 2));
    (chars.into_iter().map(|(c, _)| c).collect(), unlimited)
}

/// `a` followed by `b` without its continuation prefix.
fn merged(a: &str, b: &str) -> String {
    let mut m = String::with_capacity(a.len() + b.len());
    m.push_str(a);
    m.push_str(b.strip_prefix(CONTINUATION).unwrap_or(b));
    m
}

/// One word of the table being merged: its pieces as token ids and how often it occurs.
struct Word {
    pieces: Vec<u32>,
    count: u64,
}

/// An adjacent pair of the word table and how often it occurs.
struct Pair {
    a: u32,
    b: u32,
    count: u64,
}

/// The merge state: tokens, the word table and the pairs in it.
struct Trainer {
    tokens: Vec<String>,
    index: HashMap<String, u32>,
    words: Vec<Word>,
    pairs: Vec<Pair>,
    pair_slots: HashMap<(u32, u32), usize>,
    /// The words a pair occurs in, possibly with repeats and words it has left since.
    pair_words: HashMap<(u32, u32), Vec<u32>>,
}

impl Trainer {
    fn add_token(&mut self, token: String) -> u32 {
        let id = self.tokens.len() as u32;
        self.index.insert(token.clone(), id);
        self.tokens.push(token);
        id
    }

    fn add_pair(&mut self, pair: (u32, u32), count: u64) {
        match self.pair_slots.get(&pair) {
            Some(&slot) => self.pairs[slot].count += count,
            None => {
                self.pair_slots.insert(pair, self.pairs.len());
                self.pairs.push(Pair {
                    a: pair.0,
                    b: pair.1,
                    count,
                });
            }
        }
    }

    fn remove_pair(&mut self, pair: (u32, u32), count: u64) {
        let slot = self.pair_slots[&pair];
        self.pairs[slot].count -= count;
    }

    /// Whether pair `x` ranks before pair `y`: a higher pair count (coordinator decision D25),
    /// then the lower byte order of the merged token, then of `a`.
    fn ranks_before(&self, x: &Pair, y: &Pair) -> bool {
        if x.count != y.count {
            return x.count > y.count;
        }
        let (xa, xb) = (&self.tokens[x.a as usize], &self.tokens[x.b as usize]);
        let (ya, yb) = (&self.tokens[y.a as usize], &self.tokens[y.b as usize]);
        let xm = xa
            .bytes()
            .chain(xb.strip_prefix(CONTINUATION).unwrap_or(xb).bytes());
        let ym = ya
            .bytes()
            .chain(yb.strip_prefix(CONTINUATION).unwrap_or(yb).bytes());
        match xm.cmp(ym) {
            std::cmp::Ordering::Equal => xa < ya,
            order => order.is_lt(),
        }
    }

    /// The pair to merge next, if any pair is left.
    fn best(&self) -> Option<(u32, u32)> {
        let mut best: Option<&Pair> = None;
        for pair in self.pairs.iter().filter(|p| p.count > 0) {
            if best.is_none_or(|b| self.ranks_before(pair, b)) {
                best = Some(pair);
            }
        }
        best.map(|p| (p.a, p.b))
    }

    /// Replaces every adjacent `(a, b)` of the word table by `m`, left to right.
    fn merge(&mut self, (a, b): (u32, u32), m: u32) {
        let mut words = self.pair_words.remove(&(a, b)).unwrap_or_default();
        words.sort_unstable();
        words.dedup();
        for w in words {
            let count = self.words[w as usize].count;
            let old = std::mem::take(&mut self.words[w as usize].pieces);
            let mut pieces = Vec::with_capacity(old.len());
            let mut i = 0;
            let mut merged_any = false;
            while i < old.len() {
                if i + 1 < old.len() && old[i] == a && old[i + 1] == b {
                    pieces.push(m);
                    merged_any = true;
                    i += 2;
                } else {
                    pieces.push(old[i]);
                    i += 1;
                }
            }
            if merged_any {
                for pair in old.windows(2) {
                    self.remove_pair((pair[0], pair[1]), count);
                }
                for pair in pieces.windows(2) {
                    let pair = (pair[0], pair[1]);
                    self.add_pair(pair, count);
                    if pair.0 == m || pair.1 == m {
                        self.pair_words.entry(pair).or_default().push(w);
                    }
                }
            }
            self.words[w as usize].pieces = pieces;
        }
        if self.pairs.len() > 1024
            && self.pairs.iter().filter(|p| p.count == 0).count() * 2 > self.pairs.len()
        {
            self.pairs.retain(|p| p.count > 0);
            self.pair_slots = self
                .pairs
                .iter()
                .enumerate()
                .map(|(slot, p)| ((p.a, p.b), slot))
                .collect();
        }
    }
}

/// Trains a tokenizer on word counts to exactly `vocab_size` tokens (coordinator decisions D23
/// and D25).
pub fn train_counts(counts: &WordCounts, vocab_size: usize) -> Result<Tokenizer, TrainError> {
    let specials = SPECIAL_TOKENS.len();
    if vocab_size < specials {
        return Err(TrainError::BelowSpecialTokens {
            requested: vocab_size,
        });
    }
    let (alphabet, unlimited) = alphabet(counts, vocab_size - specials);
    let mut trainer = Trainer {
        tokens: Vec::new(),
        index: HashMap::new(),
        words: Vec::new(),
        pairs: Vec::new(),
        pair_slots: HashMap::new(),
        pair_words: HashMap::new(),
    };
    for token in SPECIAL_TOKENS {
        trainer.add_token(token.to_string());
    }
    let mut initial: HashMap<char, (u32, u32)> = HashMap::new();
    for &c in &alphabet {
        let first = trainer.add_token(c.to_string());
        let next = trainer.add_token(format!("{CONTINUATION}{c}"));
        initial.insert(c, (first, next));
    }

    // The word table in byte order, so that word indices do not depend on hash order.
    let mut table: Vec<(&String, u64)> = counts.iter().map(|(w, c)| (w, *c)).collect();
    table.sort_unstable();
    for (word, count) in table {
        if word.chars().count() > MAX_WORD_CHARS {
            continue;
        }
        let pieces: Option<Vec<u32>> = word
            .chars()
            .enumerate()
            .map(|(i, c)| {
                initial
                    .get(&c)
                    .map(|ids| if i == 0 { ids.0 } else { ids.1 })
            })
            .collect();
        let Some(pieces) = pieces else { continue };
        let w = trainer.words.len() as u32;
        for pair in pieces.windows(2) {
            let pair = (pair[0], pair[1]);
            trainer.add_pair(pair, count);
            trainer.pair_words.entry(pair).or_default().push(w);
        }
        trainer.words.push(Word { pieces, count });
    }

    while trainer.tokens.len() < vocab_size {
        let Some((a, b)) = trainer.best() else {
            let reached = trainer.tokens.len();
            let requested = vocab_size;
            return Err(if alphabet.len() < unlimited {
                TrainError::AlphabetTooLarge {
                    reached,
                    requested,
                    alphabet_tokens: specials + 2 * unlimited,
                }
            } else {
                TrainError::TooSmall { reached, requested }
            });
        };
        let token = merged(&trainer.tokens[a as usize], &trainer.tokens[b as usize]);
        let m = match trainer.index.get(&token) {
            Some(&id) => id,
            None => trainer.add_token(token),
        };
        trainer.merge((a, b), m);
    }

    let tokens = trainer.tokens;
    let mut id = blake3::Hasher::new();
    for token in &tokens {
        id.update(token.as_bytes());
        id.update(b"\n");
    }
    Ok(Tokenizer {
        tokenizer_id: uuid_of(id.finalize().as_bytes()),
        kind: TokenizerKind::WordPiece,
        vocabulary: Vocabulary {
            tokens,
            mask_token_id: MASK_TOKEN_ID,
            pad_token_id: PAD_TOKEN_ID,
            unk_token_id: UNK_TOKEN_ID,
            cls_token_id: CLS_TOKEN_ID,
            sep_token_id: SEP_TOKEN_ID,
        },
        state: TokenizerState::Recorded,
    })
}

impl Tokenizer {
    /// The encoder and decoder of this tokenizer's vocabulary.
    pub fn codec(&self) -> Codec<'_> {
        Codec::new(&self.vocabulary)
    }
}

/// Encodes text to token ids and decodes ids to text with one vocabulary.
pub struct Codec<'a> {
    tokens: &'a [String],
    index: HashMap<&'a str, u32>,
}

impl<'a> Codec<'a> {
    pub fn new(vocabulary: &'a Vocabulary) -> Self {
        let index = vocabulary
            .tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.as_str(), id as u32))
            .collect();
        Codec {
            tokens: &vocabulary.tokens,
            index,
        }
    }

    /// The ids of `text`: each word of [`for_each_word`] matched greedily, longest match first;
    /// a word over [`MAX_WORD_CHARS`] characters or with a remainder no token matches is
    /// `[UNK]` whole. No `[CLS]`/`[SEP]` frame is added.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut piece = String::new();
        for_each_word(text, |word| {
            let ends: Vec<usize> = word
                .char_indices()
                .map(|(i, _)| i)
                .skip(1)
                .chain([word.len()])
                .collect();
            if ends.len() > MAX_WORD_CHARS {
                ids.push(UNK_TOKEN_ID);
                return;
            }
            let start_len = ids.len();
            let mut start = 0;
            while start < word.len() {
                let found = ends
                    .iter()
                    .rev()
                    .take_while(|&&end| end > start)
                    .find_map(|&end| {
                        piece.clear();
                        if start > 0 {
                            piece.push_str(CONTINUATION);
                        }
                        piece.push_str(&word[start..end]);
                        self.index.get(piece.as_str()).map(|&id| (id, end))
                    });
                match found {
                    Some((id, end)) => {
                        ids.push(id);
                        start = end;
                    }
                    None => {
                        ids.truncate(start_len);
                        ids.push(UNK_TOKEN_ID);
                        return;
                    }
                }
            }
        });
        ids
    }

    /// The text of `ids`: a token starting with `##` joins the previous one without a space and
    /// without its `##`, every other token starts a word, and words are joined by one space.
    /// `None` when an id is outside the vocabulary.
    pub fn decode(&self, ids: &[u32]) -> Option<String> {
        let mut text = String::new();
        for &id in ids {
            let token = self.tokens.get(id as usize)?;
            match token.strip_prefix(CONTINUATION) {
                Some(rest) => text.push_str(rest),
                None => {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(token);
                }
            }
        }
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        let mut words = Vec::new();
        for_each_word(text, |w| words.push(w.to_string()));
        words
    }

    #[test]
    fn normalization_lowercases_splits_punctuation_and_drops_controls() {
        assert_eq!(
            words(" Copper\u{7}Bridge, in 1910 ;\tÉté\n"),
            ["copperbridge", ",", "in", "1910", ";", "été"]
        );
        assert_eq!(words("a-b"), ["a", "-", "b"]);
        assert!(words(" \n \r\n").is_empty());
    }

    #[test]
    fn alphabet_leaves_room_for_two_tokens_per_character() {
        let mut counts = WordCounts::new();
        counts.insert("abc".into(), 1);
        counts.insert("c".into(), 1);
        assert_eq!(alphabet(&counts, 5), (vec!['c', 'a'], 3));
        assert_eq!(alphabet(&counts, 100), (vec!['c', 'a', 'b'], 3));
    }

    #[test]
    fn too_small_names_the_size_reached() {
        let mut counts = WordCounts::new();
        counts.insert("ab".into(), 1);
        assert_eq!(
            train_counts(&counts, 11).unwrap_err(),
            TrainError::TooSmall {
                reached: 10,
                requested: 11
            }
        );
        assert_eq!(
            train_counts(&counts, 10).unwrap().vocabulary.tokens[9],
            "ab"
        );
        assert_eq!(
            train_counts(&counts, 4).unwrap_err(),
            TrainError::BelowSpecialTokens { requested: 4 }
        );
    }
}
