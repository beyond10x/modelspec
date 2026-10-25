//! A deterministic synthetic corpus with structure a small masked-language model can learn, in
//! the layout `modelspec data prepare` reads (coordinator decision D72).
//!
//! The vocabulary is 200 invented two-syllable words. Each word has 2 to 4 fixed successors drawn
//! uniformly, the first of them the most likely ([`WEIGHTS`]), so a word's neighbours narrow it
//! to a few words where its frequency alone leaves about 200: the bigram entropy of the text is
//! far below its unigram entropy. The text is a walk of that chain,
//! cut into sentences of 6 to 12 words that end with ` .`, paragraphs of 4 sentences and articles
//! of 3 paragraphs under a heading ` = <title> = ` with a blank line (one space) before and after
//! it, as the release files of the corpus are laid out. `wiki.train.raw` and `wiki.valid.raw` walk
//! the same chain from other seeds, so the held-out text is not the training text.

use std::path::Path;

/// The words of the chain.
const WORDS: usize = 200;
/// The seed the vocabulary and the successors are drawn with.
const CHAIN_SEED: u64 = 0x6d61_726b_6f76_0001;
const TRAIN_SEED: u64 = 0x6d61_726b_6f76_0002;
const VALID_SEED: u64 = 0x6d61_726b_6f76_0003;
/// The weight of a word's first, second, third and fourth successor: the first is taken 3 times
/// in 4 or more, so a word's left neighbour mostly determines it. With successors of equal
/// weight the twin's held-out loss stayed at its unigram level for 1,000 steps (fix pass 2 of
/// story:bert-tiny-cpu-model).
const WEIGHTS: [usize; 4] = [12, 2, 1, 1];
/// Articles of the train and of the valid file.
const TRAIN_ARTICLES: usize = 40;
const VALID_ARTICLES: usize = 10;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The invented words and each word's successors.
struct Chain {
    words: Vec<String>,
    successors: Vec<Vec<usize>>,
}

impl Chain {
    fn new() -> Self {
        const ONSETS: [&str; 12] = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t"];
        const VOWELS: [&str; 5] = ["a", "e", "i", "o", "u"];
        let mut rng = Rng(CHAIN_SEED);
        let mut words: Vec<String> = Vec::new();
        while words.len() < WORDS {
            let word: String = (0..2)
                .map(|_| format!("{}{}", ONSETS[rng.below(12)], VOWELS[rng.below(5)]))
                .collect();
            if !words.contains(&word) {
                words.push(word);
            }
        }
        let successors = (0..WORDS)
            .map(|_| (0..2 + rng.below(3)).map(|_| rng.below(WORDS)).collect())
            .collect();
        Chain { words, successors }
    }

    /// The successor of `word` the walk takes next: the `r`-th of its successors with weight
    /// `WEIGHTS[r]`, so the first is the most likely.
    fn next(&self, word: usize, rng: &mut Rng) -> usize {
        let next = &self.successors[word];
        let total: usize = WEIGHTS[..next.len()].iter().sum();
        let mut x = rng.below(total);
        for (&w, &weight) in next.iter().zip(&WEIGHTS) {
            if x < weight {
                return w;
            }
            x -= weight;
        }
        unreachable!("x is below the total weight")
    }

    /// `articles` articles of a walk drawn with `seed`.
    fn text(&self, seed: u64, articles: usize) -> String {
        let mut rng = Rng(seed);
        let mut at = rng.below(WORDS);
        let mut step = |rng: &mut Rng| {
            at = self.next(at, rng);
            self.words[at].as_str()
        };
        let mut out = String::new();
        for _ in 0..articles {
            let mut title: Vec<String> = Vec::new();
            for _ in 0..2 {
                let word = step(&mut rng);
                let mut chars = word.chars();
                let first = chars.next().unwrap().to_ascii_uppercase();
                title.push(format!("{first}{}", chars.as_str()));
            }
            out.push_str(&format!(" \n = {} = \n \n", title.join(" ")));
            for _ in 0..3 {
                let mut line = String::new();
                for _ in 0..4 {
                    for _ in 0..6 + rng.below(7) {
                        line.push(' ');
                        line.push_str(step(&mut rng));
                    }
                    line.push_str(" .");
                }
                out.push_str(&line);
                out.push_str(" \n \n");
            }
        }
        out
    }
}

/// Writes `wiki.train.raw` and `wiki.valid.raw` of the synthetic corpus into `dir`.
pub fn write(dir: &Path) {
    std::fs::create_dir_all(dir).expect("corpus directory");
    let chain = Chain::new();
    for (file, seed, articles) in [
        ("wiki.train.raw", TRAIN_SEED, TRAIN_ARTICLES),
        ("wiki.valid.raw", VALID_SEED, VALID_ARTICLES),
    ] {
        std::fs::write(dir.join(file), chain.text(seed, articles)).expect("corpus file");
    }
}
