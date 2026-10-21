//! Tokenizer for `.mspec` source.
//!
//! Whitespace and `//` line comments are skipped. Numeric literals keep their text; the parser
//! converts them, so that an out-of-range literal is reported where it is used.

use std::ops::Range;

use logos::Logos;

/// The reserved words: every `#[token]` of [`Token`] that is a word. The site's code highlighter
/// is generated from this list.
pub const KEYWORDS: &[&str] = &[
    "model", "param", "fn", "let", "return", "train", "input", "target",
];

/// A token of the M0 subset, including the operator characters the parser rejects by name.
#[derive(Logos, Clone, Copy, Debug, PartialEq, Eq)]
#[logos(skip r"[ \t\r\n\f]+")]
#[logos(skip(r"//[^\n]*", allow_greedy = true))]
pub enum Token {
    #[token("model")]
    Model,
    #[token("param")]
    Param,
    #[token("fn")]
    Fn,
    #[token("let")]
    Let,
    #[token("return")]
    Return,
    #[token("train")]
    Train,
    #[token("input")]
    Input,
    #[token("target")]
    Target,

    #[regex(r"[A-Za-z_][A-Za-z0-9_]*")]
    Ident,
    #[regex(r"[0-9]+")]
    Int,
    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?")]
    #[regex(r"[0-9]+[eE][+-]?[0-9]+")]
    Float,

    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("->")]
    Arrow,
    #[token(":")]
    Colon,
    #[token(";")]
    Semi,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("=")]
    Eq,

    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("^")]
    Caret,
}

impl Token {
    /// Whether the token is an arithmetic or comparison operator, none of which M0 has.
    pub fn is_operator(self) -> bool {
        matches!(
            self,
            Token::Plus
                | Token::Minus
                | Token::Star
                | Token::Slash
                | Token::Percent
                | Token::Caret
                | Token::Lt
                | Token::Gt
        )
    }
}

/// One lexed item: a token, or `None` for a character no token starts with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lexeme {
    pub token: Option<Token>,
    /// Byte range in the source.
    pub range: Range<usize>,
}

/// Splits `source` into lexemes, in order. Unrecognised input becomes a lexeme with no token
/// that starts at the first character no token starts with.
pub fn lex(source: &str) -> Vec<Lexeme> {
    Token::lexer(source)
        .spanned()
        .map(|(result, range)| Lexeme {
            token: result.ok(),
            range,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keywords_are_the_word_tokens_of_the_lexer() {
        let words: Vec<&str> = include_str!("lexer.rs")
            .lines()
            .filter_map(|line| line.trim().strip_prefix("#[token(\""))
            .filter_map(|rest| rest.split('"').next())
            .filter(|text| text.chars().all(|c| c.is_ascii_alphabetic()))
            .collect();
        assert_eq!(words, KEYWORDS);
        for keyword in KEYWORDS {
            let lexemes = lex(keyword);
            assert_eq!(lexemes.len(), 1, "{keyword}");
            assert_ne!(lexemes[0].token, Some(Token::Ident), "{keyword}");
        }
    }
}
