//! One release file of the corpus: copied byte for byte while its bytes are hashed, checked as
//! UTF-8 and its documents counted, so a release file of hundreds of megabytes is read once and
//! never held whole.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use super::PrepareError;

/// What reading one split file found.
pub(super) struct SplitSummary {
    pub hash: blake3::Hash,
    pub document_count: u64,
    pub byte_count: u64,
}

/// `line` without its line terminator, `\n` or `\r\n`.
fn content(line: &str) -> &str {
    let line = line.strip_suffix('\n').unwrap_or(line);
    line.strip_suffix('\r').unwrap_or(line)
}

/// Whether `line` has the shape of an article heading of the release, ` = <title> = `: one
/// leading space, `= `, a title that does not start with `=` and is not blank, ` = `, then the
/// line's end. A section heading ` = = <section> = = ` does not have it. The shape alone does not
/// start an article: the release also holds table keys and equation fragments of this shape
/// inside articles, and only a line with a blank line directly before and after it is a heading
/// (see [`DocumentCounter`]).
pub fn is_article_heading(line: &str) -> bool {
    content(line)
        .strip_prefix(" = ")
        .and_then(|rest| rest.strip_suffix(" = "))
        .is_some_and(|title| !title.starts_with('=') && !title.trim().is_empty())
}

/// Whether `line` is a blank line of the release: a single space.
fn is_blank(line: &str) -> bool {
    content(line) == " "
}

/// Counts the documents of a split as its lines arrive (coordinator decision D22). A document
/// starts at a line of the shape [`is_article_heading`] with a blank line directly before and
/// directly after it; any non-blank text before the first such heading is one document too.
#[derive(Default)]
struct DocumentCounter {
    headings: u64,
    text_before_first_heading: bool,
    /// Whether the line before the current one was blank.
    previous_blank: bool,
    /// Whether the line before the current one had the heading shape and a blank line before
    /// it, so that it is a heading if the current line is blank.
    candidate: bool,
}

impl DocumentCounter {
    fn line(&mut self, line: &str) {
        let blank = is_blank(line);
        if std::mem::take(&mut self.candidate) {
            if blank {
                self.headings += 1;
            } else {
                self.text();
            }
        }
        if self.previous_blank && is_article_heading(line) {
            self.candidate = true;
        } else if !line.trim().is_empty() {
            self.text();
        }
        self.previous_blank = blank;
    }

    /// A line of text that is not a heading.
    fn text(&mut self) {
        if self.headings == 0 {
            self.text_before_first_heading = true;
        }
    }

    /// The documents counted so far; a heading-shaped last line has no blank line after it and
    /// is text.
    fn count(&self) -> u64 {
        let text = self.text_before_first_heading || (self.candidate && self.headings == 0);
        self.headings + u64::from(text)
    }
}

/// Copies `source` to `target`, returning its blake3 hash, document count and length. A file
/// that is not UTF-8 or holds no document is refused, naming `source`.
pub(super) fn copy_split(source: &Path, target: &Path) -> Result<SplitSummary, PrepareError> {
    let input = File::open(source).map_err(|e| PrepareError::new(source, e))?;
    let output = File::create(target).map_err(|e| PrepareError::new(target, e))?;
    let mut writer = BufWriter::new(output);
    let summary = copy_lines(BufReader::new(input), &mut writer, source, target)?;
    writer
        .into_inner()
        .map_err(|e| PrepareError::new(target, e.into_error()))?
        .sync_all()
        .map_err(|e| PrepareError::new(target, e))?;
    Ok(summary)
}

/// [`copy_split`] over a reader and a writer; `source` and `target` only name them in errors.
fn copy_lines(
    mut reader: impl BufRead,
    writer: &mut impl Write,
    source: &Path,
    target: &Path,
) -> Result<SplitSummary, PrepareError> {
    let mut hasher = blake3::Hasher::new();
    let mut counter = DocumentCounter::default();
    let mut byte_count: u64 = 0;
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = reader
            .read_until(b'\n', &mut line)
            .map_err(|e| PrepareError::new(source, e))?;
        if read == 0 {
            break;
        }
        let text = std::str::from_utf8(&line).map_err(|e| {
            let at = byte_count + e.valid_up_to() as u64;
            PrepareError::new(source, format!("not UTF-8 at byte {at}"))
        })?;
        counter.line(text);
        hasher.update(&line);
        writer
            .write_all(&line)
            .map_err(|e| PrepareError::new(target, e))?;
        byte_count += read as u64;
    }
    writer.flush().map_err(|e| PrepareError::new(target, e))?;
    let document_count = counter.count();
    if document_count == 0 {
        return Err(PrepareError::new(source, "holds no document"));
    }
    Ok(SplitSummary {
        hash: hasher.finalize(),
        document_count,
        byte_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn documents(text: &str) -> u64 {
        let mut counter = DocumentCounter::default();
        for line in text.split_inclusive('\n') {
            counter.line(line);
        }
        counter.count()
    }

    #[test]
    fn article_heading_shape_is_space_equals_title_equals_space() {
        assert!(is_article_heading(" = Copper Bridge = \n"));
        assert!(is_article_heading(" = Copper Bridge = \r\n"));
        assert!(is_article_heading(" = Copper Bridge = "));
        assert!(!is_article_heading("= A =\n"));
        assert!(!is_article_heading("  = A = \n"));
        assert!(!is_article_heading(" = A =\n"));
        assert!(!is_article_heading(" = = Repairs = = \n"));
        assert!(!is_article_heading(" = = = Fair = = = \n"));
        assert!(!is_article_heading(" =   = \n"));
        assert!(!is_article_heading(" a = b = \n"));
        assert!(!is_article_heading(" \n"));
    }

    #[test]
    fn documents_are_articles_not_paragraphs() {
        let text =
            " \n = One = \n \n para . \n \n = = Sub = = \n \n more . \n \n = Two = \n \n x . \n";
        assert_eq!(documents(text), 2);
    }

    #[test]
    fn a_heading_needs_a_blank_line_before_and_after() {
        let head = " \n = One = \n \n p . \n";
        assert_eq!(
            documents(&format!("{head} Key : \n = K ; V = \n v \n \n")),
            1
        );
        assert_eq!(documents(&format!("{head} \n = Before only = \n v \n")), 1);
        assert_eq!(documents(&format!("{head} r \n = c + k t = \n \n")), 1);
        assert_eq!(documents(&format!("{head} \n = Two = \n \n")), 2);
        assert_eq!(documents(&format!("{head} \r\n = Two = \r\n \r\n")), 2);
        assert_eq!(
            documents(&format!("{head}\n = Empty line before = \n \n")),
            1
        );
        assert_eq!(documents(&format!("{head} \n = At the end = \n")), 1);
    }

    #[test]
    fn text_before_the_first_heading_is_one_document() {
        assert_eq!(documents(" lead . \n \n = One = \n \n x . \n"), 2);
        assert_eq!(documents(" \n \n = One = \n \n x . \n"), 1);
        assert_eq!(documents(" = First line = \n \n x . \n"), 1);
        assert_eq!(documents(" \n = Never confirmed = \n x . \n"), 1);
        assert_eq!(documents(" \n = Unconfirmed at the end = \n"), 1);
        assert_eq!(
            documents(" no heading at all . \n \n second paragraph . \n"),
            1
        );
        assert_eq!(documents(" \n \n"), 0);
    }

    fn copy(bytes: &[u8]) -> (Result<SplitSummary, PrepareError>, Vec<u8>) {
        let mut out = Vec::new();
        let summary = copy_lines(bytes, &mut out, Path::new("in"), Path::new("out"));
        (summary, out)
    }

    #[test]
    fn copy_keeps_the_bytes_and_hashes_them() {
        let text = b" \n = One = \n \n caf\xc3\xa9 . \r\n \r\n = Two = \r\n \r\n no final newline";
        let (summary, out) = copy(text);
        let summary = summary.unwrap();
        assert_eq!(out, text);
        assert_eq!(summary.hash, blake3::hash(text));
        assert_eq!(summary.byte_count, text.len() as u64);
        assert_eq!(summary.document_count, 2);
    }

    #[test]
    fn copy_refuses_what_is_not_utf8_or_holds_no_document() {
        let error = copy(b" = One = \n ok \n bad \xff byte \n").0.err().unwrap();
        assert_eq!(error.to_string(), "in: not UTF-8 at byte 20");
        let error = copy(b" \n \n").0.err().unwrap();
        assert_eq!(error.to_string(), "in: holds no document");
        let error = copy(b"").0.err().unwrap();
        assert_eq!(error.to_string(), "in: holds no document");
    }
}
