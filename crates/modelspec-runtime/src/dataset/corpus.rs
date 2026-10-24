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

/// Whether `line` starts an article of the release: a top-level heading `= <title> =`, one `=`
/// on each side and surrounding whitespace ignored. A section heading `= = <section> = =` is
/// not one.
pub fn is_article_heading(line: &str) -> bool {
    let t = line.trim();
    t.len() > 3
        && t.starts_with("= ")
        && t.ends_with(" =")
        && !t.starts_with("= =")
        && !t[2..t.len() - 2].trim().is_empty()
}

/// Counts the documents of a split as its lines arrive: one per article heading, plus one for
/// any non-blank text before the first heading.
#[derive(Default)]
struct DocumentCounter {
    headings: u64,
    text_before_first_heading: bool,
}

impl DocumentCounter {
    fn line(&mut self, line: &str) {
        if is_article_heading(line) {
            self.headings += 1;
        } else if self.headings == 0 && !line.trim().is_empty() {
            self.text_before_first_heading = true;
        }
    }

    fn count(&self) -> u64 {
        self.headings + u64::from(self.text_before_first_heading)
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
    fn article_heading_is_one_equals_sign_on_each_side() {
        assert!(is_article_heading(" = Copper Bridge = \n"));
        assert!(is_article_heading("= A ="));
        assert!(!is_article_heading(" = = Repairs = = \n"));
        assert!(!is_article_heading(" = = = Fair = = = \n"));
        assert!(!is_article_heading(" = = \n"));
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
    fn text_before_the_first_heading_is_one_document() {
        assert_eq!(documents(" lead . \n \n = One = \n x . \n"), 2);
        assert_eq!(documents(" \n \n = One = \n x . \n"), 1);
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
        let text = b" \n = One = \n \n caf\xc3\xa9 . \r\n = Two = \n no final newline";
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
