//! Adversary cases for story:parser-span-lookup-not-linear-in-column: line ends other than `\n`,
//! tabs, four-byte and combining characters, adjacent unrecognised characters, empty input, end of
//! input and positions reported after error recovery keep the line and column of the base
//! implementation (columns count characters; only `\n` ends a line).

use modelspec_core::syntax::{MAX_NESTING, codes, parse};

fn positions(source: &str) -> Vec<(String, u32, u32)> {
    let diagnostics = match parse("test.mspec", source) {
        Ok(_) => panic!("expected diagnostics for {source:?}"),
        Err(diagnostics) => diagnostics,
    };
    diagnostics
        .iter()
        .map(|d| {
            let span = d
                .span
                .as_ref()
                .unwrap_or_else(|| panic!("syntax diagnostic without span: {d}"));
            (d.code.as_str().to_owned(), span.line, span.column)
        })
        .collect()
}

fn lines_and_columns(source: &str) -> Vec<(u32, u32)> {
    positions(source)
        .into_iter()
        .map(|(_, line, column)| (line, column))
        .collect()
}

/// The base implementation's rule, written independently: the line is one plus the `\n`s before
/// the offset, the column one plus the characters since the last `\n`.
fn oracle(source: &str, offset: usize) -> (u32, u32) {
    let before = &source[..offset];
    let line = before.matches('\n').count() + 1;
    let start = before.rfind('\n').map_or(0, |i| i + 1);
    let column = source[start..offset].chars().count() + 1;
    (u32::try_from(line).unwrap(), u32::try_from(column).unwrap())
}

#[test]
fn empty_and_blank_sources() {
    for (code, line, column) in positions("") {
        assert_eq!((line, column), (1, 1), "{code} on empty source");
    }
    for (code, line, column) in positions("\r\n\r\n") {
        assert_eq!((line, column), (3, 1), "{code} after two CRLF");
    }
    for (code, line, column) in positions("\r\r") {
        assert_eq!(
            (line, column),
            (1, 3),
            "{code}: a lone CR is a column, not a line end"
        );
    }
    for (code, line, column) in positions("\t\t") {
        assert_eq!((line, column), (1, 3), "{code}: a tab is one column");
    }
}

#[test]
fn crlf_lone_cr_and_tab_before_a_multi_byte_character() {
    assert_eq!(
        positions("model M {\r\n\t\u{e9} }")[0],
        (codes::UNRECOGNISED_CHARACTER.as_str().to_owned(), 2, 2)
    );
    assert_eq!(
        positions("model M {\r\u{e9} }")[0],
        (codes::UNRECOGNISED_CHARACTER.as_str().to_owned(), 1, 11)
    );
    assert_eq!(
        positions("model M {\r\r\r\n\r\t \u{6f22} }")[0],
        (codes::UNRECOGNISED_CHARACTER.as_str().to_owned(), 2, 4)
    );
}

#[test]
fn four_byte_combining_and_adjacent_unrecognised_characters() {
    // Two four-byte characters with nothing between them, then a token right after.
    let source = "model M { \u{1f600}\u{1f600}} train T { \u{1f600} }";
    let found = lines_and_columns(source);
    assert_eq!(found[0], (1, 11), "{source:?}: {found:?}");

    // An unrecognised character glued to the end of a token.
    let source = "model M {param\u{1f600} }";
    let found = lines_and_columns(source);
    assert!(found.contains(&(1, 15)), "{source:?}: {found:?}");

    // A combining mark after an identifier character: the mark is its own column.
    let source = "model M { e\u{301} }";
    let found = lines_and_columns(source);
    assert!(
        found
            .iter()
            .all(|&(line, column)| line == 1 && (column == 11 || column == 12 || column == 15)),
        "{source:?}: {found:?}"
    );

    // Unrecognised characters of every UTF-8 width, back to back, then the end of input.
    let source = "model M { \u{e9}\u{6f22}\u{1f600}\u{7f}";
    let found = lines_and_columns(source);
    assert_eq!(found[0], (1, 11), "{source:?}: {found:?}");
    let chars = u32::try_from(source.chars().count()).unwrap();
    for &(line, column) in &found {
        assert!(
            line == 1 && (11..=chars + 1).contains(&column),
            "{source:?}: {found:?}"
        );
    }

    // A byte-order mark is an unrecognised character at 1:1.
    let source = "\u{feff}model M { }";
    assert_eq!(lines_and_columns(source)[0], (1, 1));
}

#[test]
fn positions_after_error_recovery_across_line_ends() {
    let source = "model M { \u{e9} }\r\ntrain T { \u{6f22} }\rmodel N { \u{1f600} }\n\ttrain U {";
    let found = positions(source);
    assert_eq!(
        found[..2],
        [
            (codes::UNRECOGNISED_CHARACTER.as_str().to_owned(), 1, 11),
            (codes::UNRECOGNISED_CHARACTER.as_str().to_owned(), 2, 11),
        ],
        "{found:?}"
    );
    // Everything reported after recovery sits on line 2 (the lone CR does not end it) or on line
    // 3 at or after the tab.
    for (code, line, column) in &found[2..] {
        assert!(
            (*line == 2 && *column >= 15) || (*line == 3 && *column >= 2),
            "{code} at {line}:{column}: {found:?}"
        );
    }
}

#[test]
fn nesting_too_deep_far_right_after_multi_byte_comment() {
    let calls = "f(".repeat(MAX_NESTING + 1);
    let closes = ")".repeat(MAX_NESTING + 1);
    let source = format!(
        "// \u{6f22}\u{1f600}\r\nmodel M {{ fn forward(x: Tensor<f32, [1]>) -> Tensor<f32, [1]> {{ return {calls}x{closes}; }} }}"
    );
    let open = source.match_indices('(').nth(MAX_NESTING + 1).unwrap().0;
    let (line, column) = oracle(&source, open);
    let found = positions(&source);
    assert_eq!(
        found[0],
        (codes::NESTING_TOO_DEEP.as_str().to_owned(), line, column),
        "{found:?}"
    );
}
