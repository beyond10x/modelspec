//! Span lookup in the parser (story:parser-span-lookup-not-linear-in-column): the cost of a
//! diagnostic position does not grow with its column, and every position keeps its line and its
//! column in characters, multi-byte characters included.

use std::time::{Duration, Instant};

use modelspec_core::diag::Diagnostic;
use modelspec_core::syntax::parser::span_lookup_scan_count;
use modelspec_core::syntax::{codes, parse};
use modelspec_core::types::check_source;

/// A valid program whose function body holds `lets` statements, each separated from the next by
/// `separator`.
fn program(lets: usize, separator: &str) -> String {
    let mut body = String::new();
    for i in 0..lets {
        body.push_str(&format!("let a{i} = x;"));
        body.push_str(separator);
    }
    format!(
        "model M {{\n    param w: Tensor<f32, [2, 1]> = xavier_uniform();\n    fn forward(x: Tensor<f32, [3, 2]>) -> Tensor<f32, [3, 1]> {{\n{body}return matmul(x, w);\n    }}\n}}\ntrain T {{\n    model = M;\n    input x: Tensor<f32, [3, 2]>;\n    target y: Tensor<f32, [3, 1]>;\n    loss = mse(model.forward(x), y);\n    optimizer = sgd(lr = 0.1);\n    steps = 1;\n    seed = 1;\n    device = cpu;\n}}\n"
    )
}

fn check_time(source: &str) -> Duration {
    let start = Instant::now();
    let result = check_source("test.mspec", source);
    let elapsed = start.elapsed();
    if let Err(diagnostics) = result {
        panic!("expected a check, got:\n{}", render(&diagnostics));
    }
    elapsed
}

fn render(diagnostics: &[Diagnostic]) -> String {
    diagnostics
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}

const LETS: usize = 20_000;

/// long-line-check-time, gated on work rather than time so that load cannot turn it red: parsing
/// 20 000 `let`s on one line scans at most 1.2x the characters the span lookup scans for the same
/// 20 000 one per line.
#[test]
fn long_line_span_lookup_scans() {
    let one_line = program(LETS, " ");
    let per_line = program(LETS, "\n");
    assert_eq!(one_line.len(), per_line.len());
    parse("test.mspec", &one_line).expect("the one-line program parses");
    parse("test.mspec", &per_line).expect("the one-per-line program parses");

    let one_line_scans = span_lookup_scan_count("test.mspec", &one_line);
    let per_line_scans = span_lookup_scan_count("test.mspec", &per_line);
    eprintln!(
        "long_line_span_lookup_scans: {LETS} lets, one per line {per_line_scans} characters, \
         one line {one_line_scans} characters"
    );
    assert!(
        per_line_scans > 0,
        "the span lookup reported no characters scanned; the counter is not counting"
    );
    assert!(
        one_line_scans * 10 <= per_line_scans * 12,
        "one line scans {one_line_scans} characters vs one per line {per_line_scans}: more than 1.2x"
    );
}

/// long-line-check-time as a wall-clock figure (story:parser-span-lookup-not-linear-in-column).
/// A measurement, not a gate: load on the host moves it, so it is ignored by default and the
/// gated test is `long_line_span_lookup_scans`. Run it with
/// `cargo test --release -p modelspec-core --test span_lookup -- --ignored --nocapture`.
#[test]
#[ignore = "measurement, not a gate: wall-clock time depends on host load"]
fn long_line_check_time_measurement() {
    let one_line = program(LETS, " ");
    let per_line = program(LETS, "\n");
    let (mut per_line_best, mut one_line_best) = (Duration::MAX, Duration::MAX);
    for run in 0..5 {
        if run % 2 == 0 {
            per_line_best = per_line_best.min(check_time(&per_line));
            one_line_best = one_line_best.min(check_time(&one_line));
        } else {
            one_line_best = one_line_best.min(check_time(&one_line));
            per_line_best = per_line_best.min(check_time(&per_line));
        }
    }
    let ratio = one_line_best.as_secs_f64() / per_line_best.as_secs_f64();
    eprintln!(
        "long_line_check_time_measurement: {LETS} lets, one per line {per_line_best:?}, one line \
         {one_line_best:?}, ratio {ratio:.3} (minimum of 5, alternating)"
    );
}

/// `(code, line, column)` of each diagnostic.
fn positions(source: &str) -> Vec<(&'static str, u32, u32)> {
    let diagnostics = match parse("test.mspec", source) {
        Ok(_) => panic!("expected diagnostics for:\n{source}"),
        Err(diagnostics) => diagnostics,
    };
    diagnostics
        .iter()
        .map(|d| {
            let span = d
                .span
                .as_ref()
                .unwrap_or_else(|| panic!("syntax diagnostic without span: {d}"));
            (d.code.as_str(), span.line, span.column)
        })
        .collect()
}

/// columns-unchanged: a column counts characters, not bytes, after multi-byte characters earlier
/// on the same line, and the line count is not thrown off by them.
#[test]
fn columns_count_characters_after_multi_byte_characters() {
    // Both items on one line; each holds one unrecognised multi-byte character.
    let source = "model M { é } train T { ü }";
    assert_eq!(
        positions(source),
        vec![
            (codes::UNRECOGNISED_CHARACTER.as_str(), 1, 11),
            (codes::UNRECOGNISED_CHARACTER.as_str(), 1, 25),
        ]
    );

    // A comment of multi-byte characters on the line before does not move the next line's
    // columns.
    let source = "// ééé ü 漢字\nmodel M { ü } train T { 漢 }";
    assert_eq!(
        positions(source),
        vec![
            (codes::UNRECOGNISED_CHARACTER.as_str(), 2, 11),
            (codes::UNRECOGNISED_CHARACTER.as_str(), 2, 25),
        ]
    );

    // Three-byte characters before a token on a long line.
    let prefix = "漢".repeat(500);
    let source = format!("model M {{ {prefix} }}");
    let found = positions(&source);
    assert_eq!(found[0], (codes::UNRECOGNISED_CHARACTER.as_str(), 1, 11));
}

/// columns-unchanged: the end of input after a trailing multi-byte comment without a newline is
/// one past its last character.
#[test]
fn end_of_input_column_counts_characters() {
    let source = "model M { param // ééé";
    let found = positions(source);
    let chars = u32::try_from(source.chars().count()).unwrap();
    assert_eq!(
        found,
        vec![
            (codes::UNEXPECTED_END.as_str(), 1, chars + 1),
            (codes::MISSING_ITEM.as_str(), 1, chars + 1),
        ]
    );

    let source = "model M {\n\tparam\r\n// ü\n";
    let found = positions(source);
    assert_eq!(found[0], (codes::UNEXPECTED_END.as_str(), 4, 1));
}

/// columns-unchanged: on one long ASCII line, the column of a diagnostic far to the right is its
/// character offset plus one.
#[test]
fn far_right_column_on_a_long_line() {
    let mut source = program(2_000, " ");
    let at = source.find("return").unwrap();
    source.replace_range(at..at, "bogus ");
    let line = source[..at].matches('\n').count() + 1;
    let line_start = source[..at].rfind('\n').map_or(0, |i| i + 1);
    let column = source[line_start..at].chars().count() + 1;
    assert_eq!(
        positions(&source),
        vec![(
            codes::UNEXPECTED_TOKEN.as_str(),
            u32::try_from(line).unwrap(),
            u32::try_from(column).unwrap()
        )]
    );
}
