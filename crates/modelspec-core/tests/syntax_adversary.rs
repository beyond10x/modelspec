//! Adversary cases for the M0 parser: inputs that must never panic, recovery that must not
//! invent diagnostics, and round-trip and lexer rules the unit's own suite leaves unpinned.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use modelspec_core::diag::Diagnostic;
use modelspec_core::syntax::{parse, print};

const BASE: &str = "\
model M {
    param w: Tensor<f32, [2, 1]> = zeros();
    fn forward(x: Tensor<f32, [3, 2]>) -> Tensor<f32, [3, 1]> {
        return matmul(x, w);
    }
}
train T {
    model = M;
    input x: Tensor<f32, [3, 2]>;
    target y: Tensor<f32, [3, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 10;
    seed = 1;
    device = cpu;
}
";

fn linear_regression() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/linear_regression.mspec");
    std::fs::read_to_string(path).expect("read linear_regression.mspec")
}

fn summary(diagnostics: &[Diagnostic]) -> Vec<(&'static str, u32, u32)> {
    diagnostics
        .iter()
        .map(|d| {
            let span = d.span.as_ref().expect("syntax diagnostic has a span");
            (d.code.as_str(), span.line, span.column)
        })
        .collect()
}

/// Parses `source` and returns the panic message if parsing panicked.
fn panics(source: &str) -> Option<String> {
    catch_unwind(AssertUnwindSafe(|| {
        let _ = parse("fuzz.mspec", source);
    }))
    .err()
    .map(|payload| {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_default()
    })
}

fn assert_round_trip(source: &str) {
    let mut first = parse("a.mspec", source)
        .unwrap_or_else(|d| panic!("source does not parse: {d:?}\n{source}"));
    let printed = print(&first);
    let mut second = parse("b.mspec", &printed)
        .unwrap_or_else(|d| panic!("printed form does not parse: {d:?}\n{printed}"));
    first.erase_spans();
    second.erase_spans();
    assert_eq!(first, second, "printed:\n{printed}");
}

#[test]
fn every_prefix_of_linear_regression_returns_without_panicking() {
    let source = linear_regression();
    let mut failures = Vec::new();
    for (end, _) in source.char_indices().chain([(source.len(), ' ')]) {
        if let Some(message) = panics(&source[..end]) {
            failures.push((end, message));
        }
    }
    assert!(failures.is_empty(), "panicking prefixes: {failures:?}");
}

#[test]
fn every_single_character_edit_of_linear_regression_returns_without_panicking() {
    let source = linear_regression();
    let mut failures = Vec::new();
    let boundaries: Vec<usize> = source
        .char_indices()
        .map(|(i, _)| i)
        .chain([source.len()])
        .collect();
    for window in boundaries.windows(2) {
        let deleted = format!("{}{}", &source[..window[0]], &source[window[1]..]);
        if let Some(message) = panics(&deleted) {
            failures.push(("delete", window[0], message));
        }
    }
    for &at in &boundaries {
        for insert in ["é", "€", "😀", "{", "}", "<", "@", "\u{feff}", "\r"] {
            let inserted = format!("{}{insert}{}", &source[..at], &source[at..]);
            if let Some(message) = panics(&inserted) {
                failures.push(("insert", at, message));
            }
        }
    }
    assert!(failures.is_empty(), "panicking edits: {failures:?}");
}

#[test]
fn multibyte_unrecognised_characters_are_one_diagnostic_at_the_first() {
    let source = BASE.replacen("param w:", "param wé€😀:", 1);
    let diagnostics = parse("t.mspec", &source).expect_err("unrecognised characters");
    assert_eq!(summary(&diagnostics), [("E-SYNTAX-001", 2, 12)]);
}

/// A model whose closing `}` is missing is one error in the model item. The train item that
/// follows exists; the parser must not report it missing.
#[test]
fn unclosed_model_does_not_report_the_following_train_as_missing() {
    let split = BASE.find("train").unwrap();
    let model_close = BASE[..split].rfind('}').unwrap();
    let source = format!("{}{}", &BASE[..model_close], &BASE[split..]);
    let diagnostics = parse("t.mspec", &source).expect_err("unclosed model");
    assert!(
        diagnostics
            .iter()
            .all(|d| d.code.as_str() != "E-SYNTAX-020"),
        "the file has a train item, yet: {:?}",
        summary(&diagnostics)
    );
}

/// The same recovery path through a function whose closing `}` is missing.
#[test]
fn unclosed_function_does_not_report_the_following_train_as_missing() {
    let source = BASE.replacen(
        "        return matmul(x, w);\n    }\n",
        "        return matmul(x, w);\n",
        1,
    );
    let diagnostics = parse("t.mspec", &source).expect_err("unclosed function");
    assert!(
        diagnostics
            .iter()
            .all(|d| d.code.as_str() != "E-SYNTAX-020"),
        "the file has a train item, yet: {:?}",
        summary(&diagnostics)
    );
}

/// Deeply nested calls must end in `Ok` or `Err`, not a stack overflow. Run in a child process,
/// on an 8 MiB thread like a binary's main thread, so that an overflow fails this case only.
#[test]
fn deeply_nested_calls_do_not_overflow_the_stack() {
    const DEPTH: usize = 10_000;
    if std::env::var_os("SYNTAX_ADVERSARY_CHILD").is_some() {
        let depth = std::env::var("SYNTAX_ADVERSARY_DEPTH")
            .ok()
            .and_then(|d| d.parse().ok())
            .unwrap_or(DEPTH);
        let nested = format!("{}x{}", "f(".repeat(depth), ")".repeat(depth));
        let source = BASE.replacen("matmul(x, w)", &nested, 1);
        std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(move || {
                let _ = parse("deep.mspec", &source);
            })
            .unwrap()
            .join()
            .unwrap();
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "deeply_nested_calls_do_not_overflow_the_stack",
            "--exact",
            "--nocapture",
        ])
        .env("SYNTAX_ADVERSARY_CHILD", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child exited {:?}; stderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn literal_extremes_round_trip() {
    for lr in [
        "1e-7",
        "1e16",
        "1e15",
        "0.30000000000000004",
        "1.7976931348623157e308",
        "5e-324",
        "123456789.123456789",
        "0.0",
        "1E5",
        "2.5e+3",
    ] {
        assert_round_trip(&BASE.replacen("lr = 0.1", &format!("lr = {lr}"), 1));
        assert_round_trip(&BASE.replacen("matmul(x, w)", &format!("mul(x, {lr})"), 1));
    }
    assert_round_trip(&BASE.replacen("seed = 1", "seed = 18446744073709551615", 1));
    assert_round_trip(&BASE.replacen("[2, 1]", "[18446744073709551615, 007]", 1));
}

#[test]
fn named_and_nested_arguments_round_trip() {
    assert_round_trip(&BASE.replacen(
        "matmul(x, w)",
        "f(a = g(b = model.forward(x, k = 1), 2.0), x, c = w)",
        1,
    ));
}

#[test]
fn identifiers_that_start_with_a_keyword_are_identifiers() {
    let source = BASE
        .replacen("param w:", "param modelx:", 1)
        .replacen("fn forward(x:", "fn fnord(inputs:", 1)
        .replacen(
            "        return matmul(x, w);",
            "        let letter = trains;\n        let targets = returned;\n        return matmul(letter, targets);",
            1,
        );
    assert_round_trip(&source);
}

/// Seeded token soup spliced into `BASE`: no input panics, and every input that parses
/// round-trips.
#[test]
fn seeded_token_soup_never_panics_and_accepted_inputs_round_trip() {
    const PIECES: &[&str] = &[
        "model", "train", "param", "fn", "let", "return", "input", "target", "x", "w", "f", "lr",
        "sgd", "cpu", "vulkan", "Tensor", "f32", "0", "7", "1.5", "2e3", "{", "}", "(", ")", "[",
        "]", "<", ">", "->", ":", ";", ",", ".", "=", "-", "//", "\n", " ", "é", "\t", "\r\n",
    ];
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let boundaries: Vec<usize> = BASE.char_indices().map(|(i, _)| i).collect();
    let mut failures = Vec::new();
    for _ in 0..20_000 {
        let at = boundaries[(next() % boundaries.len() as u64) as usize];
        let len = (next() % 4) as usize;
        let soup: String = (0..len)
            .map(|_| PIECES[(next() % PIECES.len() as u64) as usize])
            .collect::<Vec<_>>()
            .join(" ");
        let source = format!("{}{soup}{}", &BASE[..at], &BASE[at..]);
        if let Some(message) = panics(&source) {
            failures.push((source, message));
            continue;
        }
        if parse("s.mspec", &source).is_ok()
            && catch_unwind(AssertUnwindSafe(|| assert_round_trip(&source))).is_err()
        {
            failures.push((source, "round trip".to_owned()));
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures, first: {:?}",
        failures.len(),
        failures.first()
    );
}

#[test]
fn comment_at_end_of_input_without_newline_is_skipped() {
    assert_round_trip(&format!("{BASE}// trailing"));
    assert_round_trip(&format!("{BASE}//"));
    assert_round_trip(&format!("// only é\r\n{BASE}"));
}
