//! Adversary cases for `modelspec check` (story m0-check-types-shapes, pass 1).
//!
//! Sources are inline: a subdirectory under `tests/language/check/` would itself be listed by
//! `check.rs`'s `every_fixture_file_is_a_test_case`.

use std::path::PathBuf;
use std::process::Command;

use modelspec_core::diag::Diagnostic;
use modelspec_core::hir::{Device, FnRef, Hir, HirExprKind, LetRef};
use modelspec_core::ir::tensor::TensorOp;
use modelspec_core::types::check_source;

/// Each source operator name becomes its own Tensor IR op in the HIR (DESIGN.md § 2.2), not
/// merely an op with the same typing rule: `add`/`sub`/`mul` and `sum`/`mean` type alike.
#[test]
fn each_source_operator_is_its_own_op_in_hir() {
    for (call, want) in [
        ("matmul(x, w)", TensorOp::MatMul),
        ("add(x, x)", TensorOp::Add),
        ("sub(x, x)", TensorOp::Sub),
        ("mul(x, x)", TensorOp::Mul),
        ("relu(x)", TensorOp::Relu),
        ("transpose(x)", TensorOp::Transpose),
        ("sum(x)", TensorOp::ReduceSum),
        ("mean(x)", TensorOp::ReduceMean),
        ("mse(x, x)", TensorOp::Mse),
    ] {
        let source = with_train(&format!(
            "model LinearReg {{
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {{
        let t = {call};
        return matmul(x, w);
    }}
}}
"
        ));
        let hir = hir(&source);
        let (_, forward) = hir.model.function("forward").expect("forward");
        match &forward.lets[0].value.kind {
            HirExprKind::Op { op, .. } => assert_eq!(*op, want, "{call}"),
            other => panic!("{call} is {other:?}"),
        }
    }
}

const TRAIN: &str = "
train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
";

const LINEAR_REGRESSION: &str = include_str!("../../../examples/linear_regression.mspec");

fn with_train(model: &str) -> String {
    format!("{model}{TRAIN}")
}

/// `(code, line, column)` of each diagnostic, in the order returned.
fn positions(diagnostics: &[Diagnostic]) -> Vec<(String, u32, u32)> {
    diagnostics
        .iter()
        .map(|d| {
            let span = d.span.as_ref().expect("source diagnostic has a span");
            (d.code.as_str().to_owned(), span.line, span.column)
        })
        .collect()
}

fn diagnostics(source: &str) -> Vec<(String, u32, u32)> {
    match check_source("a.mspec", source) {
        Ok(_) => Vec::new(),
        Err(d) => positions(&d),
    }
}

fn hir(source: &str) -> Hir {
    match check_source("a.mspec", source) {
        Ok(hir) => hir,
        Err(d) => panic!(
            "expected a clean check, got:\n{}",
            d.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        ),
    }
}

/// 1-based column of `needle` on 1-based line `line` of `source`.
fn column(source: &str, line: u32, needle: &str) -> u32 {
    let text = source.lines().nth(line as usize - 1).expect("line");
    text.find(needle).expect("needle on line") as u32 + 1
}

fn e(code: &str, line: u32, column: u32) -> (String, u32, u32) {
    (code.to_owned(), line, column)
}

// ---------------------------------------------------------------------------------------------
// Cascades after a duplicate model item (resolve.rs ModelNames::collect).
// ---------------------------------------------------------------------------------------------

/// `param forward` then `fn forward`: the duplicate is reported once; the train block's
/// `model.forward(x)` must not add an E-NAME-008 "has no function `forward`" for a function the
/// file plainly declares.
#[test]
fn function_named_like_earlier_parameter_reports_only_the_duplicate() {
    let source = with_train(
        "model LinearReg {
    param forward: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, forward);
    }
}
",
    );
    assert_eq!(diagnostics(&source), [e("E-NAME-003", 4, 8)]);
}

/// `fn w` then `param w`: the duplicate is reported once; `forward`'s use of `w` must not add an
/// E-NAME-001 "unknown name `w`" for a parameter the file plainly declares.
#[test]
fn parameter_named_like_earlier_function_reports_only_the_duplicate() {
    let source = with_train(
        "model LinearReg {
    fn w(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 4]> {
        return x;
    }

    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}
",
    );
    assert_eq!(diagnostics(&source), [e("E-NAME-003", 6, 11)]);
}

// ---------------------------------------------------------------------------------------------
// Rules no implementor case isolates (each kills a mutant the suite lets live).
// ---------------------------------------------------------------------------------------------

/// 2^62 f32 elements fit `u64`; 2^64 bytes do not (DESIGN.md § 2.2 "element/byte counts").
#[test]
fn byte_count_overflow_without_element_overflow_is_e_type_016() {
    let source = LINEAR_REGRESSION.replace(
        "    target y: Tensor<f32, [8, 1]>;\n",
        "    target y: Tensor<f32, [8, 1]>;\n    input big: Tensor<f32, [4611686018427387904]>;\n",
    );
    let line = source
        .lines()
        .position(|l| l.contains("input big"))
        .unwrap() as u32
        + 1;
    assert_eq!(
        diagnostics(&source),
        [e("E-TYPE-016", line, column(&source, line, "Tensor"))]
    );
}

/// Diagnostics come in source order even when the train block precedes the model, so that the
/// checker's emission order (model first) differs from source order.
#[test]
fn diagnostics_sorted_when_train_block_precedes_model() {
    let source = "train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 0;
    seed = 42;
    device = cpu;
}

model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, v);
    }
}
";
    assert_eq!(
        diagnostics(source),
        [e("E-TYPE-011", 7, 5), e("E-NAME-001", 16, 26)]
    );

    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("check_adversary");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("train_first.mspec");
    std::fs::write(&path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .arg("check")
        .arg(&path)
        .output()
        .expect("modelspec runs");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8(output.stderr).unwrap();
    let heads: Vec<&str> = stderr
        .lines()
        .filter(|l| !l.starts_with("  "))
        .map(|l| l.split(':').next().unwrap())
        .collect();
    assert_eq!(heads, ["E-TYPE-011", "E-NAME-001"], "{stderr}");
}

/// check.rs: "every diagnostic is printed to stderr ... in source order". Parser recovery after
/// `model  {` reports 11:12 then 11:5; `check` prints them as the parser returned them.
#[test]
fn syntax_diagnostics_printed_in_source_order() {
    let source = LINEAR_REGRESSION.replace("    model = LinearReg;", "    model  {  LinearReg;");
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("check_adversary");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("syntax_order.mspec");
    std::fs::write(&path, &source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .arg("check")
        .arg(&path)
        .output()
        .expect("modelspec runs");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8(output.stderr).unwrap();
    let locations: Vec<(u32, u32)> = stderr
        .lines()
        .filter_map(|l| l.strip_prefix(&format!("  {}:", path.display())))
        .map(|rest| {
            let (line, column) = rest.split_once(':').unwrap();
            (line.parse().unwrap(), column.parse().unwrap())
        })
        .collect();
    assert!(locations.len() >= 2, "{stderr}");
    let mut sorted = locations.clone();
    sorted.sort();
    assert_eq!(locations, sorted, "{stderr}");
}

/// `xavier_uniform()` is rank 2 only: rank 3 and rank 0 are E-TYPE-014 as well as rank 1.
#[test]
fn xavier_uniform_on_rank_3_and_rank_0_is_e_type_014() {
    let source = with_train(
        "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();
    param c: Tensor<f32, [2, 2, 2]> = xavier_uniform();
    param s: Tensor<f32, []> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}
",
    );
    assert_eq!(
        diagnostics(&source),
        [
            e("E-TYPE-014", 3, column(&source, 3, "xavier")),
            e("E-TYPE-014", 4, column(&source, 4, "xavier")),
        ]
    );
}

/// A `let` sees only earlier bindings: a use before its definition and a self-reference are
/// E-NAME-001.
#[test]
fn let_used_before_definition_or_by_itself_is_e_name_001() {
    let before = with_train(
        "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        let a = relu(b);
        let b = matmul(x, w);
        return b;
    }
}
",
    );
    assert_eq!(diagnostics(&before), [e("E-NAME-001", 5, 22)]);

    let itself = with_train(
        "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        let h = relu(h);
        return matmul(x, w);
    }
}
",
    );
    assert_eq!(diagnostics(&itself), [e("E-NAME-001", 5, 22)]);
}

// ---------------------------------------------------------------------------------------------
// Typed HIR content that later stories read (hir.rs: "Later stories read the train configuration
// from `HirTrain` and do not edit this file").
// ---------------------------------------------------------------------------------------------

#[test]
fn vulkan_device_index_reaches_hir_train() {
    let source = LINEAR_REGRESSION.replace("device = cpu;", "device = vulkan(3);");
    assert_eq!(hir(&source).train.device, Device::Vulkan(3));
}

/// Each `let` use resolves to its own binding, not to another one of the same type.
#[test]
fn let_uses_resolve_to_their_own_binding_in_hir() {
    let source = with_train(
        "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        let a = matmul(x, w);
        let b = relu(a);
        return b;
    }
}
",
    );
    let hir = hir(&source);
    let (_, forward) = hir.model.function("forward").expect("forward");
    assert_eq!(forward.result.kind, HirExprKind::Let(LetRef(1)));
    match &forward.lets[1].value.kind {
        HirExprKind::Op { args, .. } => assert_eq!(args[0].kind, HirExprKind::Let(LetRef(0))),
        other => panic!("let b = {other:?}"),
    }
}

/// `model.forward` reaches `forward` when it is not the model's first function.
#[test]
fn model_forward_resolves_when_forward_is_not_the_first_function() {
    let source = with_train(
        "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn helper(a: Tensor<f32, [1, 8]>) -> Tensor<f32, [1, 8]> {
        return a;
    }

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}
",
    );
    let hir = hir(&source);
    match &hir.train.loss.kind {
        HirExprKind::Op { args, .. } => match &args[0].kind {
            HirExprKind::Call { function, .. } => assert_eq!(*function, FnRef(1)),
            other => panic!("mse lhs is {other:?}"),
        },
        other => panic!("loss is {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// "It reports every diagnostic of the file, not only the first" (types.rs module doc).
// ---------------------------------------------------------------------------------------------

/// A wrong initializer leaves the parameter's declared type valid and known, so a contracting
/// mismatch on that parameter elsewhere is a second, independent diagnostic.
#[test]
fn wrong_initializer_does_not_hide_an_independent_matmul_mismatch() {
    let source = with_train(
        "model LinearReg {
    param w: Tensor<f32, [3, 1]> = zeros();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}
",
    );
    assert_eq!(
        diagnostics(&source),
        [e("E-TYPE-013", 2, 36), e("E-TYPE-012", 5, 16)]
    );
}

// ---------------------------------------------------------------------------------------------
// Never panic; E-INTERNAL-001 unreachable; diagnostics well-formed on edited sources.
// ---------------------------------------------------------------------------------------------

fn tokens(source: &str) -> Vec<(usize, usize)> {
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.')
            {
                i += 1;
            }
            out.push((start, i));
        } else if c == b'-' && bytes.get(i + 1) == Some(&b'>') {
            out.push((i, i + 2));
            i += 2;
        } else if c == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else {
            let len = source[i..].chars().next().unwrap().len_utf8();
            out.push((i, i + len));
            i += len;
        }
    }
    out
}

fn violations(source: &str, out: &mut Vec<String>) {
    let result = std::panic::catch_unwind(|| check_source("fuzz.mspec", source));
    let problem = match result {
        Err(_) => Some("panicked".to_owned()),
        Ok(Ok(hir)) => {
            let t = &hir.train;
            if t.loss.ty.rank() != 0 || !(t.lr.is_finite() && t.lr > 0.0) || t.steps == 0 {
                Some(format!("clean check broke a train rule: {t:?}"))
            } else if hir.model.params.iter().any(|p| p.ty.rank() != 2) {
                Some("clean check kept a non-rank-2 parameter".to_owned())
            } else {
                None
            }
        }
        Ok(Err(diagnostics)) => {
            let mut problem = None;
            if diagnostics.is_empty() {
                problem = Some("Err with no diagnostic".to_owned());
            }
            for d in &diagnostics {
                if d.code.as_str() == "E-INTERNAL-001" {
                    problem = Some(format!("reached E-INTERNAL-001: {d}"));
                }
                if d.span.is_none() {
                    problem = Some(format!("diagnostic without span: {d}"));
                }
            }
            let keys: Vec<(u32, u32)> = diagnostics
                .iter()
                .filter_map(|d| d.span.as_ref().map(|s| (s.line, s.column)))
                .collect();
            if keys.windows(2).any(|w| w[0] > w[1]) {
                problem = Some(format!("diagnostics out of source order: {keys:?}"));
            }
            for (i, a) in diagnostics.iter().enumerate() {
                if diagnostics[i + 1..].iter().any(|b| b == a) {
                    problem = Some(format!("duplicate diagnostic: {a}"));
                }
            }
            problem
        }
    };
    if let Some(problem) = problem {
        out.push(format!("{problem}\n--- source ---\n{source}"));
    }
}

#[test]
fn edited_linear_regression_never_panics_or_reaches_internal() {
    let base = LINEAR_REGRESSION;
    let toks = tokens(base);
    let mut vocab: Vec<String> = toks.iter().map(|&(a, b)| base[a..b].to_owned()).collect();
    for extra in [
        "matmul",
        "add",
        "sub",
        "mul",
        "relu",
        "transpose",
        "sum",
        "mean",
        "mse",
        "0",
        "1",
        "0.0",
        "1e39",
        "1e-50",
        "4294967296",
        "18446744073709551615",
        "18446744073709551616",
        "4611686018427387904",
        "vulkan",
        "cpu",
        "xavier_uniform",
        "zeros",
        "model",
        "input",
        "target",
        "param",
        "fn",
        "let",
        "return",
        "Tensor",
        "f32",
        "f16",
        "w",
        "x",
        "y",
        "forward",
        "LinearReg",
        "(",
        ")",
        "[",
        "]",
        "<",
        ">",
        ",",
        ";",
        "=",
        "{",
        "}",
        ".",
        "->",
        "+",
        ":",
    ] {
        vocab.push(extra.to_owned());
    }
    vocab.sort();
    vocab.dedup();

    let mut out = Vec::new();
    let mut runs = 0usize;
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    for (i, _) in base.char_indices() {
        violations(&base[..i], &mut out);
        runs += 1;
    }
    for &(a, b) in &toks {
        violations(&format!("{}{}", &base[..a], &base[b..]), &mut out);
        runs += 1;
        for word in &vocab {
            violations(&format!("{} {word} {}", &base[..a], &base[b..]), &mut out);
            violations(&format!("{} {word} {}", &base[..a], &base[a..]), &mut out);
            runs += 2;
        }
    }
    std::panic::set_hook(default_hook);
    assert!(runs > 10_000, "only {runs} edits ran");
    assert!(
        out.is_empty(),
        "{} of {runs} edited sources broke a property; first:\n{}",
        out.len(),
        out.iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n=====\n")
    );
}
