//! Parser for the M0 language subset: round-trip of `examples/linear_regression.mspec` and the
//! syntax diagnostics (DESIGN.md § 2, § 3.3 pass 1, § 9 gate 1).
//!
//! Every `E-SYNTAX` code has one negative fixture under `tests/language/syntax/` (repository
//! root). A fixture's first line reads `// expect <code> <line>:<column> <message fragment>`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use modelspec_core::diag::Diagnostic;
use modelspec_core::syntax::ast::{Ast, DType, DataRole, Device, ExprKind, ModelItem, Optimizer};
use modelspec_core::syntax::{codes, parse, print};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(relative: &str) -> String {
    let path = repo_root().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn render(diagnostics: &[Diagnostic]) -> String {
    diagnostics
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_ok(source: &str) -> Ast {
    parse("test.mspec", source).unwrap_or_else(|d| panic!("expected a parse, got:\n{}", render(&d)))
}

fn parse_err(source: &str) -> Vec<Diagnostic> {
    match parse("test.mspec", source) {
        Ok(ast) => panic!("expected diagnostics, got a parse:\n{}", print(&ast)),
        Err(diagnostics) => diagnostics,
    }
}

/// `(code, line, column)` of each diagnostic.
fn summary(diagnostics: &[Diagnostic]) -> Vec<(&'static str, u32, u32)> {
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

/// Asserts exactly one diagnostic with this code and position, whose message contains `fragment`.
#[track_caller]
fn assert_single(source: &str, code: &str, line: u32, column: u32, fragment: &str) {
    let diagnostics = parse_err(source);
    assert_eq!(
        summary(&diagnostics),
        vec![(code, line, column)],
        "source:\n{source}\ndiagnostics:\n{}",
        render(&diagnostics)
    );
    assert!(
        diagnostics[0].message.contains(fragment),
        "message {:?} lacks {fragment:?}",
        diagnostics[0].message
    );
}

/// A valid program in the M0 subset; the negative cases below edit one line of it.
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

/// `BASE` with the first occurrence of `from` replaced by `to`.
fn edit(from: &str, to: &str) -> String {
    assert!(BASE.contains(from), "BASE lacks {from:?}");
    BASE.replacen(from, to, 1)
}

// ---------------------------------------------------------------------------------------------
// Negative fixtures: one per E-SYNTAX code.
// ---------------------------------------------------------------------------------------------

const FIXTURE_DIR: &str = "tests/language/syntax";

fn check_fixture(name: &str) {
    let relative = format!("{FIXTURE_DIR}/{name}");
    let source = read(&relative);
    let header = source
        .lines()
        .next()
        .and_then(|l| l.strip_prefix("// expect "))
        .unwrap_or_else(|| panic!("{name}: first line is not `// expect ...`"));
    let mut words = header.splitn(3, ' ');
    let code = words.next().unwrap();
    let position = words.next().expect("position");
    let fragment = words.next().expect("message fragment");
    let (line, column) = position.split_once(':').expect("line:column");
    let (line, column): (u32, u32) = (line.parse().unwrap(), column.parse().unwrap());
    assert!(
        name.starts_with(&format!("{code}-")),
        "{name} expects {code}; the file name must start with the code it pins"
    );

    let diagnostics = match parse(&relative, &source) {
        Ok(_) => panic!("{name}: parsed without diagnostics"),
        Err(d) => d,
    };
    assert_eq!(
        summary(&diagnostics),
        vec![(code, line, column)],
        "{name}:\n{}",
        render(&diagnostics)
    );
    let diagnostic = &diagnostics[0];
    assert_eq!(diagnostic.span.as_ref().unwrap().file, relative);
    assert!(
        diagnostic.message.contains(fragment),
        "{name}: message {:?} lacks {fragment:?}",
        diagnostic.message
    );
}

macro_rules! fixtures {
    ($($test:ident => $file:literal,)*) => {
        const FIXTURES: &[&str] = &[$($file),*];
        $(
            #[test]
            fn $test() {
                check_fixture($file);
            }
        )*
    };
}

fixtures! {
    fixture_e_syntax_001_unrecognised_character => "E-SYNTAX-001-unrecognised-character.mspec",
    fixture_e_syntax_002_unexpected_token => "E-SYNTAX-002-unexpected-token.mspec",
    fixture_e_syntax_003_unexpected_end_of_input => "E-SYNTAX-003-unexpected-end-of-input.mspec",
    fixture_e_syntax_004_module => "E-SYNTAX-004-module.mspec",
    fixture_e_syntax_005_layer => "E-SYNTAX-005-layer.mspec",
    fixture_e_syntax_006_generic_parameters => "E-SYNTAX-006-generic-parameters.mspec",
    fixture_e_syntax_007_generic_arguments => "E-SYNTAX-007-generic-arguments.mspec",
    fixture_e_syntax_008_symbolic_dimension => "E-SYNTAX-008-symbolic-dimension.mspec",
    fixture_e_syntax_009_operator => "E-SYNTAX-009-operator.mspec",
    fixture_e_syntax_010_const => "E-SYNTAX-010-const.mspec",
    fixture_e_syntax_011_repeat => "E-SYNTAX-011-repeat.mspec",
    fixture_e_syntax_012_dtype => "E-SYNTAX-012-dtype.mspec",
    fixture_e_syntax_013_optimizer => "E-SYNTAX-013-optimizer.mspec",
    fixture_e_syntax_014_device => "E-SYNTAX-014-device.mspec",
    fixture_e_syntax_015_literal_out_of_range => "E-SYNTAX-015-literal-out-of-range.mspec",
    fixture_e_syntax_016_unknown_train_setting => "E-SYNTAX-016-unknown-train-setting.mspec",
    fixture_e_syntax_017_duplicate_train_setting => "E-SYNTAX-017-duplicate-train-setting.mspec",
    fixture_e_syntax_018_missing_train_setting => "E-SYNTAX-018-missing-train-setting.mspec",
    fixture_e_syntax_019_second_item => "E-SYNTAX-019-second-item.mspec",
    fixture_e_syntax_020_missing_item => "E-SYNTAX-020-missing-item.mspec",
    fixture_e_syntax_021_nesting_too_deep => "E-SYNTAX-021-nesting-too-deep.mspec",
}

fn fixture_files() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(repo_root().join(FIXTURE_DIR))
        .expect("fixture directory")
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

#[test]
fn every_fixture_file_is_a_test_case() {
    let mut listed: Vec<String> = FIXTURES.iter().map(|s| (*s).to_owned()).collect();
    listed.sort();
    assert_eq!(fixture_files(), listed);
}

#[test]
fn every_syntax_code_has_exactly_one_fixture() {
    let declared: Vec<String> = codes::ALL.iter().map(|c| c.as_str().to_owned()).collect();
    let fixture_codes: Vec<String> = fixture_files()
        .into_iter()
        .map(|f| f[.."E-SYNTAX-0nn".len()].to_owned())
        .collect();
    assert_eq!(
        fixture_codes, declared,
        "fixture codes, in file order, must equal syntax::codes::ALL, one each"
    );
}

#[test]
fn syntax_codes_are_distinct_and_in_the_syntax_family() {
    let mut seen = BTreeSet::new();
    for code in codes::ALL {
        assert!(code.as_str().starts_with("E-SYNTAX-0"), "{code}");
        assert!(seen.insert(code.as_str()), "{code} declared twice");
    }
    assert!(!codes::ALL.is_empty());
}

// ---------------------------------------------------------------------------------------------
// Round trip and canonical printing of examples/linear_regression.mspec.
// ---------------------------------------------------------------------------------------------

const LINEAR_REGRESSION: &str = "examples/linear_regression.mspec";

const LINEAR_REGRESSION_CANONICAL: &str = "\
model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

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

fn linear_regression() -> Ast {
    parse(LINEAR_REGRESSION, &read(LINEAR_REGRESSION)).unwrap_or_else(|d| panic!("{}", render(&d)))
}

#[test]
fn linear_regression_round_trips_to_an_equal_ast_up_to_spans() {
    let first = linear_regression();
    let printed = print(&first);
    let second = parse("printed.mspec", &printed).unwrap_or_else(|d| panic!("{}", render(&d)));
    assert_ne!(
        first, second,
        "the printed form moves every node, so the ASTs must differ before spans are erased"
    );
    let (mut a, mut b) = (first, second);
    a.erase_spans();
    b.erase_spans();
    assert_eq!(a, b);
}

#[test]
fn linear_regression_prints_canonically() {
    assert_eq!(print(&linear_regression()), LINEAR_REGRESSION_CANONICAL);
}

/// A program using every node kind of the AST, laid out unlike the canonical form so that
/// every span moves when it is printed and parsed again.
const EVERY_NODE: &str = "\
// every node kind
model Every { param a: Tensor<f32, []> = init(scale = 2, 0.5);
  fn f(x: Tensor<f32, [2, 3]>, y: Tensor<f32, [3]>) -> Tensor<f32, [2]> {
    let h = matmul(x, y); let g = other.method(h, k = 1.5);
    return add(g, 7); }
  param b: Tensor<f32, [1]> = zeros(); }
train Run { target t: Tensor<f32, [2]>; input x: Tensor<f32, [2, 3]>;
  device = vulkan(1); seed = 9; steps = 3; optimizer = sgd(lr = 1e-3);
  loss = mse(model.f(x, y = x), t); model = Every; }
";

#[test]
fn every_node_kind_round_trips_to_an_equal_ast_up_to_spans() {
    let first = parse_ok(EVERY_NODE);
    let second = parse_ok(&print(&first));
    assert_ne!(first, second);
    let (mut a, mut b) = (first, second);
    a.erase_spans();
    b.erase_spans();
    assert_eq!(a, b);
}

#[test]
fn canonical_print_is_a_fixed_point() {
    let ast = parse_ok(LINEAR_REGRESSION_CANONICAL);
    assert_eq!(print(&ast), LINEAR_REGRESSION_CANONICAL);
}

#[test]
fn linear_regression_ast_holds_the_source_values() {
    let ast = linear_regression();
    assert_eq!(ast.model.name.name, "LinearReg");
    assert_eq!(ast.model.items.len(), 2);
    let ModelItem::Param(w) = &ast.model.items[0] else {
        panic!("first item is not a param")
    };
    assert_eq!(w.name.name, "w");
    assert_eq!(w.ty.dtype.kind, DType::F32);
    assert_eq!(w.ty.dims.iter().map(|d| d.size).collect::<Vec<_>>(), [4, 1]);
    assert_eq!(w.init.callee.name, "xavier_uniform");
    assert!(w.init.args.is_empty());
    let ModelItem::Fn(forward) = &ast.model.items[1] else {
        panic!("second item is not a fn")
    };
    assert_eq!(forward.name.name, "forward");
    assert_eq!(forward.params.len(), 1);
    assert_eq!(forward.params[0].name.name, "x");
    assert_eq!(
        forward
            .output
            .dims
            .iter()
            .map(|d| d.size)
            .collect::<Vec<_>>(),
        [8, 1]
    );
    assert!(forward.lets.is_empty());
    let ExprKind::Call(call) = &forward.result.value.kind else {
        panic!("return value is not a call")
    };
    assert_eq!(call.callee.name, "matmul");
    let args: Vec<_> = call.args.iter().map(|a| (&a.name, &a.value.kind)).collect();
    assert_eq!(
        args,
        [
            (&None, &ExprKind::Var("x".to_owned())),
            (&None, &ExprKind::Var("w".to_owned()))
        ]
    );

    let train = &ast.train;
    assert_eq!(train.name.name, "Fit");
    assert_eq!(train.model.value.name, "LinearReg");
    let data: Vec<_> = train
        .data
        .iter()
        .map(|d| (d.role, d.name.name.as_str()))
        .collect();
    assert_eq!(data, [(DataRole::Input, "x"), (DataRole::Target, "y")]);
    let ExprKind::Call(loss) = &train.loss.value.kind else {
        panic!("loss is not a call")
    };
    assert_eq!(loss.callee.name, "mse");
    let ExprKind::MemberCall(forward_call) = &loss.args[0].value.kind else {
        panic!("first mse argument is not a member call")
    };
    assert_eq!(forward_call.receiver.name, "model");
    assert_eq!(forward_call.method.name, "forward");
    let Optimizer::Sgd { lr, .. } = train.optimizer.value else {
        panic!("expected sgd");
    };
    assert_eq!(lr, 0.1);
    assert_eq!(train.steps.value, 500);
    assert_eq!(train.seed.value, 42);
    assert_eq!(train.device.value, Device::Cpu);
}

#[test]
fn linear_regression_nodes_carry_source_positions() {
    let ast = linear_regression();
    let pos = |s: &modelspec_core::diag::SourceSpan| {
        assert_eq!(s.file, LINEAR_REGRESSION);
        (s.line, s.column)
    };
    assert_eq!(pos(&ast.model.span), (2, 1));
    assert_eq!(pos(&ast.model.name.span), (2, 7));
    let ModelItem::Param(w) = &ast.model.items[0] else {
        panic!()
    };
    assert_eq!(pos(&w.span), (3, 5));
    assert_eq!(pos(&w.name.span), (3, 11));
    assert_eq!(pos(&w.ty.span), (3, 14));
    assert_eq!(pos(&w.ty.dtype.span), (3, 21));
    assert_eq!(pos(&w.ty.dims[0].span), (3, 27));
    assert_eq!(pos(&w.ty.dims[1].span), (3, 30));
    assert_eq!(pos(&w.init.span), (3, 36));
    let ModelItem::Fn(forward) = &ast.model.items[1] else {
        panic!()
    };
    assert_eq!(pos(&forward.span), (5, 5));
    assert_eq!(pos(&forward.params[0].span), (5, 16));
    assert_eq!(pos(&forward.output.span), (6, 12));
    assert_eq!(pos(&forward.result.span), (7, 9));
    assert_eq!(pos(&forward.result.value.span), (7, 16));
    assert_eq!(pos(&ast.train.span), (11, 1));
    assert_eq!(pos(&ast.train.model.span), (12, 5));
    assert_eq!(pos(&ast.train.data[1].span), (14, 5));
    assert_eq!(pos(&ast.train.loss.span), (15, 5));
    let ExprKind::Call(loss) = &ast.train.loss.value.kind else {
        panic!()
    };
    assert_eq!(pos(&loss.args[0].span), (15, 16));
    assert_eq!(pos(&loss.args[1].value.span), (15, 34));
    let Optimizer::Sgd { lr_span, .. } = &ast.train.optimizer.value else {
        panic!("expected sgd");
    };
    assert_eq!(pos(lr_span), (16, 26));
    assert_eq!(pos(&ast.train.device.span), (19, 5));
}

// ---------------------------------------------------------------------------------------------
// Grammar rules, accepted forms.
// ---------------------------------------------------------------------------------------------

#[test]
fn base_program_parses() {
    parse_ok(BASE);
}

#[test]
fn line_comments_are_skipped_anywhere() {
    let source = format!("// head\n{}", BASE.replace("{\n", "{ // trailing\n"));
    let mut with_comments = parse_ok(&source);
    let mut plain = parse_ok(BASE);
    with_comments.erase_spans();
    plain.erase_spans();
    assert_eq!(with_comments, plain);
}

#[test]
fn rank_zero_type_has_no_dimensions_and_prints_as_empty_brackets() {
    let ast = parse_ok(&edit("Tensor<f32, [2, 1]>", "Tensor<f32, []>"));
    let ModelItem::Param(w) = &ast.model.items[0] else {
        panic!()
    };
    assert!(w.ty.dims.is_empty());
    assert!(print(&ast).contains("param w: Tensor<f32, []> = zeros();"));
}

#[test]
fn rank_three_type_keeps_every_dimension() {
    let ast = parse_ok(&edit("Tensor<f32, [2, 1]>", "Tensor<f32, [2, 3, 5]>"));
    let ModelItem::Param(w) = &ast.model.items[0] else {
        panic!()
    };
    assert_eq!(
        w.ty.dims.iter().map(|d| d.size).collect::<Vec<_>>(),
        [2, 3, 5]
    );
}

#[test]
fn let_bindings_precede_return_in_order() {
    let ast = parse_ok(&edit(
        "        return matmul(x, w);",
        "        let h = matmul(x, w);\n        let r = relu(h);\n        return r;",
    ));
    let ModelItem::Fn(forward) = &ast.model.items[1] else {
        panic!()
    };
    let names: Vec<_> = forward.lets.iter().map(|l| l.name.name.as_str()).collect();
    assert_eq!(names, ["h", "r"]);
    assert_eq!(forward.result.value.kind, ExprKind::Var("r".to_owned()));
    assert!(print(&ast).contains("        let h = matmul(x, w);\n        let r = relu(h);\n"));
}

#[test]
fn calls_take_positional_and_named_arguments_and_literals() {
    let ast = parse_ok(&edit(
        "return matmul(x, w);",
        "return mean(mul(x, 2), axis = 1, scale = 0.5);",
    ));
    let ModelItem::Fn(forward) = &ast.model.items[1] else {
        panic!()
    };
    let ExprKind::Call(mean) = &forward.result.value.kind else {
        panic!()
    };
    let names: Vec<_> = mean
        .args
        .iter()
        .map(|a| a.name.as_ref().map(|n| n.name.as_str()))
        .collect();
    assert_eq!(names, [None, Some("axis"), Some("scale")]);
    assert_eq!(mean.args[1].value.kind, ExprKind::Int(1));
    assert_eq!(mean.args[2].value.kind, ExprKind::Float(0.5));
    let ExprKind::Call(mul) = &mean.args[0].value.kind else {
        panic!()
    };
    assert_eq!(mul.args[1].value.kind, ExprKind::Int(2));
    assert!(print(&ast).contains("return mean(mul(x, 2), axis = 1, scale = 0.5);"));
}

#[test]
fn member_call_works_on_any_receiver() {
    let ast = parse_ok(&edit("return matmul(x, w);", "return fc.forward(x);"));
    let ModelItem::Fn(forward) = &ast.model.items[1] else {
        panic!()
    };
    let ExprKind::MemberCall(call) = &forward.result.value.kind else {
        panic!()
    };
    assert_eq!(
        (call.receiver.name.as_str(), call.method.name.as_str()),
        ("fc", "forward")
    );
    assert!(print(&ast).contains("return fc.forward(x);"));
}

#[test]
fn functions_take_zero_or_several_parameters() {
    let zero = parse_ok(&edit("fn forward(x: Tensor<f32, [3, 2]>)", "fn forward()"));
    let ModelItem::Fn(forward) = &zero.model.items[1] else {
        panic!()
    };
    assert!(forward.params.is_empty());
    assert!(print(&zero).contains("fn forward() -> "));

    let two = parse_ok(&edit(
        "fn forward(x: Tensor<f32, [3, 2]>)",
        "fn forward(x: Tensor<f32, [3, 2]>, b: Tensor<f32, []>)",
    ));
    let ModelItem::Fn(forward) = &two.model.items[1] else {
        panic!()
    };
    let names: Vec<_> = forward
        .params
        .iter()
        .map(|p| p.name.name.as_str())
        .collect();
    assert_eq!(names, ["x", "b"]);
    assert!(print(&two).contains("fn forward(x: Tensor<f32, [3, 2]>, b: Tensor<f32, []>) -> "));
}

#[test]
fn model_items_keep_source_order_and_an_empty_model_parses() {
    let ast = parse_ok(&edit(
        "    param w: Tensor<f32, [2, 1]> = zeros();\n",
        "    fn f() -> Tensor<f32, []> {\n        return z();\n    }\n    param w: Tensor<f32, [2, 1]> = zeros();\n",
    ));
    let kinds: Vec<_> = ast
        .model
        .items
        .iter()
        .map(|i| match i {
            ModelItem::Param(_) => "param",
            ModelItem::Fn(_) => "fn",
        })
        .collect();
    assert_eq!(kinds, ["fn", "param", "fn"]);

    let empty = format!("model E {{\n}}\n{}", &BASE[BASE.find("train").unwrap()..]);
    assert!(parse_ok(&empty).model.items.is_empty());
}

#[test]
fn vulkan_device_takes_an_index() {
    let ast = parse_ok(&edit("device = cpu;", "device = vulkan(3);"));
    assert!(matches!(
        ast.train.device.value,
        Device::Vulkan { index: 3, .. }
    ));
    assert!(print(&ast).contains("    device = vulkan(3);\n"));
}

#[test]
fn float_literals_accept_exponents_and_print_back_exactly() {
    for (text, value) in [
        ("1e-3", 1e-3),
        ("2.5e2", 250.0),
        ("0.001", 0.001),
        ("1e-7", 1e-7),
    ] {
        let ast = parse_ok(&edit("lr = 0.1", &format!("lr = {text}")));
        let Optimizer::Sgd { lr, .. } = ast.train.optimizer.value else {
            panic!("expected sgd");
        };
        assert_eq!(lr, value, "{text}");
        let mut again = parse_ok(&print(&ast));
        let mut first = ast;
        again.erase_spans();
        first.erase_spans();
        assert_eq!(again, first, "{text}");
    }
}

#[test]
fn train_before_model_parses_and_prints_model_first() {
    let split = BASE.find("train").unwrap();
    let swapped = format!("{}{}", &BASE[split..], &BASE[..split]);
    let ast = parse_ok(&swapped);
    assert_eq!(print(&ast), print(&parse_ok(BASE)));
}

#[test]
fn train_settings_in_any_order_print_in_canonical_order() {
    let reordered = edit(
        "    steps = 10;\n    seed = 1;\n",
        "    seed = 1;\n    steps = 10;\n",
    );
    let reordered = reordered.replacen("    model = M;\n", "", 1).replacen(
        "    device = cpu;\n",
        "    device = cpu;\n    model = M;\n",
        1,
    );
    assert_eq!(print(&parse_ok(&reordered)), print(&parse_ok(BASE)));
}

#[test]
fn inputs_and_targets_keep_source_order() {
    let ast = parse_ok(&edit(
        "    target y: Tensor<f32, [3, 1]>;\n",
        "    target y: Tensor<f32, [3, 1]>;\n    input z: Tensor<f32, []>;\n",
    ));
    let data: Vec<_> = ast
        .train
        .data
        .iter()
        .map(|d| (d.role, d.name.name.as_str()))
        .collect();
    assert_eq!(
        data,
        [
            (DataRole::Input, "x"),
            (DataRole::Target, "y"),
            (DataRole::Input, "z")
        ]
    );
    assert!(
        print(&ast).contains("    target y: Tensor<f32, [3, 1]>;\n    input z: Tensor<f32, []>;\n")
    );
}

#[test]
fn crlf_line_endings_parse_and_count_lines() {
    let crlf = BASE.replace('\n', "\r\n");
    parse_ok(&crlf);
    let broken = edit("param w: Tensor", "param w Tensor").replace('\n', "\r\n");
    assert_single(&broken, "E-SYNTAX-002", 2, 13, "expected `:`");
}

// ---------------------------------------------------------------------------------------------
// Grammar rules, rejected forms, beyond the one fixture per code.
// ---------------------------------------------------------------------------------------------

#[test]
fn unrecognised_character_is_reported_whole() {
    assert_single(&edit("param w:", "param wé:"), "E-SYNTAX-001", 2, 12, "`é`");
}

#[test]
fn columns_count_characters_not_bytes() {
    let split = BASE.find("train").unwrap();
    let source = format!("{}train T {{ // üé", &BASE[..split]);
    assert_single(&source, "E-SYNTAX-003", 7, 16, "unexpected end of input");
}

#[test]
fn tab_counts_as_one_column() {
    assert_single(
        &edit("    param w: Tensor", "\tparam w Tensor"),
        "E-SYNTAX-002",
        2,
        10,
        "expected `:`",
    );
}

#[test]
fn keyword_is_not_an_identifier() {
    assert_single(
        &edit("param w:", "param model:"),
        "E-SYNTAX-002",
        2,
        11,
        "`model`",
    );
}

#[test]
fn type_must_be_tensor() {
    assert_single(
        &edit(
            "param w: Tensor<f32, [2, 1]>",
            "param w: Matrix<f32, [2, 1]>",
        ),
        "E-SYNTAX-002",
        2,
        14,
        "expected `Tensor`",
    );
}

#[test]
fn dimension_list_rejects_a_trailing_comma() {
    assert_single(&edit("[2, 1]>", "[2, 1,]>"), "E-SYNTAX-002", 2, 32, "`]`");
}

#[test]
fn dimension_out_of_range_is_reported() {
    assert_single(
        &edit("[2, 1]>", "[2, 18446744073709551616]>"),
        "E-SYNTAX-015",
        2,
        30,
        "out of range",
    );
}

#[test]
fn param_initializer_must_be_a_call() {
    assert_single(
        &edit("= zeros();", "= zeros;"),
        "E-SYNTAX-002",
        2,
        41,
        "expected `(`",
    );
}

#[test]
fn function_requires_an_output_type() {
    assert_single(
        &edit(") -> Tensor<f32, [3, 1]> {", ") {"),
        "E-SYNTAX-002",
        3,
        40,
        "expected `->`",
    );
}

#[test]
fn function_body_must_end_with_return() {
    assert_single(
        &edit(
            "        return matmul(x, w);\n",
            "        let h = matmul(x, w);\n",
        ),
        "E-SYNTAX-002",
        5,
        5,
        "expected `let` or `return`",
    );
}

#[test]
fn nothing_may_follow_return() {
    assert_single(
        &edit(
            "        return matmul(x, w);\n",
            "        return matmul(x, w);\n        let h = x;\n",
        ),
        "E-SYNTAX-002",
        5,
        9,
        "expected `}`",
    );
}

#[test]
fn member_access_must_be_a_call() {
    assert_single(
        &edit("mse(model.forward(x), y)", "mse(model.forward, y)"),
        "E-SYNTAX-002",
        11,
        29,
        "expected `(`",
    );
}

#[test]
fn every_operator_is_reported_by_name() {
    for op in ["+", "-", "*", "/", "%", "^", "<", ">"] {
        assert_single(
            &edit("return matmul(x, w);", &format!("return x {op} w;")),
            "E-SYNTAX-009",
            4,
            18,
            &format!("operator `{op}`"),
        );
    }
}

#[test]
fn prefix_minus_is_an_operator() {
    assert_single(
        &edit("return matmul(x, w);", "return matmul(-x, w);"),
        "E-SYNTAX-009",
        4,
        23,
        "operator `-`",
    );
}

#[test]
fn generic_parameters_on_a_function_are_reported() {
    assert_single(
        &edit("fn forward(", "fn forward<B: dim>("),
        "E-SYNTAX-006",
        3,
        15,
        "generic parameter list",
    );
}

#[test]
fn sgd_learning_rate_must_be_a_float_literal() {
    assert_single(
        &edit("lr = 0.1", "lr = 1"),
        "E-SYNTAX-002",
        12,
        26,
        "float literal",
    );
}

#[test]
fn sgd_takes_only_a_named_lr() {
    assert_single(
        &edit("sgd(lr = 0.1)", "sgd(0.1)"),
        "E-SYNTAX-002",
        12,
        21,
        "`lr`",
    );
    assert_single(
        &edit("sgd(lr = 0.1)", "sgd(lr = 0.1, momentum = 0.9)"),
        "E-SYNTAX-002",
        12,
        29,
        "expected `)`",
    );
}

#[test]
fn float_literal_out_of_range_is_reported() {
    assert_single(
        &edit("lr = 0.1", "lr = 1e999"),
        "E-SYNTAX-015",
        12,
        26,
        "out of range",
    );
}

#[test]
fn vulkan_index_out_of_range_is_reported() {
    assert_single(
        &edit("device = cpu;", "device = vulkan(4294967296);"),
        "E-SYNTAX-015",
        15,
        21,
        "out of range",
    );
}

#[test]
fn step_count_must_be_an_integer() {
    assert_single(
        &edit("steps = 10;", "steps = 1.5;"),
        "E-SYNTAX-002",
        13,
        13,
        "integer literal",
    );
}

#[test]
fn every_train_setting_is_required() {
    for (line, name) in [
        ("    model = M;\n", "model"),
        ("    loss = mse(model.forward(x), y);\n", "loss"),
        ("    optimizer = sgd(lr = 0.1);\n", "optimizer"),
        ("    steps = 10;\n", "steps"),
        ("    seed = 1;\n", "seed"),
        ("    device = cpu;\n", "device"),
    ] {
        assert_single(&edit(line, ""), "E-SYNTAX-018", 15, 1, &format!("`{name}`"));
    }
}

#[test]
fn every_train_setting_appears_once() {
    for (line, name, at) in [
        ("    model = M;\n", "model", 9),
        ("    loss = mse(model.forward(x), y);\n", "loss", 12),
        ("    optimizer = sgd(lr = 0.1);\n", "optimizer", 13),
        ("    steps = 10;\n", "steps", 14),
        ("    seed = 1;\n", "seed", 15),
        ("    device = cpu;\n", "device", 16),
    ] {
        assert_single(
            &edit(line, &format!("{line}{line}")),
            "E-SYNTAX-017",
            at,
            5,
            &format!("`{name}`"),
        );
    }
}

#[test]
fn second_train_is_reported() {
    let split = BASE.find("train").unwrap();
    let source = format!("{BASE}{}", &BASE[split..]);
    assert_single(&source, "E-SYNTAX-019", 17, 1, "second `train`");
}

#[test]
fn missing_model_is_reported_at_end_of_input() {
    let split = BASE.find("train").unwrap();
    assert_single(&BASE[split..], "E-SYNTAX-020", 11, 1, "`model`");
}

#[test]
fn empty_source_reports_both_missing_items() {
    let diagnostics = parse_err("");
    assert_eq!(
        summary(&diagnostics),
        [("E-SYNTAX-020", 1, 1), ("E-SYNTAX-020", 1, 1)]
    );
    assert!(diagnostics[0].message.contains("`model`"));
    assert!(diagnostics[1].message.contains("`train`"));
}

#[test]
fn top_level_garbage_is_an_unexpected_token() {
    assert_single(
        &format!("foo\n{BASE}"),
        "E-SYNTAX-002",
        1,
        1,
        "expected `model` or `train`",
    );
}

#[test]
fn unsupported_constructs_are_named_in_every_position() {
    let positions = [
        ("top level", "model M {", 1, 1),
        ("model body", "    param w:", 2, 5),
        ("function body", "        return", 4, 9),
        ("train body", "    model = M;", 8, 5),
    ];
    for (construct, code) in [
        ("module", "E-SYNTAX-004"),
        ("layer", "E-SYNTAX-005"),
        ("const", "E-SYNTAX-010"),
        ("repeat", "E-SYNTAX-011"),
    ] {
        for (place, anchor, line, column) in positions {
            let indent: String = anchor.chars().take_while(|c| *c == ' ').collect();
            let insert = format!("{indent}{construct} k {{ }}\n{anchor}");
            let source = edit(anchor, &insert);
            let diagnostics = parse_err(&source);
            assert_eq!(
                summary(&diagnostics),
                vec![(code, line, column)],
                "{construct} at {place}:\n{}",
                render(&diagnostics)
            );
            assert!(diagnostics[0].message.contains(&format!("`{construct}`")));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Reporting: first error per item, rendering through diag.rs.
// ---------------------------------------------------------------------------------------------

#[test]
fn only_the_first_error_of_an_item_is_reported() {
    let source = edit("param w:", "param w").replacen("return matmul(x, w);", "return x * w;", 1);
    assert_eq!(summary(&parse_err(&source)), [("E-SYNTAX-002", 2, 13)]);
}

#[test]
fn each_item_reports_its_own_first_error() {
    let source = edit("param w:", "param w").replacen("device = cpu;", "device = gpu;", 1);
    assert_eq!(
        summary(&parse_err(&source)),
        [("E-SYNTAX-002", 2, 13), ("E-SYNTAX-014", 15, 14)]
    );
}

// ---------------------------------------------------------------------------------------------
// Recovery: an item with an error never swallows the item after it.
// ---------------------------------------------------------------------------------------------

#[test]
fn unclosed_model_is_one_error_and_the_following_train_parses() {
    let split = BASE.find("train").unwrap();
    let model_close = BASE[..split].rfind('}').unwrap();
    let source = format!("{}{}", &BASE[..model_close], &BASE[split..]);
    assert_eq!(summary(&parse_err(&source)), [("E-SYNTAX-002", 6, 1)]);
}

#[test]
fn unclosed_function_is_one_error_and_the_following_train_parses() {
    let source = edit(
        "        return matmul(x, w);\n    }\n",
        "        return matmul(x, w);\n",
    );
    assert_eq!(summary(&parse_err(&source)), [("E-SYNTAX-002", 6, 1)]);
}

/// Every head a `model` item can start with stops the recovery of an unclosed `train` before
/// it, so the model is parsed on its own and reports only its own first error.
#[test]
fn unclosed_train_stops_at_every_model_item_head() {
    let split = BASE.find("train").unwrap();
    let train = BASE[split..].replacen("    device = cpu;\n}\n", "    device = cpu;\n", 1);
    for (head, expected) in [
        ("model M {", vec![("E-SYNTAX-017", 10, 1)]),
        (
            "model M<B: dim> {",
            vec![("E-SYNTAX-017", 10, 1), ("E-SYNTAX-006", 10, 8)],
        ),
        (
            "model {",
            vec![("E-SYNTAX-017", 10, 1), ("E-SYNTAX-002", 10, 7)],
        ),
    ] {
        let model = BASE[..split].replacen("model M {", head, 1);
        let source = format!("{train}{model}");
        assert_eq!(summary(&parse_err(&source)), expected, "{head}:\n{source}");
    }
}

/// `model` inside a train item (`model = M;`, `model.forward(x)`, a malformed `model M;`) does
/// not start an item, so recovery from an earlier error in the train skips it.
#[test]
fn recovery_skips_model_inside_a_train_item() {
    assert_single(
        &edit("    model = M;\n", "    steps = 1.5;\n    model = M;\n"),
        "E-SYNTAX-002",
        8,
        13,
        "integer literal",
    );
    assert_single(
        &edit("input x: Tensor", "input x Tensor"),
        "E-SYNTAX-002",
        9,
        13,
        "expected `:`",
    );
    assert_single(
        &edit("    model = M;", "    model M;"),
        "E-SYNTAX-002",
        8,
        11,
        "expected `=`",
    );
}

// ---------------------------------------------------------------------------------------------
// Nesting limit: argument lists nest at most 64 deep, on every recursive path.
// ---------------------------------------------------------------------------------------------

/// The nesting limit the parser documents; its diagnostic names it.
const LIMIT: usize = 64;

/// `depth` argument lists, each opened by `open` and closed by `)`, around `inner`.
fn nested(open: &str, depth: usize, inner: &str) -> String {
    format!("{}{inner}{}", open.repeat(depth), ")".repeat(depth))
}

/// At the limit the source parses; one level deeper it is one E-SYNTAX-021 at the `(` that
/// opens the argument list past the limit, and the train item after it still parses.
#[track_caller]
fn assert_nesting_limit(make: impl Fn(usize) -> String, line: u32, column: u32) {
    parse_ok(&make(LIMIT));
    assert_single(
        &make(LIMIT + 1),
        "E-SYNTAX-021",
        line,
        column,
        &format!("more than {LIMIT}"),
    );
}

#[test]
fn nesting_limit_holds_for_calls() {
    assert_nesting_limit(
        |depth| edit("matmul(x, w)", &nested("f(", depth, "x")),
        4,
        15 + 2 * (LIMIT as u32 + 1),
    );
}

#[test]
fn nesting_limit_holds_for_member_calls() {
    assert_nesting_limit(
        |depth| edit("matmul(x, w)", &nested("a.f(", depth, "x")),
        4,
        15 + 4 * (LIMIT as u32 + 1),
    );
}

#[test]
fn nesting_limit_holds_for_named_arguments() {
    assert_nesting_limit(
        |depth| edit("matmul(x, w)", &nested("f(a = ", depth, "x")),
        4,
        11 + 6 * (LIMIT as u32 + 1),
    );
}

#[test]
fn nesting_limit_holds_for_param_initializers() {
    assert_nesting_limit(
        |depth| edit("zeros()", &nested("f(", depth, "")),
        2,
        35 + 2 * (LIMIT as u32 + 1),
    );
}

#[test]
fn nesting_limit_holds_for_train_loss() {
    assert_nesting_limit(
        |depth| edit("mse(model.forward(x), y)", &nested("f(", depth, "y")),
        11,
        11 + 2 * (LIMIT as u32 + 1),
    );
}

/// The limit is the documented constant, and every path parses at the limit on a thread with
/// half the 2 MiB stack std gives spawned threads, so the limit is reached before the stack is.
#[test]
fn nesting_limit_fits_in_half_a_default_thread_stack() {
    assert_eq!(modelspec_core::syntax::MAX_NESTING, LIMIT);
    let sources = [
        edit("matmul(x, w)", &nested("f(", LIMIT, "x")),
        edit("matmul(x, w)", &nested("a.f(", LIMIT, "x")),
        edit("matmul(x, w)", &nested("f(a = ", LIMIT, "x")),
        edit("zeros()", &nested("f(", LIMIT, "")),
        edit("mse(model.forward(x), y)", &nested("f(", LIMIT, "y")),
    ];
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            for source in &sources {
                parse_ok(source);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Far past the limit: the parser stops at the limit instead of overflowing the stack.
#[test]
fn nesting_far_past_the_limit_is_one_diagnostic() {
    assert_single(
        &edit("matmul(x, w)", &nested("f(", 100_000, "x")),
        "E-SYNTAX-021",
        4,
        15 + 2 * (LIMIT as u32 + 1),
        &format!("more than {LIMIT}"),
    );
}

/// Parentheses, prefix operators and infix operators are not M0 expressions: the first one is
/// the item's error, and no amount of them recurses.
#[test]
fn deep_parentheses_stop_at_the_first() {
    assert_single(
        &edit("matmul(x, w)", &nested("(", 100_000, "x")),
        "E-SYNTAX-002",
        4,
        16,
        "expected an expression, found `(`",
    );
}

#[test]
fn deep_prefix_operators_stop_at_the_first() {
    assert_single(
        &edit("matmul(x, w)", &format!("{}x", "-".repeat(100_000))),
        "E-SYNTAX-009",
        4,
        16,
        "operator `-`",
    );
}

#[test]
fn deep_infix_operators_stop_at_the_first() {
    assert_single(
        &edit("matmul(x, w)", &format!("x{}", " + x".repeat(100_000))),
        "E-SYNTAX-009",
        4,
        18,
        "operator `+`",
    );
}

#[test]
fn tiny_mlp_reports_each_construct_outside_m0() {
    let diagnostics = parse("examples/tiny_mlp.mspec", &read("examples/tiny_mlp.mspec"))
        .expect_err("tiny_mlp uses iteration-3 constructs");
    assert_eq!(
        summary(&diagnostics),
        [
            ("E-SYNTAX-004", 2, 1),
            ("E-SYNTAX-006", 11, 14),
            ("E-SYNTAX-007", 23, 20)
        ]
    );
}

#[test]
fn syntax_diagnostic_renders_in_the_design_layout() {
    let diagnostics = parse(
        "model.mspec",
        &read(&format!("{FIXTURE_DIR}/E-SYNTAX-005-layer.mspec")),
    )
    .expect_err("layer fixture");
    let rendered = diagnostics[0].to_string();
    let mut lines = rendered.lines();
    assert!(
        lines.next().unwrap().starts_with("E-SYNTAX-005: "),
        "{rendered}"
    );
    assert_eq!(lines.next(), Some("  model.mspec:3:5"), "{rendered}");
}
