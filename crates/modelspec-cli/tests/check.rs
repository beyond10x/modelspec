//! `modelspec check`: name resolution, type and shape checking (DESIGN.md § 3, § 8, § 8.1).
//!
//! Every fixture under `tests/language/check/` (repository root) is one test case. A fixture's
//! leading lines read `// expect <code> <line>:<column> <message fragment>`, one per diagnostic in
//! the order `check` prints them, or `// expect ok` for a file that checks clean. Every `E-NAME`
//! and `E-TYPE` code has at least one fixture, and every train-block rule of the story has its
//! own, including the rules the parser already enforces (`seed` fits `u64`, `sgd`, finite `lr`).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::{resolve, types};

const FIXTURE_DIR: &str = "tests/language/check";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs `modelspec check <relative>` from the repository root.
fn check(relative: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["check", relative])
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr is UTF-8")
}

#[derive(Debug)]
struct Expected {
    code: String,
    line: u32,
    column: u32,
    fragment: String,
}

/// The `// expect` lines of a fixture; empty for `// expect ok`.
fn expectations(name: &str, source: &str) -> Vec<Expected> {
    let headers: Vec<&str> = source
        .lines()
        .map_while(|l| l.strip_prefix("// expect "))
        .collect();
    assert!(
        !headers.is_empty(),
        "{name}: first line is not `// expect ...`"
    );
    if headers == ["ok"] {
        return Vec::new();
    }
    headers
        .iter()
        .map(|header| {
            let mut words = header.splitn(3, ' ');
            let code = words.next().unwrap().to_owned();
            let position = words.next().expect("position");
            let fragment = words.next().expect("message fragment").to_owned();
            let (line, column) = position.split_once(':').expect("line:column");
            Expected {
                code,
                line: line.parse().unwrap(),
                column: column.parse().unwrap(),
                fragment,
            }
        })
        .collect()
}

/// Splits rendered diagnostics into blocks: a block starts at a line without indentation.
fn blocks(text: &str) -> Vec<Vec<&str>> {
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    for line in text.lines() {
        if line.starts_with("  ") {
            blocks
                .last_mut()
                .unwrap_or_else(|| panic!("detail line before any diagnostic: {line:?}"))
                .push(line);
        } else {
            blocks.push(vec![line]);
        }
    }
    blocks
}

fn check_fixture(name: &str) {
    let relative = format!("{FIXTURE_DIR}/{name}");
    let source = std::fs::read_to_string(repo_root().join(&relative)).expect("fixture");
    let expected = expectations(name, &source);
    let output = check(&relative);
    let text = stderr(&output);
    assert!(
        output.stdout.is_empty(),
        "{name}: check wrote to stdout: {:?}",
        String::from_utf8_lossy(&output.stdout)
    );
    if expected.is_empty() {
        assert_eq!(output.status.code(), Some(0), "{name}:\n{text}");
        assert_eq!(text, "", "{name}: a clean check prints nothing");
        return;
    }
    assert_eq!(output.status.code(), Some(1), "{name}:\n{text}");
    let blocks = blocks(&text);
    assert_eq!(
        blocks.len(),
        expected.len(),
        "{name}: expected {} diagnostic(s), got:\n{text}",
        expected.len()
    );
    for (block, want) in blocks.iter().zip(&expected) {
        let head = block[0];
        let prefix = format!("{}: ", want.code);
        assert!(
            head.starts_with(&prefix),
            "{name}: expected {}, got {head:?}\n{text}",
            want.code
        );
        assert!(
            head.contains(&want.fragment),
            "{name}: {head:?} lacks {:?}",
            want.fragment
        );
        assert_eq!(
            block.get(1).copied(),
            Some(format!("  {relative}:{}:{}", want.line, want.column).as_str()),
            "{name}: location of {}\n{text}",
            want.code
        );
    }
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
    e_name_001_unknown_name => "E-NAME-001-unknown-name.mspec",
    e_name_001_parameter_in_train => "E-NAME-001-parameter-in-train.mspec",
    e_name_002_unknown_function => "E-NAME-002-unknown-function.mspec",
    e_name_003_duplicate_model_item => "E-NAME-003-duplicate-model-item.mspec",
    e_name_003_function_named_like_parameter => "E-NAME-003-function-named-like-parameter.mspec",
    e_name_004_duplicate_argument => "E-NAME-004-duplicate-argument.mspec",
    e_name_004_let_shadows_argument => "E-NAME-004-let-shadows-argument.mspec",
    e_name_004_argument_shadows_parameter => "E-NAME-004-argument-shadows-parameter.mspec",
    e_name_005_duplicate_data => "E-NAME-005-duplicate-data.mspec",
    e_name_006_unknown_model => "E-NAME-006-unknown-model.mspec",
    e_name_007_unknown_receiver => "E-NAME-007-unknown-receiver.mspec",
    e_name_007_model_call_inside_model => "E-NAME-007-model-call-inside-model.mspec",
    e_name_008_unknown_method => "E-NAME-008-unknown-method.mspec",
    e_type_001_operator_argument_count => "E-TYPE-001-operator-argument-count.mspec",
    e_type_001_forward_argument_count => "E-TYPE-001-forward-argument-count.mspec",
    e_type_002_named_argument => "E-TYPE-002-named-argument.mspec",
    e_type_002_named_forward_argument => "E-TYPE-002-named-forward-argument.mspec",
    e_type_003_shape_mismatch => "E-TYPE-003-shape-mismatch.mspec",
    e_type_003_mse_shape_mismatch => "E-TYPE-003-mse-shape-mismatch.mspec",
    e_type_004_rank => "E-TYPE-004-rank.mspec",
    e_type_004_transpose_rank => "E-TYPE-004-transpose-rank.mspec",
    e_type_005_integer_literal => "E-TYPE-005-integer-literal.mspec",
    e_type_006_float_literal_range => "E-TYPE-006-float-literal-range.mspec",
    e_type_007_return_type => "E-TYPE-007-return-type.mspec",
    e_type_008_argument_type => "E-TYPE-008-argument-type.mspec",
    e_type_009_loss_rank => "E-TYPE-009-loss-rank.mspec",
    e_type_010_learning_rate_zero => "E-TYPE-010-learning-rate-zero.mspec",
    e_type_010_learning_rate_underflow => "E-TYPE-010-learning-rate-underflow.mspec",
    e_type_010_learning_rate_overflow => "E-TYPE-010-learning-rate-overflow.mspec",
    e_type_011_steps_zero => "E-TYPE-011-steps-zero.mspec",
    e_type_012_matmul_mismatch => "matmul_mismatch.mspec",
    e_type_013_initializer => "E-TYPE-013-initializer.mspec",
    e_type_013_initializer_arguments => "E-TYPE-013-initializer-arguments.mspec",
    e_type_014_initializer_rank => "E-TYPE-014-initializer-rank.mspec",
    e_type_015_zero_dimension => "E-TYPE-015-zero-dimension.mspec",
    e_type_015_zero_dimension_parameter => "E-TYPE-015-zero-dimension-parameter.mspec",
    e_type_016_count_overflow => "E-TYPE-016-count-overflow.mspec",
    e_type_016_result_overflow => "E-TYPE-016-result-overflow.mspec",
    e_type_017_no_forward_call => "E-TYPE-017-no-forward-call.mspec",
    e_type_017_no_forward_function => "E-TYPE-017-no-forward-function.mspec",
    e_type_017_two_forward_calls => "E-TYPE-017-two-forward-calls.mspec",
    e_type_017_nested_forward_calls => "E-TYPE-017-nested-forward-calls.mspec",
    e_type_018_i32_operand => "E-TYPE-018-i32-operand.mspec",
    e_type_019_heads_not_a_literal => "E-TYPE-019-heads-not-a-literal.mspec",
    e_type_020_heads_not_dividing_width => "E-TYPE-020-heads-not-dividing-width.mspec",
    e_type_021_key_mask_reads_parameter => "E-TYPE-021-key-mask-reads-parameter.mspec",
    e_type_022_dropout_rate_outside_0_1 => "E-TYPE-022-dropout-rate-outside-0-1.mspec",
    e_type_023_adamw_beta1_one => "E-TYPE-023-adamw-beta1-one.mspec",
    e_type_024_warmup_not_below_steps => "E-TYPE-024-warmup-not-below-steps.mspec",
    rule_seed_fits_u64 => "rule-seed-fits-u64.mspec",
    rule_optimizer_is_sgd => "rule-optimizer-is-sgd.mspec",
    rule_learning_rate_finite => "rule-learning-rate-finite.mspec",
    ok_vulkan_device_not_validated => "ok-vulkan-device-not-validated.mspec",
    ok_every_operator => "ok-every-operator.mspec",
    every_diagnostic_in_source_order => "every-diagnostic-in-source-order.mspec",
}

#[test]
fn linear_regression_example_checks_clean() {
    let output = check("examples/linear_regression.mspec");
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert!(output.stdout.is_empty());
    assert_eq!(stderr(&output), "");
}

/// The acceptance example: DESIGN.md § 8.1, line for line.
#[test]
fn matmul_mismatch_prints_the_design_diagnostic() {
    let output = check("tests/language/check/matmul_mismatch.mspec");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        stderr(&output),
        "E-TYPE-012: matmul contracting dimension mismatch\n  \
         tests/language/check/matmul_mismatch.mspec:7:16\n  \
         lhs: Tensor<f32, [64, 32]>\n  \
         rhs: Tensor<f32, [64, 128]>\n  \
         expected rhs dimension[0] = 32; got 64\n"
    );
}

#[test]
fn unreadable_file_exits_1_naming_the_file() {
    let output = check("tests/language/check/no-such-file.mspec");
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("tests/language/check/no-such-file.mspec"),
        "{}",
        stderr(&output)
    );
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
fn every_name_and_type_code_has_a_fixture_named_for_it() {
    let declared: Vec<&str> = resolve::codes::ALL
        .iter()
        .chain(types::codes::ALL)
        .map(|c| c.as_str())
        .collect();
    let mut pinned: Vec<String> = Vec::new();
    for name in FIXTURES {
        let source =
            std::fs::read_to_string(repo_root().join(FIXTURE_DIR).join(name)).expect("fixture");
        let expected = expectations(name, &source);
        for want in &expected {
            if !want.code.starts_with("E-SYNTAX-") {
                assert!(
                    declared.contains(&want.code.as_str()),
                    "{name} expects {}, which no codes::ALL declares",
                    want.code
                );
            }
        }
        if let [only] = expected.as_slice() {
            let named = name.starts_with(&format!("{}-", only.code))
                || (*name == "matmul_mismatch.mspec" && only.code == "E-TYPE-012");
            if named {
                pinned.push(only.code.clone());
            }
        }
    }
    for code in resolve::codes::ALL.iter().chain(types::codes::ALL) {
        assert!(
            pinned.iter().any(|p| p == code.as_str()),
            "{code} has no single-diagnostic fixture named for it under {FIXTURE_DIR}"
        );
    }
}
