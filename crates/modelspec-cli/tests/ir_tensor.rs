//! `modelspec ir --stage tensor`: Typed HIR lowered to verified Tensor IR (DESIGN.md § 2.1, § 2.3,
//! § 3, § 3.3 passes 2–3, § 8).
//!
//! The golden `tests/golden/linear_regression.tensor.ir` is written by hand from DESIGN.md § 3 and
//! the printer's line format (`ir/print.rs`): the train block's `input` and `target` declarations
//! become `Input` nodes in source order, then each `param` one `ParamRead`, then the loss with the
//! model's `forward` inlined. The pipeline cases call `modelspec_core::compile` directly.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::compile::{self, codes};
use modelspec_core::ir::print::print;
use modelspec_core::ir::tensor::{TensorOp, TensorProgram, ValueId};
use modelspec_core::ir::verify::verify;
use modelspec_core::lower::LoweredProgram;

const EXAMPLE: &str = "examples/linear_regression.mspec";
const GOLDEN: &str = include_str!("golden/linear_regression.tensor.ir");

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs `modelspec <args>` from the repository root.
fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

fn example_source() -> String {
    std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example exists")
}

/// The linear-regression example with `model` replacing its model block.
fn with_model(model: &str) -> String {
    let source = example_source();
    let train = source
        .find("train Fit")
        .expect("the example has `train Fit`");
    format!("{model}\n\n{}", &source[train..])
}

fn compiled(source: &str) -> LoweredProgram {
    match compile::compile_tensor("test.mspec", source) {
        Ok(lowered) => lowered,
        Err(diagnostics) => panic!("compile_tensor refused the source: {diagnostics:#?}"),
    }
}

fn ops(program: &TensorProgram) -> Vec<&'static str> {
    program.nodes.iter().map(|n| n.op.name()).collect()
}

fn node(program: &TensorProgram, id: ValueId) -> &modelspec_core::ir::tensor::TensorNode {
    program
        .node(id)
        .unwrap_or_else(|| panic!("{id} is not defined"))
}

// The acceptance: exit 0, exactly the golden, identical bytes on a second run.

#[test]
fn ir_stage_tensor_prints_exactly_the_golden() {
    let output = modelspec(&["ir", EXAMPLE, "--stage", "tensor"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert_eq!(text(&output.stdout), GOLDEN);
    assert_eq!(text(&output.stderr), "");
}

#[test]
fn ir_stage_tensor_prints_identical_bytes_on_a_second_run() {
    let first = modelspec(&["ir", EXAMPLE, "--stage", "tensor"]);
    let second = modelspec(&["ir", EXAMPLE, "--stage", "tensor"]);
    assert_eq!(first.status.code(), Some(0), "{}", text(&first.stderr));
    assert_eq!(second.status.code(), Some(0), "{}", text(&second.stderr));
    assert!(!first.stdout.is_empty());
    assert_eq!(first.stdout, second.stdout);
}

#[test]
fn golden_is_a_param_read_of_linear_reg_w_feeding_an_8_by_1_matmul_feeding_mse() {
    let lowered = compiled(&example_source());
    let program = &lowered.program;
    verify(program).expect("the lowered program verifies");
    assert_eq!(print(program), GOLDEN);

    assert_eq!(program.outputs.len(), 1, "the loss is the single output");
    let mse = node(program, program.outputs[0]);
    assert_eq!(mse.op, TensorOp::Mse);
    assert_eq!(mse.ty.to_string(), "Tensor<f32, []>");
    let matmul = node(program, mse.operands[0]);
    assert_eq!(matmul.op, TensorOp::MatMul);
    assert_eq!(matmul.ty.to_string(), "Tensor<f32, [8, 1]>");
    let read = node(program, matmul.operands[1]);
    let TensorOp::ParamRead { param } = read.op else {
        panic!("matmul's rhs is {:?}, not a ParamRead", read.op);
    };
    assert_eq!(program.params[&param].path, "LinearReg.w");
    assert_eq!(program.params[&param].init.to_string(), "xavier_uniform");
    assert_eq!(
        node(program, matmul.operands[0]).op,
        TensorOp::Input { name: "x".into() }
    );
    assert_eq!(
        node(program, mse.operands[1]).op,
        TensorOp::Input { name: "y".into() }
    );
    assert_eq!(program.inputs, vec![ValueId(0), ValueId(1)]);
}

#[test]
fn prediction_is_the_value_of_the_inlined_forward_call() {
    let lowered = compiled(&example_source());
    let mse = node(&lowered.program, lowered.program.outputs[0]);
    assert_eq!(lowered.prediction, mse.operands[0]);
    assert_eq!(
        node(&lowered.program, lowered.prediction).op,
        TensorOp::MatMul
    );
}

#[test]
fn every_node_carries_the_span_of_its_source_construct() {
    let source = example_source();
    let hir = compile::compile_hir("test.mspec", &source).expect("the example checks");
    let lowered = compiled(&source);
    let program = &lowered.program;
    let span = |id: ValueId| node(program, id).source_span.clone();
    let (_, forward) = hir.model.function("forward").expect("forward");

    assert_eq!(span(ValueId(0)), Some(hir.train.data[0].span.clone()));
    assert_eq!(span(ValueId(1)), Some(hir.train.data[1].span.clone()));
    assert_eq!(span(ValueId(2)), Some(hir.model.params[0].span.clone()));
    assert_eq!(span(ValueId(3)), Some(forward.result.span.clone()));
    assert_eq!(span(ValueId(4)), Some(hir.train.loss.span.clone()));
    assert_ne!(hir.train.data[0].span, hir.train.data[1].span);
    assert_ne!(forward.result.span, hir.train.loss.span);
}

#[test]
fn every_operator_lowers_and_verifies() {
    let source =
        std::fs::read_to_string(repo_root().join("tests/language/check/ok-every-operator.mspec"))
            .expect("fixture exists");
    let lowered = compiled(&source);
    verify(&lowered.program).expect("verifies");
    let ops = ops(&lowered.program);
    for op in [
        "Input",
        "ParamRead",
        "Scalar",
        "Add",
        "Sub",
        "Mul",
        "MatMul",
        "Relu",
        "Transpose",
        "ReduceSum",
        "ReduceMean",
        "Mse",
    ] {
        assert!(ops.contains(&op), "{op} missing from {ops:?}");
    }
    assert_eq!(
        ops.iter().filter(|&&o| o == "ParamRead").count(),
        2,
        "one ParamRead per parameter, though `w` is read twice: {}",
        print(&lowered.program)
    );
    assert_eq!(lowered.program.outputs.len(), 1);
    assert_eq!(
        node(&lowered.program, lowered.program.outputs[0])
            .ty
            .to_string(),
        "Tensor<f32, []>"
    );
}

// Dead-code elimination (DESIGN.md § 3.3 pass 3).

#[test]
fn dce_removes_a_let_the_output_does_not_reach() {
    let source = with_model(
        "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        let unused = relu(x);
        return matmul(x, w);
    }
}",
    );
    let lowered = compiled(&source);
    assert!(
        !ops(&lowered.program).contains(&"Relu"),
        "{}",
        print(&lowered.program)
    );
    assert_eq!(
        ops(&lowered.program),
        ["Input", "Input", "ParamRead", "MatMul", "Mse"]
    );
    verify(&lowered.program).expect("verifies after DCE");
}

#[test]
fn dce_keeps_every_node_the_output_reaches() {
    let source = with_model(
        "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        let h = relu(x);
        let p = matmul(h, w);
        return p;
    }
}",
    );
    let lowered = compiled(&source);
    assert_eq!(
        ops(&lowered.program),
        ["Input", "Input", "ParamRead", "Relu", "MatMul", "Mse"]
    );
}

#[test]
fn dce_removes_an_unread_parameter_read_and_keeps_its_table_entry() {
    let source = with_model(
        "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();
    param v: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}",
    );
    let lowered = compiled(&source);
    let printed = print(&lowered.program);
    assert!(
        printed.contains("param @LinearReg.v: Tensor<f32, [4, 1]> = xavier_uniform\n"),
        "{printed}"
    );
    assert!(!printed.contains("ParamRead @LinearReg.v"), "{printed}");
    assert!(printed.contains("ParamRead @LinearReg.w"), "{printed}");
}

#[test]
fn dce_removes_an_input_the_loss_does_not_read_from_nodes_and_inputs() {
    let source = example_source().replace(
        "    target y: Tensor<f32, [8, 1]>;",
        "    input z: Tensor<f32, [8, 4]>;\n    target y: Tensor<f32, [8, 1]>;",
    );
    let lowered = compiled(&source);
    let program = &lowered.program;
    assert!(
        !program
            .nodes
            .iter()
            .any(|n| n.op == TensorOp::Input { name: "z".into() }),
        "{}",
        print(program)
    );
    assert_eq!(program.inputs.len(), 2);
    verify(program).expect("verifies after DCE");
}

#[test]
fn an_input_the_loss_ignores_is_declared_but_not_a_graph_input() {
    let source = example_source().replace(
        "    target y: Tensor<f32, [8, 1]>;",
        "    input z: Tensor<f32, [8, 4]>;\n    target y: Tensor<f32, [8, 1]>;",
    );
    let lowered = compiled(&source);
    let declared: Vec<(String, String)> = lowered
        .declared_inputs
        .iter()
        .map(|d| (d.name.clone(), d.ty.to_string()))
        .collect();
    assert_eq!(
        declared,
        [
            ("x".to_owned(), "Tensor<f32, [8, 4]>".to_owned()),
            ("z".to_owned(), "Tensor<f32, [8, 4]>".to_owned()),
            ("y".to_owned(), "Tensor<f32, [8, 1]>".to_owned()),
        ]
    );
    let graph: Vec<String> = lowered
        .program
        .inputs
        .iter()
        .map(|&id| match &node(&lowered.program, id).op {
            TensorOp::Input { name } => name.clone(),
            other => panic!("input {id} is {other:?}"),
        })
        .collect();
    assert_eq!(graph, ["x", "y"], "{}", print(&lowered.program));
}

#[test]
fn the_prediction_survives_dce() {
    let source = with_model(
        "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        let unused = relu(x);
        let p = matmul(x, w);
        return relu(p);
    }
}",
    );
    let lowered = compiled(&source);
    let prediction = node(&lowered.program, lowered.prediction);
    assert_eq!(prediction.op, TensorOp::Relu);
    assert_eq!(
        node(&lowered.program, prediction.operands[0]).op,
        TensorOp::MatMul
    );
    let mse = node(&lowered.program, lowered.program.outputs[0]);
    assert_eq!(mse.operands[0], lowered.prediction);
    assert_eq!(
        ops(&lowered.program)
            .iter()
            .filter(|&&o| o == "Relu")
            .count(),
        1
    );
}

// The forward call (E-TYPE-017): the checker refuses a loss without exactly one, so `check` and
// `ir` agree and lowering never sees one.

/// `check` and `ir --stage tensor` on `fixture` both exit 1 with the same single E-TYPE-017
/// diagnostic on stderr and nothing on stdout.
fn refused_by_check_and_ir_with_e_type_017(fixture: &str) {
    let check = modelspec(&["check", fixture]);
    let ir = modelspec(&["ir", fixture, "--stage", "tensor"]);
    assert_eq!(check.status.code(), Some(1), "{}", text(&check.stderr));
    assert_eq!(ir.status.code(), Some(1), "{}", text(&ir.stderr));
    assert_eq!(text(&ir.stdout), "");
    assert_eq!(text(&ir.stderr), text(&check.stderr));
    let source = std::fs::read_to_string(repo_root().join(fixture)).expect("fixture exists");
    let diagnostics = compile::compile_tensor(fixture, &source).expect_err("refused");
    assert_eq!(diagnostics.len(), 1, "{diagnostics:#?}");
    assert_eq!(diagnostics[0].code.as_str(), "E-TYPE-017");
    assert!(modelspec_core::types::codes::ALL.contains(&diagnostics[0].code));
    assert_ne!(diagnostics[0].code, codes::COMPILER_BUG);
    assert!(diagnostics[0].span.is_some());
}

#[test]
fn a_loss_without_a_forward_call_is_refused_by_check_and_ir() {
    refused_by_check_and_ir_with_e_type_017(
        "tests/language/check/E-TYPE-017-no-forward-call.mspec",
    );
}

#[test]
fn a_loss_that_calls_forward_twice_is_refused_by_check_and_ir() {
    refused_by_check_and_ir_with_e_type_017(
        "tests/language/check/E-TYPE-017-two-forward-calls.mspec",
    );
}

#[test]
fn a_loss_with_a_nested_forward_call_is_refused_by_check_and_ir() {
    refused_by_check_and_ir_with_e_type_017(
        "tests/language/check/E-TYPE-017-nested-forward-calls.mspec",
    );
}

// The command line.

#[test]
fn a_stage_other_than_tensor_is_refused_as_a_usage_error() {
    let output = modelspec(&["ir", EXAMPLE, "--stage", "training"]);
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(text(&output.stdout), "");
    let stderr = text(&output.stderr);
    assert!(stderr.contains("'training'"), "{stderr}");
    assert!(stderr.contains("[possible values: tensor]"), "{stderr}");
}

#[test]
fn ir_without_a_stage_is_a_usage_error() {
    let output = modelspec(&["ir", EXAMPLE]);
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(text(&output.stdout), "");
    assert!(
        text(&output.stderr).contains("--stage"),
        "{}",
        text(&output.stderr)
    );
}

#[test]
fn ir_reports_diagnostics_as_check_does() {
    let fixture = "tests/language/check/matmul_mismatch.mspec";
    let check = modelspec(&["check", fixture]);
    let ir = modelspec(&["ir", fixture, "--stage", "tensor"]);
    assert_eq!(check.status.code(), Some(1));
    assert_eq!(ir.status.code(), Some(1));
    assert_eq!(text(&ir.stdout), "");
    assert!(!ir.stderr.is_empty());
    assert_eq!(text(&ir.stderr), text(&check.stderr));
}

#[test]
fn ir_on_a_missing_file_exits_1_naming_it() {
    let output = modelspec(&["ir", "no-such-model.mspec", "--stage", "tensor"]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(text(&output.stdout), "");
    let stderr = text(&output.stderr);
    assert!(stderr.contains("no-such-model.mspec"), "{stderr}");
    assert!(!stderr.contains("not implemented"), "{stderr}");
}

#[test]
fn ir_with_a_closed_stdout_pipe_exits_1_without_a_panic() {
    let (reader, writer) = std::io::pipe().expect("a pipe");
    drop(reader);
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["ir", EXAMPLE, "--stage", "tensor"])
        .current_dir(repo_root())
        .stdout(writer)
        .stderr(std::process::Stdio::piped())
        .output()
        .expect("modelspec runs");
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(stderr.contains("cannot write the IR"), "{stderr}");
}
