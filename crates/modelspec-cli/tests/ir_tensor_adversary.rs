//! Adversary cases for `modelspec ir --stage tensor` (story m0-lower-tensor-ir, pass 1).
//!
//! Sources are inline: a file under `tests/language/check/` would be listed by `check.rs`'s
//! `every_fixture_file_is_a_test_case`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use modelspec_core::autodiff::differentiate;
use modelspec_core::compile::{compile_hir, compile_tensor};
use modelspec_core::hir::HirExprKind;
use modelspec_core::ir::print::print;
use modelspec_core::ir::tensor::{Dim, ParamId, TensorOp, TensorProgram, ValueId};
use modelspec_core::ir::verify::{verify, verify_training};
use modelspec_core::lower;
use modelspec_core::passes::dce;
use modelspec_runtime::cpu::{HostTensor, evaluate, evaluate_all};

const LINEAR_REGRESSION: &str = include_str!("../../../examples/linear_regression.mspec");

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Writes `source` to a fresh file under this test binary's own target directory.
fn source_file(name: &str, source: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("ir_tensor_adversary");
    std::fs::create_dir_all(&dir).expect("scratch directory");
    let path = dir.join(name);
    std::fs::write(&path, source).expect("source written");
    path
}

fn tensor(shape: &[u64], data: &[f32]) -> HostTensor<f32> {
    HostTensor::new(shape.to_vec(), data.to_vec()).expect("a well-formed host tensor")
}

fn param_id(program: &TensorProgram, path: &str) -> ParamId {
    *program
        .params
        .iter()
        .find(|(_, p)| p.path == path)
        .unwrap_or_else(|| panic!("no parameter `{path}`"))
        .0
}

// ---------------------------------------------------------------------------------------------
// The CLI contract: 0 success, 1 diagnostics or runtime error, 2 usage (src/lib.rs, AGENTS.md).
// ---------------------------------------------------------------------------------------------

/// `ir` is the first subcommand that writes to stdout. A write that fails is a runtime error
/// (exit 1), not a panic (exit 101).
#[cfg(target_os = "linux")]
#[test]
fn ir_with_an_unwritable_stdout_exits_1_without_a_panic() {
    let full = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/full")
        .expect("/dev/full opens for writing");
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args([
            "ir",
            "examples/linear_regression.mspec",
            "--stage",
            "tensor",
        ])
        .current_dir(repo_root())
        .stdout(Stdio::from(full))
        .stderr(Stdio::piped())
        .output()
        .expect("modelspec runs");
    let stderr = text(&output.stderr);
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert_eq!(output.status.code(), Some(1), "{stderr}");
}

/// DESIGN.md § 2: "`forward` is a regular pure function, not a magical runtime hook." A model
/// whose only function is `predict` is a file `check` accepts; `ir` must not then refuse it as a
/// user error that `check` never reports.
#[test]
fn check_and_ir_agree_on_a_model_without_a_forward_function() {
    let source = LINEAR_REGRESSION
        .replace("fn forward(", "fn predict(")
        .replace("model.forward(x)", "model.predict(x)");
    let path = source_file("predict.mspec", &source);
    let path = path.to_str().expect("UTF-8 path");
    let check = modelspec(&["check", path]);
    let ir = modelspec(&["ir", path, "--stage", "tensor"]);
    assert_eq!(
        check.status.code(),
        ir.status.code(),
        "check exits {:?} with stderr {:?}; ir exits {:?} with stderr {:?}",
        check.status.code(),
        text(&check.stderr),
        ir.status.code(),
        text(&ir.stderr),
    );
}

// ---------------------------------------------------------------------------------------------
// Lowering against the CPU interpreter: the lowered program computes what the source says.
// ---------------------------------------------------------------------------------------------

const SEMANTICS: &str = "model Semantics {
    param w: Tensor<f32, [2, 2]> = xavier_uniform();
    param v: Tensor<f32, [2, 1]> = xavier_uniform();

    fn forward(a: Tensor<f32, [2, 2]>, b: Tensor<f32, [2, 2]>) -> Tensor<f32, [2, 1]> {
        let d = sub(a, b);
        let dead = relu(d);
        let s = mul(3.0, d);
        let m = matmul(s, w);
        let r = relu(sub(m, 10.0));
        return matmul(r, v);
    }
}

train Fit {
    model = Semantics;
    input p: Tensor<f32, [2, 2]>;
    input q: Tensor<f32, [2, 2]>;
    target y: Tensor<f32, [2, 1]>;
    loss = add(mse(model.forward(p, q), y), mul(0.5, mean(transpose(p))));
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}
";

/// Hand-computed: `d = p - q = [[4, -1], [-1, 3]]`, `s = 3d = [[12, -3], [-3, 9]]`,
/// `m = s w = [[12, 21], [-3, 3]]`, `r = relu(m - 10) = [[2, 11], [0, 0]]`, prediction
/// `r v = [[-9], [0]]`, `mse = (100 + 0) / 2 = 50`, `0.5 * mean(transpose(p)) = 0.5 * 15 / 4 =
/// 1.875`, loss `51.875`. Every value is exact in `f32`. Two arguments of one type, a square
/// `matmul`, a scalar on each side of an operator and a dead `let` before a live one: a lowering
/// that binds arguments or `let`s by the wrong index, swaps operands, or drops a literal's value
/// still type-checks and verifies, and only the value shows it.
#[test]
fn the_lowered_program_computes_the_source_on_the_cpu_interpreter() {
    let lowered = match compile_tensor("semantics.mspec", SEMANTICS) {
        Ok(lowered) => lowered,
        Err(d) => panic!("compile_tensor refused the source: {d:#?}"),
    };
    let program = &lowered.program;
    let inputs = BTreeMap::from([
        ("p".to_owned(), tensor(&[2, 2], &[5.0, 1.0, 2.0, 7.0])),
        ("q".to_owned(), tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0])),
        ("y".to_owned(), tensor(&[2, 1], &[1.0, 0.0])),
    ]);
    let params = BTreeMap::from([
        (
            param_id(program, "Semantics.w"),
            tensor(&[2, 2], &[1.0, 2.0, 0.0, 1.0]),
        ),
        (
            param_id(program, "Semantics.v"),
            tensor(&[2, 1], &[1.0, -1.0]),
        ),
    ]);
    let evaluation = evaluate_all(program, &inputs, &params).expect("the interpreter runs it");
    assert_eq!(evaluation.outputs.len(), 1, "{}", print(program));
    assert_eq!(evaluation.outputs[0].shape(), &[] as &[u64]);
    assert_eq!(
        evaluation.outputs[0].data(),
        &[51.875_f32],
        "{}",
        print(program)
    );
    let prediction = &evaluation.values[&lowered.prediction];
    assert_eq!(prediction.shape(), &[2, 1]);
    assert_eq!(prediction.data(), &[-9.0_f32, 0.0], "{}", print(program));
}

// ---------------------------------------------------------------------------------------------
// The order and spans lower.rs documents.
// ---------------------------------------------------------------------------------------------

/// lower.rs: "one `Input` node per `input` and `target` declaration, in source order".
#[test]
fn a_target_declared_before_its_input_is_the_first_input_node() {
    let source = LINEAR_REGRESSION.replace(
        "    input x: Tensor<f32, [8, 4]>;\n    target y: Tensor<f32, [8, 1]>;",
        "    target y: Tensor<f32, [8, 1]>;\n    input x: Tensor<f32, [8, 4]>;",
    );
    assert_ne!(source, LINEAR_REGRESSION, "the edit applied");
    let lowered = match compile_tensor("target-first.mspec", &source) {
        Ok(lowered) => lowered,
        Err(d) => panic!("compile_tensor refused the source: {d:#?}"),
    };
    assert_eq!(
        print(&lowered.program),
        "param @LinearReg.w: Tensor<f32, [4, 1]> = xavier_uniform
%0: Tensor<f32, [8, 1]> = Input \"y\"
%1: Tensor<f32, [8, 4]> = Input \"x\"
%2: Tensor<f32, [4, 1]> = ParamRead @LinearReg.w
%3: Tensor<f32, [8, 1]> = MatMul %1, %2
%4: Tensor<f32, []> = Mse %3, %0
output %4
"
    );
}

/// lower.rs: every node carries "the expression" span; a `Scalar` node is no exception.
#[test]
fn a_scalar_node_carries_the_span_of_its_literal() {
    let source = LINEAR_REGRESSION.replace(
        "loss = mse(model.forward(x), y);",
        "loss = mul(0.5, mse(model.forward(x), y));",
    );
    let hir = compile_hir("scalar.mspec", &source).expect("the edited example checks");
    let HirExprKind::Op { args, .. } = &hir.train.loss.kind else {
        panic!("the loss is an operator call");
    };
    assert!(matches!(args[0].kind, HirExprKind::Scalar(_)));
    let literal = args[0].span.clone();
    let lowered = compile_tensor("scalar.mspec", &source).expect("the edited example lowers");
    let scalar = lowered
        .program
        .nodes
        .iter()
        .find(|n| matches!(n.op, TensorOp::Scalar(_)))
        .expect("a Scalar node");
    assert_eq!(scalar.op, TensorOp::Scalar(0.5));
    assert_eq!(scalar.source_span, Some(literal));
}

// ---------------------------------------------------------------------------------------------
// Properties over edited sources: what `check` accepts lowers, verifies, runs and differentiates.
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

/// The values `program`'s outputs depend on, by a worklist from the outputs: an algorithm other
/// than dce.rs's reverse sweep.
fn reachable(program: &TensorProgram) -> BTreeSet<ValueId> {
    let by_id: BTreeMap<ValueId, &Vec<ValueId>> =
        program.nodes.iter().map(|n| (n.id, &n.operands)).collect();
    let mut seen = BTreeSet::new();
    let mut work: Vec<ValueId> = program.outputs.clone();
    while let Some(v) = work.pop() {
        if seen.insert(v)
            && let Some(operands) = by_id.get(&v)
        {
            work.extend(operands.iter().copied());
        }
    }
    seen
}

fn zeros(shape: &[Dim]) -> HostTensor<f32> {
    let shape: Vec<u64> = shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
            Dim::Symbol(s) => panic!("symbolic dimension ?{s}"),
        })
        .collect();
    let n = shape.iter().product::<u64>() as usize;
    HostTensor::new(shape, vec![0.0; n]).expect("zeros")
}

fn lowering_violation(source: &str) -> Option<String> {
    let Ok(hir) = compile_hir("fuzz.mspec", source) else {
        return None;
    };
    let lowered = match compile_tensor("fuzz.mspec", source) {
        Ok(lowered) => lowered,
        Err(d) if d.len() == 1 && d[0].code.as_str() == "E-LOWER-001" => return None,
        Err(d) => return Some(format!("checked source refused: {d:#?}")),
    };
    let program = &lowered.program;
    if let Err(e) = verify(program) {
        return Some(format!("does not verify: {e}"));
    }
    if program.outputs.len() != 1 {
        return Some(format!("{} outputs", program.outputs.len()));
    }
    let loss = program.outputs[0];
    let live = reachable(program);
    if !live.contains(&lowered.prediction) {
        return Some(format!("prediction {} is not live", lowered.prediction));
    }
    if let Some(dead) = program.nodes.iter().find(|n| !live.contains(&n.id)) {
        return Some(format!("DCE kept dead node {}", dead.id));
    }
    let undone = match lower::lower(&hir) {
        Ok(undone) => undone.program,
        Err(e) => return Some(format!("lower refused checked HIR: {e}")),
    };
    let want: Vec<ValueId> = undone
        .nodes
        .iter()
        .map(|n| n.id)
        .filter(|id| reachable(&undone).contains(id))
        .collect();
    let got: Vec<ValueId> = dce::eliminate(undone).nodes.iter().map(|n| n.id).collect();
    if got != want {
        return Some(format!("DCE kept {got:?}; the outputs reach {want:?}"));
    }
    let again = compile_tensor("fuzz.mspec", source).expect("compiled once");
    if print(&again.program) != print(program) || again.prediction != lowered.prediction {
        return Some("a second compile differs".to_owned());
    }
    let inputs: BTreeMap<String, HostTensor<f32>> = program
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            TensorOp::Input { name } => Some((name.clone(), zeros(&n.ty.shape))),
            _ => None,
        })
        .collect();
    let params: BTreeMap<ParamId, HostTensor<f32>> = program
        .params
        .iter()
        .map(|(&id, p)| (id, zeros(&p.ty.shape)))
        .collect();
    if let Err(e) = evaluate(program, &inputs, &params) {
        return Some(format!("the interpreter refuses it: {e}"));
    }
    match differentiate(program, loss) {
        Ok(training) => verify_training(&training)
            .err()
            .map(|e| format!("training program does not verify: {e}")),
        Err(e) => Some(format!("autodiff refuses it: {e}")),
    }
}

#[test]
fn every_edit_check_accepts_lowers_verifies_runs_and_differentiates() {
    let mut sources = Vec::new();
    for base in [LINEAR_REGRESSION, SEMANTICS] {
        let toks = tokens(base);
        let mut vocab: Vec<String> = toks.iter().map(|&(a, b)| base[a..b].to_owned()).collect();
        for extra in [
            "add",
            "sub",
            "mul",
            "relu",
            "transpose",
            "sum",
            "mean",
            "mse",
            "matmul",
            "0.5",
            "1.0",
            "w",
            "x",
            "y",
            "p",
            "q",
            "a",
            "b",
            "d",
            "s",
            "(",
            ")",
            ",",
            ";",
            "let",
            "return",
            "model.forward(x)",
            "model.forward(p, q)",
        ] {
            vocab.push(extra.to_owned());
        }
        vocab.sort();
        vocab.dedup();
        for &(a, b) in &toks {
            sources.push(format!("{}{}", &base[..a], &base[b..]));
            for word in &vocab {
                sources.push(format!("{} {word} {}", &base[..a], &base[b..]));
                sources.push(format!("{} {word} {}", &base[..a], &base[a..]));
            }
        }
    }
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut accepted = 0usize;
    let mut out = Vec::new();
    for source in &sources {
        if compile_hir("fuzz.mspec", source).is_ok() {
            accepted += 1;
        }
        match std::panic::catch_unwind(|| lowering_violation(source)) {
            Ok(None) => {}
            Ok(Some(problem)) => out.push(format!("{problem}\n--- source ---\n{source}")),
            Err(_) => out.push(format!("panicked\n--- source ---\n{source}")),
        }
    }
    std::panic::set_hook(default_hook);
    assert!(sources.len() > 10_000, "only {} edits ran", sources.len());
    assert!(accepted > 100, "only {accepted} edits checked clean");
    assert!(
        out.is_empty(),
        "{} of {} edits ({accepted} checked clean) break lowering; first:\n{}",
        out.len(),
        sources.len(),
        out[0]
    );
}
