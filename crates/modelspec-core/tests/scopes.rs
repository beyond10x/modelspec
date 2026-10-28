//! Scope provenance in the Tensor IR (DESIGN.md § 3.2, § 13.2, story:diagram-scope-provenance).
//!
//! Every node of a lowered program, its training step and the passes after them carries the scope
//! it came from: the train block, each inlined function with its instance and call site, the
//! gradient rule of one forward node, the update of one parameter. The scope table prints nothing.

use std::collections::BTreeSet;
use std::path::Path;

use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact, compile_hir, compile_training};
use modelspec_core::diag::SourceSpan;
use modelspec_core::ir::tensor::{
    ScopeSegment, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::training::Subgraph;
use modelspec_core::ir::verify::{VerifyError, verify};
use modelspec_core::lower::lower;
use modelspec_core::passes::dce;

fn repository_file(relative: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The program of the dropout-site test in `lower.rs`: dropout in `forward`, in `extra`, which
/// the loss inlines twice, and in the loss itself.
const SITES: &str = "model Reg {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        let h = matmul(dropout(x, p = 0.5), w);
        return dropout(relu(dropout(h, p = 0.25)), p = 0.0);
    }

    fn extra(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        return dropout(matmul(x, w), p = 0.5);
    }
}

train Fit {
    model = Reg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(dropout(add(model.forward(x), add(model.extra(x), model.extra(x))), p = 0.1), y);
    optimizer = sgd(lr = 0.05);
    steps = 2;
    seed = 1;
    device = cpu;
}
";

/// The span of the `n`th occurrence of `needle` in `source`, from 0, as the parser counts it.
fn span_of(file: &str, source: &str, needle: &str, n: usize) -> SourceSpan {
    let offset = source
        .match_indices(needle)
        .nth(n)
        .unwrap_or_else(|| panic!("`{needle}` occurs {} times", n + 1))
        .0;
    let before = &source[..offset];
    let line = before.matches('\n').count() as u32 + 1;
    let column = (offset - before.rfind('\n').map_or(0, |i| i + 1)) as u32 + 1;
    SourceSpan {
        file: file.to_owned(),
        line,
        column,
    }
}

/// The scope key of `value`, which must have a scope.
fn key(program: &TensorProgram, value: ValueId) -> String {
    program
        .scopes
        .key_of(value)
        .unwrap_or_else(|| panic!("{value} has no scope"))
}

/// The innermost segment of the scope of `value`.
fn innermost(program: &TensorProgram, value: ValueId) -> &ScopeSegment {
    let scope = program.scopes.scope_of(value).expect("a scope");
    &program.scopes.entry(scope).expect("an entry").segment
}

/// scope.lowering-stamps-functions: lowering enters the train block first and each inlining of a
/// function with its instance and call site, so the scope keys and the dropout site names agree.
#[test]
fn scope_lowering_stamps_functions() {
    let hir = compile_hir("sites.mspec", SITES).expect("it checks");
    let program = lower(&hir).expect("it lowers").program;
    verify(&program).expect("the lowered program verifies");

    let allowed = ["Fit", "Fit/forward", "Fit/extra", "Fit/extra@1"];
    let mut seen = BTreeSet::new();
    for node in &program.nodes {
        let key = key(&program, node.id);
        assert!(allowed.contains(&key.as_str()), "{}: {key}", node.id);
        seen.insert(key);
    }
    assert_eq!(
        seen,
        allowed.iter().map(|k| k.to_string()).collect(),
        "every scope holds a node"
    );

    let functions: Vec<&ScopeSegment> = program
        .scopes
        .entries
        .iter()
        .map(|e| &e.segment)
        .filter(|s| matches!(s, ScopeSegment::Function { name, .. } if name == "extra"))
        .collect();
    let second_call = span_of("sites.mspec", SITES, "model.extra(x)", 1);
    let first_call = span_of("sites.mspec", SITES, "model.extra(x)", 0);
    assert_eq!(
        functions,
        [
            &ScopeSegment::Function {
                name: "extra".into(),
                instance: 0,
                call_span: Some(first_call),
            },
            &ScopeSegment::Function {
                name: "extra".into(),
                instance: 1,
                call_span: Some(second_call),
            },
        ]
    );

    let mut tokens = 0;
    for node in &program.nodes {
        if let TensorOp::Input { name } = &node.op {
            let expected = match name.as_str() {
                n if n.starts_with("Reg.extra@1.dropout#") => "Fit/extra@1",
                n if n.starts_with("Reg.extra.dropout#") => "Fit/extra",
                n if n.starts_with("Reg.forward.dropout#") => "Fit/forward",
                _ => "Fit",
            };
            if name.starts_with("Reg.extra@1.dropout#") {
                tokens += 1;
            }
            assert_eq!(key(&program, node.id), expected, "{name}");
        }
    }
    assert_eq!(
        tokens, 1,
        "the second inlining of `extra` has its own token"
    );
}

/// scope.builder-nests: the builder stamps each node with the scope entered last, scopes nest as
/// deep as they are entered, and the table's parent chain leads back to the Program scope.
#[test]
fn scope_builder_nests() {
    let mut b = TensorProgramBuilder::new();
    b.enter_scope(ScopeSegment::Program("T".into()));
    b.enter_scope(ScopeSegment::Function {
        name: "a".into(),
        instance: 0,
        call_span: None,
    });
    b.enter_scope(ScopeSegment::Function {
        name: "b".into(),
        instance: 1,
        call_span: None,
    });
    let first = b.input("x", TensorType::f32(&[2, 2])).unwrap();
    b.exit_scope();
    b.exit_scope();
    let second = b.relu(first).unwrap();
    b.output(second).unwrap();
    let program = b.finish();

    assert_eq!(key(&program, first), "T/a/b@1");
    assert_eq!(key(&program, second), "T");
    let scopes = &program.scopes;
    let mut chain = Vec::new();
    let mut at = scopes
        .entry(scopes.scope_of(first).unwrap())
        .unwrap()
        .parent;
    while let Some(scope) = at {
        chain.push(scopes.key(scope));
        at = scopes.entry(scope).unwrap().parent;
    }
    assert_eq!(chain, ["T/a", "T"]);
    verify(&program).expect("the program verifies");
}

/// scope.adjoints-and-updates: autodiff stamps every node a gradient rule emits `Adjoint(%k)`
/// under the forward node's scope, constants and transposes tagged `forward` alike, and the SGD
/// step stamps each parameter's update `Update(<path>)`; the seed and the shared learning rate
/// stay in the Program scope.
#[test]
fn scope_adjoints_and_updates() {
    let path = "crates/modelspec-docs/models/two_layer_relu.mspec";
    let compiled = compile_training(path, &repository_file(path)).expect("it compiles");
    let training = &compiled.training.training;
    let program = &training.program;
    let forward: BTreeSet<ValueId> = compiled
        .forward
        .program
        .nodes
        .iter()
        .map(|n| n.id)
        .collect();
    let op = |k: u32| &program.node(ValueId(k)).expect("defined").op;

    let adjoint_of_forward = |value: ValueId| match innermost(program, value) {
        ScopeSegment::Adjoint(k) => {
            assert!(forward.contains(k), "{value}: {k} is not a forward node");
            let scope = program.scopes.scope_of(value).unwrap();
            let parent = program.scopes.entry(scope).unwrap().parent;
            assert_eq!(
                parent,
                program.scopes.scope_of(*k),
                "{value}: under {k}'s scope"
            );
        }
        other => panic!("{value} lies in {other:?}, not an Adjoint"),
    };
    let mut gradients = 0;
    for node in &program.nodes {
        if training.subgraphs[&node.id] == Subgraph::Gradient {
            adjoint_of_forward(node.id);
            gradients += 1;
        }
    }
    assert!(gradients > 0);

    assert_eq!(op(9), &TensorOp::Sub);
    assert!(matches!(op(10), TensorOp::Scalar(_)));
    for k in [9, 10, 13, 15, 18] {
        assert_eq!(training.subgraphs[&ValueId(k)], Subgraph::Forward, "%{k}");
        assert!(!forward.contains(&ValueId(k)), "%{k} is a rule's value");
        if k >= 13 {
            assert_eq!(op(k), &TensorOp::Transpose, "%{k}");
        }
        adjoint_of_forward(ValueId(k));
    }

    for (k, path) in [
        (21, "TwoLayer.w1"),
        (22, "TwoLayer.w1"),
        (23, "TwoLayer.w2"),
        (24, "TwoLayer.w2"),
    ] {
        assert_eq!(training.subgraphs[&ValueId(k)], Subgraph::Optimizer, "%{k}");
        assert_eq!(
            innermost(program, ValueId(k)),
            &ScopeSegment::Update(path.into()),
            "%{k}"
        );
        assert_eq!(
            key(program, ValueId(k)),
            format!("FitTwoLayer/update@{path}")
        );
    }
    assert_eq!(op(20), &TensorOp::Scalar(0.05));
    assert_eq!(op(8), &TensorOp::Scalar(1.0));
    assert_eq!(training.subgraphs[&ValueId(8)], Subgraph::Loss);
    for k in [8, 20] {
        assert_eq!(key(program, ValueId(k)), "FitTwoLayer", "%{k}");
    }

    for node in &program.nodes {
        assert!(program.scopes.scope_of(node.id).is_some(), "{}", node.id);
    }
    verify(program).expect("the step's program verifies");
}

/// scope.survives-dce: dead-code elimination keeps the scope of each surviving node and drops
/// the entry of each removed node.
#[test]
fn scope_survives_dce() {
    let source = SITES.replace(
        "fn extra(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {\n        return",
        "fn extra(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {\n        let unused = relu(x);\n        return",
    );
    let hir = compile_hir("sites.mspec", &source).expect("it checks");
    let before = lower(&hir).expect("it lowers").program;
    let after = dce::eliminate(before.clone());
    assert!(after.nodes.len() < before.nodes.len(), "DCE removes a node");
    for node in &after.nodes {
        assert_eq!(
            after.scopes.scope_of(node.id),
            before.scopes.scope_of(node.id),
            "{}",
            node.id
        );
        assert_eq!(key(&after, node.id), key(&before, node.id));
    }
    let kept: BTreeSet<ValueId> = after.nodes.iter().map(|n| n.id).collect();
    let stamped: BTreeSet<ValueId> = after.scopes.values.keys().copied().collect();
    assert_eq!(stamped, kept, "an entry for exactly the surviving nodes");
    verify(&after).expect("the program after DCE verifies");
}

/// scope.verifier-refuses-missing: a program with one node's scope entry removed is refused with
/// the scope rule's code, and so is a scope whose parent chain does not reach the Program scope.
#[test]
fn scope_verifier_refuses_missing() {
    let hir = compile_hir("sites.mspec", SITES).expect("it checks");
    let program = dce::eliminate(lower(&hir).expect("it lowers").program);
    verify(&program).expect("the program verifies");

    let victim = program.nodes[program.nodes.len() / 2].id;
    let mut missing = program.clone();
    missing.scopes.values.remove(&victim);
    let error = verify(&missing).expect_err("a node without a scope is refused");
    assert_eq!(error, VerifyError::NodeWithoutScope { node: victim });
    assert_eq!(error.rule(), Some("V-SCOPE-001"));

    let mut orphan = program.clone();
    let function = orphan
        .scopes
        .entries
        .iter()
        .position(|e| matches!(e.segment, ScopeSegment::Function { .. }))
        .expect("a function scope");
    orphan.scopes.entries[function].parent = None;
    let error = verify(&orphan).expect_err("a scope outside the Program scope is refused");
    assert!(
        matches!(error, VerifyError::ScopeOutsideProgram { .. }),
        "{error:?}"
    );
    assert_eq!(error.rule(), Some("V-SCOPE-002"));
}

/// scope.ir-text-unchanged: the scope table prints nothing, so the `ir_hash` of
/// `examples/bert_tiny.mspec`, blake3 of `print_step`, is the value recorded before the table.
#[test]
fn scope_ir_text_unchanged() {
    const BEFORE: &str = "578248c4cfa39356e86d8736005b6ed762c2c139c39450dccd64b08be156f9f6";
    let path = "examples/bert_tiny.mspec";
    let artifact =
        compile_artifact(path, &repository_file(path), Backend::Cpu).expect("the example compiles");
    assert_eq!(artifact.ir_hash.0, BEFORE);
}
