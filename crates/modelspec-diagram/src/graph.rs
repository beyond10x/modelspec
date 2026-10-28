//! The `DiagramGraph` of a verified program (DESIGN.md § 13.2), built from
//! [`compile_training`](modelspec_core::compile::compile_training)'s output; nothing is parsed.
//!
//! The model view is the forward slice of the training step: every Tensor IR node the model's
//! prediction ([`LoweredProgram::prediction`](modelspec_core::lower::LoweredProgram)) reaches
//! through its operands, in definition order, dropouts and their RNG tokens included, so each node
//! is a line `modelspec ir --stage tensor` prints. The loss and the targets only it reads are not
//! in it. One edge per operand carries the producer's dtype and shape. Every node sits in the
//! Program scope, the train block; function, adjoint and update scopes arrive with the IR's scope
//! table (story:diagram-scope-provenance) and are drawn by story:diagram-zoom.

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::compile::CompiledTraining;
use modelspec_core::ir::print::{print_step, write_node};
use modelspec_core::ir::tensor::{self, Dim, TensorNode, TensorOp, TensorProgram, ValueId};
use modelspec_core::ir::training::Subgraph as IrSubgraph;
use modelspec_core::ops;

use crate::glyph::{InputRole, family_of, glyph_of};
use crate::model::{
    DType, DiagramEdge, DiagramGraph, DiagramGraphId, DiagramGraphState, DiagramNode, DiagramView,
    GlyphId, NodeKind, Scope, ScopeKey, ScopeKind, ScopeSegment,
};

/// The model view of `compiled`, the train block named `program`, compiled from `source`.
pub fn model(compiled: &CompiledTraining, program: &str, source: &str) -> DiagramGraph {
    let training = &compiled.training.training;
    let ir = &training.program;
    let by_id: BTreeMap<ValueId, &TensorNode> = ir.nodes.iter().map(|n| (n.id, n)).collect();
    let slice = reaches(&by_id, compiled.forward.prediction);
    let roles = input_roles(ir, &training.subgraphs);
    let root = ScopeKey(program.to_owned());
    let ir_hash = blake3::hash(print_step(&compiled.training).as_bytes())
        .to_hex()
        .to_string();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for node in ir.nodes.iter().filter(|n| slice.contains(&n.id)) {
        let role = roles.get(&node.id).copied().unwrap_or(InputRole::Data);
        nodes.push(operator(ir, node, role, &root));
        for (operand, value) in node.operands.iter().enumerate() {
            let producer = by_id[value];
            edges.push(DiagramEdge {
                producer: key(*value),
                consumer: key(node.id),
                operand: operand as i64,
                dtype: dtype(producer.ty.dtype),
                shape: shape(&producer.ty.shape),
            });
        }
    }
    DiagramGraph {
        graph_id: DiagramGraphId(format!("{}/{ir_hash}", DiagramView::Model.name())),
        view: DiagramView::Model,
        program: program.to_owned(),
        function: None,
        source_hash: blake3::hash(source.as_bytes()).to_hex().to_string(),
        ir_hash,
        scopes: vec![Scope {
            key: root.clone(),
            parent: None,
            path: vec![ScopeSegment {
                kind: ScopeKind::Program,
                name: program.to_owned(),
                instance: 0,
                call_site: None,
            }],
            stage: None,
            glyph: GlyphId("scope-function".to_owned()),
            ports: Vec::new(),
        }],
        nodes,
        edges,
        state: DiagramGraphState::Derived,
    }
}

/// The line `modelspec ir` prints for each node, by node key; a `DiagramNode` does not carry it,
/// so the SVG's `<title>`s take it from here.
pub type IrLines = BTreeMap<String, String>;

/// The line of every node of `compiled`'s training step, as `modelspec ir` prints it
/// (`%2: Tensor<f32, [16, 8]> = MatMul %0, %1`): the same line under `--stage tensor` for each
/// node of the forward program.
pub fn ir_lines(compiled: &CompiledTraining) -> IrLines {
    let program = &compiled.training.training.program;
    program
        .nodes
        .iter()
        .map(|node| {
            let mut line = String::new();
            let _ = write_node(&mut line, program, node);
            (key(node.id), line)
        })
        .collect()
}

/// The node key of a value, `%<k>`.
pub fn key(value: ValueId) -> String {
    format!("%{}", value.0)
}

/// Every value `from` reaches through operands, itself included.
fn reaches(by_id: &BTreeMap<ValueId, &TensorNode>, from: ValueId) -> BTreeSet<ValueId> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![from];
    while let Some(id) = stack.pop() {
        if seen.insert(id)
            && let Some(node) = by_id.get(&id)
        {
            stack.extend(node.operands.iter().copied());
        }
    }
    seen
}

/// The glyph role of each `Input` other than a plain data input: the RNG token a `Dropout` or
/// `DropoutGrad` reads at its token operand, and a host input of the optimizer step.
fn input_roles(
    program: &TensorProgram,
    subgraphs: &BTreeMap<ValueId, IrSubgraph>,
) -> BTreeMap<ValueId, InputRole> {
    let mut roles = BTreeMap::new();
    for node in &program.nodes {
        if let Some(&token) = ops::token_operand(&node.op).and_then(|i| node.operands.get(i)) {
            roles.insert(token, InputRole::DropoutToken);
        }
    }
    for node in &program.nodes {
        if matches!(node.op, TensorOp::Input { .. })
            && subgraphs.get(&node.id) == Some(&IrSubgraph::Optimizer)
        {
            roles.entry(node.id).or_insert(InputRole::HostStep);
        }
    }
    roles
}

fn operator(
    program: &TensorProgram,
    node: &TensorNode,
    role: InputRole,
    root: &ScopeKey,
) -> DiagramNode {
    let dtype = dtype(node.ty.dtype);
    let shape = shape(&node.ty.shape);
    let name = match &node.op {
        TensorOp::Input { name } => Some(name.clone()),
        TensorOp::ParamRead { param } => {
            program
                .params
                .get(param)
                .map(|p| match p.path.split_once('.') {
                    Some((_, rest)) => rest.to_owned(),
                    None => p.path.clone(),
                })
        }
        TensorOp::Scalar(value) => Some(format!("{value:?}")),
        _ => None,
    };
    let dims: Vec<String> = shape.iter().map(u64::to_string).collect();
    let ty = format!("{}[{}]", dtype.name(), dims.join(","));
    let second = match name {
        Some(name) => format!("{name} {ty}"),
        None => ty,
    };
    DiagramNode {
        node_key: key(node.id),
        kind: NodeKind::Operator,
        scope: root.clone(),
        composite_of: None,
        value_id: Some(i64::from(node.id.0)),
        op: Some(node.op.name().to_owned()),
        family: Some(match role {
            InputRole::Data => family_of(&node.op),
            InputRole::DropoutToken => crate::glyph::OperatorFamily::Stochastic,
            InputRole::HostStep => crate::glyph::OperatorFamily::Optimizer,
        }),
        glyph: glyph_of(&node.op, role),
        label: format!("{} {}\n{second}", node.op.name(), key(node.id)),
        dtype: Some(dtype),
        shape: Some(shape),
        subgraph: None,
        endpoint: None,
        source_span: node.source_span.as_ref().map(ToString::to_string),
    }
}

fn dtype(dtype: tensor::DType) -> DType {
    match dtype {
        tensor::DType::F32 => DType::F32,
        tensor::DType::I32 => DType::I32,
    }
}

/// A verified program has only static dimensions (DESIGN.md § 2.2); a symbolic one, which no
/// verified program holds, is written 0.
fn shape(dims: &[Dim]) -> Vec<u64> {
    dims.iter()
        .map(|dim| match dim {
            Dim::Static(n) => *n,
            Dim::Symbol(_) => 0,
        })
        .collect()
}
