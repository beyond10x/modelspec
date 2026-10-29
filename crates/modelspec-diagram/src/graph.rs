//! The `DiagramGraph` of a verified program (DESIGN.md § 13.2), built from
//! [`compile_training`](modelspec_core::compile::compile_training)'s output; nothing is parsed.
//!
//! The model view is the forward slice of the Tensor IR before autodiff
//! ([`CompiledTraining::tensor`]): every node the model's prediction
//! ([`LoweredProgram::prediction`](modelspec_core::lower::LoweredProgram)) reaches through its
//! operands, in definition order, dropouts and their RNG tokens included, and every node a
//! `StopGradient` reaches, the `StopGradient` too: the branch that computes a target the model
//! sets itself, such as a joint-embedding model's target embedding. Each node is a line
//! `modelspec ir --stage tensor` prints. The loss and the targets only it reads are not in it.
//! One edge per operand carries the producer's dtype and shape.
//!
//! Every node sits in the scope the IR's scope table stamped on it (DESIGN.md § 13.2): the
//! Program scope, the train block, or an inlined function's, keyed `<program>/<fn>` for its first
//! inlining and `<program>/<fn>@<c>` from the second, as dropout sites are named. The graph holds
//! the Program scope and every scope that holds a node of the slice, with their ancestors, parents
//! before children, each with its ports: one per value crossing its boundary, In (defined outside,
//! read inside) by ValueId, then Out (defined inside, read outside) by ValueId.

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::compile::CompiledTraining;
use modelspec_core::ir::print::{print_step, write_node};
use modelspec_core::ir::tensor::{
    self, Dim, ScopeId, ScopeTable, TensorNode, TensorOp, TensorProgram, ValueId,
};
use modelspec_core::ir::training::Subgraph as IrSubgraph;
use modelspec_core::ops;

use crate::glyph::{InputRole, family_of, glyph_of};
use crate::model::{
    DType, DiagramEdge, DiagramGraph, DiagramGraphId, DiagramGraphState, DiagramNode, DiagramView,
    GlyphId, NodeKind, Port, PortDirection, Scope, ScopeKey, ScopeKind, ScopeSegment,
};

/// The model view of `compiled`, the train block named `program`, compiled from `source`.
pub fn model(compiled: &CompiledTraining, program: &str, source: &str) -> DiagramGraph {
    let training = &compiled.training.training;
    let ir = &compiled.tensor.program;
    let by_id: BTreeMap<ValueId, &TensorNode> = ir.nodes.iter().map(|n| (n.id, n)).collect();
    let mut slice = reaches(&by_id, compiled.tensor.prediction);
    for stop in ir.nodes.iter().filter(|n| n.op == TensorOp::StopGradient) {
        slice.extend(reaches(&by_id, stop.id));
    }
    let roles = input_roles(&training.program, &training.subgraphs);
    let root = ScopeKey(program.to_owned());
    let ir_hash = blake3::hash(print_step(&compiled.training).as_bytes())
        .to_hex()
        .to_string();
    let table = &ir.scopes;
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for node in ir.nodes.iter().filter(|n| slice.contains(&n.id)) {
        let role = roles.get(&node.id).copied().unwrap_or(InputRole::Data);
        let scope = match table.scope_of(node.id) {
            Some(s) if table.entry(s).is_some_and(|e| e.parent.is_some()) => ScopeKey(table.key(s)),
            _ => root.clone(),
        };
        nodes.push(operator(ir, node, role, &scope));
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
        scopes: scopes(table, program, &nodes, &edges),
        nodes,
        edges,
        state: DiagramGraphState::Derived,
    }
}

/// The scopes of a graph whose nodes and edges are `nodes` and `edges`: the Program scope, then
/// every scope of `table` below it that holds a node, with its ancestors, in table order (a
/// parent before its children), each with its ports.
fn scopes(
    table: &ScopeTable,
    program: &str,
    nodes: &[DiagramNode],
    edges: &[DiagramEdge],
) -> Vec<Scope> {
    let root = Scope {
        key: ScopeKey(program.to_owned()),
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
    };
    let held: BTreeSet<&str> = nodes.iter().map(|n| n.scope.0.as_str()).collect();
    let mut kept: BTreeSet<usize> = BTreeSet::new();
    for (i, entry) in table.entries.iter().enumerate() {
        if entry.parent.is_none() || !held.contains(table.key(ScopeId(i as u32)).as_str()) {
            continue;
        }
        let mut at = Some(ScopeId(i as u32));
        while let Some(scope) = at {
            let Some(entry) = table.entry(scope) else {
                break;
            };
            if entry.parent.is_none() || !kept.insert(scope.0 as usize) {
                break;
            }
            at = entry.parent;
        }
    }
    let scope_of: BTreeMap<&str, &ScopeKey> = nodes
        .iter()
        .map(|n| (n.node_key.as_str(), &n.scope))
        .collect();
    let mut out = vec![root];
    for i in kept {
        let id = ScopeId(i as u32);
        let path: Vec<ScopeSegment> = table.path(id).into_iter().map(segment).collect();
        let parent = table.entries[i].parent.map(|p| ScopeKey(table.key(p)));
        let key = ScopeKey(table.key(id));
        let glyph = scope_glyph(path.last().map(|s| s.kind));
        let ports = ports(&key, &scope_of, edges);
        out.push(Scope {
            key,
            parent,
            path,
            stage: None,
            glyph,
            ports,
        });
    }
    out
}

/// `key` is `scope` or a scope inside it.
pub fn within(key: &ScopeKey, scope: &ScopeKey) -> bool {
    key.0 == scope.0
        || key
            .0
            .strip_prefix(scope.0.as_str())
            .is_some_and(|rest| rest.starts_with('/'))
}

/// The ports of `scope`: every value an edge carries across its boundary, In ports by ValueId,
/// then Out ports by ValueId.
fn ports(
    scope: &ScopeKey,
    scope_of: &BTreeMap<&str, &ScopeKey>,
    edges: &[DiagramEdge],
) -> Vec<Port> {
    let mut crossing: BTreeMap<(PortDirection, i64), Port> = BTreeMap::new();
    for edge in edges {
        let (Some(p), Some(c)) = (
            scope_of.get(edge.producer.as_str()),
            scope_of.get(edge.consumer.as_str()),
        ) else {
            continue;
        };
        let direction = match (within(p, scope), within(c, scope)) {
            (false, true) => PortDirection::In,
            (true, false) => PortDirection::Out,
            _ => continue,
        };
        let Some(value_id) = edge.producer.strip_prefix('%').and_then(|v| v.parse().ok()) else {
            continue;
        };
        crossing.entry((direction, value_id)).or_insert(Port {
            direction,
            value_id,
            dtype: edge.dtype,
            shape: edge.shape.clone(),
        });
    }
    crossing.into_values().collect()
}

fn segment(segment: &tensor::ScopeSegment) -> ScopeSegment {
    let (kind, name, instance, call_site) = match segment {
        tensor::ScopeSegment::Program(name) => (ScopeKind::Program, name.clone(), 0, None),
        tensor::ScopeSegment::Function {
            name,
            instance,
            call_span,
        } => (
            ScopeKind::Function,
            name.clone(),
            *instance,
            call_span.as_ref().map(ToString::to_string),
        ),
        tensor::ScopeSegment::Adjoint(_) => (ScopeKind::Adjoint, segment.key(), 0, None),
        tensor::ScopeSegment::Update(path) => (ScopeKind::Update, path.clone(), 0, None),
        tensor::ScopeSegment::Module(name) => (ScopeKind::Module, name.clone(), 0, None),
        tensor::ScopeSegment::Layer(name) => (ScopeKind::Layer, name.clone(), 0, None),
    };
    ScopeSegment {
        kind,
        name,
        instance,
        call_site,
    }
}

/// The composite glyph of a scope whose innermost segment is `kind` (DESIGN.md § 13.1); an
/// Update scope's optimizer is the pipeline view's to name, so it takes `scope-update-sgd` here.
fn scope_glyph(kind: Option<ScopeKind>) -> GlyphId {
    GlyphId(
        match kind {
            Some(ScopeKind::Adjoint) => "scope-adjoint",
            Some(ScopeKind::Update) => "scope-update-sgd",
            Some(ScopeKind::Module) => "scope-module",
            Some(ScopeKind::Layer) => "scope-layer",
            Some(ScopeKind::Program | ScopeKind::Function | ScopeKind::Stage) | None => {
                "scope-function"
            }
        }
        .to_owned(),
    )
}

/// The line `modelspec ir` prints for each node, by node key; a `DiagramNode` does not carry it,
/// so the SVG's `<title>`s take it from here.
pub type IrLines = BTreeMap<String, String>;

/// The line of every node of `compiled`'s training step, as `modelspec ir` prints it
/// (`%2: Tensor<f32, [16, 8]> = MatMul %0, %1`): the same line under `--stage tensor` for each
/// node of the forward program; and the `--stage tensor` line of each node autodiff removes, a
/// `StopGradient`.
pub fn ir_lines(compiled: &CompiledTraining) -> IrLines {
    let step = &compiled.training.training.program;
    let tensor = &compiled.tensor.program;
    let mut lines: IrLines = step
        .nodes
        .iter()
        .map(|node| {
            let mut line = String::new();
            let _ = write_node(&mut line, step, node);
            (key(node.id), line)
        })
        .collect();
    for node in &tensor.nodes {
        lines.entry(key(node.id)).or_insert_with(|| {
            let mut line = String::new();
            let _ = write_node(&mut line, tensor, node);
            line
        });
    }
    lines
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
