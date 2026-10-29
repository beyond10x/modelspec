//! The masked-row gather of a weighted `cross_entropy` loss (DESIGN.md § 3.3,
//! story:vulkan-mlm-masked-row-gather).
//!
//! A masked-language-model loss weighs most rows by zero: only the selected positions count. The
//! row-local head that turns the encoder's rows into logits (each node a `matmul` by a value that
//! depends on no input, an `add_rows`, a `gelu` or a `layer_norm`, each read by the next alone)
//! then computes thousands of rows the loss multiplies by zero. [`gather`] adds to the program a
//! copy of that head that runs on `capacity` gathered rows: an `embedding` of the head's first
//! operand (the frontier) at the host-supplied row ids [`ROWS_INPUT`], the copied head, and the
//! loss reading the gathered logits with the gathered labels and weights. The original head
//! stays: it still computes the prediction that evaluation and inference read, and nothing the
//! loss depends on reads it. Autodiff differentiates the `embedding`, so the frontier's gradient
//! is the scatter of the gathered rows' gradients into a dense zero gradient, and the head's
//! parameters, the tied token table included, collect their gradients over the gathered rows
//! only, in ascending row order.
//!
//! The host fills the three inputs at every step ([`MaskedRowsSite`]): the rows of non-zero
//! weight in ascending order, each with its label and weight, then padding rows of row `0`,
//! label `0` and weight `+0.0` up to the capacity. A batch with more selected rows than the
//! capacity is refused, never truncated.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::tensor::{DType, Dim, TensorNode, TensorOp, TensorProgram, TensorType, ValueId};
use crate::ops;

/// The `i32[capacity]` input of the gathered row ids. Its dotted name is no identifier, so no
/// declaration of a `train` block takes it.
pub const ROWS_INPUT: &str = "masked.rows";

/// The prefix of the `i32[capacity]` input of the gathered rows' labels; the declared labels'
/// name follows it.
pub const LABELS_PREFIX: &str = "masked.labels.";

/// The prefix of the `f32[capacity]` input of the gathered rows' weights; the declared weights'
/// name follows it.
pub const WEIGHTS_PREFIX: &str = "masked.weights.";

/// The masked-row gather of a program: what the host feeds at every step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskedRowsSite {
    /// How many rows the gather holds: the length of the three inputs.
    pub capacity: u64,
    /// The declared `i32[rows]` labels the full loss reads.
    pub labels: String,
    /// The declared `f32[rows]` weights the full loss reads.
    pub weights: String,
}

impl MaskedRowsSite {
    /// The name of the gathered labels' input.
    pub fn labels_input(&self) -> String {
        format!("{LABELS_PREFIX}{}", self.labels)
    }

    /// The name of the gathered weights' input.
    pub fn weights_input(&self) -> String {
        format!("{WEIGHTS_PREFIX}{}", self.weights)
    }
}

/// The three inputs [`gather`] added to `program`, in input order: none for a program it left
/// unchanged. The host supplies them at every step, so they are no data of the `train` block.
pub fn site_inputs(program: &TensorProgram) -> Vec<ValueId> {
    program
        .inputs
        .iter()
        .copied()
        .filter(|&value| {
            matches!(
                program.node(value).map(|n| &n.op),
                Some(TensorOp::Input { name })
                    if name == ROWS_INPUT
                        || name.starts_with(LABELS_PREFIX)
                        || name.starts_with(WEIGHTS_PREFIX)
            )
        })
        .collect()
}

/// The site of a program [`gather`] transformed, read from its three inputs; `None` for every
/// program [`gather`] left unchanged.
pub fn site(program: &TensorProgram) -> Option<MaskedRowsSite> {
    let mut capacity = None;
    let mut labels = None;
    let mut weights = None;
    for &value in &program.inputs {
        let node = program.node(value)?;
        let TensorOp::Input { name } = &node.op else {
            continue;
        };
        if name == ROWS_INPUT {
            if let [Dim::Static(n)] = node.ty.shape[..] {
                capacity = Some(n);
            }
        } else if let Some(declared) = name.strip_prefix(LABELS_PREFIX) {
            labels = Some(declared.to_owned());
        } else if let Some(declared) = name.strip_prefix(WEIGHTS_PREFIX) {
            weights = Some(declared.to_owned());
        }
    }
    Some(MaskedRowsSite {
        capacity: capacity?,
        labels: labels?,
        weights: weights?,
    })
}

/// Gathers the loss of `program` at `capacity` rows (module docs). The program is unchanged when
/// its one output is not a `cross_entropy` of declared labels and weights over static `[rows, K]`
/// logits, when `capacity` is `0` or at least `rows`, or when no row-local head over a rank-2
/// `f32` value computes the logits.
///
/// The `cross_entropy` node keeps its id and now reads the gathered logits and the two gathered
/// inputs. New nodes take ids above the largest one and are defined right before the loss; the
/// program's inputs list them in node order. In a program that records scopes (DESIGN.md
/// § 13.2), each copied head node takes the scope of the node it copies, and the three inputs
/// and the `embedding` take the loss's scope.
pub fn gather(program: TensorProgram, capacity: u64) -> TensorProgram {
    let Some(plan) = plan(&program, capacity) else {
        return program;
    };
    let TensorProgram {
        nodes,
        params,
        inputs: _,
        outputs,
        mut scopes,
    } = program;
    let mut next = nodes.iter().map(|n| n.id.0).max().unwrap_or(0) + 1;
    let mut types: BTreeMap<ValueId, TensorType> =
        nodes.iter().map(|n| (n.id, n.ty.clone())).collect();
    let loss = nodes[plan.loss_at].clone();
    let span = loss.source_span.clone();
    let mut added: Vec<TensorNode> = Vec::new();
    let mut likes: Vec<(ValueId, ValueId)> = Vec::new();
    let mut add = |op: TensorOp, operands: Vec<ValueId>, ty: Option<TensorType>, like: ValueId| {
        let ty = match ty {
            Some(ty) => ty,
            None => {
                let operand_types: Vec<&TensorType> = operands.iter().map(|v| &types[v]).collect();
                ops::infer(&op, &operand_types, None).ok()?
            }
        };
        let id = ValueId(next);
        next += 1;
        types.insert(id, ty.clone());
        added.push(TensorNode {
            id,
            op,
            operands,
            ty,
            source_span: span.clone(),
        });
        likes.push((id, like));
        Some(id)
    };
    let input = |name: String| TensorOp::Input { name };
    let built = (|| {
        let rows = add(
            input(ROWS_INPUT.to_owned()),
            Vec::new(),
            Some(TensorType::i32(&[capacity])),
            loss.id,
        )?;
        let labels = add(
            input(format!("{LABELS_PREFIX}{}", plan.labels)),
            Vec::new(),
            Some(TensorType::i32(&[capacity])),
            loss.id,
        )?;
        let weights = add(
            input(format!("{WEIGHTS_PREFIX}{}", plan.weights)),
            Vec::new(),
            Some(TensorType::f32(&[capacity])),
            loss.id,
        )?;
        let mut current = add(
            TensorOp::Embedding,
            vec![plan.frontier, rows],
            None,
            loss.id,
        )?;
        for (original, op, rest) in &plan.chain {
            let mut operands = vec![current];
            operands.extend(rest.iter().copied());
            current = add(op.clone(), operands, None, *original)?;
        }
        Some((current, labels, weights))
    })();
    let Some((logits, labels, weights)) = built else {
        return TensorProgram {
            nodes,
            params,
            inputs: Vec::new(),
            outputs,
            scopes,
        }
        .with_inputs_in_node_order();
    };
    for (value, like) in likes {
        if let Some(scope) = scopes.scope_of(like) {
            scopes.stamp(value, scope);
        }
    }
    let mut result = Vec::with_capacity(nodes.len() + added.len());
    for (at, node) in nodes.into_iter().enumerate() {
        if at == plan.loss_at {
            result.append(&mut added);
            result.push(TensorNode {
                operands: vec![logits, labels, weights],
                ..node
            });
        } else {
            result.push(node);
        }
    }
    TensorProgram {
        nodes: result,
        params,
        inputs: Vec::new(),
        outputs,
        scopes,
    }
    .with_inputs_in_node_order()
}

impl TensorProgram {
    /// `self` with [`TensorProgram::inputs`] every `Input` node, in node order.
    fn with_inputs_in_node_order(mut self) -> Self {
        self.inputs = self
            .nodes
            .iter()
            .filter(|n| matches!(n.op, TensorOp::Input { .. }))
            .map(|n| n.id)
            .collect();
        self
    }
}

/// What [`gather`] adds: the loss's position, the declared labels' and weights' names, the
/// frontier, and the head from the frontier to the logits, each node's id, its op and its
/// operands after the first.
struct Plan {
    loss_at: usize,
    labels: String,
    weights: String,
    frontier: ValueId,
    chain: Vec<(ValueId, TensorOp, Vec<ValueId>)>,
}

fn plan(program: &TensorProgram, capacity: u64) -> Option<Plan> {
    let [loss] = program.outputs[..] else {
        return None;
    };
    let loss_at = program.nodes.iter().position(|n| n.id == loss)?;
    let node = &program.nodes[loss_at];
    if node.op != TensorOp::CrossEntropy {
        return None;
    }
    let input_name = |value: ValueId| match &program.node(value)?.op {
        TensorOp::Input { name } => Some(name.clone()),
        _ => None,
    };
    let labels = input_name(node.operands[1])?;
    let weights = input_name(node.operands[2])?;
    let rows = match program.node(node.operands[0])?.ty.shape[..] {
        [Dim::Static(rows), Dim::Static(_)] => rows,
        _ => return None,
    };
    if capacity == 0 || capacity >= rows {
        return None;
    }
    let row_free = row_free(program);
    let mut consumers: BTreeMap<ValueId, usize> = BTreeMap::new();
    for n in &program.nodes {
        for operand in &n.operands {
            *consumers.entry(*operand).or_default() += 1;
        }
    }
    for output in &program.outputs {
        *consumers.entry(*output).or_default() += 1;
    }
    let mut chain: Vec<(ValueId, TensorOp, Vec<ValueId>)> = Vec::new();
    let mut current = program.node(node.operands[0])?;
    loop {
        let row_local = matches!(
            current.op,
            TensorOp::AddRows | TensorOp::MatMul | TensorOp::Gelu | TensorOp::LayerNorm
        );
        let rest = current.operands.get(1..).unwrap_or_default();
        if !row_local
            || consumers.get(&current.id) != Some(&1)
            || !rest.iter().all(|v| row_free.contains(v))
        {
            break;
        }
        chain.push((current.id, current.op.clone(), rest.to_vec()));
        current = program.node(current.operands[0])?;
    }
    if chain.is_empty() || current.ty.dtype != DType::F32 || current.ty.rank() != 2 {
        return None;
    }
    chain.reverse();
    Some(Plan {
        loss_at,
        labels,
        weights,
        frontier: current.id,
        chain,
    })
}

/// The values of `program` that depend on no `Input`: parameters, constants and what is computed
/// from them alone, the same for every row of a batch.
fn row_free(program: &TensorProgram) -> BTreeSet<ValueId> {
    let mut free = BTreeSet::new();
    for node in &program.nodes {
        let leaf = matches!(node.op, TensorOp::ParamRead { .. } | TensorOp::Scalar(_));
        let derived = !node.operands.is_empty()
            && !matches!(node.op, TensorOp::Input { .. })
            && node.operands.iter().all(|v| free.contains(v));
        if leaf || derived {
            free.insert(node.id);
        }
    }
    free
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::differentiate;
    use crate::ir::tensor::{Initializer, ScopeSegment, TensorProgramBuilder};
    use crate::ir::verify::verify;

    /// `cross_entropy(add_rows(matmul(gelu(x), w), b), labels, weights)` over `rows` rows.
    fn head(rows: u64) -> TensorProgram {
        let mut b = TensorProgramBuilder::new();
        let x = b.input("x", TensorType::f32(&[rows, 4])).unwrap();
        let labels = b.input("labels", TensorType::i32(&[rows])).unwrap();
        let weights = b.input("weights", TensorType::f32(&[rows])).unwrap();
        let w = b
            .declare_param("w", TensorType::f32(&[4, 6]), Initializer::XavierUniform)
            .unwrap();
        let bias = b
            .declare_param("bias", TensorType::f32(&[6]), Initializer::Zeros)
            .unwrap();
        let w = b.param_read(w).unwrap();
        let bias = b.param_read(bias).unwrap();
        let g = b.gelu(x).unwrap();
        let m = b.matmul(g, w).unwrap();
        let logits = b.add_rows(m, bias).unwrap();
        let loss = b.cross_entropy(logits, labels, weights).unwrap();
        b.output(loss).unwrap();
        b.finish()
    }

    /// `gelu`, `matmul` and `add_rows` are row-local and each read once, so the frontier is `x`
    /// and the gathered head is one `embedding` of it and a copy of the three on 4 rows.
    #[test]
    fn the_head_is_copied_onto_the_gathered_rows() {
        let program = gather(head(16), 4);
        verify(&program).expect("the gathered program verifies");
        let site = site(&program).expect("the program has a site");
        assert_eq!(
            site,
            MaskedRowsSite {
                capacity: 4,
                labels: "labels".into(),
                weights: "weights".into(),
            }
        );
        let loss = program.node(program.outputs[0]).unwrap();
        let logits = program.node(loss.operands[0]).unwrap();
        assert_eq!(logits.ty, TensorType::f32(&[4, 6]));
        let ops: Vec<&str> = program.nodes.iter().map(|n| n.op.name()).collect();
        assert_eq!(
            ops.iter().filter(|op| **op == "Embedding").count(),
            1,
            "{ops:?}"
        );
        assert_eq!(ops.iter().filter(|op| **op == "Gelu").count(), 2);
        differentiate(&program, program.outputs[0]).expect("the gathered loss differentiates");
    }

    /// In a program that records scopes, every gathered node has one: a copied head node its
    /// original's, the inputs and the `embedding` the loss's (DESIGN.md § 13.2).
    #[test]
    fn the_gathered_nodes_keep_the_scopes_of_what_they_copy() {
        let mut b = TensorProgramBuilder::new();
        b.enter_scope(ScopeSegment::Program("t".into()));
        let x = b.input("x", TensorType::f32(&[16, 4])).unwrap();
        let labels = b.input("labels", TensorType::i32(&[16])).unwrap();
        let weights = b.input("weights", TensorType::f32(&[16])).unwrap();
        let w = b
            .declare_param("w", TensorType::f32(&[4, 6]), Initializer::XavierUniform)
            .unwrap();
        let w = b.param_read(w).unwrap();
        let head = b.enter_scope(ScopeSegment::Function {
            name: "head".into(),
            instance: 0,
            call_span: None,
        });
        let g = b.gelu(x).unwrap();
        let logits = b.matmul(g, w).unwrap();
        b.exit_scope();
        let loss = b.cross_entropy(logits, labels, weights).unwrap();
        b.output(loss).unwrap();
        let original = b.finish();
        let loss_scope = original.scopes.scope_of(loss).unwrap();
        let program = gather(original.clone(), 4);
        verify(&program).expect("the gathered program verifies with its scopes");
        let added: Vec<&TensorNode> = program
            .nodes
            .iter()
            .filter(|n| original.node(n.id).is_none())
            .collect();
        assert_eq!(
            added.len(),
            6,
            "three inputs, the embedding and two head copies"
        );
        for node in added {
            let expected = match node.op {
                TensorOp::Gelu | TensorOp::MatMul => head,
                _ => loss_scope,
            };
            assert_eq!(program.scopes.scope_of(node.id), Some(expected), "{node:?}");
        }
    }

    /// A capacity of the full row count, or none, leaves the program unchanged.
    #[test]
    fn a_capacity_of_every_row_changes_nothing() {
        assert_eq!(gather(head(16), 16), head(16));
        assert_eq!(gather(head(16), 0), head(16));
        assert_eq!(site(&head(16)), None);
    }

    /// A loss that is no `cross_entropy` has no site.
    #[test]
    fn a_program_without_cross_entropy_is_unchanged() {
        let mut b = TensorProgramBuilder::new();
        let x = b.input("x", TensorType::f32(&[16, 4])).unwrap();
        let y = b.input("y", TensorType::f32(&[16, 4])).unwrap();
        let loss = b.mse(x, y).unwrap();
        b.output(loss).unwrap();
        let program = b.finish();
        assert_eq!(gather(program.clone(), 4), program);
    }
}
