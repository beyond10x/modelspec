//! Removal of `stop_gradient` after autodiff (DESIGN.md § 3.3, § 4; story:jepa-stop-gradient).
//!
//! `StopGradient` is the identity in the forward pass and has no gradient: autodiff does not walk
//! through it ([`crate::autodiff`]), so once the backward graph exists the node only copies its
//! operand. [`eliminate`] replaces every use of a `StopGradient` value by its operand, through a
//! chain of them to the first value that is not one, and removes the nodes, so neither the CPU
//! interpreter nor the scheduler sees one and the op needs no kernel. Surviving nodes keep their
//! ids. [`eliminate_training`] does the same to a training program, its subgraph tags, loss and
//! gradients included.

use std::collections::BTreeMap;

use crate::ir::tensor::{TensorOp, TensorProgram, ValueId};
use crate::ir::training::{Subgraph, TrainingProgram};

/// The values [`eliminate`] removed, each with the value that now stands for it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Replacements(BTreeMap<ValueId, ValueId>);

impl Replacements {
    /// The value that stands for `value`: its replacement if the pass removed it, else itself.
    pub fn of(&self, value: ValueId) -> ValueId {
        self.0.get(&value).copied().unwrap_or(value)
    }

    /// Whether the pass removed no node.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Removes every `StopGradient` node of `program`, each use of one reading its operand instead,
/// and returns the program and what was replaced. Outputs that named a removed value name its
/// replacement; inputs and the parameter table are unchanged.
///
/// Precondition: every operand is defined by an earlier node, as [`crate::ir::verify::verify`]
/// requires; one forward sweep then resolves chains.
pub fn eliminate(mut program: TensorProgram) -> (TensorProgram, Replacements) {
    let mut replaced = Replacements::default();
    let mut nodes = Vec::with_capacity(program.nodes.len());
    for mut node in std::mem::take(&mut program.nodes) {
        for operand in &mut node.operands {
            *operand = replaced.of(*operand);
        }
        if node.op == TensorOp::StopGradient {
            replaced.0.insert(node.id, node.operands[0]);
        } else {
            nodes.push(node);
        }
    }
    program.nodes = nodes;
    for output in &mut program.outputs {
        *output = replaced.of(*output);
    }
    (program, replaced)
}

/// [`eliminate`] on the program of `training`: the removed nodes lose their subgraph tags, and
/// the loss and each gradient name their replacement. A loss that was a `StopGradient` is
/// replaced by its operand, which is tagged [`Subgraph::Loss`] in its place.
pub fn eliminate_training(training: TrainingProgram) -> TrainingProgram {
    let TrainingProgram {
        program,
        mut subgraphs,
        loss,
        gradients,
    } = training;
    let (program, replaced) = eliminate(program);
    if replaced.is_empty() {
        return TrainingProgram {
            program,
            subgraphs,
            loss,
            gradients,
        };
    }
    subgraphs.retain(|value, _| replaced.of(*value) == *value);
    let new_loss = replaced.of(loss);
    if new_loss != loss {
        subgraphs.insert(new_loss, Subgraph::Loss);
    }
    TrainingProgram {
        program,
        subgraphs,
        loss: new_loss,
        gradients: gradients
            .into_iter()
            .map(|(param, value)| (param, replaced.of(value)))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::differentiate;
    use crate::ir::tensor::{Initializer, TensorProgramBuilder, TensorType};
    use crate::ir::verify::{verify, verify_training};

    /// `relu(stop_gradient(stop_gradient(x)))` as the output, `x` an input: the chain resolves to
    /// `x`, both nodes go, and the other ids stay.
    #[test]
    fn a_chain_of_stop_gradients_resolves_to_the_first_other_value() {
        let mut b = TensorProgramBuilder::new();
        let x = b.input("x", TensorType::f32(&[2, 3])).unwrap();
        let first = b.stop_gradient(x).unwrap();
        let second = b.stop_gradient(first).unwrap();
        let y = b.relu(second).unwrap();
        b.output(y).unwrap();
        let (program, replaced) = eliminate(b.finish());
        verify(&program).expect("the result verifies");
        assert_eq!(replaced.of(first), x);
        assert_eq!(replaced.of(second), x);
        assert_eq!(replaced.of(y), y);
        let ids: Vec<ValueId> = program.nodes.iter().map(|n| n.id).collect();
        assert_eq!(ids, [x, y]);
        assert_eq!(program.node(y).unwrap().operands, [x]);
    }

    /// An output that is a `StopGradient` names its operand afterwards.
    #[test]
    fn an_output_that_is_removed_names_its_replacement() {
        let mut b = TensorProgramBuilder::new();
        let x = b.input("x", TensorType::f32(&[4])).unwrap();
        let y = b.relu(x).unwrap();
        let stopped = b.stop_gradient(y).unwrap();
        b.output(stopped).unwrap();
        let (program, _) = eliminate(b.finish());
        assert_eq!(program.outputs, [y]);
        verify(&program).expect("the result verifies");
    }

    /// A program without `StopGradient` is unchanged.
    #[test]
    fn a_program_without_stop_gradient_is_unchanged() {
        let mut b = TensorProgramBuilder::new();
        let x = b.input("x", TensorType::f32(&[4])).unwrap();
        let y = b.relu(x).unwrap();
        b.output(y).unwrap();
        let program = b.finish();
        let (after, replaced) = eliminate(program.clone());
        assert_eq!(after, program);
        assert!(replaced.is_empty());
    }

    /// `mse(mul(stop_gradient(matmul(x, w)), matmul(x, w)), t)`: after autodiff and the pass the
    /// training program verifies, holds no `StopGradient` and no tag of the removed node, and the
    /// gradient rule of `Mul` reads the stopped operand directly.
    #[test]
    fn a_training_program_loses_the_node_and_its_tag_and_still_verifies() {
        let mut b = TensorProgramBuilder::new();
        let x = b.input("x", TensorType::f32(&[3, 2])).unwrap();
        let w = b
            .declare_param("M.w", TensorType::f32(&[2, 2]), Initializer::XavierUniform)
            .unwrap();
        let read = b.param_read(w).unwrap();
        let target = b.matmul(x, read).unwrap();
        let stopped = b.stop_gradient(target).unwrap();
        let online = b.matmul(x, read).unwrap();
        let product = b.mul(stopped, online).unwrap();
        let t = b.input("t", TensorType::f32(&[3, 2])).unwrap();
        let loss = b.mse(product, t).unwrap();
        b.output(loss).unwrap();
        let training = differentiate(&b.finish(), loss).unwrap();
        let training = eliminate_training(training);
        verify_training(&training).expect("the training program verifies");
        assert!(
            training
                .program
                .nodes
                .iter()
                .all(|n| n.op != TensorOp::StopGradient)
        );
        assert!(!training.subgraphs.contains_key(&stopped));
        assert_eq!(training.loss, loss);
        assert!(
            training
                .program
                .nodes
                .iter()
                .any(|n| training.subgraphs[&n.id] == Subgraph::Gradient
                    && n.op == TensorOp::Mul
                    && n.operands.contains(&target)),
            "the adjoint of the online term reads the stopped operand"
        );
    }
}
