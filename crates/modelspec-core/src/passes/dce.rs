//! Dead-code elimination over Tensor IR (DESIGN.md § 3.3 pass 3).
//!
//! A node is live when a program output depends on it through operands; [`eliminate`] removes
//! every other node. Surviving nodes keep their ids, so printed ids stay those lowering assigned.
//! An `Input` node that is removed leaves the program's input list too. The parameter table is
//! not changed: a parameter no output reads keeps its entry, and its initial value. Each surviving
//! node keeps its scope; the scope of every node no longer in the program is dropped, the scopes
//! themselves stay (DESIGN.md § 13.2).

use std::collections::BTreeSet;

use crate::ir::tensor::{TensorProgram, ValueId};

/// Removes the nodes of `program` that no output depends on.
///
/// Precondition: every operand is defined by an earlier node, as [`crate::ir::verify::verify`]
/// requires; a single reverse sweep then finds every live node.
pub fn eliminate(mut program: TensorProgram) -> TensorProgram {
    let mut live: BTreeSet<ValueId> = program.outputs.iter().copied().collect();
    for node in program.nodes.iter().rev() {
        if live.contains(&node.id) {
            live.extend(node.operands.iter().copied());
        }
    }
    program.nodes.retain(|node| live.contains(&node.id));
    program.inputs.retain(|value| live.contains(value));
    let kept: BTreeSet<ValueId> = program.nodes.iter().map(|node| node.id).collect();
    program
        .scopes
        .values
        .retain(|value, _| kept.contains(value));
    program
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::tensor::{Initializer, TensorOp, TensorProgramBuilder, TensorType};
    use crate::ir::verify::verify;

    /// `x: [8, 4]`, `w: [4, 1]`, live `MatMul x, w` as the output, and two dead nodes: `Relu x`
    /// and a second `ParamRead w`.
    fn program_with_dead_nodes() -> (TensorProgram, ValueId, ValueId) {
        let mut b = TensorProgramBuilder::new();
        let w = b
            .declare_param("M.w", TensorType::f32(&[4, 1]), Initializer::XavierUniform)
            .unwrap();
        let x = b.input("x", TensorType::f32(&[8, 4])).unwrap();
        let dead_relu = b.relu(x).unwrap();
        let read = b.param_read(w).unwrap();
        let dead_read = b.param_read(w).unwrap();
        let out = b.matmul(x, read).unwrap();
        b.output(out).unwrap();
        assert_eq!((dead_relu, dead_read), (ValueId(1), ValueId(3)));
        (b.finish(), dead_relu, dead_read)
    }

    #[test]
    fn removes_every_node_no_output_reaches() {
        let (program, dead_relu, dead_read) = program_with_dead_nodes();
        let after = eliminate(program);
        assert!(after.node(dead_relu).is_none());
        assert!(after.node(dead_read).is_none());
        assert_eq!(after.nodes.len(), 3);
        verify(&after).expect("DCE output verifies");
    }

    #[test]
    fn keeps_the_nodes_an_output_reaches_with_their_ids_and_the_parameter_table() {
        let (program, _, _) = program_with_dead_nodes();
        let params = program.params.clone();
        let after = eliminate(program.clone());
        let kept: Vec<ValueId> = after.nodes.iter().map(|n| n.id).collect();
        assert_eq!(kept, [ValueId(0), ValueId(2), ValueId(4)]);
        for node in &after.nodes {
            assert_eq!(Some(node), program.node(node.id));
        }
        assert_eq!(after.outputs, program.outputs);
        assert_eq!(after.params, params);
        assert_eq!(after.inputs, [ValueId(0)]);
    }

    #[test]
    fn removes_a_dead_input_from_the_input_list() {
        let mut b = TensorProgramBuilder::new();
        let x = b.input("x", TensorType::f32(&[2, 2])).unwrap();
        let unused = b.input("unused", TensorType::f32(&[2, 2])).unwrap();
        let out = b.relu(x).unwrap();
        b.output(out).unwrap();
        let after = eliminate(b.finish());
        assert_eq!(after.inputs, [x]);
        assert!(after.node(unused).is_none());
        assert!(
            !after
                .nodes
                .iter()
                .any(|n| matches!(&n.op, TensorOp::Input { name } if name == "unused"))
        );
        verify(&after).expect("DCE output verifies");
    }

    #[test]
    fn a_program_with_no_dead_node_is_unchanged() {
        let (program, _, _) = program_with_dead_nodes();
        let once = eliminate(program);
        assert_eq!(eliminate(once.clone()), once);
    }
}
