//! Adversary cases for `story:observatory-input-adjoint`: the duplicated-name claim, a target
//! that is itself an input, and structural cases of shared use. The one-time comparison of
//! `differentiate` against the base commit `ab9ed27` is recorded in the wave's review-result;
//! the CLI golden `linear_regression.training.ir` holds the training path from here on.

use modelspec_core::autodiff::{differentiate_inputs, output_element_target};
use modelspec_core::compile::compile_training;
use modelspec_core::ir::tensor::{
    TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::verify::verify;

const LINEAR: &str = include_str!("../../../examples/linear_regression.mspec");

fn loss_of(program: &TensorProgram) -> ValueId {
    let [loss] = program.outputs[..] else {
        panic!("one output")
    };
    loss
}

/// The claim "a name asked for twice is listed twice with one adjoint": the same value, for a
/// read and for an unread input, and one zero for the unread one.
#[test]
fn adversary_a_name_asked_twice_has_one_adjoint_value() {
    let compiled = compile_training("linear_regression.mspec", LINEAR).unwrap();
    let forward = &compiled.forward.program;
    let loss = loss_of(forward);
    let once = differentiate_inputs(forward, loss, &["x"]).unwrap();
    let twice = differentiate_inputs(forward, loss, &["x", "x"]).unwrap();
    assert_eq!(twice.adjoints[0].1, twice.adjoints[1].1, "read input");
    assert_eq!(
        twice.program.nodes.len(),
        once.program.nodes.len(),
        "no second walk for a repeated name"
    );

    let mut b = TensorProgramBuilder::new();
    let a = b.input("a", TensorType::f32(&[2])).unwrap();
    b.input("c", TensorType::f32(&[4])).unwrap();
    let target = b.reduce_sum(a).unwrap();
    let forward = b.finish();
    let twice = differentiate_inputs(&forward, target, &["c", "a", "c"]).unwrap();
    assert_eq!(twice.adjoints[0].1, twice.adjoints[2].1, "unread input");
    let broadcasts = twice
        .program
        .nodes
        .iter()
        .filter(|n| matches!(n.op, TensorOp::BroadcastScalar { .. }))
        .count();
    assert_eq!(broadcasts, 2, "one for reduce_sum's rule, one for c's zero");
    verify(&twice.program).unwrap();
}

/// A target that is itself a rank-0 input: its adjoint is the seed `1.0`.
#[test]
fn adversary_a_target_that_is_the_input_itself() {
    let mut b = TensorProgramBuilder::new();
    let s = b.input("s", TensorType::f32_scalar()).unwrap();
    let forward = b.finish();
    let adjoint = differentiate_inputs(&forward, s, &["s"]).unwrap();
    let g = adjoint.program.node(adjoint.adjoints[0].1).unwrap();
    assert_eq!(g.op, TensorOp::Scalar(1.0));
    assert_eq!(adjoint.program.outputs, vec![s, g.id]);
    verify(&adjoint.program).unwrap();
}

/// An input read by both operands of one node and by a second node: the walk sums three
/// contributions into one adjoint (two `Add`s over the three uses).
#[test]
fn adversary_an_input_read_three_times_sums_three_contributions() {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[2, 2])).unwrap();
    let xx = b.mul(x, x).unwrap();
    let a = b.add(xx, x).unwrap();
    let target = b.reduce_sum(a).unwrap();
    let forward = b.finish();
    let adjoint = differentiate_inputs(&forward, target, &["x"]).unwrap();
    let g = adjoint.program.node(adjoint.adjoints[0].1).unwrap();
    assert_eq!(g.op, TensorOp::Add, "{}", adjoint.program);
    let inner = adjoint.program.node(g.operands[0]).unwrap();
    assert_eq!(inner.op, TensorOp::Add, "{}", adjoint.program);
    verify(&adjoint.program).unwrap();
}

/// `output_element_target` of a rank-0 prediction: the empty index, offset 0.
#[test]
fn adversary_output_element_target_of_a_rank_0_prediction() {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[3])).unwrap();
    let p = b.reduce_sum(x).unwrap();
    let forward = b.finish();
    let target = output_element_target(&forward, p, &[]).unwrap();
    assert_eq!(target.offset, 0);
    let adjoint = differentiate_inputs(&target.program, target.target, &["x"]).unwrap();
    verify(&adjoint.program).unwrap();
}
