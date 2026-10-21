//! Adversary cases for reverse-mode autodiff and the Training IR verifier: inconsistent training
//! programs the verifier accepts, and forward programs `differentiate` cannot take.

use std::panic::{AssertUnwindSafe, catch_unwind};

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{
    Initializer, ParamId, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::training::{Subgraph, TrainingProgram};
use modelspec_core::ir::verify::verify_training;

fn ty(shape: &[u64]) -> TensorType {
    TensorType::f32(shape)
}

/// `mse(matmul(x, w), y)` with the shapes of `examples/linear_regression.mspec`.
fn linear_forward() -> (TensorProgram, ValueId) {
    let mut b = TensorProgramBuilder::new();
    let w = b
        .declare_param("LinearReg.w", ty(&[4, 1]), Initializer::XavierUniform)
        .unwrap();
    let w = b.param_read(w).unwrap();
    let x = b.input("x", ty(&[8, 4])).unwrap();
    let y = b.input("y", ty(&[8, 1])).unwrap();
    let h = b.matmul(x, w).unwrap();
    let loss = b.mse(h, y).unwrap();
    (b.finish(), loss)
}

struct Linear {
    training: TrainingProgram,
    w: ParamId,
    w_read: ValueId,
    h: ValueId,
}

fn linear() -> Linear {
    let (forward, loss) = linear_forward();
    let training = differentiate(&forward, loss).unwrap();
    verify_training(&training).unwrap();
    let find =
        |want: &dyn Fn(&TensorOp) -> bool| forward.nodes.iter().find(|n| want(&n.op)).unwrap().id;
    Linear {
        w: ParamId(0),
        w_read: find(&|op| matches!(op, TensorOp::ParamRead { .. })),
        h: find(&|op| *op == TensorOp::MatMul),
        training,
    }
}

/// The gradient of `w` replaced by the forward read of `w` itself: right type, right output
/// layout, but a "gradient" that is tagged `Forward` and does not depend on the loss.
#[test]
fn verifier_refuses_a_gradient_that_does_not_depend_on_the_loss() {
    let mut l = linear();
    l.training.gradients.insert(l.w, l.w_read);
    l.training.program.outputs[1] = l.w_read;
    let result = verify_training(&l.training);
    assert!(
        result.is_err(),
        "verify_training accepted the parameter read {} as the gradient of {}: {result:?}",
        l.w_read,
        l.w
    );
}

/// A forward node tagged `Loss` makes a forward node tagged `Gradient` pass the "every Gradient
/// node depends on the loss" check, although neither depends on the loss value or the seed.
#[test]
fn verifier_refuses_a_gradient_node_whose_only_loss_dependency_is_a_mistagged_forward_node() {
    let mut l = linear();
    l.training.subgraphs.insert(l.w_read, Subgraph::Loss);
    l.training.subgraphs.insert(l.h, Subgraph::Gradient);
    let result = verify_training(&l.training);
    assert!(
        result.is_err(),
        "verify_training accepted forward matmul {} tagged Gradient because its operand {} was \
         tagged Loss: {result:?}",
        l.h,
        l.w_read
    );
}

/// The gradient of `w` is computed from the seed but tagged `Forward`.
#[test]
fn verifier_refuses_a_forward_tag_on_a_node_that_depends_on_the_seed() {
    let mut l = linear();
    let g = l.training.gradients[&l.w];
    l.training.subgraphs.insert(g, Subgraph::Forward);
    let result = verify_training(&l.training);
    assert!(
        result.is_err(),
        "verify_training accepted gradient {g}, computed from the backward seed, tagged Forward: \
         {result:?}"
    );
}

/// `w` is loss-reachable, but the training program has no gradient for it: the map and the
/// outputs agree with each other and not with the graph.
#[test]
fn verifier_refuses_a_missing_gradient_for_a_loss_reachable_parameter() {
    let mut l = linear();
    l.training.gradients.clear();
    l.training.program.outputs.truncate(1);
    let result = verify_training(&l.training);
    assert!(
        result.is_err(),
        "verify_training accepted a training program with no gradient for loss-reachable {}: \
         {result:?}",
        l.w
    );
}

/// The nodes `differentiate` added that no output depends on: an adjoint built for an input or
/// target ("targets receive no gradient", DESIGN.md § 4) and then dropped.
fn dead_backward_nodes(forward: &TensorProgram, training: &TrainingProgram) -> Vec<String> {
    let program = &training.program;
    let mut live: std::collections::BTreeSet<ValueId> = program.outputs.iter().copied().collect();
    for node in program.nodes.iter().rev() {
        if live.contains(&node.id) {
            live.extend(node.operands.iter().copied());
        }
    }
    program.nodes[forward.nodes.len()..]
        .iter()
        .filter(|n| !live.contains(&n.id))
        .map(|n| format!("{} = {:?}{:?}", n.id, n.op, n.operands))
        .collect()
}

/// Each binary op with a trainable operand and an input operand, in both orders and with the
/// input of rank 0: the pass builds nothing for the input.
#[test]
fn no_adjoint_is_built_for_an_input_operand() {
    type Build = fn(&mut TensorProgramBuilder, ValueId, ValueId) -> ValueId;
    let ops: [(&str, Build); 4] = [
        ("add", |b, l, r| b.add(l, r).unwrap()),
        ("sub", |b, l, r| b.sub(l, r).unwrap()),
        ("mul", |b, l, r| b.mul(l, r).unwrap()),
        ("mse", |b, l, r| b.mse(l, r).unwrap()),
    ];
    let mut dead = Vec::new();
    for (name, op) in ops {
        for input_shape in [&[2u64, 3][..], &[][..]] {
            if name == "mse" && input_shape.is_empty() {
                continue;
            }
            for param_first in [true, false] {
                let mut b = TensorProgramBuilder::new();
                let p = b
                    .declare_param("p", ty(&[2, 3]), Initializer::XavierUniform)
                    .unwrap();
                let p = b.param_read(p).unwrap();
                let x = b.input("x", ty(input_shape)).unwrap();
                let y = if param_first {
                    op(&mut b, p, x)
                } else {
                    op(&mut b, x, p)
                };
                let loss = if name == "mse" {
                    y
                } else {
                    let t = b.input("t", ty(&[2, 3])).unwrap();
                    b.mse(y, t).unwrap()
                };
                let forward = b.finish();
                let training = differentiate(&forward, loss).unwrap();
                let found = dead_backward_nodes(&forward, &training);
                if !found.is_empty() {
                    dead.push(format!(
                        "{name}(param first: {param_first}, input {input_shape:?}): {found:?}"
                    ));
                }
            }
        }
    }
    assert!(dead.is_empty(), "adjoints built for inputs: {dead:#?}");
}

/// A forward program that verifies, its value ids moved to the top of the `u32` range. Value ids
/// are opaque typed ids (DESIGN.md § 3.2); `differentiate` must return a result, not panic.
#[test]
fn differentiate_does_not_panic_on_the_largest_value_ids() {
    let (mut forward, loss) = linear_forward();
    let offset = u32::MAX - (forward.nodes.len() as u32 - 1);
    let moved = |v: ValueId| ValueId(v.0 + offset);
    for node in &mut forward.nodes {
        node.id = moved(node.id);
        for o in &mut node.operands {
            *o = moved(*o);
        }
    }
    for v in forward.inputs.iter_mut().chain(forward.outputs.iter_mut()) {
        *v = moved(*v);
    }
    modelspec_core::ir::verify::verify(&forward).unwrap();
    let loss = moved(loss);
    let outcome = catch_unwind(AssertUnwindSafe(|| differentiate(&forward, loss)));
    assert!(
        outcome.is_ok(),
        "differentiate panicked on a verified program whose largest value id is u32::MAX"
    );
}
