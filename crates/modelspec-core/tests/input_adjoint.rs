//! The input adjoint (DESIGN.md § 3.3 pass 4, § 4): [`differentiate_inputs`] and
//! [`output_element_target`], their refusals, the programs they return, and the training program
//! they leave as it was.
//!
//! Scenarios `observatory.adjoint.training-unchanged` and `observatory.adjoint.refusals` of
//! `story:observatory-input-adjoint`; the finite-difference checks are in
//! `crates/modelspec-runtime/tests/input_adjoint.rs`.

use modelspec_core::autodiff::{
    AutodiffError, InputAdjoint, InputAdjointError, OutputTarget, TARGET_MASK,
    differentiate_inputs, output_element_target,
};
use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::{
    Initializer, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::verify::{VerifyError, verify};

const LINEAR: &str = include_str!("../../../examples/linear_regression.mspec");
const TWO_LAYER: &str = include_str!("../../modelspec-docs/models/two_layer_relu.mspec");

/// A model whose forward does not read `x`: dead-code elimination removes the input.
const UNREAD_X: &str = "model Const {
    param b: Tensor<f32, [8, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 1]> {
        return b;
    }
}

train Fit {
    model = Const;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
";

fn compiled(name: &str, source: &str) -> CompiledTraining {
    compile_training(name, source).expect("the model compiles")
}

const SOURCES: [(&str, &str); 2] = [
    ("linear_regression.mspec", LINEAR),
    ("two_layer_relu.mspec", TWO_LAYER),
];

fn models() -> [(&'static str, CompiledTraining); 2] {
    SOURCES.map(|(name, source)| (name, compiled(name, source)))
}

fn loss_of(program: &TensorProgram) -> ValueId {
    let [loss] = program.outputs[..] else {
        panic!("the forward program has the loss as its one output")
    };
    loss
}

fn input_named(program: &TensorProgram, name: &str) -> ValueId {
    program
        .inputs
        .iter()
        .copied()
        .find(|&v| matches!(&program.node(v).unwrap().op, TensorOp::Input { name: n } if n == name))
        .unwrap_or_else(|| panic!("no input `{name}`"))
}

/// The layout every input adjoint has: the forward nodes kept as a prefix, the inputs and the
/// parameter table unchanged, outputs the target then one adjoint per name in the order asked,
/// each of its input's type; the program verifies.
fn assert_layout(forward: &TensorProgram, target: ValueId, names: &[&str], adjoint: &InputAdjoint) {
    let program = &adjoint.program;
    verify(program).expect("the input adjoint program verifies");
    assert_eq!(adjoint.target, target);
    assert_eq!(&program.nodes[..forward.nodes.len()], &forward.nodes[..]);
    assert_eq!(program.inputs, forward.inputs);
    assert_eq!(program.params, forward.params);
    let asked: Vec<&str> = adjoint.adjoints.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(asked, names);
    let mut outputs = vec![target];
    outputs.extend(adjoint.adjoints.iter().map(|&(_, v)| v));
    assert_eq!(program.outputs, outputs);
    for (name, value) in &adjoint.adjoints {
        let input = input_named(forward, name);
        assert_eq!(
            program.node(*value).unwrap().ty,
            forward.node(input).unwrap().ty,
            "adjoint of `{name}`"
        );
    }
}

// observatory.adjoint.training-unchanged

#[test]
fn observatory_adjoint_training_unchanged_outputs_are_loss_gradients_then_updates() {
    for (name, compiled) in models() {
        let step = &compiled.training;
        let training = &step.training;
        let mut outputs = vec![training.loss];
        outputs.extend(training.gradients.values().copied());
        outputs.extend(step.optimizer.updates.values().copied());
        assert_eq!(training.program.outputs, outputs, "{name}");
        assert_eq!(
            training.gradients.len(),
            compiled.forward.program.params.len(),
            "{name}: a gradient per parameter"
        );
        assert_eq!(
            training.loss,
            loss_of(&compiled.forward.program),
            "{name}: the loss is the forward program's output"
        );
    }
}

#[test]
fn observatory_adjoint_training_unchanged_by_asking_for_an_input_adjoint() {
    for (name, source) in SOURCES {
        let compiled = compiled(name, source);
        let forward = &compiled.forward.program;
        let target = output_element_target(forward, compiled.forward.prediction, &[0, 0])
            .expect("the target builds");
        differentiate_inputs(forward, loss_of(forward), &["x", "y"]).expect("the adjoint builds");
        differentiate_inputs(&target.program, target.target, &["x"]).expect("the adjoint builds");
        assert_eq!(
            compile_training(name, source).unwrap(),
            compiled,
            "{name}: compiling again gives the same step"
        );
    }
}

// observatory.adjoint.refusals

#[test]
fn observatory_adjoint_refusals_a_name_that_is_no_input() {
    let compiled = compiled("linear_regression.mspec", LINEAR);
    let forward = &compiled.forward.program;
    assert_eq!(
        differentiate_inputs(forward, loss_of(forward), &["x", "z"]),
        Err(InputAdjointError::UnknownInput { name: "z".into() })
    );
    assert_eq!(
        differentiate_inputs(forward, loss_of(forward), &["LinearReg.w"]),
        Err(InputAdjointError::UnknownInput {
            name: "LinearReg.w".into()
        }),
        "a parameter is no input"
    );
}

/// Open point 1: dead-code elimination drops an input the loss does not read, so a compiled
/// program has no `Input` for it and the name is refused, not zeroed.
#[test]
fn observatory_adjoint_refusals_a_declared_input_dce_removed() {
    let compiled = compiled("unread_x.mspec", UNREAD_X);
    let forward = &compiled.forward.program;
    assert!(
        compiled
            .forward
            .declared_inputs
            .iter()
            .any(|d| d.name == "x"),
        "the source declares x"
    );
    assert_eq!(
        differentiate_inputs(forward, loss_of(forward), &["x"]),
        Err(InputAdjointError::UnknownInput { name: "x".into() })
    );
}

#[test]
fn observatory_adjoint_refusals_a_target_that_is_not_rank_0() {
    let compiled = compiled("two_layer_relu.mspec", TWO_LAYER);
    let forward = &compiled.forward.program;
    let prediction = compiled.forward.prediction;
    assert_eq!(
        differentiate_inputs(forward, prediction, &["x"]),
        Err(InputAdjointError::TargetNotScalar {
            target: prediction,
            ty: TensorType::f32(&[16, 1]),
        })
    );
}

#[test]
fn observatory_adjoint_refusals_an_undefined_target() {
    let compiled = compiled("linear_regression.mspec", LINEAR);
    let forward = &compiled.forward.program;
    assert_eq!(
        differentiate_inputs(forward, ValueId(999), &["x"]),
        Err(InputAdjointError::UndefinedTarget {
            target: ValueId(999)
        })
    );
}

#[test]
fn observatory_adjoint_refusals_a_prediction_that_is_no_value() {
    let compiled = compiled("linear_regression.mspec", LINEAR);
    let forward = &compiled.forward.program;
    assert_eq!(
        output_element_target(forward, ValueId(999), &[0, 0]),
        Err(InputAdjointError::UndefinedPrediction {
            prediction: ValueId(999)
        })
    );
}

#[test]
fn observatory_adjoint_refusals_an_index_of_the_wrong_rank() {
    let compiled = compiled("two_layer_relu.mspec", TWO_LAYER);
    let forward = &compiled.forward.program;
    let prediction = compiled.forward.prediction;
    for index in [&[][..], &[0], &[0, 0, 0]] {
        assert_eq!(
            output_element_target(forward, prediction, index),
            Err(InputAdjointError::IndexRank {
                expected: 2,
                got: index.len()
            }),
            "{index:?}"
        );
    }
}

#[test]
fn observatory_adjoint_refusals_an_index_out_of_bounds() {
    let compiled = compiled("two_layer_relu.mspec", TWO_LAYER);
    let forward = &compiled.forward.program;
    let prediction = compiled.forward.prediction;
    for index in [[16, 0], [0, 1], [u64::MAX, u64::MAX]] {
        assert_eq!(
            output_element_target(forward, prediction, &index),
            Err(InputAdjointError::IndexOutOfBounds {
                index: index.to_vec(),
                ty: TensorType::f32(&[16, 1]),
            }),
            "{index:?}"
        );
    }
}

/// Open point 2: the builder's input name is fixed, so a program that already has an input of
/// that name is refused rather than given a second one or a renamed one.
#[test]
fn observatory_adjoint_refusals_a_program_with_an_input_named_target_mask() {
    let mut b = TensorProgramBuilder::new();
    let w = b
        .declare_param("w", TensorType::f32(&[4, 1]), Initializer::XavierUniform)
        .unwrap();
    let x = b.input(TARGET_MASK, TensorType::f32(&[8, 4])).unwrap();
    let r = b.param_read(w).unwrap();
    let h = b.matmul(x, r).unwrap();
    let loss = b.reduce_sum(h).unwrap();
    b.output(loss).unwrap();
    let forward = b.finish();
    assert_eq!(
        output_element_target(&forward, h, &[0, 0]),
        Err(InputAdjointError::MaskNameTaken)
    );
}

#[test]
fn observatory_adjoint_refusals_a_program_that_does_not_verify() {
    let compiled = compiled("linear_regression.mspec", LINEAR);
    let mut forward = compiled.forward.program.clone();
    let loss = loss_of(&forward);
    forward.outputs.push(ValueId(999));
    let refused = InputAdjointError::Verify(VerifyError::UndefinedOutput {
        output: ValueId(999),
    });
    assert_eq!(
        differentiate_inputs(&forward, loss, &["x"]),
        Err(refused.clone())
    );
    assert_eq!(
        output_element_target(&forward, compiled.forward.prediction, &[0, 0]),
        Err(refused)
    );
}

#[test]
fn observatory_adjoint_refusals_an_op_without_a_gradient_rule() {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[3])).unwrap();
    let g = b.input("g", TensorType::f32(&[3])).unwrap();
    let y = b.relu_grad(x, g).unwrap();
    let target = b.reduce_sum(y).unwrap();
    let forward = b.finish();
    assert_eq!(
        differentiate_inputs(&forward, target, &["x"]),
        Err(InputAdjointError::Autodiff(AutodiffError::NoGradientRule {
            node: y,
            op: "ReluGrad"
        }))
    );
}

#[test]
fn observatory_adjoint_refusals_value_ids_past_u32_max() {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[3])).unwrap();
    let t = b.reduce_sum(x).unwrap();
    let mut forward = b.finish();
    let shift = u32::MAX - 1;
    for node in &mut forward.nodes {
        node.id.0 += shift;
        node.operands.iter_mut().for_each(|v| v.0 += shift);
    }
    forward.inputs.iter_mut().for_each(|v| v.0 += shift);
    let (x, t) = (ValueId(x.0 + shift), ValueId(t.0 + shift));
    assert_eq!(
        differentiate_inputs(&forward, t, &["x"]),
        Err(InputAdjointError::ValueIdsExhausted)
    );
    assert_eq!(
        output_element_target(&forward, x, &[0]),
        Err(InputAdjointError::ValueIdsExhausted)
    );
}

// The programs the two functions return.

#[test]
fn observatory_adjoint_refusals_output_element_target_two_layer_relu_at_0_0() {
    let compiled = compiled("two_layer_relu.mspec", TWO_LAYER);
    let forward = &compiled.forward.program;
    let prediction = compiled.forward.prediction;
    let OutputTarget {
        program,
        target,
        mask,
        offset,
    } = output_element_target(forward, prediction, &[0, 0]).expect("the target builds");
    verify(&program).expect("the target program verifies");
    assert_eq!(offset, 0);
    assert_eq!(program.node(target).unwrap().ty, TensorType::f32_scalar());
    assert_eq!(program.outputs, vec![target]);
    let mask_node = program.node(mask).unwrap();
    assert_eq!(
        mask_node.op,
        TensorOp::Input {
            name: TARGET_MASK.into()
        }
    );
    assert_eq!(mask_node.ty, forward.node(prediction).unwrap().ty);
    assert_eq!(input_named(&program, TARGET_MASK), mask);
    let mut inputs = forward.inputs.clone();
    inputs.push(mask);
    assert_eq!(program.inputs, inputs);
    assert_eq!(&program.nodes[..forward.nodes.len()], &forward.nodes[..]);

    let tail: Vec<(&TensorOp, &[ValueId])> = program.nodes[forward.nodes.len()..]
        .iter()
        .map(|n| (&n.op, &n.operands[..]))
        .collect();
    let product = program.nodes[forward.nodes.len() + 1].id;
    assert_eq!(
        tail,
        vec![
            (
                &TensorOp::Input {
                    name: TARGET_MASK.into()
                },
                &[][..]
            ),
            (&TensorOp::Mul, &[prediction, mask][..]),
            (&TensorOp::ReduceSum, &[product][..]),
        ],
        "from existing ops only"
    );

    let adjoint = differentiate_inputs(&program, target, &["x"]).expect("the adjoint builds");
    assert_layout(&program, target, &["x"], &adjoint);
}

#[test]
fn observatory_adjoint_refusals_the_offset_is_row_major() {
    let compiled = compiled("two_layer_relu.mspec", TWO_LAYER);
    let forward = &compiled.forward.program;
    let prediction = compiled.forward.prediction;
    let at = |index: &[u64]| {
        output_element_target(forward, prediction, index)
            .unwrap()
            .offset
    };
    assert_eq!(at(&[3, 0]), 3);
    assert_eq!(at(&[15, 0]), 15);

    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[2, 3, 4])).unwrap();
    let forward = b.finish();
    let target = output_element_target(&forward, x, &[1, 2, 3]).unwrap();
    assert_eq!(target.offset, 12 + 2 * 4 + 3);
    verify(&target.program).unwrap();
}

#[test]
fn observatory_adjoint_refusals_differentiate_inputs_layout_for_the_loss() {
    for (name, compiled) in models() {
        let forward = &compiled.forward.program;
        let loss = loss_of(forward);
        for names in [&["x"][..], &["y", "x"], &["x", "x"], &[]] {
            let adjoint = differentiate_inputs(forward, loss, names)
                .unwrap_or_else(|e| panic!("{name}: {names:?}: {e}"));
            assert_layout(forward, loss, names, &adjoint);
        }
    }
}

/// Open point 3: the `Mse` rule gives its operand `B` an adjoint exactly when `B` is asked for.
#[test]
fn observatory_adjoint_refusals_the_mse_target_gets_an_adjoint_only_when_asked() {
    let compiled = compiled("linear_regression.mspec", LINEAR);
    let forward = &compiled.forward.program;
    let loss = loss_of(forward);
    let x_only = differentiate_inputs(forward, loss, &["x"]).unwrap();
    let both = differentiate_inputs(forward, loss, &["x", "y"]).unwrap();
    let negations = |a: &InputAdjoint| {
        a.program.nodes[forward.nodes.len()..]
            .iter()
            .filter(|n| n.op == TensorOp::Scalar(-1.0))
            .count()
    };
    assert_eq!(negations(&x_only), 0, "no -dA for the target y");
    assert_eq!(negations(&both), 1, "-dA for the target y");
    let backward = |a: &InputAdjoint| a.program.nodes.len() - forward.nodes.len();
    assert_eq!(
        backward(&x_only),
        7,
        "seed, the mse rule for the prediction (Sub, Scalar, Mul, Mul) and the matmul rule for \
         x (Transpose, MatMul)\n{}",
        x_only.program
    );
}

/// An input the target does not read is not refused: its adjoint is zeros of its shape.
#[test]
fn observatory_adjoint_refusals_none_for_an_unread_input() {
    let mut b = TensorProgramBuilder::new();
    let a = b.input("a", TensorType::f32(&[3, 4])).unwrap();
    let _c = b.input("c", TensorType::f32(&[2])).unwrap();
    let _s = b.input("s", TensorType::f32_scalar()).unwrap();
    let target = b.reduce_sum(a).unwrap();
    let forward = b.finish();
    let adjoint = differentiate_inputs(&forward, target, &["c", "a", "s"]).unwrap();
    assert_layout(&forward, target, &["c", "a", "s"], &adjoint);
    let c_adjoint = adjoint.program.node(adjoint.adjoints[0].1).unwrap();
    assert_eq!(
        c_adjoint.op,
        TensorOp::BroadcastScalar {
            shape: TensorType::f32(&[2]).shape
        }
    );
    let zero = adjoint.program.node(c_adjoint.operands[0]).unwrap();
    assert_eq!(zero.op, TensorOp::Scalar(0.0));
    let s_adjoint = adjoint.program.node(adjoint.adjoints[2].1).unwrap();
    assert_eq!(
        s_adjoint.op,
        TensorOp::Scalar(0.0),
        "rank 0: the zero itself"
    );
}
