//! Finite-difference gradient checks (DESIGN.md § 4, § 9 gate 3).
//!
//! Each case builds a forward program, differentiates it with
//! [`modelspec_core::autodiff::differentiate`] and runs the resulting training program on the CPU
//! interpreter. Its gradients, evaluated in `f64` and in `f32`, are compared with central finite
//! differences of the loss evaluated in `f64`. Inputs and parameters are drawn from a fixed
//! test-local seed, at least 0.1 away from zero, and every relu operand is kept at least
//! [`KINK_MARGIN`] away from the kink. The harness, its tolerances and its generator are
//! [`finite_difference`], which `tests/gradients_embedding.rs` shares.
//!
//! The error of a gradient is `max_i |g_i - d_i| / max_i |d_i|` over its elements, `d` the
//! differences. A wrong rule (a dropped transpose, a missing factor 2 in `mse`, a missing `1/n` in
//! `mean`, an adjoint counted once instead of per use) is off by a factor of order one, far above
//! either tolerance.

use std::collections::BTreeMap;

use modelspec_core::autodiff::{AutodiffError, differentiate};
use modelspec_core::ir::tensor::{
    Dim, Initializer, ParamId, TensorNode, TensorOp, TensorProgram, TensorProgramBuilder,
    TensorType, ValueId,
};
use modelspec_core::ir::training::{Subgraph, TrainingProgram};
use modelspec_core::ir::verify::{TrainingVerifyError, VerifyError, verify_training};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{
    H, Ids, KINK_MARGIN, MIN_SCALE, Point, Rng, TOL_F32, TOL_F64, assert_structure, gradient_check,
    input, kink_distance, loss_at, mse_to_target, param, relative_error, run, seed_of, ty,
};

// Each M0 op alone.

#[test]
fn add_same_shape() {
    gradient_check("add_same_shape", |b| {
        let p = param(b, "p", &[3, 4]);
        let q = param(b, "q", &[3, 4]);
        let y = b.add(p, q).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn add_rank0_rhs() {
    gradient_check("add_rank0_rhs", |b| {
        let p = param(b, "p", &[2, 3]);
        let s = param(b, "s", &[]);
        let y = b.add(p, s).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn add_rank0_lhs() {
    gradient_check("add_rank0_lhs", |b| {
        let s = param(b, "s", &[]);
        let p = param(b, "p", &[2, 3]);
        let y = b.add(s, p).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn sub_same_shape() {
    gradient_check("sub_same_shape", |b| {
        let p = param(b, "p", &[3, 4]);
        let q = param(b, "q", &[3, 4]);
        let y = b.sub(p, q).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn sub_rank0_rhs() {
    gradient_check("sub_rank0_rhs", |b| {
        let p = param(b, "p", &[2, 3]);
        let s = param(b, "s", &[]);
        let y = b.sub(p, s).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn sub_rank0_lhs() {
    gradient_check("sub_rank0_lhs", |b| {
        let s = param(b, "s", &[]);
        let p = param(b, "p", &[2, 3]);
        let y = b.sub(s, p).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn mul_same_shape() {
    gradient_check("mul_same_shape", |b| {
        let p = param(b, "p", &[3, 4]);
        let q = param(b, "q", &[3, 4]);
        let y = b.mul(p, q).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn mul_rank0_rhs() {
    gradient_check("mul_rank0_rhs", |b| {
        let p = param(b, "p", &[2, 3]);
        let s = param(b, "s", &[]);
        let y = b.mul(p, s).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn mul_rank0_lhs() {
    gradient_check("mul_rank0_lhs", |b| {
        let s = param(b, "s", &[]);
        let p = param(b, "p", &[2, 3]);
        let y = b.mul(s, p).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn mul_both_rank0() {
    gradient_check("mul_both_rank0", |b| {
        let s = param(b, "s", &[]);
        let r = param(b, "r", &[]);
        let y = b.mul(s, r).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn matmul_non_square() {
    gradient_check("matmul_non_square", |b| {
        let a = param(b, "a", &[3, 4]);
        let c = param(b, "c", &[4, 2]);
        let y = b.matmul(a, c).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn relu() {
    gradient_check("relu", |b| {
        let p = param(b, "p", &[3, 4]);
        let y = b.relu(p).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn mean() {
    gradient_check("mean", |b| {
        let p = param(b, "p", &[3, 5]);
        let y = b.reduce_mean(p).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn sum() {
    gradient_check("sum", |b| {
        let p = param(b, "p", &[3, 5]);
        let y = b.reduce_sum(p).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn transpose() {
    gradient_check("transpose", |b| {
        let p = param(b, "p", &[2, 3]);
        let y = b.transpose(p).unwrap();
        mse_to_target(b, y)
    });
}

/// `mse` as the loss itself: its incoming adjoint is the seed.
#[test]
fn mse_as_loss() {
    gradient_check("mse_as_loss", |b| {
        let p = param(b, "p", &[3, 5]);
        mse_to_target(b, p)
    });
}

/// `mse` under a scale, so its rule must use the incoming adjoint.
#[test]
fn mse_scaled() {
    gradient_check("mse_scaled", |b| {
        let p = param(b, "p", &[3, 5]);
        let m = mse_to_target(b, p);
        let c = input(b, "c", &[]);
        b.mul(m, c).unwrap()
    });
}

/// A trainable second operand receives `-dY`; the targets of the other cases are inputs and
/// receive nothing.
#[test]
fn mse_both_operands() {
    gradient_check("mse_both_operands", |b| {
        let p = param(b, "p", &[3, 5]);
        let q = param(b, "q", &[3, 5]);
        b.mse(p, q).unwrap()
    });
}

// Complete graphs.

/// `mse(matmul(x, w), y)` with the shapes of `examples/linear_regression.mspec`.
fn linear_graph(b: &mut TensorProgramBuilder) -> ValueId {
    let w = param(b, "LinearReg.w", &[4, 1]);
    let x = input(b, "x", &[8, 4]);
    let y = input(b, "y", &[8, 1]);
    let h = b.matmul(x, w).unwrap();
    b.mse(h, y).unwrap()
}

#[test]
fn linear_regression() {
    let training = gradient_check("linear_regression", linear_graph);
    assert_eq!(training.gradients.len(), 1, "a gradient for w only");
    assert_eq!(training.program.outputs.len(), 2, "loss and one gradient");
    let x_type = ty(&[8, 4]);
    let backward: Vec<_> = training.program.nodes[5..]
        .iter()
        .filter(|n| training.subgraphs[&n.id] != Subgraph::Loss)
        .collect();
    for node in &backward {
        assert_ne!(node.ty, x_type, "{}: an adjoint for input x", node.id);
    }
    assert_eq!(
        backward.len(),
        6,
        "the mse rule for the prediction only (Sub, Scalar, Mul, Mul) and the matmul rule for w \
         only (Transpose, MatMul): nothing for input x or target y\n{}",
        training.program
    );
}

#[test]
fn two_layer_relu() {
    gradient_check("two_layer_relu", |b| {
        let w1 = param(b, "w1", &[4, 5]);
        let w2 = param(b, "w2", &[5, 2]);
        let x = input(b, "x", &[8, 4]);
        let y = input(b, "y", &[8, 2]);
        let h = b.matmul(x, w1).unwrap();
        let a = b.relu(h).unwrap();
        let o = b.matmul(a, w2).unwrap();
        b.mse(o, y).unwrap()
    });
}

/// Coordinates of each parameter checked by [`mnist_mlp_sampled`].
const SAMPLES: usize = 16;

/// The complete graph of `examples/mnist_mlp.mspec`, compiled from the file, checked at
/// [`SAMPLES`] coordinates of each parameter drawn from the case's seed. A full check would take
/// 2 x 50,816 forward runs of the 784 -> 64 -> 10 model at batch 100.
///
/// The point is drawn as in every other case, then each row of `x` is redrawn until all 64 of its
/// hidden pre-activations are [`KINK_MARGIN`] away from 0: a whole-point redraw would succeed
/// about once in 135 tries (6,400 operands, each within the margin with probability about 8e-4).
/// [`kink_distance`] then checks the result on the training program itself.
#[test]
fn mnist_mlp_sampled() {
    let name = "mnist_mlp_sampled";
    let path = "examples/mnist_mlp.mspec";
    let source = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path),
    )
    .expect("the example exists");
    let forward = modelspec_core::compile::compile_tensor(path, &source)
        .expect("the example lowers")
        .program;
    let [loss] = forward.outputs[..] else {
        panic!("the loss is the one output: {:?}", forward.outputs)
    };
    let training = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);
    let id_of = |path: &str| {
        forward
            .params
            .iter()
            .find(|(_, decl)| decl.path == path)
            .map(|(&id, _)| id)
            .unwrap_or_else(|| panic!("no parameter {path}"))
    };
    let (w1, w2) = (id_of("MnistMlp.w1"), id_of("MnistMlp.w2"));
    assert_eq!(
        training.gradients.keys().copied().collect::<Vec<_>>(),
        vec![w1, w2],
        "a gradient per parameter"
    );

    let mut rng = Rng::new(seed_of(name));
    let mut point = Point::draw(&forward, &Ids::new(), &mut rng);
    let (inputs, hidden) = match &point.params[&w1].0[..] {
        &[i, h] => (i as usize, h as usize),
        shape => panic!("w1 shape {shape:?}"),
    };
    let w1_values: Vec<f64> = point.params[&w1].1.iter().map(|&v| v.into()).collect();
    let x = &mut point.inputs.get_mut("x").expect("input x").1;
    for row in x.chunks_mut(inputs) {
        let clear = |row: &[f32]| {
            (0..hidden).all(|j| {
                let h: f64 = (0..inputs)
                    .map(|i| f64::from(row[i]) * w1_values[i * hidden + j])
                    .sum();
                h.abs() >= KINK_MARGIN
            })
        };
        let mut tries = 0;
        while !clear(row) {
            tries += 1;
            assert!(
                tries < 1000,
                "{name}: no row of x away from every relu kink"
            );
            row.iter_mut().for_each(|v| *v = rng.value());
        }
    }
    assert!(
        kink_distance(&forward, &training, &point) >= KINK_MARGIN,
        "{name}: a relu operand within {KINK_MARGIN} of the kink"
    );

    let in_f64 = run(&training, &point, f64::from);
    let in_f32 = run(&training, &point, |v| v);
    for (k, (&param, _)) in training.gradients.iter().enumerate() {
        let path = &forward.params[&param].path;
        let n = point.params[&param].1.len();
        let mut coordinates = Vec::new();
        while coordinates.len() < SAMPLES {
            let i = (rng.next_u64() % n as u64) as usize;
            if !coordinates.contains(&i) {
                coordinates.push(i);
            }
        }
        let differences: Vec<f64> = coordinates
            .iter()
            .map(|&i| {
                (loss_at(&forward, &point, param, i, H) - loss_at(&forward, &point, param, i, -H))
                    / (2.0 * H)
            })
            .collect();
        let scale = differences.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            scale >= MIN_SCALE,
            "{name}: {path}: differences {differences:?} too small to compare"
        );
        let g64: Vec<f64> = coordinates
            .iter()
            .map(|&i| in_f64[1 + k].data()[i])
            .collect();
        let g32: Vec<f64> = coordinates
            .iter()
            .map(|&i| f64::from(in_f32[1 + k].data()[i]))
            .collect();
        let e64 = relative_error(&g64, &differences);
        let e32 = relative_error(&g32, &differences);
        println!("{name}: {path} at {coordinates:?}: f64 error {e64:e}, f32 error {e32:e}");
        assert!(
            e64 <= TOL_F64,
            "{name}: {path}: f64 error {e64:e}; gradient {g64:?}, differences {differences:?}"
        );
        assert!(
            e32 <= TOL_F32,
            "{name}: {path}: f32 error {e32:e}; gradient {g32:?}, differences {differences:?}"
        );
    }
}

// Shared uses: every incoming adjoint accumulated exactly once.

#[test]
fn shared_mul_a_a() {
    gradient_check("shared_mul_a_a", |b| {
        let a = param(b, "a", &[3, 4]);
        let y = b.mul(a, a).unwrap();
        mse_to_target(b, y)
    });
}

#[test]
fn shared_add_h_h() {
    gradient_check("shared_add_h_h", |b| {
        let w = param(b, "w", &[4, 3]);
        let x = input(b, "x", &[5, 4]);
        let h = b.matmul(x, w).unwrap();
        let y = b.add(h, h).unwrap();
        mse_to_target(b, y)
    });
}

/// One value used by two different consumers.
#[test]
fn shared_fan_out() {
    gradient_check("shared_fan_out", |b| {
        let w = param(b, "w", &[4, 3]);
        let x = input(b, "x", &[5, 4]);
        let h = b.matmul(x, w).unwrap();
        let r = b.relu(h).unwrap();
        let y = b.add(h, r).unwrap();
        mse_to_target(b, y)
    });
}

/// Two reads of one parameter: its gradient is the sum over both.
#[test]
fn shared_parameter_read_twice() {
    gradient_check("shared_parameter_read_twice", |b| {
        let id = b
            .declare_param("p", ty(&[3, 4]), Initializer::XavierUniform)
            .unwrap();
        let r1 = b.param_read(id).unwrap();
        let r2 = b.param_read(id).unwrap();
        let y = b.mul(r1, r2).unwrap();
        mse_to_target(b, y)
    });
}

// Relu at its kink.

/// `d relu(x) / dx` at exactly 0 is 0 (DESIGN.md § 4), in `f32` and `f64`.
#[test]
fn relu_gradient_at_zero_is_zero() {
    let mut b = TensorProgramBuilder::new();
    let id = b
        .declare_param("p", ty(&[5]), Initializer::XavierUniform)
        .unwrap();
    let p = b.param_read(id).unwrap();
    let r = b.relu(p).unwrap();
    let loss = b.reduce_sum(r).unwrap();
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);
    let values = [0.0f32, 0.5, -0.5, -0.0, 2.0];
    let point = Point {
        inputs: BTreeMap::new(),
        ids: Ids::new(),
        params: BTreeMap::from([(id, (vec![5], values.to_vec()))]),
    };
    let want = [0.0, 1.0, 0.0, 0.0, 1.0];
    let g64 = run(&training, &point, f64::from).remove(1);
    assert_eq!(g64.data(), &want);
    let g32 = run(&training, &point, |v| v).remove(1);
    let want32: Vec<f32> = want.iter().map(|&v| v as f32).collect();
    assert_eq!(g32.data(), &want32[..]);
}

// What receives a gradient.

#[test]
fn unreachable_parameter_receives_no_gradient() {
    let mut b = TensorProgramBuilder::new();
    let p = param(&mut b, "p", &[3]);
    let q = param(&mut b, "q", &[3]);
    let _unused = b.add(q, q).unwrap();
    let loss = mse_to_target(&mut b, p);
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);
    assert_eq!(
        training.gradients.keys().copied().collect::<Vec<_>>(),
        vec![ParamId(0)]
    );
}

#[test]
fn loss_that_is_a_parameter_read_has_the_seed_as_its_gradient() {
    let mut b = TensorProgramBuilder::new();
    let loss = param(&mut b, "s", &[]);
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);
    let g = training.gradients[&ParamId(0)];
    assert_eq!(
        training.subgraphs[&g],
        Subgraph::Loss,
        "the gradient is the seed"
    );
    assert_eq!(training.program.node(g).unwrap().op, TensorOp::Scalar(1.0));
}

#[test]
fn loss_without_parameters_has_no_gradient_nodes() {
    let mut b = TensorProgramBuilder::new();
    let x = input(&mut b, "x", &[3]);
    let loss = mse_to_target(&mut b, x);
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);
    assert!(training.gradients.is_empty());
    assert_eq!(training.program.nodes.len(), forward.nodes.len());
    assert_eq!(training.program.outputs, vec![loss]);
}

// Refusals of the pass.

#[test]
fn differentiate_refuses_an_undefined_loss() {
    let mut b = TensorProgramBuilder::new();
    linear_graph(&mut b);
    let forward = b.finish();
    assert_eq!(
        differentiate(&forward, ValueId(99)),
        Err(AutodiffError::UndefinedLoss { loss: ValueId(99) })
    );
}

#[test]
fn differentiate_refuses_a_non_scalar_loss() {
    let mut b = TensorProgramBuilder::new();
    let p = param(&mut b, "p", &[3]);
    let r = b.relu(p).unwrap();
    let forward = b.finish();
    assert_eq!(
        differentiate(&forward, r),
        Err(AutodiffError::LossNotScalar {
            loss: r,
            ty: ty(&[3])
        })
    );
}

#[test]
fn differentiate_refuses_a_program_that_does_not_verify() {
    let mut b = TensorProgramBuilder::new();
    let loss = linear_graph(&mut b);
    let mut forward = b.finish();
    forward.outputs.push(ValueId(99));
    assert_eq!(
        differentiate(&forward, loss),
        Err(AutodiffError::Verify(VerifyError::UndefinedOutput {
            output: ValueId(99)
        }))
    );
}

#[test]
fn differentiate_refuses_internal_gradient_ops() {
    let mut b = TensorProgramBuilder::new();
    let p = param(&mut b, "p", &[3]);
    let g = input(&mut b, "g", &[3]);
    let y = b.relu_grad(p, g).unwrap();
    let loss = mse_to_target(&mut b, y);
    let forward = b.finish();
    assert_eq!(
        differentiate(&forward, loss),
        Err(AutodiffError::NoGradientRule {
            node: y,
            op: "ReluGrad"
        })
    );

    let mut b = TensorProgramBuilder::new();
    let s = param(&mut b, "s", &[]);
    let y = b.broadcast_scalar(s, vec![Dim::Static(3)]).unwrap();
    let loss = mse_to_target(&mut b, y);
    let forward = b.finish();
    assert_eq!(
        differentiate(&forward, loss),
        Err(AutodiffError::NoGradientRule {
            node: y,
            op: "BroadcastScalar"
        })
    );
}

/// `forward` with every value id moved up by `offset`.
fn moved_ids(forward: &TensorProgram, offset: u32) -> TensorProgram {
    let mut moved = forward.clone();
    let shift = |v: &mut ValueId| v.0 += offset;
    for node in &mut moved.nodes {
        shift(&mut node.id);
        node.operands.iter_mut().for_each(shift);
    }
    moved.inputs.iter_mut().for_each(shift);
    moved.outputs.iter_mut().for_each(shift);
    moved
}

#[test]
fn differentiate_refuses_value_ids_past_u32_max() {
    let mut b = TensorProgramBuilder::new();
    let loss = linear_graph(&mut b);
    let forward = b.finish();
    let backward_nodes = differentiate(&forward, loss).unwrap().program.nodes.len() as u32
        - forward.nodes.len() as u32;
    let largest = forward.nodes.iter().map(|n| n.id.0).max().unwrap();

    let room = u32::MAX - largest - backward_nodes;
    let fits = moved_ids(&forward, room);
    let training = differentiate(&fits, ValueId(loss.0 + room)).unwrap();
    assert_eq!(
        training.program.nodes.iter().map(|n| n.id.0).max(),
        Some(u32::MAX),
        "the last backward node takes u32::MAX"
    );

    for short_by in [1, 2, backward_nodes] {
        let offset = room + short_by;
        let tight = moved_ids(&forward, offset);
        assert_eq!(
            differentiate(&tight, ValueId(loss.0 + offset)),
            Err(AutodiffError::ValueIdsExhausted),
            "{short_by} ids short"
        );
    }
}

// Training IR verifier: one case per check.

struct Linear {
    training: TrainingProgram,
    w: ParamId,
    w_read: ValueId,
    x: ValueId,
    h: ValueId,
}

/// The linear-regression training program and the values the cases corrupt.
fn linear() -> Linear {
    let mut b = TensorProgramBuilder::new();
    let loss = linear_graph(&mut b);
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    let find =
        |op: &dyn Fn(&TensorOp) -> bool| forward.nodes.iter().find(|n| op(&n.op)).unwrap().id;
    Linear {
        w: ParamId(0),
        w_read: find(&|op| matches!(op, TensorOp::ParamRead { .. })),
        x: find(&|op| matches!(op, TensorOp::Input { name } if name == "x")),
        h: find(&|op| *op == TensorOp::MatMul),
        training,
    }
}

#[test]
fn verifier_checks_each_gradient_has_its_parameter_type() {
    let mut l = linear();
    l.training.gradients.insert(l.w, l.x);
    l.training.program.outputs[1] = l.x;
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::GradientType {
            param: l.w,
            expected: ty(&[4, 1]),
            got: ty(&[8, 4]),
        })
    );
}

#[test]
fn verifier_checks_each_gradient_node_depends_on_the_loss() {
    let mut l = linear();
    l.training.subgraphs.insert(l.h, Subgraph::Gradient);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::GradientIndependentOfLoss { node: l.h })
    );

    let mut l = linear();
    l.training.subgraphs.insert(l.w_read, Subgraph::Gradient);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::GradientIndependentOfLoss { node: l.w_read })
    );

    let mut l = linear();
    let constant = l.training.program.nodes[5..]
        .iter()
        .find(|n| matches!(n.op, TensorOp::Scalar(v) if v != 1.0))
        .expect("the mse rule's 2 / numel constant")
        .id;
    l.training.subgraphs.insert(constant, Subgraph::Gradient);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::GradientIndependentOfLoss { node: constant })
    );
}

#[test]
fn verifier_checks_the_tensor_program() {
    let mut l = linear();
    l.training.program.outputs.push(ValueId(999));
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::Tensor(VerifyError::UndefinedOutput {
            output: ValueId(999)
        }))
    );
}

#[test]
fn verifier_checks_every_node_is_tagged() {
    let mut l = linear();
    l.training.subgraphs.remove(&l.h);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::UntaggedNode { node: l.h })
    );
}

#[test]
fn verifier_checks_every_tag_names_a_node() {
    let mut l = linear();
    l.training.subgraphs.insert(ValueId(999), Subgraph::Forward);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::TagWithoutNode {
            value: ValueId(999)
        })
    );
}

#[test]
fn verifier_checks_the_loss_is_defined() {
    let mut l = linear();
    l.training.loss = ValueId(999);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::UndefinedLoss { loss: ValueId(999) })
    );
}

#[test]
fn verifier_checks_the_loss_is_in_the_loss_subgraph() {
    let mut l = linear();
    let loss = l.training.loss;
    l.training.subgraphs.insert(loss, Subgraph::Forward);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::LossNotInLossSubgraph {
            loss,
            tag: Subgraph::Forward
        })
    );
}

#[test]
fn verifier_checks_the_loss_is_scalar() {
    let mut l = linear();
    l.training.loss = l.h;
    l.training.subgraphs.insert(l.h, Subgraph::Loss);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::LossNotScalar {
            loss: l.h,
            ty: ty(&[8, 1])
        })
    );
}

#[test]
fn verifier_checks_each_gradient_parameter_is_declared() {
    let mut l = linear();
    let g = l.training.gradients[&l.w];
    l.training.gradients.insert(ParamId(7), g);
    l.training.program.outputs.push(g);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::GradientForUndeclaredParam { param: ParamId(7) })
    );
}

#[test]
fn verifier_checks_each_gradient_is_defined() {
    let mut l = linear();
    l.training.gradients.insert(l.w, ValueId(999));
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::UndefinedGradient {
            param: l.w,
            value: ValueId(999)
        })
    );
}

#[test]
fn verifier_checks_only_the_loss_and_the_seed_are_tagged_loss() {
    let mut l = linear();
    l.training.subgraphs.insert(l.w_read, Subgraph::Loss);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::StrayLossTag { node: l.w_read })
    );

    let mut l = linear();
    let second = ValueId(900);
    l.training.program.nodes.push(TensorNode {
        id: second,
        op: TensorOp::Scalar(1.0),
        operands: Vec::new(),
        ty: TensorType::f32_scalar(),
        source_span: None,
    });
    l.training.subgraphs.insert(second, Subgraph::Loss);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::StrayLossTag { node: second })
    );
}

#[test]
fn verifier_checks_each_node_depending_on_the_seed_is_tagged_gradient() {
    let mut l = linear();
    let g = l.training.gradients[&l.w];
    l.training.subgraphs.insert(g, Subgraph::Forward);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::ForwardDependsOnSeed { node: g })
    );
}

#[test]
fn verifier_checks_each_gradient_depends_on_the_seed() {
    let mut l = linear();
    l.training.gradients.insert(l.w, l.w_read);
    l.training.program.outputs[1] = l.w_read;
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::GradientNotFromSeed {
            param: l.w,
            value: l.w_read
        })
    );
}

#[test]
fn verifier_checks_each_parameter_the_loss_reads_has_a_gradient() {
    let mut l = linear();
    l.training.gradients.clear();
    l.training.program.outputs.truncate(1);
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::MissingGradient { param: l.w })
    );
}

#[test]
fn verifier_checks_no_gradient_is_given_for_a_parameter_the_loss_does_not_read() {
    let mut b = TensorProgramBuilder::new();
    let p = param(&mut b, "p", &[3]);
    param(&mut b, "q", &[3]);
    let loss = mse_to_target(&mut b, p);
    let forward = b.finish();
    let mut training = differentiate(&forward, loss).unwrap();
    let g = training.gradients[&ParamId(0)];
    training.gradients.insert(ParamId(1), g);
    training.program.outputs.push(g);
    assert_eq!(
        verify_training(&training),
        Err(TrainingVerifyError::GradientForUnreachableParam { param: ParamId(1) })
    );
}

#[test]
fn verifier_checks_the_output_layout() {
    let mut l = linear();
    let expected = l.training.program.outputs.clone();
    l.training.program.outputs.reverse();
    assert_eq!(
        verify_training(&l.training),
        Err(TrainingVerifyError::OutputLayout {
            expected: expected.clone(),
            got: l.training.program.outputs.clone(),
        })
    );
}
