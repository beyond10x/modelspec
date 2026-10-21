//! Adversary cases for the Tensor IR core: invariants the builder enforces that the verifier must
//! also enforce for a program made or changed by other means, and registry and printer rules the
//! unit's own suite leaves unpinned.

use modelspec_core::ir::print::print;
use modelspec_core::ir::tensor::{
    BuildError, Dim, Initializer, ParamDecl, ParamId, TensorOp, TensorProgram,
    TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::verify::verify;
use modelspec_core::ops::OpError;

fn f32(shape: &[u64]) -> TensorType {
    TensorType::f32(shape)
}

/// Two parameters, each read once, summed.
fn two_params() -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let w1 = b
        .declare_param("M.a.w", f32(&[2, 2]), Initializer::XavierUniform)
        .unwrap();
    let w2 = b
        .declare_param("M.b.w", f32(&[2, 2]), Initializer::XavierUniform)
        .unwrap();
    let r1 = b.param_read(w1).unwrap();
    let r2 = b.param_read(w2).unwrap();
    let s = b.add(r1, r2).unwrap();
    b.output(s).unwrap();
    b.finish()
}

/// `x + y` over two named inputs.
fn two_inputs() -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", f32(&[2])).unwrap();
    let y = b.input("y", f32(&[2])).unwrap();
    let s = b.add(x, y).unwrap();
    b.output(s).unwrap();
    b.finish()
}

#[test]
fn adv_verifier_rejects_two_parameters_with_one_canonical_path() {
    let mut program = two_params();
    program.params.insert(
        ParamId(1),
        ParamDecl {
            path: "M.a.w".to_owned(),
            ty: f32(&[2, 2]),
            init: Initializer::XavierUniform,
        },
    );
    assert!(
        verify(&program).is_err(),
        "two ParamIds share canonical path `M.a.w` (DESIGN.md § 2.3: stable canonical path, no \
         accidental aliasing; the builder refuses it as DuplicateParamPath) yet verify accepted:\n{}",
        print(&program)
    );
}

#[test]
fn adv_verifier_rejects_two_input_nodes_with_one_name() {
    let mut program = two_inputs();
    program.nodes[1].op = TensorOp::Input {
        name: "x".to_owned(),
    };
    assert!(
        verify(&program).is_err(),
        "two Input nodes are named `x` (the builder refuses it as DuplicateInput) yet verify \
         accepted:\n{}",
        print(&program)
    );
}

#[test]
fn adv_verifier_rejects_input_node_missing_from_inputs_list() {
    let mut program = two_inputs();
    program.inputs.retain(|&v| v != ValueId(1));
    assert!(
        verify(&program).is_err(),
        "Input node %1 is absent from `inputs` (documented as the program's Input nodes) yet \
         verify accepted: inputs = {:?}",
        program.inputs
    );
}

#[test]
fn adv_verifier_rejects_inputs_list_naming_one_node_twice() {
    let mut program = two_inputs();
    program.inputs = vec![ValueId(0), ValueId(0), ValueId(1)];
    assert!(
        verify(&program).is_err(),
        "`inputs` lists %0 twice yet verify accepted: inputs = {:?}",
        program.inputs
    );
}

#[test]
fn adv_relu_grad_rejects_operands_of_different_shapes() {
    let mut b = TensorProgramBuilder::new();
    let a = b.input("a", f32(&[2, 3])).unwrap();
    let g = b.input("g", f32(&[3, 2])).unwrap();
    assert_eq!(
        b.relu_grad(a, g),
        Err(BuildError::Op(OpError::ShapeMismatch {
            op: "ReluGrad",
            lhs: f32(&[2, 3]),
            rhs: f32(&[3, 2]),
        }))
    );
}

#[test]
fn adv_matmul_rejects_rank3_rhs() {
    let mut b = TensorProgramBuilder::new();
    let a = b.input("a", f32(&[2, 3])).unwrap();
    let c = b.input("c", f32(&[3, 4, 5])).unwrap();
    assert_eq!(
        b.matmul(a, c),
        Err(BuildError::Op(OpError::Rank {
            op: "MatMul",
            expected: 2,
            got: f32(&[3, 4, 5]),
        }))
    );
}

#[test]
fn adv_printer_pins_scalar_and_broadcast_scalar_lines() {
    let mut b = TensorProgramBuilder::new();
    let s = b.scalar(0.5);
    let bs = b
        .broadcast_scalar(s, vec![Dim::Static(2), Dim::Static(3)])
        .unwrap();
    b.output(bs).unwrap();
    let program = b.finish();
    assert_eq!(verify(&program), Ok(()));
    assert_eq!(
        print(&program),
        "%0: Tensor<f32, []> = Scalar 0.5\n\
         %1: Tensor<f32, [2, 3]> = BroadcastScalar [2, 3] %0\n\
         output %1\n"
    );
}
