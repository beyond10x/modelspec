//! Tensor IR core: builder, operator registry, verifier and printer (DESIGN.md § 2.2, § 3.1, § 3.2,
//! § 8).

use modelspec_core::ir::print::print;
use modelspec_core::ir::tensor::{
    BuildError, Dim, Initializer, ParamId, TensorOp, TensorProgram, TensorProgramBuilder,
    TensorType, ValueId,
};
use modelspec_core::ir::verify::{Site, VerifyError, verify};
use modelspec_core::ops::OpError;

fn f32(shape: &[u64]) -> TensorType {
    TensorType::f32(shape)
}

/// `mse(matmul(x, w), y)` with the shapes of `examples/linear_regression.mspec`.
fn linear_regression_loss() -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let w = b
        .declare_param("LinearReg.w", f32(&[4, 1]), Initializer::XavierUniform)
        .unwrap();
    let x = b.input("x", f32(&[8, 4])).unwrap();
    let wv = b.param_read(w).unwrap();
    let prediction = b.matmul(x, wv).unwrap();
    let y = b.input("y", f32(&[8, 1])).unwrap();
    let loss = b.mse(prediction, y).unwrap();
    b.output(loss).unwrap();
    b.finish()
}

fn node_index(program: &TensorProgram, id: ValueId) -> usize {
    program.nodes.iter().position(|n| n.id == id).unwrap()
}

#[test]
fn builder_made_linear_regression_loss_verifies() {
    let program = linear_regression_loss();
    assert_eq!(verify(&program), Ok(()));
    assert_eq!(program.outputs, vec![ValueId(4)]);
    assert_eq!(program.inputs, vec![ValueId(0), ValueId(3)]);
    assert_eq!(program.nodes[4].ty, TensorType::f32(&[]));
}

#[test]
fn printer_golden_for_mse_of_matmul() {
    let golden = include_str!("golden/mse_matmul.tensor.ir");
    assert_eq!(print(&linear_regression_loss()), golden);
}

#[test]
fn verifier_rejects_use_before_definition() {
    let mut program = linear_regression_loss();
    let matmul = program.nodes.remove(node_index(&program, ValueId(2)));
    program.nodes.insert(0, matmul);
    assert_eq!(
        verify(&program),
        Err(VerifyError::UseBeforeDefinition {
            node: ValueId(2),
            operand: ValueId(0),
        })
    );
}

#[test]
fn verifier_rejects_operand_never_defined() {
    let mut program = linear_regression_loss();
    let i = node_index(&program, ValueId(4));
    program.nodes[i].operands[1] = ValueId(99);
    assert_eq!(
        verify(&program),
        Err(VerifyError::UndefinedOperand {
            node: ValueId(4),
            operand: ValueId(99),
        })
    );
}

#[test]
fn verifier_rejects_undeclared_param_id() {
    let mut program = linear_regression_loss();
    let i = node_index(&program, ValueId(1));
    program.nodes[i].op = TensorOp::ParamRead { param: ParamId(7) };
    assert_eq!(
        verify(&program),
        Err(VerifyError::UndeclaredParam {
            node: ValueId(1),
            param: ParamId(7),
        })
    );
}

#[test]
fn verifier_rejects_matmul_with_mismatched_contracting_dimension() {
    let mut program = linear_regression_loss();
    program.params.get_mut(&ParamId(0)).unwrap().ty = f32(&[5, 1]);
    let i = node_index(&program, ValueId(1));
    program.nodes[i].ty = f32(&[5, 1]);
    assert_eq!(
        verify(&program),
        Err(VerifyError::Op {
            node: ValueId(2),
            source: OpError::MatMulContracting {
                lhs: f32(&[8, 4]),
                rhs: f32(&[5, 1]),
                expected: Dim::Static(4),
                got: Dim::Static(5),
            },
        })
    );
}

#[test]
fn builder_rejects_matmul_with_mismatched_contracting_dimension() {
    let mut b = TensorProgramBuilder::new();
    let a = b.input("a", f32(&[64, 32])).unwrap();
    let c = b.input("c", f32(&[64, 128])).unwrap();
    assert_eq!(
        b.matmul(a, c),
        Err(BuildError::Op(OpError::MatMulContracting {
            lhs: f32(&[64, 32]),
            rhs: f32(&[64, 128]),
            expected: Dim::Static(32),
            got: Dim::Static(64),
        }))
    );
}

#[test]
fn verifier_rejects_declared_type_that_disagrees_with_registry() {
    let mut program = linear_regression_loss();
    let i = node_index(&program, ValueId(2));
    program.nodes[i].ty = f32(&[8, 2]);
    assert_eq!(
        verify(&program),
        Err(VerifyError::TypeMismatch {
            node: ValueId(2),
            declared: f32(&[8, 2]),
            inferred: f32(&[8, 1]),
        })
    );
}

#[test]
fn verifier_rejects_param_read_whose_type_differs_from_table() {
    let mut program = linear_regression_loss();
    program.params.get_mut(&ParamId(0)).unwrap().ty = f32(&[4, 2]);
    assert_eq!(
        verify(&program),
        Err(VerifyError::TypeMismatch {
            node: ValueId(1),
            declared: f32(&[4, 1]),
            inferred: f32(&[4, 2]),
        })
    );
}

#[test]
fn verifier_rejects_undefined_output() {
    let mut program = linear_regression_loss();
    program.outputs.push(ValueId(42));
    assert_eq!(
        verify(&program),
        Err(VerifyError::UndefinedOutput {
            output: ValueId(42)
        })
    );
}

#[test]
fn verifier_rejects_duplicate_value_id() {
    let mut program = linear_regression_loss();
    let i = node_index(&program, ValueId(3));
    program.nodes[i].id = ValueId(0);
    assert_eq!(
        verify(&program),
        Err(VerifyError::DuplicateValue { id: ValueId(0) })
    );
}

#[test]
fn verifier_rejects_input_list_entry_that_is_not_an_input_node() {
    let mut program = linear_regression_loss();
    program.inputs.push(ValueId(2));
    assert_eq!(
        verify(&program),
        Err(VerifyError::NotAnInput { value: ValueId(2) })
    );
}

#[test]
fn verifier_rejects_two_params_with_one_canonical_path() {
    let mut program = linear_regression_loss();
    let mut twin = program.params[&ParamId(0)].clone();
    twin.ty = f32(&[4, 1]);
    program.params.insert(ParamId(1), twin);
    assert_eq!(
        verify(&program),
        Err(VerifyError::DuplicateParamPath {
            path: "LinearReg.w".to_owned(),
            first: ParamId(0),
            second: ParamId(1),
        })
    );
}

#[test]
fn verifier_rejects_two_input_nodes_with_one_name() {
    let mut program = linear_regression_loss();
    let i = node_index(&program, ValueId(3));
    program.nodes[i].op = TensorOp::Input {
        name: "x".to_owned(),
    };
    assert_eq!(
        verify(&program),
        Err(VerifyError::DuplicateInputName {
            name: "x".to_owned(),
            first: ValueId(0),
            second: ValueId(3),
        })
    );
}

#[test]
fn verifier_rejects_input_node_missing_from_inputs() {
    let mut program = linear_regression_loss();
    program.inputs.retain(|&v| v != ValueId(3));
    assert_eq!(
        verify(&program),
        Err(VerifyError::InputNotListed { value: ValueId(3) })
    );
}

#[test]
fn verifier_rejects_inputs_entry_listed_twice() {
    let mut program = linear_regression_loss();
    program.inputs = vec![ValueId(0), ValueId(0), ValueId(3)];
    assert_eq!(
        verify(&program),
        Err(VerifyError::DuplicateInputEntry { value: ValueId(0) })
    );
}

#[test]
fn verifier_rejects_inputs_out_of_node_order() {
    let mut program = linear_regression_loss();
    program.inputs.reverse();
    assert_eq!(
        verify(&program),
        Err(VerifyError::InputsOutOfOrder {
            expected: vec![ValueId(0), ValueId(3)],
            got: vec![ValueId(3), ValueId(0)],
        })
    );
}

#[test]
fn printer_keeps_the_fraction_of_a_whole_scalar() {
    let mut b = TensorProgramBuilder::new();
    let one = b.scalar(1.0);
    b.output(one).unwrap();
    assert_eq!(
        print(&b.finish()),
        "%0: Tensor<f32, []> = Scalar 1.0\noutput %0\n"
    );
}

#[test]
fn verifier_rejects_unresolved_symbolic_dimension() {
    let mut b = TensorProgramBuilder::new();
    let x = b
        .input(
            "x",
            TensorType {
                dtype: modelspec_core::ir::tensor::DType::F32,
                shape: vec![Dim::Symbol(0), Dim::Static(4)],
            },
        )
        .unwrap();
    b.output(x).unwrap();
    assert_eq!(
        verify(&b.finish()),
        Err(VerifyError::UnresolvedDimension {
            site: Site::Value(ValueId(0)),
            symbol: 0,
        })
    );
}

#[test]
fn verifier_rejects_zero_dimension_in_param_table() {
    let mut program = linear_regression_loss();
    program.params.get_mut(&ParamId(0)).unwrap().ty = f32(&[0, 1]);
    assert_eq!(
        verify(&program),
        Err(VerifyError::ZeroDimension {
            site: Site::Param(ParamId(0)),
        })
    );
}

#[test]
fn verifier_rejects_element_or_byte_count_overflow() {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", f32(&[u64::MAX / 2, 3])).unwrap();
    b.output(x).unwrap();
    assert_eq!(
        verify(&b.finish()),
        Err(VerifyError::CountOverflow {
            site: Site::Value(ValueId(0)),
        })
    );

    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", f32(&[u64::MAX / 2])).unwrap();
    b.output(x).unwrap();
    assert_eq!(
        verify(&b.finish()),
        Err(VerifyError::CountOverflow {
            site: Site::Value(ValueId(0)),
        })
    );
}

#[test]
fn elementwise_ops_accept_same_shape_or_one_rank0_operand() {
    let mut b = TensorProgramBuilder::new();
    let a = b.input("a", f32(&[2, 3])).unwrap();
    let c = b.input("c", f32(&[2, 3])).unwrap();
    let s = b.scalar(0.5);
    let sum = b.add(a, c).unwrap();
    let diff = b.sub(s, sum).unwrap();
    let prod = b.mul(diff, s).unwrap();
    for v in [sum, diff, prod] {
        assert_eq!(b.type_of(v), Some(&f32(&[2, 3])));
    }
    let both_scalar = b.mul(s, s).unwrap();
    assert_eq!(b.type_of(both_scalar), Some(&f32(&[])));

    let row = b.input("row", f32(&[1, 3])).unwrap();
    assert_eq!(
        b.add(a, row),
        Err(BuildError::Op(OpError::ShapeMismatch {
            op: "Add",
            lhs: f32(&[2, 3]),
            rhs: f32(&[1, 3]),
        }))
    );
}

#[test]
fn unary_reduction_and_gradient_ops_follow_the_registry() {
    let mut b = TensorProgramBuilder::new();
    let a = b.input("a", f32(&[2, 3])).unwrap();
    let g = b.input("g", f32(&[2, 3])).unwrap();
    let r = b.relu(a).unwrap();
    let t = b.transpose(a).unwrap();
    let s = b.reduce_sum(a).unwrap();
    let m = b.reduce_mean(a).unwrap();
    let rg = b.relu_grad(a, g).unwrap();
    let bs = b
        .broadcast_scalar(m, vec![Dim::Static(2), Dim::Static(3)])
        .unwrap();
    assert_eq!(b.type_of(r), Some(&f32(&[2, 3])));
    assert_eq!(b.type_of(t), Some(&f32(&[3, 2])));
    assert_eq!(b.type_of(s), Some(&f32(&[])));
    assert_eq!(b.type_of(m), Some(&f32(&[])));
    assert_eq!(b.type_of(rg), Some(&f32(&[2, 3])));
    assert_eq!(b.type_of(bs), Some(&f32(&[2, 3])));

    let v = b.input("v", f32(&[3])).unwrap();
    assert_eq!(
        b.transpose(v),
        Err(BuildError::Op(OpError::Rank {
            op: "Transpose",
            expected: 2,
            got: f32(&[3]),
        }))
    );
    assert_eq!(
        b.matmul(v, a),
        Err(BuildError::Op(OpError::Rank {
            op: "MatMul",
            expected: 2,
            got: f32(&[3]),
        }))
    );
    assert_eq!(
        b.broadcast_scalar(a, vec![Dim::Static(2), Dim::Static(3)]),
        Err(BuildError::Op(OpError::Rank {
            op: "BroadcastScalar",
            expected: 0,
            got: f32(&[2, 3]),
        }))
    );
    assert_eq!(
        b.mse(a, t),
        Err(BuildError::Op(OpError::ShapeMismatch {
            op: "Mse",
            lhs: f32(&[2, 3]),
            rhs: f32(&[3, 2]),
        }))
    );
}

#[test]
fn verifier_rejects_wrong_arity() {
    let mut program = linear_regression_loss();
    let i = node_index(&program, ValueId(2));
    program.nodes[i].operands.pop();
    assert_eq!(
        verify(&program),
        Err(VerifyError::Op {
            node: ValueId(2),
            source: OpError::Arity {
                op: "MatMul",
                expected: 2,
                got: 1,
            },
        })
    );
}

#[test]
fn builder_rejects_unknown_operands_and_duplicates() {
    let mut b = TensorProgramBuilder::new();
    assert_eq!(
        b.relu(ValueId(3)),
        Err(BuildError::UnknownValue(ValueId(3)))
    );
    assert_eq!(
        b.param_read(ParamId(1)),
        Err(BuildError::UndeclaredParam(ParamId(1)))
    );
    b.input("x", f32(&[1])).unwrap();
    assert_eq!(
        b.input("x", f32(&[1])),
        Err(BuildError::DuplicateInput("x".to_owned()))
    );
    b.declare_param("M.w", f32(&[1, 1]), Initializer::XavierUniform)
        .unwrap();
    assert_eq!(
        b.declare_param("M.w", f32(&[1, 1]), Initializer::XavierUniform),
        Err(BuildError::DuplicateParamPath("M.w".to_owned()))
    );
}

#[test]
fn verification_errors_render_with_stable_ids() {
    let error = VerifyError::UseBeforeDefinition {
        node: ValueId(2),
        operand: ValueId(0),
    };
    assert_eq!(error.to_string(), "%2 uses %0 before its definition");
    let error = OpError::MatMulContracting {
        lhs: f32(&[64, 32]),
        rhs: f32(&[64, 128]),
        expected: Dim::Static(32),
        got: Dim::Static(64),
    };
    assert_eq!(
        error.to_string(),
        "matmul contracting dimension mismatch: lhs Tensor<f32, [64, 32]>, \
         rhs Tensor<f32, [64, 128]>, expected rhs dimension[0] = 32; got 64"
    );
}
