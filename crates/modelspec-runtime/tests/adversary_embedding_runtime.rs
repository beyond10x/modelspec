//! Adversary cases for story:bert-embedding-lookup on the CPU interpreter and the Tensor IR
//! verifier: id boundaries, the order of the range check, repeated runs of one plan, the
//! determinism and exactness of `EmbeddingGrad`, and the dtype rule of the verifier on programs
//! the builder would refuse.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate_inputs;
use modelspec_core::ir::tensor::{
    DType, Dim, TensorNode, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::verify::{VerifyError, verify};
use modelspec_core::ops::OpError;
use modelspec_runtime::cpu::interpreter::Prepared;
use modelspec_runtime::cpu::{
    HostTensor, RuntimeError, evaluate, evaluate_all_with_ids, evaluate_with_ids,
};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{Ids, gradient_check_with_ids, id_input, mse_to_target, param};

type Floats = BTreeMap<String, HostTensor<f32>>;
type IdMap = BTreeMap<String, HostTensor<i32>>;

/// `embedding(table, ids)` over a `[3, 2]` table input and `n` ids; its one output the rows.
fn lookup(n: u64) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let table = b.input("table", TensorType::f32(&[3, 2])).unwrap();
    let ids = b.input("ids", TensorType::i32(&[n])).unwrap();
    let rows = b.embedding(table, ids).unwrap();
    b.output(rows).unwrap();
    b.finish()
}

fn table() -> Floats {
    BTreeMap::from([(
        "table".to_owned(),
        HostTensor::new(vec![3, 2], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    )])
}

fn ids(values: &[i32]) -> IdMap {
    BTreeMap::from([(
        "ids".to_owned(),
        HostTensor::new(vec![values.len() as u64], values.to_vec()).unwrap(),
    )])
}

/// Attack 3: `V`, `-1`, `i32::MIN` and `i32::MAX` are each the out-of-range error at their
/// position, never a panic, and the first offending position is the one named.
#[test]
fn adversary_embedding_every_out_of_range_id_is_a_runtime_error() {
    let program = lookup(4);
    for (bad, at) in [
        ([0, 1, 2, 3], 3usize),
        ([-1, 0, 0, 0], 0),
        ([0, i32::MIN, 0, 0], 1),
        ([0, 0, i32::MAX, 0], 2),
        ([0, 3, -1, i32::MAX], 1),
    ] {
        let result = evaluate_with_ids(&program, &table(), &ids(&bad), &BTreeMap::new());
        assert_eq!(
            result,
            Err(RuntimeError::IdOutOfRange {
                name: "ids".to_owned(),
                index: at,
                id: bad[at],
                rows: 3,
            }),
            "ids {bad:?}"
        );
    }
    let ok = evaluate_with_ids(&program, &table(), &ids(&[2, 2, 0, 1]), &BTreeMap::new());
    assert_eq!(
        ok.unwrap()[0].data(),
        &[5.0, 6.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0]
    );
}

/// Attack 3: an ids tensor shorter or longer than `N` is the input-shape error; an unknown name
/// among the ids is an unexpected input; ids for an `f32` input and floats for an `i32` input are
/// the dtype error.
#[test]
fn adversary_embedding_ids_of_the_wrong_length_or_map_are_refused() {
    let program = lookup(4);
    for wrong in [&[0, 1, 2][..], &[0, 1, 2, 0, 1][..]] {
        assert_eq!(
            evaluate_with_ids(&program, &table(), &ids(wrong), &BTreeMap::new()),
            Err(RuntimeError::InputShape {
                name: "ids".to_owned(),
                declared: TensorType::i32(&[4]),
                got: vec![wrong.len() as u64],
            }),
            "{} ids",
            wrong.len()
        );
    }
    let mut extra = ids(&[0, 1, 2, 0]);
    extra.insert(
        "positions".to_owned(),
        HostTensor::new(vec![1], vec![0]).unwrap(),
    );
    assert_eq!(
        evaluate_with_ids(&program, &table(), &extra, &BTreeMap::new()),
        Err(RuntimeError::UnexpectedInput {
            name: "positions".to_owned()
        })
    );
    let mut floats_for_ids = table();
    floats_for_ids.insert(
        "ids".to_owned(),
        HostTensor::new(vec![4], vec![0.0, 1.0, 2.0, 0.0]).unwrap(),
    );
    assert_eq!(
        evaluate_with_ids(
            &program,
            &floats_for_ids,
            &ids(&[0, 1, 2, 0]),
            &BTreeMap::new()
        ),
        Err(RuntimeError::InputDType {
            name: "ids".to_owned(),
            declared: TensorType::i32(&[4]),
        })
    );
}

/// Attack 3: the range check runs before the first node. A node before the lookup that cannot be
/// allocated (2^50 elements) would be the allocation error if it ran first; the out-of-range id
/// is reported instead.
#[test]
fn adversary_embedding_range_check_precedes_every_node() {
    let mut b = TensorProgramBuilder::new();
    let s = b.scalar(1.0);
    let huge = b
        .broadcast_scalar(s, vec![Dim::Static(1 << 25), Dim::Static(1 << 25)])
        .unwrap();
    let table = b.input("table", TensorType::f32(&[3, 2])).unwrap();
    let ids_in = b.input("ids", TensorType::i32(&[2])).unwrap();
    let rows = b.embedding(table, ids_in).unwrap();
    b.output(huge).unwrap();
    b.output(rows).unwrap();
    let program = b.finish();
    assert_eq!(
        evaluate_with_ids(&program, &table_map(), &ids(&[0, 3]), &BTreeMap::new()),
        Err(RuntimeError::IdOutOfRange {
            name: "ids".to_owned(),
            index: 1,
            id: 3,
            rows: 3,
        })
    );
    assert!(
        matches!(
            evaluate_with_ids(&program, &table_map(), &ids(&[0, 2]), &BTreeMap::new()),
            Err(RuntimeError::Allocation { .. })
        ),
        "with valid ids the huge node runs and is refused"
    );
}

fn table_map() -> Floats {
    table()
}

/// Attack 3: one plan run many times with different ids gives each run's rows; a refused run in
/// between leaves the plan usable.
#[test]
fn adversary_embedding_one_plan_runs_again_with_new_ids() {
    let program = lookup(3);
    let mut prepared = Prepared::<f32>::new(&program).unwrap();
    let floats = table();
    let none = BTreeMap::new();
    let first = prepared
        .run_with_ids(&floats, &ids(&[0, 1, 2]), &none)
        .unwrap();
    assert_eq!(first[0].data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(matches!(
        prepared.run_with_ids(&floats, &ids(&[0, 7, 2]), &none),
        Err(RuntimeError::IdOutOfRange {
            index: 1,
            id: 7,
            ..
        })
    ));
    let second = prepared
        .run_with_ids(&floats, &ids(&[2, 2, 1]), &none)
        .unwrap();
    assert_eq!(second[0].data(), &[5.0, 6.0, 5.0, 6.0, 3.0, 4.0]);
    assert_eq!(
        prepared.run(&floats, &none),
        Err(RuntimeError::MissingInput {
            name: "ids".to_owned()
        }),
        "the plain run supplies no ids"
    );
}

/// The program `EmbeddingGrad [v] (ids, g)` with `g: f32[n, h]`.
fn scatter(v: u64, n: u64, h: u64) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let ids_in = b.input("ids", TensorType::i32(&[n])).unwrap();
    let g = b.input("g", TensorType::f32(&[n, h])).unwrap();
    let grad = b.embedding_grad(ids_in, g, Dim::Static(v)).unwrap();
    b.output(grad).unwrap();
    b.finish()
}

/// SplitMix64 for the property cases.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value of magnitude between 1e-8 and 1e8, either sign, sometimes -0.0.
    fn wide(&mut self) -> f32 {
        let r = self.next();
        if r.is_multiple_of(17) {
            return -0.0;
        }
        let exponent = (r >> 8) % 17;
        let mantissa = 1.0 + ((r >> 16) % 1000) as f32 / 1000.0;
        let v = mantissa * 10f32.powi(exponent as i32 - 8);
        if (r >> 40) & 1 == 0 { v } else { -v }
    }
}

/// Attack 4: for seeded ids that repeat rows and adjoint rows of magnitudes 1e-8 to 1e8, the
/// gradient equals a serial `f32` sum in ascending `n` from `+0.0`, bit for bit, on two runs; a
/// row no id reads is `+0.0` exactly. The descending sum differs in at least one case, so the
/// property has teeth.
#[test]
fn adversary_embedding_grad_is_the_ascending_serial_sum_bit_for_bit() {
    let mut rng = Mix(0x5EED_0E4B);
    let mut descending_differs = false;
    for case in 0..64 {
        let v = 1 + rng.next() % 6;
        let h = 1 + rng.next() % 3;
        let n = 1 + rng.next() % 24;
        let id_values: Vec<i32> = (0..n).map(|_| (rng.next() % v) as i32).collect();
        let g: Vec<f32> = (0..n * h).map(|_| rng.wide()).collect();
        let program = scatter(v, n, h);
        let floats = BTreeMap::from([(
            "g".to_owned(),
            HostTensor::new(vec![n, h], g.clone()).unwrap(),
        )]);
        let id_map = BTreeMap::from([(
            "ids".to_owned(),
            HostTensor::new(vec![n], id_values.clone()).unwrap(),
        )]);
        let got = evaluate_with_ids(&program, &floats, &id_map, &BTreeMap::new()).unwrap();
        let again = evaluate_with_ids(&program, &floats, &id_map, &BTreeMap::new()).unwrap();
        let (h, n) = (h as usize, n as usize);
        let mut want = vec![0.0f32; v as usize * h];
        let mut down = vec![0.0f32; v as usize * h];
        for k in 0..n {
            let row = id_values[k] as usize;
            for c in 0..h {
                want[row * h + c] += g[k * h + c];
            }
        }
        for k in (0..n).rev() {
            let row = id_values[k] as usize;
            for c in 0..h {
                down[row * h + c] += g[k * h + c];
            }
        }
        let bits = |x: &[f32]| x.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(got[0].data()), bits(&want), "case {case}");
        assert_eq!(
            bits(again[0].data()),
            bits(&want),
            "case {case}, second run"
        );
        descending_differs |= bits(&down) != bits(&want);
        for row in 0..v as usize {
            if !id_values.contains(&(row as i32)) {
                assert!(
                    got[0].data()[row * h..(row + 1) * h]
                        .iter()
                        .all(|x| x.to_bits() == 0),
                    "case {case}: unread row {row} is not +0.0"
                );
            }
        }
    }
    assert!(
        descending_differs,
        "no case separates ascending from descending order"
    );
}

/// Attack 4: `V = 1`, `H = 1`: every id reads the one row; the gradient check still holds.
#[test]
fn adversary_embedding_one_row_one_column_gradient_matches() {
    let fixed: Ids = BTreeMap::from([("ids".to_owned(), (vec![3], vec![0, 0, 0]))]);
    gradient_check_with_ids(
        "adversary_embedding_one_row_one_column_gradient_matches",
        &fixed,
        |b| {
            let table = param(b, "table", &[1, 1]);
            let ids_in = id_input(b, "ids", &[3]);
            let rows = b.embedding(table, ids_in).unwrap();
            mse_to_target(b, rows)
        },
    );
}

/// Attack 4: the table is computed from parameters (`matmul(a, b)`) and the rows feed a further
/// matmul, so `EmbeddingGrad` sits between two other rules.
#[test]
fn adversary_embedding_between_two_matmuls_gradient_matches() {
    let fixed: Ids = BTreeMap::from([("ids".to_owned(), (vec![5], vec![3, 0, 3, 1, 3]))]);
    gradient_check_with_ids(
        "adversary_embedding_between_two_matmuls_gradient_matches",
        &fixed,
        |b| {
            let a = param(b, "a", &[4, 2]);
            let w = param(b, "w", &[2, 3]);
            let out = param(b, "out", &[3, 2]);
            let table = b.matmul(a, w).unwrap();
            let ids_in = id_input(b, "ids", &[5]);
            let rows = b.embedding(table, ids_in).unwrap();
            let y = b.matmul(rows, out).unwrap();
            mse_to_target(b, y)
        },
    );
}

/// Attack 4: the adjoint of an `f32` table input (the saliency path) is an `EmbeddingGrad` of the
/// ids, and evaluates to the scatter of the rows' adjoint.
#[test]
fn adversary_embedding_input_adjoint_of_a_table_input_is_the_scatter() {
    let mut b = TensorProgramBuilder::new();
    let table_in = b.input("table", TensorType::f32(&[3, 2])).unwrap();
    let ids_in = b.input("ids", TensorType::i32(&[4])).unwrap();
    let rows = b.embedding(table_in, ids_in).unwrap();
    let loss = b.reduce_sum(rows).unwrap();
    b.output(loss).unwrap();
    let forward = b.finish();
    let adjoint = differentiate_inputs(&forward, loss, &["table"]).unwrap();
    let out = evaluate_with_ids(
        &adjoint.program,
        &table(),
        &ids(&[2, 0, 2, 2]),
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(out[0].data(), &[36.0], "rows 2, 0, 2, 2: 11 + 3 + 11 + 11");
    assert_eq!(out[1].data(), &[1.0, 1.0, 0.0, 0.0, 3.0, 3.0]);
    assert!(out[1].data()[2].is_sign_positive() && out[1].data()[3].is_sign_positive());
}

/// Attack 3: `Evaluation.values` leaves out the `i32` input and holds every other node.
#[test]
fn adversary_embedding_evaluation_values_omit_only_the_ids() {
    let program = lookup(2);
    let evaluation =
        evaluate_all_with_ids(&program, &table(), &ids(&[1, 0]), &BTreeMap::new()).unwrap();
    let ids_value = program.inputs[1];
    let keys: Vec<ValueId> = evaluation.values.keys().copied().collect();
    let want: Vec<ValueId> = program
        .nodes
        .iter()
        .map(|n| n.id)
        .filter(|&id| id != ids_value)
        .collect();
    assert_eq!(keys, want);
    assert_eq!(evaluation.outputs[0].data(), &[3.0, 4.0, 1.0, 2.0]);
}

/// A node defining `id` with `op`, `operands` and type `ty`.
fn node(id: u32, op: TensorOp, operands: &[u32], ty: TensorType) -> TensorNode {
    TensorNode {
        id: ValueId(id),
        op,
        operands: operands.iter().map(|&o| ValueId(o)).collect(),
        ty,
        source_span: None,
    }
}

/// Inputs `x: f32[2, 2]` (%0) and `ids: i32[2, 2]` (%1) and the nodes after them.
fn hand_built(rest: Vec<TensorNode>, outputs: &[u32]) -> TensorProgram {
    let mut nodes = vec![
        node(
            0,
            TensorOp::Input { name: "x".into() },
            &[],
            TensorType::f32(&[2, 2]),
        ),
        node(
            1,
            TensorOp::Input { name: "ids".into() },
            &[],
            TensorType::i32(&[2, 2]),
        ),
    ];
    nodes.extend(rest);
    TensorProgram {
        nodes,
        params: BTreeMap::new(),
        inputs: vec![ValueId(0), ValueId(1)],
        outputs: outputs.iter().map(|&o| ValueId(o)).collect(),
    }
}

/// Attack 2: the verifier, not only the checker, refuses an `i32` operand of every other op
/// in a program the builder would not make, whichever type the node claims.
#[test]
fn adversary_embedding_verifier_refuses_an_i32_operand_of_every_other_op() {
    let f = TensorType::f32(&[2, 2]);
    let cases = [
        (TensorOp::MatMul, vec![1, 0], 0usize, f.clone()),
        (TensorOp::MatMul, vec![0, 1], 1, f.clone()),
        (TensorOp::Add, vec![0, 1], 1, f.clone()),
        (TensorOp::Sub, vec![1, 0], 0, f.clone()),
        (TensorOp::Mul, vec![1, 1], 0, TensorType::i32(&[2, 2])),
        (TensorOp::Relu, vec![1], 0, f.clone()),
        (TensorOp::Transpose, vec![1], 0, f.clone()),
        (TensorOp::ReduceSum, vec![1], 0, TensorType::f32_scalar()),
        (TensorOp::ReduceMean, vec![1], 0, TensorType::f32_scalar()),
        (TensorOp::Mse, vec![0, 1], 1, TensorType::f32_scalar()),
        (TensorOp::ReluGrad, vec![1, 0], 0, f.clone()),
    ];
    for (op, operands, index, ty) in cases {
        let program = hand_built(vec![node(2, op.clone(), &operands, ty)], &[0]);
        match verify(&program) {
            Err(VerifyError::Op {
                node: ValueId(2),
                source:
                    OpError::OperandDType {
                        index: got,
                        expected: DType::F32,
                        ..
                    },
            }) => assert_eq!(got, index, "{op:?}"),
            other => panic!("{op:?}: {other:?}"),
        }
    }
}

/// Attack 2: the verifier refuses an `i32` table, `f32` ids, ids of rank 2 and a table of rank 1
/// or 3, and an `EmbeddingGrad` whose ids and adjoint rows differ.
#[test]
fn adversary_embedding_verifier_refuses_misshapen_lookups() {
    let refused = |rest: Vec<TensorNode>| verify(&hand_built(rest, &[0])).unwrap_err();
    let e = refused(vec![node(
        2,
        TensorOp::Embedding,
        &[1, 1],
        TensorType::f32(&[2, 2]),
    )]);
    assert!(
        matches!(
            e,
            VerifyError::Op {
                source: OpError::OperandDType { index: 0, .. },
                ..
            }
        ),
        "{e:?}"
    );
    let e = refused(vec![node(
        2,
        TensorOp::Embedding,
        &[0, 0],
        TensorType::f32(&[2, 2]),
    )]);
    assert!(
        matches!(
            e,
            VerifyError::Op {
                source: OpError::OperandDType { index: 1, .. },
                ..
            }
        ),
        "{e:?}"
    );
    let e = refused(vec![node(
        2,
        TensorOp::Embedding,
        &[0, 1],
        TensorType::f32(&[4, 2]),
    )]);
    assert!(
        matches!(
            e,
            VerifyError::Op {
                source: OpError::Rank { expected: 1, .. },
                ..
            }
        ),
        "rank-2 ids: {e:?}"
    );
    let e = refused(vec![
        node(
            2,
            TensorOp::Input {
                name: "flat".into(),
            },
            &[],
            TensorType::i32(&[2]),
        ),
        node(3, TensorOp::ReduceSum, &[0], TensorType::f32_scalar()),
        node(4, TensorOp::Embedding, &[3, 2], TensorType::f32(&[2, 1])),
    ]);
    // Nodes are checked before the input list, so the rank-0 table is the refusal.
    assert!(
        matches!(
            e,
            VerifyError::Op {
                node: ValueId(4),
                source: OpError::Rank { expected: 2, .. },
            }
        ),
        "rank-0 table: {e:?}"
    );
    let e = refused(vec![node(
        2,
        TensorOp::EmbeddingGrad {
            rows: Dim::Static(3),
        },
        &[1, 0],
        TensorType::f32(&[3, 2]),
    )]);
    assert!(
        matches!(
            e,
            VerifyError::Op {
                source: OpError::Rank { expected: 1, .. },
                ..
            }
        ),
        "rank-2 ids of EmbeddingGrad: {e:?}"
    );
}

/// Attack 2: a declared type that disagrees with the lookup's inferred `[N, H]` is refused, and
/// an `i32` parameter or output is refused.
#[test]
fn adversary_embedding_verifier_holds_the_result_type_parameters_and_outputs() {
    let mut b = TensorProgramBuilder::new();
    let table_in = b.input("table", TensorType::f32(&[3, 2])).unwrap();
    let ids_in = b.input("ids", TensorType::i32(&[4])).unwrap();
    let rows = b.embedding(table_in, ids_in).unwrap();
    b.output(rows).unwrap();
    let good = b.finish();
    assert_eq!(verify(&good), Ok(()));
    let mut wrong = good.clone();
    wrong.nodes[2].ty = TensorType::f32(&[3, 2]);
    assert!(matches!(
        verify(&wrong),
        Err(VerifyError::TypeMismatch { .. })
    ));
    let mut int_out = good.clone();
    int_out.nodes[2].ty = TensorType::i32(&[4, 2]);
    assert!(verify(&int_out).is_err(), "an i32 lookup result is refused");
    let mut ids_out = good;
    ids_out.outputs = vec![ids_in];
    assert_eq!(
        verify(&ids_out),
        Err(VerifyError::IntegerOutput { output: ids_in })
    );
}

/// Attack 3: `evaluate` without ids on a program that reads none still works (the old entry
/// point is unchanged for `f32` programs).
#[test]
fn adversary_embedding_plain_evaluate_is_unchanged_for_f32_programs() {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[2])).unwrap();
    let y = b.relu(x).unwrap();
    b.output(y).unwrap();
    let program = b.finish();
    let out = evaluate(
        &program,
        &BTreeMap::from([(
            "x".to_owned(),
            HostTensor::new(vec![2], vec![-1.0f32, 2.0]).unwrap(),
        )]),
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(out[0].data(), &[0.0, 2.0]);
}
