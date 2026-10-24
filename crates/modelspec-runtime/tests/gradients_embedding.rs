//! The embedding lookup on the CPU interpreter (DESIGN.md § 2.2, § 4, § 9 gate 3;
//! story:bert-embedding-lookup).
//!
//! `embedding(table, ids)` reads row `ids[n]` of `table` for each `n`. Its gradient with respect
//! to `table` is the internal op `EmbeddingGrad`, a scatter-add of the output adjoint into a zero
//! table, rows accumulated in ascending `n`; the `i32` ids have none. The finite-difference cases
//! use the harness of `tests/gradients.rs` ([`finite_difference`]) with the ids held fixed.

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::{Dim, TensorOp, TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::{HostTensor, RuntimeError, evaluate, evaluate_with_ids};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{Ids, gradient_check_with_ids, id_input, mse_to_target, param};

/// Rows of the table.
const V: u64 = 7;
/// Width of a row.
const H: u64 = 3;
/// Ids looked up.
const N: u64 = 10;

/// `N` ids into a `V`-row table: row 2 three times, rows 0, 1 and 4 twice, row 6 once, rows 3
/// and 5 never.
const IDS: [i32; N as usize] = [2, 0, 4, 2, 1, 6, 2, 0, 4, 1];

/// The rows [`IDS`] does not read.
const UNREAD: [usize; 2] = [3, 5];

fn ids(name: &str, values: &[i32]) -> Ids {
    BTreeMap::from([(
        name.to_owned(),
        (vec![values.len() as u64], values.to_vec()),
    )])
}

/// `mse(embedding(table, ids), t)`: the gradient of `table` is within `TOL_F64` of central
/// differences in `f64` and `TOL_F32` in `f32`, and the rows no id reads have the gradient 0
/// exactly, in both.
#[test]
fn embedding_table_gradient_matches_central_differences() {
    let checked = gradient_check_with_ids(
        "embedding_table_gradient_matches_central_differences",
        &ids("ids", &IDS),
        |b| {
            let table = param(b, "table", &[V, H]);
            let ids = id_input(b, "ids", &[N]);
            let rows = b.embedding(table, ids).unwrap();
            assert_eq!(b.type_of(rows), Some(&TensorType::f32(&[N, H])));
            mse_to_target(b, rows)
        },
    );
    let ops: Vec<&TensorOp> = checked
        .training
        .program
        .nodes
        .iter()
        .map(|n| &n.op)
        .collect();
    assert!(
        ops.contains(&&TensorOp::EmbeddingGrad {
            rows: Dim::Static(V)
        }),
        "the table gradient is an EmbeddingGrad: {ops:?}"
    );
    let width = H as usize;
    for row in UNREAD {
        let g64 = &checked.in_f64[0].data()[row * width..(row + 1) * width];
        let g32 = &checked.in_f32[0].data()[row * width..(row + 1) * width];
        assert!(
            g64.iter().all(|&g| g == 0.0) && g32.iter().all(|&g| g == 0.0),
            "row {row} is read by no id; gradient {g64:?} in f64, {g32:?} in f32"
        );
    }
}

/// Positional embeddings are a second lookup over an `i32` input `positions`:
/// `embedding(tok, ids) + embedding(pos, positions)` passes the same check for both tables.
#[test]
fn positional_embedding_adds_a_second_table_and_both_gradients_match() {
    let positions: [i32; N as usize] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
    let mut fixed = ids("ids", &IDS);
    fixed.extend(ids("positions", &positions));
    let checked = gradient_check_with_ids(
        "positional_embedding_adds_a_second_table_and_both_gradients_match",
        &fixed,
        |b| {
            let tok = param(b, "tok", &[V, H]);
            let pos = param(b, "pos", &[5, H]);
            let ids = id_input(b, "ids", &[N]);
            let positions = id_input(b, "positions", &[N]);
            let tokens = b.embedding(tok, ids).unwrap();
            let places = b.embedding(pos, positions).unwrap();
            let sum = b.add(tokens, places).unwrap();
            mse_to_target(b, sum)
        },
    );
    assert_eq!(checked.training.gradients.len(), 2, "a gradient per table");
}

/// The program `embedding(table, ids)` with a `[3, 2]` table and 4 ids, its one output the rows.
fn lookup() -> modelspec_core::ir::tensor::TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let table = b.input("table", TensorType::f32(&[3, 2])).unwrap();
    let ids = b.input("ids", TensorType::i32(&[4])).unwrap();
    let rows = b.embedding(table, ids).unwrap();
    b.output(rows).unwrap();
    b.finish()
}

fn table<T: modelspec_runtime::cpu::Element>(convert: impl Fn(f32) -> T) -> HostTensor<T> {
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    HostTensor::new(vec![3, 2], values.iter().map(|&v| convert(v)).collect()).unwrap()
}

fn id_tensor(values: [i32; 4]) -> BTreeMap<String, HostTensor<i32>> {
    BTreeMap::from([(
        "ids".to_owned(),
        HostTensor::new(vec![4], values.to_vec()).unwrap(),
    )])
}

/// Row `n` of the result is row `ids[n]` of the table, in `f32` and `f64`.
#[test]
fn embedding_reads_row_ids_n_of_the_table() {
    let program = lookup();
    let ids = id_tensor([2, 0, 2, 1]);
    let out = evaluate_with_ids(
        &program,
        &BTreeMap::from([("table".to_owned(), table(|v| v))]),
        &ids,
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(out[0].shape(), &[4, 2]);
    assert_eq!(out[0].data(), &[5.0, 6.0, 1.0, 2.0, 5.0, 6.0, 3.0, 4.0]);
    let out = evaluate_with_ids(
        &program,
        &BTreeMap::from([("table".to_owned(), table(f64::from))]),
        &ids,
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(out[0].data(), &[5.0, 6.0, 1.0, 2.0, 5.0, 6.0, 3.0, 4.0]);
}

/// An id of `V` or `-1` is the interpreter's out-of-range error, not a panic, and nothing runs.
#[test]
fn embedding_id_out_of_range_is_a_runtime_error() {
    let program = lookup();
    for (bad, at) in [([0, 3, 1, 2], 1usize), ([0, 1, 2, -1], 3)] {
        let id = bad[at];
        let result = evaluate_with_ids(
            &program,
            &BTreeMap::from([("table".to_owned(), table(|v| v))]),
            &id_tensor(bad),
            &BTreeMap::new(),
        );
        assert_eq!(
            result,
            Err(RuntimeError::IdOutOfRange {
                name: "ids".to_owned(),
                index: at,
                id,
                rows: 3,
            }),
            "ids {bad:?}"
        );
        let message = result.unwrap_err().to_string();
        assert!(message.contains(&format!("{id}")), "{message}");
    }
}

/// The ids are supplied as `i32` tensors only: missing, or given among the `f32` inputs, the
/// run is refused before anything is evaluated.
#[test]
fn ids_are_supplied_as_i32_tensors() {
    let program = lookup();
    let floats = BTreeMap::from([("table".to_owned(), table(|v| v))]);
    assert_eq!(
        evaluate(&program, &floats, &BTreeMap::new()),
        Err(RuntimeError::MissingInput {
            name: "ids".to_owned()
        })
    );
    let mut wrong = floats.clone();
    wrong.insert(
        "ids".to_owned(),
        HostTensor::new(vec![4], vec![0.0f32, 1.0, 2.0, 0.0]).unwrap(),
    );
    assert_eq!(
        evaluate(&program, &wrong, &BTreeMap::new()),
        Err(RuntimeError::InputDType {
            name: "ids".to_owned(),
            declared: TensorType::i32(&[4]),
        })
    );
    let mut extra = id_tensor([0, 1, 2, 0]);
    extra.insert(
        "table".to_owned(),
        HostTensor::new(vec![3, 2], vec![0; 6]).unwrap(),
    );
    assert_eq!(
        evaluate_with_ids(&program, &floats, &extra, &BTreeMap::new()),
        Err(RuntimeError::InputDType {
            name: "table".to_owned(),
            declared: TensorType::f32(&[3, 2]),
        })
    );
}

/// `EmbeddingGrad` adds the adjoint rows of one id in ascending `n`: with `f32` rows `1`, `1e8`
/// and `-1e8` for id 0, `(1 + 1e8) - 1e8` rounds to 0, where the descending order would give 1.
#[test]
fn embedding_grad_accumulates_rows_in_ascending_n() {
    let mut b = TensorProgramBuilder::new();
    let ids = b.input("ids", TensorType::i32(&[3])).unwrap();
    let g = b.input("g", TensorType::f32(&[3, 1])).unwrap();
    let grad = b.embedding_grad(ids, g, Dim::Static(2)).unwrap();
    b.output(grad).unwrap();
    let program = b.finish();
    let out = evaluate_with_ids(
        &program,
        &BTreeMap::from([(
            "g".to_owned(),
            HostTensor::new(vec![3, 1], vec![1.0f32, 1e8, -1e8]).unwrap(),
        )]),
        &BTreeMap::from([(
            "ids".to_owned(),
            HostTensor::new(vec![3], vec![0, 0, 0]).unwrap(),
        )]),
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(out[0].shape(), &[2, 1]);
    assert_eq!(out[0].data(), &[0.0, 0.0]);
    assert!(out[0].data()[1].is_sign_positive(), "an unread row is +0");
}
