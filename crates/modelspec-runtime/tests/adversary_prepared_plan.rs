//! Adversary cases for story:cpu-prepared-training-step: the plan of a `Prepared` program on
//! the shapes its own tests do not build. An output that is an input, an output listed twice,
//! an output that reuses the slot of a freed intermediate, an in-place transpose whose source is
//! an output, each run twice with every output handed back, against `evaluate_all` (which keeps
//! every value and moves nothing) and against literal values.

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::{
    Initializer, ParamId, TensorProgram, TensorProgramBuilder, TensorType,
};
use modelspec_runtime::cpu::interpreter::Prepared;
use modelspec_runtime::cpu::{HostTensor, evaluate_all};

/// A program, its inputs and its parameter values.
type Fixture = (
    TensorProgram,
    BTreeMap<String, HostTensor<f32>>,
    BTreeMap<ParamId, HostTensor<f32>>,
);

/// x: [2, 3] input, w: [2, 3] parameter.
/// a = relu(w); b = a + a; c = b - x (takes the slot a frees); r = relu(x);
/// m = transpose(r) @ w (the transpose read in place from r, which is an output).
/// Outputs: [c, x, c, m, r, w].
fn program() -> Fixture {
    let mut b = TensorProgramBuilder::new();
    let ty = TensorType::f32(&[2, 3]);
    let p = b
        .declare_param("M.w", ty.clone(), Initializer::XavierUniform)
        .unwrap();
    let x = b.input("x", ty.clone()).unwrap();
    let w = b.param_read(p).unwrap();
    let a = b.relu(w).unwrap();
    let s = b.add(a, a).unwrap();
    let c = b.sub(s, x).unwrap();
    let r = b.relu(x).unwrap();
    let t = b.transpose(r).unwrap();
    let m = b.matmul(t, w).unwrap();
    for out in [c, x, c, m, r, w] {
        b.output(out).unwrap();
    }
    let program = b.finish();
    let inputs = BTreeMap::from([(
        "x".to_string(),
        HostTensor::new(vec![2, 3], vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0]).unwrap(),
    )]);
    let params = BTreeMap::from([(
        p,
        HostTensor::new(vec![2, 3], vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]).unwrap(),
    )]);
    (program, inputs, params)
}

#[test]
fn every_output_shape_survives_two_runs_with_recycling() {
    let (program, inputs, params) = program();
    let expected = evaluate_all(&program, &inputs, &params).unwrap().outputs;
    // c = 2 * relu(w) - x
    assert_eq!(expected[0].data(), &[-1.0, 6.0, -3.0, 12.0, -5.0, 18.0]);
    let mut prepared = Prepared::new(&program).unwrap();
    for run in 0..3 {
        let got = prepared.run(&inputs, &params).unwrap();
        assert_eq!(got, expected, "run {run}");
        for (position, output) in got.into_iter().enumerate() {
            prepared.recycle(position, output);
        }
    }
    // the feed is untouched by the runs
    assert_eq!(inputs["x"].data(), &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
}

/// A recycled tensor of the right element count and another shape, given back to a slot an
/// earlier intermediate shares: the next run still has the declared shapes and values.
#[test]
fn a_recycled_tensor_of_another_shape_does_not_change_the_next_run() {
    let (program, inputs, params) = program();
    let expected = evaluate_all(&program, &inputs, &params).unwrap().outputs;
    let mut prepared = Prepared::new(&program).unwrap();
    let got = prepared.run(&inputs, &params).unwrap();
    drop(got);
    prepared.recycle(2, HostTensor::new(vec![6], vec![9.0; 6]).unwrap());
    prepared.recycle(3, HostTensor::new(vec![9], vec![9.0; 9]).unwrap());
    let again = prepared.run(&inputs, &params).unwrap();
    assert_eq!(again, expected);
}
