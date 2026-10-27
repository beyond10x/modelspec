//! Adversary cases against the CPU reference interpreter (story `m0-cpu-interpreter`).
//!
//! Each case drives the interpreter from DESIGN.md § 2.2 (the reduction order policy and the hard
//! error rule) or § 8.1 (runtime allocation failure is a diagnostic class), not from the code.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use modelspec_core::ir::tensor::{Dim, TensorProgram, TensorProgramBuilder, TensorType};
use modelspec_core::ir::verify::verify;
use modelspec_runtime::cpu::{HostTensor, evaluate};

fn ty(shape: &[u64]) -> TensorType {
    TensorType::f32(shape)
}

/// `op(a, b)` over inputs `a` and `b` of one shape.
fn binary(
    shape: &[u64],
    op: impl FnOnce(
        &mut TensorProgramBuilder,
        modelspec_core::ir::tensor::ValueId,
        modelspec_core::ir::tensor::ValueId,
    ) -> modelspec_core::ir::tensor::ValueId,
) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("a", ty(shape)).unwrap();
    let y = b.input("b", ty(shape)).unwrap();
    let z = op(&mut b, x, y);
    b.output(z).unwrap();
    b.finish()
}

fn mse_of<T: modelspec_runtime::cpu::Element>(shape: &[u64], a: Vec<T>, b: Vec<T>) -> T {
    let program = binary(shape, |bld, x, y| bld.mse(x, y).unwrap());
    let inputs = BTreeMap::from([
        ("a".to_string(), HostTensor::new(shape.to_vec(), a).unwrap()),
        ("b".to_string(), HostTensor::new(shape.to_vec(), b).unwrap()),
    ]);
    evaluate(&program, &inputs, &BTreeMap::new()).unwrap()[0].data()[0]
}

fn mean_of<T: modelspec_runtime::cpu::Element>(shape: &[u64], x: Vec<T>) -> T {
    let mut b = TensorProgramBuilder::new();
    let v = b.input("x", ty(shape)).unwrap();
    let m = b.reduce_mean(v).unwrap();
    b.output(m).unwrap();
    let program = b.finish();
    let inputs = BTreeMap::from([("x".to_string(), HostTensor::new(shape.to_vec(), x).unwrap())]);
    evaluate(&program, &inputs, &BTreeMap::new()).unwrap()[0].data()[0]
}

fn p(e: i32) -> f64 {
    2f64.powi(e)
}

/// DESIGN.md § 2.2: "allocations over backend limits are hard errors"; § 8.1 lists "runtime
/// allocation failure" as its own diagnostic class. The core verifier accepts a value of 2^61
/// `f32` elements (2^63 bytes fit `u64`); no host can hold it, so the interpreter must refuse it
/// with an error, not panic or abort.
#[test]
fn a_value_too_large_for_host_memory_is_a_runtime_error_not_a_panic() {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", ty(&[])).unwrap();
    let y = b.broadcast_scalar(x, vec![Dim::Static(1 << 61)]).unwrap();
    b.output(y).unwrap();
    let program = b.finish();
    verify(&program).expect("the core accepts the program");
    let inputs = BTreeMap::from([("x".to_string(), HostTensor::scalar(1.0f32))]);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        evaluate(&program, &inputs, &BTreeMap::new())
    }));
    match outcome {
        Ok(Err(_)) => {}
        Ok(Ok(_)) => panic!("a 2^61-element value was evaluated"),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            panic!("the interpreter panicked instead of returning a RuntimeError: {msg}");
        }
    }
}

/// DESIGN.md § 2.2 (coordinator decision D84): a whole-tensor `mean` divides its `f64` sum by the
/// element count in `f64` and rounds the quotient to the element type once. With 2^24 + 1
/// elements the count is exact in `f64`, though it converts to 2^24 in `f32`; the sum here is 1,
/// so the mean is `1 / (2^24 + 1)` rounded to `f32` once, one ulp below `2^-24`, the mean a
/// division by the count converted to `f32` would give.
#[test]
fn mean_divides_by_the_element_count_in_f64() {
    let n: u64 = (1 << 24) + 1;
    let mut x = vec![0.0f32; n as usize];
    x[0] = 1.0;
    let got = mean_of::<f32>(&[n], x);
    let want = (1.0f64 / n as f64) as f32;
    assert_ne!(
        want,
        1.0f32 / (n as f32),
        "the case separates the two divisions"
    );
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "got {got:e}, expected {want:e} = 1 / (2^24 + 1) in f64, rounded to f32 once"
    );
}

/// `mean([12582914, 0, 0])`: `12582914 / 3` rounds to `4194304.5` in `f32`; multiplying by the
/// rounded reciprocal `1/3` gives `4194305`. Pins "divided by", which no golden case separates.
#[test]
fn mean_divides_and_does_not_multiply_by_a_reciprocal() {
    let got = mean_of::<f32>(&[3], vec![12_582_914.0, 0.0, 0.0]);
    assert_eq!(got.to_bits(), 4_194_304.5f32.to_bits(), "got {got}");
}

/// `mse([3548, 1, 0], 0)`: squares sum to `12588305`, and `12588305 / 3` rounds to `4196101.5` in
/// `f32`; multiplying by the rounded reciprocal `1/3` gives `4196102`.
#[test]
fn mse_divides_and_does_not_multiply_by_a_reciprocal() {
    let got = mse_of::<f32>(&[3], vec![3548.0, 1.0, 0.0], vec![0.0; 3]);
    assert_eq!(got.to_bits(), 4_196_101.5f32.to_bits(), "got {got}");
}

/// DESIGN.md § 12 ADR-009: the first square of `mse` starts the sum and every later square is
/// fused into it, one rounding. `d = [2^-12, 2^-12, 1, 1 + 2^-12]`: the partial sum before the
/// last term is `1 + 2^-23`; fused, the last step is `2 + 2^-11 + 2^-23 + 2^-24`, which rounds up
/// to `2 + 2^-11 + 2^-22`. Rounding the square first (`1 + 2^-11`) would leave the tie
/// `2 + 2^-11 + 2^-23`, which rounds to even, `2 + 2^-11`. In `f64` every step is exact.
#[test]
fn mse_fuses_each_later_square_into_its_sum_in_f32() {
    let d = [p(-12), p(-12), 1.0, 1.0 + p(-12)];
    let got = mse_of::<f32>(&[4], d.iter().map(|&v| v as f32).collect(), vec![0.0; 4]);
    let want = (0.5 + p(-13) + p(-24)) as f32;
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "got {got:e}, expected {want:e}"
    );
    let got64 = mse_of::<f64>(&[4], d.to_vec(), vec![0.0; 4]);
    let want64 = 0.5 + p(-13) + p(-25) + p(-26);
    assert_eq!(got64.to_bits(), want64.to_bits(), "f64: got {got64:e}");
}

/// The same in `f64`: `d = [2^-26, 1, 1 + 2^-27, 0]`; the partial sum is `1 + 2^-52`, and fused,
/// `2 + 2^-26 + 2^-52 + 2^-54` rounds up to `2 + 2^-26 + 2^-51`. Rounding the square first
/// (`1 + 2^-26`) would leave the tie `2 + 2^-26 + 2^-52`, which rounds to `2 + 2^-26`.
#[test]
fn mse_fuses_each_later_square_into_its_sum_in_f64() {
    let d = vec![p(-26), 1.0, 1.0 + p(-27), 0.0];
    let got = mse_of::<f64>(&[4], d, vec![0.0; 4]);
    let want = 0.5 + p(-28) + p(-53);
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "got {got:e}, expected {want:e}"
    );
}
