//! Adversary cases for story:deterministic-fma on the CPU (DESIGN.md § 12 ADR-009).
//!
//! 1. A product that overflows when rounded alone and is cancelled by the running sum:
//!    `-f32::MAX + 2^64 * 2^64` is `2^104` exactly when the product is fused into the sum and
//!    `+inf` when it is rounded first. Every CPU kernel (portable and AVX2), each left operand
//!    layout, the masked and unmasked vector lanes and the interpreter's `matmul` give `2^104`;
//!    in `f64` the same with `-f64::MAX + 2^512 * 2^512 = 2^971`.
//! 2. The AVX2 kernel and the portable loop agree bit for bit with the written-out reference
//!    (`a[i, 0] * b[0, j]`, then `mul_add` per later `k`) on operands that hold subnormals,
//!    signed zeros and values near the overflow bound, at shapes whose `k` is 1, 2, 3, 5, 33,
//!    127 and 1000 and whose `n` leaves a partial vector.
//! 3. The interpreter's `mse`, at 1 to 9 terms and at a count that is not a multiple of 8, gives
//!    the bits of the written-out sum: the first square rounded, each later one fused.

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::{TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::matmul::{Kernel, Lhs, matmul};
use modelspec_runtime::cpu::{HostTensor, evaluate};

fn kernels() -> Vec<Kernel> {
    [Kernel::Portable, Kernel::Avx2]
        .into_iter()
        .filter(|kernel| kernel.is_available())
        .collect()
}

fn transposed<T: Copy>(a: &[T], m: usize, k: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(a.len());
    for p in 0..k {
        for i in 0..m {
            out.push(a[i * k + p]);
        }
    }
    out
}

/// The `[m, k]` and `[k, n]` operands whose every element is `first + big * big`: column 0 of
/// `a` is `first` against a row of ones, column 1 is `big` against a row of `big`, and every
/// later product is `0 * 0`.
fn cancelling<T: Copy>(
    m: usize,
    k: usize,
    n: usize,
    first: T,
    big: T,
    one: T,
    zero: T,
) -> (Vec<T>, Vec<T>) {
    let mut a = Vec::with_capacity(m * k);
    for _ in 0..m {
        for p in 0..k {
            a.push(match p {
                0 => first,
                1 => big,
                _ => zero,
            });
        }
    }
    let mut b = Vec::with_capacity(k * n);
    for p in 0..k {
        for _ in 0..n {
            b.push(match p {
                0 => one,
                1 => big,
                _ => zero,
            });
        }
    }
    (a, b)
}

#[test]
fn an_overflowing_product_cancelled_by_the_sum_is_finite_in_every_cpu_kernel() {
    let want32 = 2f32.powi(104);
    let big32 = 2f32.powi(64);
    assert!((big32 * big32).is_infinite(), "the product overflows alone");
    assert_eq!((-f32::MAX).mul_add(1.0, 0.0), -f32::MAX);
    assert_eq!(
        big32.mul_add(big32, -f32::MAX),
        want32,
        "fused, the sum is 2^104"
    );
    let want64 = 2f64.powi(971);
    let big64 = 2f64.powi(512);
    assert!((big64 * big64).is_infinite());
    assert_eq!(big64.mul_add(big64, -f64::MAX), want64);
    let mut wrong = Vec::new();
    let mut checked = 0;
    for &(m, k, n) in &[
        (1, 2, 1),
        (5, 3, 7),
        (6, 2, 8),
        (7, 9, 9),
        (13, 17, 21),
        (2, 1000, 33),
    ] {
        let (a, b) = cancelling(m, k, n, -f32::MAX, big32, 1.0, 0.0);
        let (a64, b64) = cancelling(m, k, n, -f64::MAX, big64, 1.0, 0.0);
        for kernel in kernels() {
            for lhs in [Lhs::RowMajor, Lhs::Transposed] {
                let bound = match lhs {
                    Lhs::RowMajor => a.clone(),
                    Lhs::Transposed => transposed(&a, m, k),
                };
                let got = matmul(kernel, &bound, lhs, &b, m, k, n).expect("the kernel runs");
                let bad = got
                    .iter()
                    .filter(|v| v.to_bits() != want32.to_bits())
                    .count();
                if bad > 0 {
                    wrong.push(format!(
                        "f32 [{m}, {k}] x [{k}, {n}] {kernel:?} {lhs:?}: {bad} of {} elements, \
                         first {:e}",
                        got.len(),
                        got.iter()
                            .find(|v| v.to_bits() != want32.to_bits())
                            .unwrap()
                    ));
                }
                checked += got.len();
            }
            let got = matmul(kernel, &a64, Lhs::RowMajor, &b64, m, k, n).expect("runs");
            let bad = got
                .iter()
                .filter(|v| v.to_bits() != want64.to_bits())
                .count();
            if bad > 0 {
                wrong.push(format!(
                    "f64 [{m}, {k}] x [{k}, {n}] {kernel:?}: {bad} elements"
                ));
            }
        }
        // The interpreter, which picks its kernel itself.
        let mut builder = TensorProgramBuilder::new();
        let x = builder
            .input("a", TensorType::f32(&[m as u64, k as u64]))
            .unwrap();
        let y = builder
            .input("b", TensorType::f32(&[k as u64, n as u64]))
            .unwrap();
        let z = builder.matmul(x, y).unwrap();
        builder.output(z).unwrap();
        let program = builder.finish();
        let inputs: BTreeMap<String, HostTensor<f32>> = [
            ("a", vec![m as u64, k as u64], a.clone()),
            ("b", vec![k as u64, n as u64], b.clone()),
        ]
        .into_iter()
        .map(|(name, shape, data)| (name.to_owned(), HostTensor::new(shape, data).unwrap()))
        .collect();
        let out = evaluate(&program, &inputs, &BTreeMap::new()).expect("evaluates");
        let got = out[0].data();
        let bad = got
            .iter()
            .filter(|v| v.to_bits() != want32.to_bits())
            .count();
        if bad > 0 {
            wrong.push(format!(
                "interpreter [{m}, {k}] x [{k}, {n}]: {bad} elements"
            ));
        }
    }
    eprintln!("adversary_fma_cpu: {checked} f32 elements of cancelling sums");
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// SplitMix64 values of every class the fused sum can meet without overflowing at `k <= 1000`:
/// normal values in [-4, 4), subnormals, signed zeros, and values near `2^60` whose products
/// stay finite.
fn hostile(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|i| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            let unit = (z >> 40) as f32 / (1u64 << 24) as f32;
            match i % 11 {
                2 => 0.0,
                5 => -0.0,
                7 => f32::from_bits((z as u32) & 0x807f_ffff),
                9 => (unit * 2.0 - 1.0) * 2f32.powi(60),
                _ => unit * 8.0 - 4.0,
            }
        })
        .collect()
}

fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = a[i * k] * b[j];
            for p in 1..k {
                acc = a[i * k + p].mul_add(b[p * n + j], acc);
            }
            out.push(acc);
        }
    }
    out
}

#[test]
fn every_cpu_kernel_gives_the_fused_reference_bits_on_hostile_operands_at_every_k() {
    let mut wrong = Vec::new();
    let mut checked = 0;
    for (case, &k) in [1usize, 2, 3, 5, 33, 127, 1000].iter().enumerate() {
        for &(m, n) in &[(1, 1), (7, 9), (6, 17), (13, 31)] {
            let a = hostile(11 + case as u64 * 7 + m as u64, m * k);
            let b = hostile(97 + case as u64 * 5 + n as u64, k * n);
            let want = reference(&a, &b, m, k, n);
            for kernel in kernels() {
                for lhs in [Lhs::RowMajor, Lhs::Transposed] {
                    let bound = match lhs {
                        Lhs::RowMajor => a.clone(),
                        Lhs::Transposed => transposed(&a, m, k),
                    };
                    let got = matmul(kernel, &bound, lhs, &b, m, k, n).expect("runs");
                    let bad: Vec<usize> = (0..got.len())
                        .filter(|&e| got[e].to_bits() != want[e].to_bits())
                        .collect();
                    if let Some(&e) = bad.first() {
                        wrong.push(format!(
                            "[{m}, {k}] x [{k}, {n}] {kernel:?} {lhs:?}: {} elements, first {e} \
                             got {:e} want {:e}",
                            bad.len(),
                            got[e],
                            want[e]
                        ));
                    }
                    checked += got.len();
                }
            }
        }
    }
    eprintln!("adversary_fma_cpu: {checked} hostile elements");
    assert!(wrong.is_empty(), "{wrong:#?}");
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    let count = a.len() as u64;
    let mut builder = TensorProgramBuilder::new();
    let x = builder.input("a", TensorType::f32(&[count])).unwrap();
    let y = builder.input("b", TensorType::f32(&[count])).unwrap();
    let z = builder.mse(x, y).unwrap();
    builder.output(z).unwrap();
    let program = builder.finish();
    let inputs: BTreeMap<String, HostTensor<f32>> = [("a", a), ("b", b)]
        .into_iter()
        .map(|(name, data)| {
            (
                name.to_owned(),
                HostTensor::new(vec![count], data.to_vec()).unwrap(),
            )
        })
        .collect();
    evaluate(&program, &inputs, &BTreeMap::new()).expect("evaluates")[0].data()[0]
}

#[test]
fn the_interpreter_mse_gives_the_fused_reference_bits_at_every_term_count() {
    let mut wrong = Vec::new();
    for count in (1..=9).chain([63, 65, 1001]) {
        let a = hostile(5 + count as u64, count);
        let t = hostile(500 + count as u64, count);
        let mut acc = 0.0f32;
        for (i, (&x, &y)) in a.iter().zip(&t).enumerate() {
            let d = x - y;
            acc = if i == 0 { d * d } else { d.mul_add(d, acc) };
        }
        let want = acc / count as f32;
        let got = mse(&a, &t);
        if got.to_bits() != want.to_bits() {
            wrong.push(format!("{count} terms: got {got:e} want {want:e}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
