//! `story:deterministic-fma`, acceptance `oracle-matches`, on the CPU (DESIGN.md § 12 ADR-009):
//! the CPU oracle computes every matmul element and the `mse` sum the way the Vulkan kernels do,
//! the `k = 0` product (the first term) as a plain product and every later product fused into
//! the sum with one rounding, `mul_add`, in ascending `k` (row-major term order).
//!
//! The reference here is written out: `acc = a[i, 0] * b[0, j]`, then
//! `acc = a[i, p].mul_add(b[p, j], acc)` for `p = 1 .. k`. Each kernel of `cpu::matmul` this CPU
//! runs (the portable loop and, with AVX2 and FMA, the vector kernel), with the left operand
//! row-major and transposed, and the interpreter's `matmul` and `mse`, in `f32` and `f64`, give
//! its bits. The hand cases are chosen so that a separate multiply and add gives other bits.

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::{TensorProgram, TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::matmul::{Kernel, Lhs, matmul};
use modelspec_runtime::cpu::{HostTensor, evaluate};

/// The written-out reference: the `k = 0` product, then each later product fused into the sum.
fn reference32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

fn reference64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
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

/// The `[m, k]` row-major `a` as the `[k, m]` row-major operand `Lhs::Transposed` reads.
fn transposed<T: Copy>(a: &[T], m: usize, k: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(a.len());
    for p in 0..k {
        for i in 0..m {
            out.push(a[i * k + p]);
        }
    }
    out
}

/// SplitMix64 values in [-4, 4) with 24 significant bits, every seventh a signed zero.
fn values(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|i| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            match i % 14 {
                3 => 0.0,
                10 => -0.0,
                _ => ((z >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0,
            }
        })
        .collect()
}

fn kernels() -> Vec<Kernel> {
    [Kernel::Portable, Kernel::Avx2]
        .into_iter()
        .filter(|kernel| kernel.is_available())
        .collect()
}

fn bits32(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn bits64(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// `(1 + 2^-12)^2 - 1` is `2^-11 + 2^-24` fused and `2^-11` with the product rounded first: row
/// `i` of the `[m, 2]` left operand is `[1, e]`, column `j` of the `[2, n]` right operand is
/// `[-1, e]`, so every element is `-1` and then `e * e` added. `m = 7` and `n = 21` take a
/// partial row block, two whole vectors and a masked one of the vector kernel.
#[test]
fn every_cpu_matmul_kernel_fuses_each_later_product_into_its_sum() {
    let e = 1.0 + 2f32.powi(-12);
    let (m, k, n) = (7, 2, 21);
    let a: Vec<f32> = (0..m).flat_map(|_| [1.0, e]).collect();
    let b: Vec<f32> = [vec![-1.0; n], vec![e; n]].concat();
    let fused = 2f32.powi(-11) + 2f32.powi(-24);
    assert_ne!(e * e - 1.0, fused, "the case separates fused from unfused");
    for kernel in kernels() {
        for lhs in [Lhs::RowMajor, Lhs::Transposed] {
            let bound = match lhs {
                Lhs::RowMajor => a.clone(),
                Lhs::Transposed => transposed(&a, m, k),
            };
            let got = matmul(kernel, &bound, lhs, &b, m, k, n).expect("the kernel runs");
            assert_eq!(
                bits32(&got),
                bits32(&vec![fused; m * n]),
                "{kernel:?} {lhs:?}: {got:?}"
            );
        }
    }
    // In `f64` the product of `1 + 2^-27` with itself is `1 + 2^-26 + 2^-54`, rounded to
    // `1 + 2^-26` before an add.
    let e = 1.0 + 2f64.powi(-27);
    let a: Vec<f64> = (0..m).flat_map(|_| [1.0, e]).collect();
    let b: Vec<f64> = [vec![-1.0; n], vec![e; n]].concat();
    let fused = 2f64.powi(-26) + 2f64.powi(-54);
    assert_ne!(e * e - 1.0, fused);
    for kernel in kernels() {
        let got = matmul(kernel, &a, Lhs::RowMajor, &b, m, k, n).expect("the kernel runs");
        assert_eq!(bits64(&got), bits64(&vec![fused; m * n]), "{kernel:?} f64");
    }
}

/// The `k = 0` product starts the sum and is not `fma(a0, b0, +0)`: `-1 * 0` is `-0`, and with a
/// second product `-0` the sum stays `-0`; a sum started at `+0` would give `+0`.
#[test]
fn the_first_product_starts_the_sum_so_a_negative_zero_stays_negative() {
    for kernel in kernels() {
        for k in [1, 2, 9] {
            let a = vec![-1.0f32; k];
            let b = vec![0.0f32; k];
            let got = matmul(kernel, &a, Lhs::RowMajor, &b, 1, k, 1).expect("runs");
            assert_eq!(got[0].to_bits(), (-0.0f32).to_bits(), "{kernel:?} k = {k}");
        }
    }
}

/// Every CPU kernel against the written-out reference, bit for bit, over drawn operands at
/// shapes that cover whole and partial row blocks and vectors, `k = 1`, and the MNIST and
/// `mlp_sweep` contracting depths.
#[test]
fn every_cpu_matmul_kernel_gives_the_bits_of_the_fused_reference() {
    let shapes = [
        (1, 1, 1),
        (1, 7, 1),
        (6, 1, 16),
        (7, 3, 17),
        (13, 100, 25),
        (100, 784, 64),
        (64, 100, 10),
        (37, 1024, 10),
        (12, 8192, 9),
    ];
    let mut compared = 0;
    for (case, &(m, k, n)) in shapes.iter().enumerate() {
        let a = values(1 + case as u64, m * k);
        let b = values(101 + case as u64, k * n);
        let want = reference32(&a, &b, m, k, n);
        for kernel in kernels() {
            for lhs in [Lhs::RowMajor, Lhs::Transposed] {
                let bound = match lhs {
                    Lhs::RowMajor => a.clone(),
                    Lhs::Transposed => transposed(&a, m, k),
                };
                let got = matmul(kernel, &bound, lhs, &b, m, k, n).expect("runs");
                let wrong = got
                    .iter()
                    .zip(&want)
                    .filter(|(g, w)| g.to_bits() != w.to_bits())
                    .count();
                assert_eq!(wrong, 0, "[{m}, {k}] x [{k}, {n}] {kernel:?} {lhs:?}");
                compared += got.len();
            }
        }
        let a64: Vec<f64> = a.iter().map(|&v| f64::from(v)).collect();
        let b64: Vec<f64> = b.iter().map(|&v| f64::from(v)).collect();
        let want64 = reference64(&a64, &b64, m, k, n);
        for kernel in kernels() {
            let got = matmul(kernel, &a64, Lhs::RowMajor, &b64, m, k, n).expect("runs");
            assert_eq!(
                bits64(&got),
                bits64(&want64),
                "[{m}, {k}] x [{k}, {n}] {kernel:?} f64"
            );
        }
    }
    eprintln!("cpu_fma_oracle: {compared} f32 elements equal the fused reference");
}

fn program(build: impl FnOnce(&mut TensorProgramBuilder)) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    build(&mut b);
    b.finish()
}

fn run32(program: &TensorProgram, inputs: Vec<(&str, Vec<u64>, Vec<f32>)>) -> Vec<f32> {
    let inputs: BTreeMap<String, HostTensor<f32>> = inputs
        .into_iter()
        .map(|(name, shape, data)| {
            (
                name.to_owned(),
                HostTensor::new(shape, data).expect("a tensor"),
            )
        })
        .collect();
    let outputs = evaluate(program, &inputs, &BTreeMap::new()).expect("evaluates");
    outputs[0].data().to_vec()
}

/// The interpreter's `mse` sums the squares of `d = a - b` from the first term, each later
/// square fused into the sum: for `d = [2^-12, 1 + 2^-12]` the sum is `2^-24`, then
/// `fma(d1, d1, 2^-24) = 1 + 2^-11 + 2^-23` exactly, where a rounded square `1 + 2^-11` plus
/// `2^-24` ties back to `1 + 2^-11`. Then the division by 2.
#[test]
fn the_interpreter_mse_fuses_each_later_square_into_its_sum() {
    let p = program(|b| {
        let x = b.input("a", TensorType::f32(&[2])).unwrap();
        let y = b.input("b", TensorType::f32(&[2])).unwrap();
        let z = b.mse(x, y).unwrap();
        b.output(z).unwrap();
    });
    let d = [2f32.powi(-12), 1.0 + 2f32.powi(-12)];
    let got = run32(
        &p,
        vec![("a", vec![2], d.to_vec()), ("b", vec![2], vec![0.0, 0.0])],
    );
    let fused = (1.0 + 2f32.powi(-11) + 2f32.powi(-23)) / 2.0;
    assert_ne!(
        (d[0] * d[0] + d[1] * d[1]) / 2.0,
        fused,
        "the case separates"
    );
    assert_eq!(got[0].to_bits(), fused.to_bits(), "{got:?}");

    // Drawn operands over the shape of the MNIST loss and of an `mlp_sweep` one.
    for (case, shape) in [(0u64, [100u64, 10u64]), (1, [8192, 10])] {
        let count = (shape[0] * shape[1]) as usize;
        let a = values(7 + case, count);
        let t = values(17 + case, count);
        let mut acc = 0.0f32;
        for (i, (&x, &y)) in a.iter().zip(&t).enumerate() {
            let d = x - y;
            acc = if i == 0 { d * d } else { d.mul_add(d, acc) };
        }
        let want = acc / count as f32;
        let p = program(|b| {
            let x = b.input("a", TensorType::f32(&shape)).unwrap();
            let y = b.input("b", TensorType::f32(&shape)).unwrap();
            let z = b.mse(x, y).unwrap();
            b.output(z).unwrap();
        });
        let got = run32(&p, vec![("a", shape.to_vec(), a), ("b", shape.to_vec(), t)]);
        assert_eq!(got[0].to_bits(), want.to_bits(), "{shape:?}");
    }
}

/// The interpreter's `matmul`, of a row-major operand and of a `transpose` it reads in place,
/// gives the fused reference's bits.
#[test]
fn the_interpreter_matmul_gives_the_bits_of_the_fused_reference() {
    let (m, k, n) = (13, 300, 21);
    let a = values(31, m * k);
    let b = values(37, k * n);
    let want = reference32(&a, &b, m, k, n);
    let plain = program(|bld| {
        let x = bld
            .input("a", TensorType::f32(&[m as u64, k as u64]))
            .unwrap();
        let y = bld
            .input("b", TensorType::f32(&[k as u64, n as u64]))
            .unwrap();
        let z = bld.matmul(x, y).unwrap();
        bld.output(z).unwrap();
    });
    let got = run32(
        &plain,
        vec![
            ("a", vec![m as u64, k as u64], a.clone()),
            ("b", vec![k as u64, n as u64], b.clone()),
        ],
    );
    assert_eq!(bits32(&got), bits32(&want), "row-major");
    let folded = program(|bld| {
        let x = bld
            .input("at", TensorType::f32(&[k as u64, m as u64]))
            .unwrap();
        let y = bld
            .input("b", TensorType::f32(&[k as u64, n as u64]))
            .unwrap();
        let xt = bld.transpose(x).unwrap();
        let z = bld.matmul(xt, y).unwrap();
        bld.output(z).unwrap();
    });
    let got = run32(
        &folded,
        vec![
            ("at", vec![k as u64, m as u64], transposed(&a, m, k)),
            ("b", vec![k as u64, n as u64], b),
        ],
    );
    assert_eq!(bits32(&got), bits32(&want), "transposed in place");
}
