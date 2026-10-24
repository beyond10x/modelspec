//! The CPU `matmul` against the fused loop, bit for bit (story `cpu-matmul-row-order`).
//!
//! DESIGN.md § 2.2 fixes the reduction order of `matmul`: each output element sums its `K`
//! products in ascending `k`, starting from the `k = 0` product, and § 12 ADR-009 fuses each later
//! product into the sum (`mul_add`, one rounding). The 0.1.1 interpreter computed the sum with
//! loop order `i`, `j`, `k`; that loop, with each later product fused, is kept here as the
//! reference, and the interpreter's result must equal it by bit pattern on every seeded case, in
//! `f32` and `f64`,
//! with signed zeros, infinities, NaN, subnormal and overflowing inputs among the operands.
//! A NaN result compares as NaN only: Rust does not fix the sign or payload of an arithmetic NaN,
//! and an optimized build of the same loop picks either (story `cpu-matmul-avx2-blocking`,
//! acceptance 2). Every other result compares by bit pattern.
//!
//! The planted defect of the story, a row accumulator that starts from `+0` instead of the
//! `k = 0` products, is kept here too, and the same cases must catch it on a `-0.0` element.
//!
//! Story `cpu-matmul-avx2-blocking` adds the AVX2 kernel and the in-place read of a transposed
//! left operand: each kernel, forced by name, and each left-operand layout must equal the same
//! loop on the same cases, and so must a `transpose(a) @ b` program through `evaluate` (which
//! reads the transpose in place) and through `evaluate_all` (which keeps every value).

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::{TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::matmul::{self, Kernel, Lhs};
use modelspec_runtime::cpu::{Element, HostTensor, evaluate, evaluate_all};

/// An element type the cases run in, with its bit pattern and its special values.
trait Real: Element {
    const NAME: &'static str;
    const SPECIALS: [Self; 9];
    fn of(v: f64) -> Self;
    fn bits(self) -> u64;
}

impl Real for f32 {
    const NAME: &'static str = "f32";
    const SPECIALS: [Self; 9] = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE / 8.0,
    ];
    fn of(v: f64) -> Self {
        v as f32
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl Real for f64 {
    const NAME: &'static str = "f64";
    const SPECIALS: [Self; 9] = [
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::MAX,
        -f64::MAX,
        f64::MIN_POSITIVE / 8.0,
    ];
    fn of(v: f64) -> Self {
        v
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

const MAX_M: usize = 64;
const MAX_K: usize = 784;
const MAX_N: usize = 64;

/// SplitMix64, the seeded source of every case.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// `1..=max`, weighted towards small sizes so the suite stays fast, reaching `max`.
    fn size(&mut self, max: usize) -> usize {
        let u = self.unit();
        1 + ((u * u * max as f64) as usize).min(max - 1)
    }
}

/// How a case draws its elements.
#[derive(Clone, Copy, Debug)]
enum Draw {
    /// Finite values of mixed sign and magnitude, a special value with probability `rate`.
    Mixed { rate: f64 },
    /// Only `+0`, `-0`, `1` and `-1`: many elements whose products are all `-0`.
    SignedZeros,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    m: usize,
    k: usize,
    n: usize,
    seed: u64,
    draw: Draw,
}

/// Row counts on both sides of the AVX2 kernel's row blocks (6) and column counts on both sides of
/// its lane widths (8 `f32`, 4 `f64`) and column panels (16 `f32`, 8 `f64`).
const EDGE_M: [usize; 8] = [1, 2, 5, 6, 7, 11, 12, 13];
const EDGE_N: [usize; 14] = [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 33, 63];

/// 2 cases at the largest shape, 40 of signed zeros at small `K`, 200 of random shapes and 112 at
/// the block edges.
fn cases() -> Vec<Case> {
    let mut rng = Rng(0x6d61_746d_756c);
    let mut cases = vec![
        Case {
            m: MAX_M,
            k: MAX_K,
            n: MAX_N,
            seed: 1,
            draw: Draw::Mixed { rate: 0.0 },
        },
        Case {
            m: MAX_M,
            k: MAX_K,
            n: MAX_N,
            seed: 2,
            draw: Draw::Mixed { rate: 1e-4 },
        },
    ];
    for _ in 0..40 {
        cases.push(Case {
            m: 1 + rng.below(8),
            k: 1 + rng.below(4),
            n: 1 + rng.below(8),
            seed: rng.next(),
            draw: Draw::SignedZeros,
        });
    }
    for i in 0..200 {
        let rate = [0.0, 1e-3, 1e-2, 0.1][i % 4];
        cases.push(Case {
            m: rng.size(MAX_M),
            k: rng.size(MAX_K),
            n: rng.size(MAX_N),
            seed: rng.next(),
            draw: Draw::Mixed { rate },
        });
    }
    for (i, (&m, &n)) in EDGE_M
        .iter()
        .flat_map(|m| EDGE_N.iter().map(move |n| (m, n)))
        .enumerate()
    {
        let draw = if i % 3 == 0 {
            Draw::SignedZeros
        } else {
            Draw::Mixed { rate: 0.05 }
        };
        cases.push(Case {
            m,
            k: [1, 2, 3, 17, 100][i % 5],
            n,
            seed: rng.next(),
            draw,
        });
    }
    cases
}

/// The left operand of a case as `[k, m]` row-major: what `transpose(a)` holds.
fn transposed<T: Copy>(a: &[T], m: usize, k: usize) -> Vec<T> {
    (0..k)
        .flat_map(|p| (0..m).map(move |i| a[i * k + p]))
        .collect()
}

/// `matmul(a, b)` by the named kernel, the left operand in the named layout.
fn kernel<T: Element>(
    kernel: Kernel,
    lhs: Lhs,
) -> impl Fn(&[T], &[T], usize, usize, usize) -> Vec<T> {
    move |a, b, m, k, n| {
        let a = match lhs {
            Lhs::RowMajor => a.to_vec(),
            Lhs::Transposed => transposed(a, m, k),
        };
        matmul::matmul(kernel, &a, lhs, b, m, k, n)
            .unwrap_or_else(|| panic!("{kernel:?} is not available for this element type"))
    }
}

/// `transpose(at) @ b` evaluated by the CPU interpreter, `at` being `a` transposed: through
/// `evaluate` when `all` is false, through `evaluate_all` otherwise.
fn interpreter_transposed<T: Real>(
    all: bool,
) -> impl Fn(&[T], &[T], usize, usize, usize) -> Vec<T> {
    move |a, b, m, k, n| {
        let at = transposed(a, m, k);
        let (m, k, n) = (m as u64, k as u64, n as u64);
        let mut bld = TensorProgramBuilder::new();
        let x = bld.input("at", TensorType::f32(&[k, m])).unwrap();
        let y = bld.input("b", TensorType::f32(&[k, n])).unwrap();
        let xt = bld.transpose(x).unwrap();
        let z = bld.matmul(xt, y).unwrap();
        bld.output(z).unwrap();
        let program = bld.finish();
        let inputs = BTreeMap::from([
            ("at".to_string(), HostTensor::new(vec![k, m], at).unwrap()),
            (
                "b".to_string(),
                HostTensor::new(vec![k, n], b.to_vec()).unwrap(),
            ),
        ]);
        let out = if all {
            let evaluation = evaluate_all(&program, &inputs, &BTreeMap::new()).unwrap();
            let bits = |v: &[T]| v.iter().map(|x| x.bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(evaluation.values[&xt].data()),
                bits(&transposed(
                    evaluation.values[&x].data(),
                    k as usize,
                    m as usize
                )),
                "evaluate_all holds the transpose's own value"
            );
            evaluation.outputs
        } else {
            evaluate(&program, &inputs, &BTreeMap::new()).unwrap()
        };
        assert_eq!(out[0].shape(), &[m, n]);
        out[0].data().to_vec()
    }
}

fn element<T: Real>(rng: &mut Rng, draw: Draw) -> T {
    match draw {
        Draw::SignedZeros => [T::of(0.0), T::of(-0.0), T::of(1.0), T::of(-1.0)][rng.below(4)],
        Draw::Mixed { rate } => {
            if rng.unit() < rate {
                T::SPECIALS[rng.below(T::SPECIALS.len())]
            } else {
                let sign = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
                let exponent = rng.below(33) as i32 - 16;
                T::of(sign * (1.0 + rng.unit()) * 2f64.powi(exponent))
            }
        }
    }
}

/// The operands `a: [m, k]` and `b: [k, n]` of a case, row-major.
fn operands<T: Real>(case: &Case) -> (Vec<T>, Vec<T>) {
    let mut rng = Rng(case.seed);
    let a = (0..case.m * case.k)
        .map(|_| element(&mut rng, case.draw))
        .collect();
    let b = (0..case.k * case.n)
        .map(|_| element(&mut rng, case.draw))
        .collect();
    (a, b)
}

/// The 0.1.1 interpreter's loop: `i`, `j`, then the `K` products in ascending `k` summed from the
/// `k = 0` product, each later product fused into the sum (DESIGN.md § 12 ADR-009).
fn matmul_fused<T: Element>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(sum_fused((0..k).map(|p| (a[i * k + p], b[p * n + j]))));
        }
    }
    data
}

/// The sequential sum of the products of `factors`, starting from the first product, each later
/// one fused into the sum.
fn sum_fused<T: Element>(mut factors: impl Iterator<Item = (T, T)>) -> T {
    match factors.next() {
        Some((x, y)) => factors.fold(x * y, |acc, (x, y)| x.mul_add(y, acc)),
        None => T::ZERO,
    }
}

/// The planted defect: loop order `i`, `k`, `j` with the row accumulator started from `+0`, each
/// product fused into it as in the reference.
fn matmul_row_from_zero<T: Element>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        let mut row = vec![T::ZERO; n];
        for p in 0..k {
            for j in 0..n {
                row[j] = a[i * k + p].mul_add(b[p * n + j], row[j]);
            }
        }
        data.extend(row);
    }
    data
}

/// `matmul(a, b)` evaluated by the CPU interpreter.
fn interpreter<T: Element>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let (m, k, n) = (m as u64, k as u64, n as u64);
    let mut bld = TensorProgramBuilder::new();
    let x = bld.input("a", TensorType::f32(&[m, k])).unwrap();
    let y = bld.input("b", TensorType::f32(&[k, n])).unwrap();
    let z = bld.matmul(x, y).unwrap();
    bld.output(z).unwrap();
    let program = bld.finish();
    let inputs = BTreeMap::from([
        (
            "a".to_string(),
            HostTensor::new(vec![m, k], a.to_vec()).unwrap(),
        ),
        (
            "b".to_string(),
            HostTensor::new(vec![k, n], b.to_vec()).unwrap(),
        ),
    ]);
    let out = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
    assert_eq!(out[0].shape(), &[m, n]);
    out[0].data().to_vec()
}

/// One element where `candidate` and the fused loop differ in bits.
struct Mismatch {
    case: Case,
    dtype: &'static str,
    index: usize,
    expected: u64,
    got: u64,
}

/// One line per mismatch: the case, the element and both bit patterns.
fn report(found: &[Mismatch]) -> String {
    let lines: Vec<String> = found
        .iter()
        .map(|m| {
            format!(
                "{} {:?} element {}: fused {:#x}, got {:#x}",
                m.dtype, m.case, m.index, m.expected, m.got
            )
        })
        .collect();
    format!("{} case(s) differ:\n{}", found.len(), lines.join("\n"))
}

/// `got` is `expected`: both NaN, or the same bit pattern.
fn same<T: Real>(expected: T, got: T) -> bool {
    #[allow(clippy::eq_op)]
    let nan = |x: T| x != x;
    (nan(expected) && nan(got)) || expected.bits() == got.bits()
}

/// Every case on which `candidate` differs from the fused loop, with its first differing element.
fn mismatches<T: Real>(
    candidate: impl Fn(&[T], &[T], usize, usize, usize) -> Vec<T>,
) -> Vec<Mismatch> {
    let mut found = Vec::new();
    for case in cases() {
        let (a, b) = operands::<T>(&case);
        let expected = matmul_fused(&a, &b, case.m, case.k, case.n);
        let got = candidate(&a, &b, case.m, case.k, case.n);
        assert_eq!(got.len(), expected.len(), "{case:?}");
        if let Some(index) = (0..got.len()).find(|&i| !same(expected[i], got[i])) {
            found.push(Mismatch {
                case,
                dtype: T::NAME,
                index,
                expected: expected[index].bits(),
                got: got[index].bits(),
            });
        }
    }
    found
}

/// The suite is what the story asks for: at least 200 shapes, the largest `[64, 784] x [784,
/// 64]`, and its operands hold `-0.0`, infinities and NaN.
#[test]
fn the_cases_cover_the_largest_shape_and_every_special_value() {
    let cases = cases();
    assert!(cases.len() >= 200, "{} cases", cases.len());
    assert!(
        cases
            .iter()
            .any(|c| (c.m, c.k, c.n) == (MAX_M, MAX_K, MAX_N))
    );
    assert!(
        cases
            .iter()
            .all(|c| c.m <= MAX_M && c.k <= MAX_K && c.n <= MAX_N)
    );
    let mut seen = [false; 3];
    for case in &cases {
        let (a, b) = operands::<f32>(case);
        for x in a.iter().chain(&b) {
            seen[0] |= x.to_bits() == (-0.0f32).to_bits();
            seen[1] |= x.is_infinite();
            seen[2] |= x.is_nan();
        }
    }
    assert_eq!(seen, [true; 3], "-0.0, infinity, NaN among the operands");
}

#[test]
fn matmul_equals_the_fused_loop_bit_for_bit_in_f32() {
    let found = mismatches::<f32>(interpreter);
    assert!(found.is_empty(), "{}", report(&found));
}

#[test]
fn matmul_equals_the_fused_loop_bit_for_bit_in_f64() {
    let found = mismatches::<f64>(interpreter);
    assert!(found.is_empty(), "{}", report(&found));
}

/// Acceptance 4: a row accumulator started from `+0` turns an element whose products are all
/// `-0.0` into `+0.0`, and the cases above see it in both element types.
#[test]
fn a_row_accumulator_started_from_zero_is_caught_on_a_negative_zero() {
    fn caught_on_negative_zero<T: Real>() -> bool {
        mismatches::<T>(matmul_row_from_zero)
            .iter()
            .any(|m| m.expected == T::of(-0.0).bits() && m.got == T::of(0.0).bits())
    }
    assert!(caught_on_negative_zero::<f32>(), "f32");
    assert!(caught_on_negative_zero::<f64>(), "f64");
}

/// The block-edge cases are there: every edge row count meets every edge column count, and
/// some of them are not a multiple of any block or lane width.
#[test]
fn the_cases_cross_every_block_edge() {
    let cases = cases();
    for &m in &EDGE_M {
        for &n in &EDGE_N {
            assert!(cases.iter().any(|c| (c.m, c.n) == (m, n)), "{m} x {n}");
        }
    }
    assert!(
        cases
            .iter()
            .any(|c| c.m % 6 != 0 && c.n % 8 != 0 && c.n > 16)
    );
}

/// The kernels this host can run: the portable one always, AVX2 where the CPU has AVX2 and FMA.
fn kernels() -> Vec<Kernel> {
    #[cfg(target_arch = "x86_64")]
    let avx2 =
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let avx2 = false;
    [Kernel::Portable]
        .into_iter()
        .chain(avx2.then_some(Kernel::Avx2))
        .collect()
}

#[test]
fn each_kernel_is_available_exactly_where_the_cpu_has_it() {
    assert!(Kernel::Portable.is_available());
    assert_eq!(
        Kernel::Avx2.is_available(),
        kernels().contains(&Kernel::Avx2)
    );
    assert_eq!(Kernel::detected(), *kernels().last().unwrap());
    assert!(matmul::matmul::<f32>(Kernel::Portable, &[], Lhs::RowMajor, &[], 0, 0, 0).is_some());
}

fn kernel_equals_the_fused_loop<T: Real>() {
    for kernel_name in kernels() {
        for lhs in [Lhs::RowMajor, Lhs::Transposed] {
            let found = mismatches::<T>(kernel(kernel_name, lhs));
            assert!(
                found.is_empty(),
                "{kernel_name:?} {lhs:?}: {}",
                report(&found)
            );
        }
    }
}

#[test]
fn each_kernel_equals_the_fused_loop_bit_for_bit_in_f32() {
    kernel_equals_the_fused_loop::<f32>();
}

#[test]
fn each_kernel_equals_the_fused_loop_bit_for_bit_in_f64() {
    kernel_equals_the_fused_loop::<f64>();
}

/// This host runs the AVX2 kernel; on one without it the test above covers the portable kernel
/// only, and this one says so by failing.
#[cfg(target_arch = "x86_64")]
#[test]
fn this_host_runs_the_avx2_kernel() {
    assert!(
        kernels().contains(&Kernel::Avx2),
        "the CPU has no AVX2: the AVX2 kernel was not tested"
    );
}

/// `m`, `k` or `n` zero: `m x n` elements, each `+0` when `k` is zero.
#[test]
fn each_kernel_handles_a_zero_dimension() {
    for kernel_name in kernels() {
        for lhs in [Lhs::RowMajor, Lhs::Transposed] {
            for (m, k, n) in [(0, 0, 0), (0, 3, 2), (2, 0, 3), (2, 3, 0), (7, 0, 17)] {
                let a = vec![-1.5f32; m * k];
                let b = vec![-2.0f32; k * n];
                let got = matmul::matmul(kernel_name, &a, lhs, &b, m, k, n).unwrap();
                let want = if k == 0 { 0.0 } else { 3.0 };
                assert_eq!(got.len(), m * n, "{kernel_name:?} {lhs:?} {m} {k} {n}");
                assert!(
                    got.iter().all(|x| x.to_bits() == f32::to_bits(want)),
                    "{kernel_name:?} {lhs:?} {m} {k} {n}: {got:?}"
                );
            }
        }
    }
}

#[test]
fn a_transposed_left_operand_equals_the_fused_loop_through_evaluate() {
    for found in [
        mismatches::<f32>(interpreter_transposed(false)),
        mismatches::<f64>(interpreter_transposed(false)),
    ] {
        assert!(found.is_empty(), "{}", report(&found));
    }
}

/// A transpose that is also an output, or also the right operand of a `matmul`, still has its
/// value through `evaluate`, and every product still equals the fused loop.
#[test]
fn a_transpose_read_elsewhere_keeps_its_value_through_evaluate() {
    let (m, k) = (5usize, 7usize);
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.5 - 3.0).collect();
    let at = transposed(&a, m, k);
    let mut bld = TensorProgramBuilder::new();
    let x = bld
        .input("at", TensorType::f32(&[k as u64, m as u64]))
        .unwrap();
    let xt = bld.transpose(x).unwrap();
    let left = bld.matmul(xt, x).unwrap();
    let right = bld.matmul(x, xt).unwrap();
    bld.output(left).unwrap();
    bld.output(right).unwrap();
    bld.output(xt).unwrap();
    let program = bld.finish();
    let inputs = BTreeMap::from([(
        "at".to_string(),
        HostTensor::new(vec![k as u64, m as u64], at.clone()).unwrap(),
    )]);
    let out = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(out[0].data()), bits(&matmul_fused(&a, &at, m, k, m)));
    assert_eq!(bits(out[1].data()), bits(&matmul_fused(&at, &a, k, m, k)));
    assert_eq!(bits(out[2].data()), bits(&a));
}

#[test]
fn a_transposed_left_operand_equals_the_fused_loop_through_evaluate_all() {
    for found in [
        mismatches::<f32>(interpreter_transposed(true)),
        mismatches::<f64>(interpreter_transposed(true)),
    ] {
        assert!(found.is_empty(), "{}", report(&found));
    }
}
