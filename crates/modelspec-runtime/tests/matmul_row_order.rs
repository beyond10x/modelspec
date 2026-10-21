//! The CPU `matmul` against the 0.1.1 loop, bit for bit (story `cpu-matmul-row-order`).
//!
//! DESIGN.md § 2.2 fixes the reduction order of `matmul`: each output element sums its `K`
//! products in ascending `k`, starting from the first term, nothing fused. The 0.1.1 interpreter
//! computed it with loop order `i`, `j`, `k`; that loop is kept here as the reference, and the
//! interpreter's result must equal it by bit pattern on every seeded case, in `f32` and `f64`,
//! with signed zeros, infinities, NaN, subnormal and overflowing inputs among the operands.
//!
//! The planted defect of the story, a row accumulator that starts from `+0` instead of the
//! `k = 0` products, is kept here too, and the same cases must catch it on a `-0.0` element.

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::{TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::{Element, HostTensor, evaluate};

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

/// 2 cases at the largest shape, 40 of signed zeros at small `K`, 200 of random shapes.
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
    cases
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

/// The 0.1.1 interpreter's `matmul`, verbatim in its loop: `i`, `j`, then the `K` products in
/// ascending `k` summed from the first term.
fn matmul_0_1_1<T: Element>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(sum_0_1_1((0..k).map(|p| a[i * k + p] * b[p * n + j])));
        }
    }
    data
}

/// The 0.1.1 interpreter's sequential sum, starting from the first term.
fn sum_0_1_1<T: Element>(mut terms: impl Iterator<Item = T>) -> T {
    match terms.next() {
        Some(first) => terms.fold(first, |acc, x| acc + x),
        None => T::ZERO,
    }
}

/// The planted defect: loop order `i`, `k`, `j` with the row accumulator started from `+0`.
fn matmul_row_from_zero<T: Element>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        let mut row = vec![T::ZERO; n];
        for p in 0..k {
            for j in 0..n {
                row[j] = row[j] + a[i * k + p] * b[p * n + j];
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
        ("a".to_string(), HostTensor::new(vec![m, k], a.to_vec()).unwrap()),
        ("b".to_string(), HostTensor::new(vec![k, n], b.to_vec()).unwrap()),
    ]);
    let out = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
    assert_eq!(out[0].shape(), &[m, n]);
    out[0].data().to_vec()
}

/// One element where `candidate` and the 0.1.1 loop differ in bits.
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
                "{} {:?} element {}: 0.1.1 {:#x}, got {:#x}",
                m.dtype, m.case, m.index, m.expected, m.got
            )
        })
        .collect();
    format!("{} case(s) differ:\n{}", found.len(), lines.join("\n"))
}

/// Every case on which `candidate` differs from the 0.1.1 loop, with its first differing element.
fn mismatches<T: Real>(
    candidate: impl Fn(&[T], &[T], usize, usize, usize) -> Vec<T>,
) -> Vec<Mismatch> {
    let mut found = Vec::new();
    for case in cases() {
        let (a, b) = operands::<T>(&case);
        let expected = matmul_0_1_1(&a, &b, case.m, case.k, case.n);
        let got = candidate(&a, &b, case.m, case.k, case.n);
        assert_eq!(got.len(), expected.len(), "{case:?}");
        if let Some(index) = (0..got.len()).find(|&i| got[i].bits() != expected[i].bits()) {
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
fn matmul_equals_the_0_1_1_loop_bit_for_bit_in_f32() {
    let found = mismatches::<f32>(interpreter);
    assert!(found.is_empty(), "{}", report(&found));
}

#[test]
fn matmul_equals_the_0_1_1_loop_bit_for_bit_in_f64() {
    let found = mismatches::<f64>(interpreter);
    assert!(found.is_empty(), "{}", report(&found));
}

/// Acceptance 4: a row accumulator started from `+0` turns an element whose products are all
/// `-0.0` into `+0.0`, and the cases above see it in both element types.
#[test]
fn a_row_accumulator_started_from_zero_is_caught_on_a_negative_zero() {
    fn caught_on_negative_zero<T: Real>() -> bool {
        mismatches::<T>(matmul_row_from_zero).iter().any(|m| {
            m.expected == T::of(-0.0).bits() && m.got == T::of(0.0).bits()
        })
    }
    assert!(caught_on_negative_zero::<f32>(), "f32");
    assert!(caught_on_negative_zero::<f64>(), "f64");
}
