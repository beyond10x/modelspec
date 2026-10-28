//! Adversary pass 2 for story `cpu-matmul-output-skips-zero-fill`: the counter behind
//! [`matmul::output_zero_fills`] moves where a fill happens.
//!
//! `matmul_output_zero_fills.rs` asserts only that the counter does not move for `f32` and `f64`.
//! That assertion also holds if the increment is deleted or the reader returns a constant 0, so
//! this binary asserts the presence side: under `Kernel::Avx2`, an element type with no vector
//! code fills its output once per call where `k > 0` and `m * n > 0` (its own hook then runs over
//! the zeros), and not at all where `k` is 0 (the zeros are the result) or `m * n` is 0. The
//! counter is global to the process, so this binary holds one test only.

use std::ops::{Add, Div, Mul, Sub};

use modelspec_runtime::cpu::Element;
use modelspec_runtime::cpu::matmul::{self, Kernel, Lhs};

/// An `f64` with no AVX2 code: the default [`Element::matmul_avx2`] hook, which declines.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct Plain(f64);

impl Add for Plain {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Plain(self.0 + rhs.0)
    }
}

impl Sub for Plain {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Plain(self.0 - rhs.0)
    }
}

impl Mul for Plain {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Plain(self.0 * rhs.0)
    }
}

impl Div for Plain {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Plain(self.0 / rhs.0)
    }
}

impl Element for Plain {
    const ZERO: Self = Plain(0.0);
    fn from_f32(value: f32) -> Self {
        Plain(f64::from(value))
    }
    fn from_count(count: u64) -> Self {
        Plain(count as f64)
    }
    fn to_f64(self) -> f64 {
        self.0
    }
    fn from_f64(value: f64) -> Self {
        Plain(value)
    }
    fn exp(self) -> Self {
        Plain(self.0.exp())
    }
    fn sqrt(self) -> Self {
        Plain(self.0.sqrt())
    }
    fn ln(self) -> Self {
        Plain(self.0.ln())
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = <f64 as Element>::sin_cos(self.0);
        (Plain(sin), Plain(cos))
    }
}

fn operands<T: Element>(m: usize, k: usize, n: usize) -> (Vec<T>, Vec<T>) {
    let a = (0..m * k)
        .map(|i| T::from_f32(i as f32 * 0.5 - 2.0))
        .collect();
    let b = (0..k * n)
        .map(|i| T::from_f32(1.5 - i as f32 * 0.25))
        .collect();
    (a, b)
}

/// The counter's movement over one `matmul` call, and the call's result length.
fn moved<T: Element>(
    kernel: Kernel,
    lhs: Lhs,
    (m, k, n): (usize, usize, usize),
) -> (usize, Option<usize>) {
    let (a, b) = operands::<T>(m, k, n);
    let before = matmul::output_zero_fills();
    let out = matmul::matmul(kernel, &a, lhs, &b, m, k, n);
    (matmul::output_zero_fills() - before, out.map(|v| v.len()))
}

#[test]
fn the_counter_moves_once_per_fill_and_not_for_k_0_m_n_0_f32_f64_or_the_portable_kernel() {
    let positive = [(1, 1, 1), (3, 4, 5), (7, 3, 17), (13, 9, 33), (6, 1, 8)];
    let k_zero = [(1, 0, 1), (4, 0, 7), (13, 0, 33)];
    let mn_zero = [(0, 3, 4), (4, 3, 0)];
    let avx2 = Kernel::Avx2.is_available();
    let mut bad = Vec::new();
    for lhs in [Lhs::RowMajor, Lhs::Transposed] {
        for s in positive {
            // A type without vector code: filled once, then declined by its hook.
            let want = if avx2 { (1, None) } else { (0, None) };
            let got = moved::<Plain>(Kernel::Avx2, lhs, s);
            if got != want {
                bad.push(format!("Plain Avx2 {lhs:?} {s:?}: {got:?}, want {want:?}"));
            }
            for (name, got) in [
                ("Plain Portable", moved::<Plain>(Kernel::Portable, lhs, s)),
                ("f32 Portable", moved::<f32>(Kernel::Portable, lhs, s)),
                ("f64 Portable", moved::<f64>(Kernel::Portable, lhs, s)),
            ] {
                if got != (0, Some(s.0 * s.2)) {
                    bad.push(format!(
                        "{name} {lhs:?} {s:?}: {got:?}, want (0, Some(m * n))"
                    ));
                }
            }
            if avx2 {
                for (name, got) in [
                    ("f32 Avx2", moved::<f32>(Kernel::Avx2, lhs, s)),
                    ("f64 Avx2", moved::<f64>(Kernel::Avx2, lhs, s)),
                ] {
                    if got != (0, Some(s.0 * s.2)) {
                        bad.push(format!(
                            "{name} {lhs:?} {s:?}: {got:?}, want (0, Some(m * n))"
                        ));
                    }
                }
            }
        }
        for s in k_zero.into_iter().chain(mn_zero) {
            let want = (0, avx2.then_some(s.0 * s.2));
            let got = moved::<Plain>(Kernel::Avx2, lhs, s);
            if got != want {
                bad.push(format!("Plain Avx2 {lhs:?} {s:?}: {got:?}, want {want:?}"));
            }
        }
    }
    assert!(bad.is_empty(), "{} case(s) differ: {bad:#?}", bad.len());
}
