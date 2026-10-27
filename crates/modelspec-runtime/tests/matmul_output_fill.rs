//! The CPU `matmul` writes each output element once (story `cpu-matmul-output-skips-zero-fill`).
//!
//! The output used to be resized with `T::ZERO` before the kernel overwrote every element, a fill
//! the result never read. An element type whose `clone` is counted makes such a fill visible:
//! `Vec::resize` and `vec![x; n]` clone their value into each element, while the kernel only
//! copies products and sums. So a `matmul` that fills its output first clones `m * n - 1` times,
//! and one that writes each element once clones none. Where `k` is 0 the result is `m * n` zeros,
//! which are written, so that case may clone.
//!
//! The bits of the result are those of the 0.1.1 loop (`matmul_row_order.rs`), unchanged here.

use std::cell::Cell;
use std::ops::{Add, Div, Mul, Sub};

use modelspec_runtime::cpu::Element;
use modelspec_runtime::cpu::matmul::{self, Kernel, Lhs};

thread_local! {
    static CLONES: Cell<usize> = const { Cell::new(0) };
}

/// An `f32` whose `clone` counts itself in [`CLONES`]; a plain copy does not.
#[derive(Copy, Debug, PartialEq, PartialOrd)]
struct Counted(f32);

#[allow(
    clippy::non_canonical_clone_impl,
    reason = "the count is what makes a fill observable"
)]
impl Clone for Counted {
    fn clone(&self) -> Self {
        CLONES.with(|c| c.set(c.get() + 1));
        Counted(self.0)
    }
}

impl Add for Counted {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Counted(self.0 + rhs.0)
    }
}

impl Sub for Counted {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Counted(self.0 - rhs.0)
    }
}

impl Mul for Counted {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Counted(self.0 * rhs.0)
    }
}

impl Div for Counted {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Counted(self.0 / rhs.0)
    }
}

impl Element for Counted {
    const ZERO: Self = Counted(0.0);

    fn from_f32(value: f32) -> Self {
        Counted(value)
    }

    fn from_count(count: u64) -> Self {
        Counted(count as f32)
    }

    fn to_f64(self) -> f64 {
        f64::from(self.0)
    }

    fn from_f64(value: f64) -> Self {
        Counted(value as f32)
    }

    fn exp(self) -> Self {
        Counted(self.0.exp())
    }

    fn sqrt(self) -> Self {
        Counted(self.0.sqrt())
    }

    fn ln(self) -> Self {
        Counted(self.0.ln())
    }
}

/// The clones `f` makes on this thread, and its result.
fn clones<R>(f: impl FnOnce() -> R) -> (usize, R) {
    CLONES.with(|c| c.set(0));
    let result = f();
    (CLONES.with(Cell::get), result)
}

fn operand(len: usize, seed: f32) -> Vec<Counted> {
    (0..len)
        .map(|i| Counted(seed + i as f32 * 0.25 - 1.5))
        .collect()
}

/// The ascending-`k` sum of DESIGN.md § 2.2, from the `k = 0` product, in `f32`.
fn reference(a: &[f32], lhs: Lhs, b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let at = |i: usize, p: usize| match lhs {
        Lhs::RowMajor => a[i * k + p],
        Lhs::Transposed => a[p * m + i],
    };
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = if k == 0 { 0.0 } else { at(i, 0) * b[j] };
            for p in 1..k {
                acc += at(i, p) * b[p * n + j];
            }
            out.push(acc);
        }
    }
    out
}

#[test]
fn the_portable_kernel_writes_each_output_element_once() {
    for lhs in [Lhs::RowMajor, Lhs::Transposed] {
        for (m, k, n) in [(1, 1, 1), (3, 4, 5), (7, 1, 9), (1, 13, 2), (16, 8, 33)] {
            let a = operand(m * k, 0.5);
            let b = operand(k * n, -0.75);
            let (cloned, got) =
                clones(|| matmul::matmul(Kernel::Portable, &a, lhs, &b, m, k, n).unwrap());
            assert_eq!(
                cloned, 0,
                "{lhs:?} {m} x {k} x {n}: the output was filled before the kernel wrote it"
            );
            let plain = |v: &[Counted]| v.iter().map(|x| x.0).collect::<Vec<_>>();
            let want = reference(&plain(&a), lhs, &plain(&b), m, k, n);
            let got = plain(&got);
            assert_eq!(
                got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "{lhs:?} {m} x {k} x {n}"
            );
        }
    }
}

#[test]
fn a_zero_dimension_still_gives_m_times_n_positive_zeros_without_a_kernel_write() {
    for (m, k, n) in [(0, 5, 3), (4, 0, 7), (1, 0, 1), (3, 9, 0), (0, 0, 0)] {
        let a = operand(m * k, 0.5);
        let b = operand(k * n, -0.75);
        let got = matmul::matmul(Kernel::Portable, &a, Lhs::RowMajor, &b, m, k, n).unwrap();
        assert_eq!(got.len(), m * n, "{m} x {k} x {n}");
        assert!(
            got.iter().all(|x| x.0.to_bits() == 0),
            "{m} x {k} x {n}: {got:?}"
        );
    }
}

/// The AVX2 kernel has no code for a type other than `f32` and `f64`: it returns `None` and the
/// caller falls back to the portable kernel, as before.
#[test]
fn the_avx2_kernel_still_declines_a_type_it_has_no_code_for() {
    let a = operand(6, 0.5);
    let b = operand(6, -0.75);
    assert!(matmul::matmul(Kernel::Avx2, &a, Lhs::RowMajor, &b, 2, 3, 2).is_none());
}
