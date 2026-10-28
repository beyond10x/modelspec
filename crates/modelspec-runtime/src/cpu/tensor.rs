//! Host tensors: a shape and dense row-major storage, generic over the element type.
//!
//! The same code runs in `f32`, the M0 dtype, and in `f64`, which the finite-difference gradient
//! checks use as reference math (DESIGN.md § 4). Row ids are `HostTensor<i32>`, read only by
//! `embedding` and its gradient.

use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use thiserror::Error;

use super::matmul::Lhs;

/// An element type the CPU interpreter computes in.
///
/// Every operation rounds to this type: the interpreter evaluates an `f32`-typed program in
/// whichever `Element` it is instantiated with.
pub trait Element:
    Copy
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Send
    + Sync
    + 'static
{
    /// Positive zero.
    const ZERO: Self;

    /// An IR `f32` constant in this type; exact for `f32` and `f64`.
    fn from_f32(value: f32) -> Self;

    /// An element count in this type, rounded to nearest.
    fn from_count(count: u64) -> Self;

    /// `self` in `f64`, exact for `f32` and `f64`: a term of a whole-tensor sum, which the
    /// interpreter accumulates in `f64` (DESIGN.md § 2.2).
    fn to_f64(self) -> f64;

    /// `value` in this type, rounded to nearest once: the result of a whole-tensor sum.
    fn from_f64(value: f64) -> Self;

    /// `self * b + c`, the product fused into the sum: for `f32` and `f64` the IEEE 754 fused
    /// multiply-add, rounded once, as the GPU's `Fma` (DESIGN.md § 12 ADR-009). The interpreter
    /// computes every product after the first of a sum this way. The default, for a type with
    /// no fused operation, rounds the product and then the sum.
    fn mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }

    /// `e` raised to `self`, as the standard library computes it for `f32` and `f64`; the
    /// interpreter's `gelu` and its gradient use it (DESIGN.md § 4).
    fn exp(self) -> Self;

    /// The square root of `self`, correctly rounded as IEEE 754 requires and the standard library
    /// computes it for `f32` and `f64`; the interpreter's `layer_norm` and its gradients use it
    /// (DESIGN.md § 4).
    fn sqrt(self) -> Self;

    /// The natural logarithm of `self`, as the standard library computes it for `f32` and `f64`;
    /// the interpreter's `cross_entropy` and its gradient use it (DESIGN.md § 4).
    fn ln(self) -> Self;

    /// `(sin(self), cos(self))`: for `f32` the shared written-out routine [`sin_cos`], the one a
    /// Vulkan kernel is to reproduce, and for `f64` the standard library, the reference math of the
    /// gradient checks. The interpreter's `SigregColumns` and its gradient use it (DESIGN.md § 4).
    fn sin_cos(self) -> (Self, Self);

    /// The AVX2 `matmul` kernel for this type ([`Kernel::Avx2`](super::matmul::Kernel::Avx2)):
    /// writes `out`, `m * n` elements, and returns true; `m`, `k` and `n` are nonzero. False,
    /// writing nothing, where the type has no such kernel or the CPU no AVX2 or no FMA, which is
    /// the default. Panics when `m * k`, `k * n` or `m * n` overflows `usize` or a slice is not
    /// that long.
    #[doc(hidden)]
    fn matmul_avx2(
        out: &mut [Self],
        a: &[Self],
        _lhs: Lhs,
        b: &[Self],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        let (mk, kn, mn) = super::matmul::element_counts(m, k, n);
        assert!(out.len() == mn && a.len() == mk && b.len() == kn);
        false
    }
}

impl Element for f32 {
    const ZERO: Self = 0.0;

    fn from_f32(value: f32) -> Self {
        value
    }

    fn from_count(count: u64) -> Self {
        count as f32
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn mul_add(self, b: Self, c: Self) -> Self {
        f32::mul_add(self, b, c)
    }

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    fn ln(self) -> Self {
        f32::ln(self)
    }

    fn sin_cos(self) -> (Self, Self) {
        sin_cos(self)
    }

    #[cfg(target_arch = "x86_64")]
    fn matmul_avx2(
        out: &mut [Self],
        a: &[Self],
        lhs: Lhs,
        b: &[Self],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        super::matmul::avx2::f32x8::matmul(out, a, lhs, b, m, k, n)
    }
}

impl Element for f64 {
    const ZERO: Self = 0.0;

    fn from_f32(value: f32) -> Self {
        f64::from(value)
    }

    fn from_count(count: u64) -> Self {
        count as f64
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn mul_add(self, b: Self, c: Self) -> Self {
        f64::mul_add(self, b, c)
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn sin_cos(self) -> (Self, Self) {
        (f64::sin(self), f64::cos(self))
    }

    #[cfg(target_arch = "x86_64")]
    fn matmul_avx2(
        out: &mut [Self],
        a: &[Self],
        lhs: Lhs,
        b: &[Self],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        super::matmul::avx2::f64x4::matmul(out, a, lhs, b, m, k, n)
    }
}

/// `2 / pi` in `f32`, the scale of the quadrant count of [`sin_cos`].
const FRAC_2_PI: f32 = std::f32::consts::FRAC_2_PI;
/// `pi / 2` in three parts, `P1 + P2 + P3`, for Cody–Waite reduction: twice the constants `DP1`,
/// `DP2` and `DP3` of Cephes `sinf`.
const PI_2_P1: f32 = 1.570_312_5;
const PI_2_P2: f32 = 4.837_513e-4;
const PI_2_P3: f32 = 7.549_79e-8;
/// The minimax polynomial of `(sin(r) / r - 1) / z` in `z = r^2` on `|r| <= pi/4`, highest degree
/// first (Cephes `sinf`).
const SIN_P: [f32; 3] = [-1.951_529_6e-4, 8.332_161e-3, -1.666_665_5e-1];
/// The minimax polynomial of `(cos(r) - 1 + z/2) / z^2` in `z = r^2` on `|r| <= pi/4`, highest
/// degree first (Cephes `cosf`).
const COS_P: [f32; 3] = [2.443_315_7e-5, -1.388_731_6e-3, 4.166_664_6e-2];

/// `(sin(x), cos(x))` in `f32` by one written-out sequence of `f32` operations, the same on every
/// platform, so that a Vulkan kernel can emit it instruction for instruction (DESIGN.md § 4,
/// story:jepa-sigreg-cpu). It is not the platform `sinf`, and not `GLSL.std.450 Sin`, which only
/// promises `2^-11` absolute error on `[-pi, pi]`.
///
/// The quadrant count is `q = round_ties_even(x * (2/pi))`. Cody–Waite reduction gives
/// `r = x - q (P1 + P2 + P3)` as three fused multiply-adds, `r = fma(-q, P1, x)`, then `P2`, then
/// `P3`. With `z = r * r`, `s = r + (z ((S0 z + S1) z + S2)) r` and
/// `c = (1 - z/2) + (z ((C0 z + C1) z + C2)) z`: Horner's rule with each step a fused
/// multiply-add, `1 - z/2` as `fma(-0.5, z, 1)`, and each final `(p * z) * r + r` and
/// `(p * z) * z + (1 - z/2)` the product `p * z` rounded and then one fused multiply-add. The
/// quadrant `q mod 4` (two's complement of `q as i32`) maps `(s, c)` to `(s, c)`, `(c, -s)`,
/// `(-s, -c)` or `(-c, s)`.
///
/// Measured: within 1.43 `f32` ulp of `f64` `sin` and `cos` at every multiple of `1/256` in
/// `[-64, 64]` (`tests/sin_cos.rs`). Outside `|x| < 2^9 pi/2` the reduction loses accuracy.
pub fn sin_cos(x: f32) -> (f32, f32) {
    let q = (x * FRAC_2_PI).round_ties_even();
    let r = (-q).mul_add(PI_2_P1, x);
    let r = (-q).mul_add(PI_2_P2, r);
    let r = (-q).mul_add(PI_2_P3, r);
    let z = r * r;
    let s = SIN_P[0].mul_add(z, SIN_P[1]).mul_add(z, SIN_P[2]);
    let s = (s * z).mul_add(r, r);
    let c = COS_P[0].mul_add(z, COS_P[1]).mul_add(z, COS_P[2]);
    let c = (c * z).mul_add(z, (-0.5f32).mul_add(z, 1.0));
    match (q as i32) & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// Why a host tensor could not be made.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TensorError {
    #[error("shape {shape:?} holds {expected} element(s); got {got}")]
    LengthMismatch {
        shape: Vec<u64>,
        expected: usize,
        got: usize,
    },
    #[error("element count of shape {shape:?} overflows")]
    CountOverflow { shape: Vec<u64> },
}

/// A dense tensor in host memory: `data` holds the elements in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor<T> {
    shape: Vec<u64>,
    data: Vec<T>,
}

/// Any `Copy` element: an `Element` the interpreter computes in, or the `i32` of row ids.
impl<T: Copy> HostTensor<T> {
    /// A tensor of shape `shape` holding `data` in row-major order; `data` has exactly as many
    /// elements as the shape.
    pub fn new(shape: Vec<u64>, data: Vec<T>) -> Result<Self, TensorError> {
        let expected = element_count(&shape)?;
        if data.len() != expected {
            return Err(TensorError::LengthMismatch {
                shape,
                expected,
                got: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    /// A rank-0 tensor.
    pub fn scalar(value: T) -> Self {
        Self {
            shape: Vec::new(),
            data: vec![value],
        }
    }

    /// `shape` and `data` whose lengths the caller has already matched.
    pub(crate) fn from_parts(shape: Vec<u64>, data: Vec<T>) -> Self {
        debug_assert_eq!(element_count(&shape).ok(), Some(data.len()));
        Self { shape, data }
    }

    /// Gives the tensor the shape `shape` and lets `fill` rewrite its elements in place, keeping
    /// both buffers: the shape is copied only when it differs, and `fill` gets the old elements
    /// and must leave exactly as many as `shape` holds.
    pub(crate) fn refill(&mut self, shape: &[u64], fill: impl FnOnce(&mut Vec<T>)) {
        if self.shape != shape {
            self.shape.clear();
            self.shape.extend_from_slice(shape);
        }
        fill(&mut self.data);
        debug_assert_eq!(element_count(&self.shape).ok(), Some(self.data.len()));
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The element count.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor has no elements (a zero dimension).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The elements in row-major order.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn into_data(self) -> Vec<T> {
        self.data
    }
}

fn element_count(shape: &[u64]) -> Result<usize, TensorError> {
    shape
        .iter()
        .try_fold(1u64, |n, &d| n.checked_mul(d))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| TensorError::CountOverflow {
            shape: shape.to_vec(),
        })
}
