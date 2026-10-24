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

    /// The AVX2 `matmul` kernel for this type ([`Kernel::Avx2`](super::matmul::Kernel::Avx2)):
    /// writes `out`, `m * n` elements, and returns true; `m`, `k` and `n` are nonzero. False,
    /// writing nothing, where the type has no such kernel or the CPU no AVX2, which is the default.
    /// Panics when `m * k`, `k * n` or `m * n` overflows `usize` or a slice is not that long.
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
