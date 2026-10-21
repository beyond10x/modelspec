//! Host tensors: a shape and dense row-major storage, generic over the element type.
//!
//! The same code runs in `f32`, the M0 dtype, and in `f64`, which the finite-difference gradient
//! checks use as reference math (DESIGN.md § 4).

use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use thiserror::Error;

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
}

impl Element for f32 {
    const ZERO: Self = 0.0;

    fn from_f32(value: f32) -> Self {
        value
    }

    fn from_count(count: u64) -> Self {
        count as f32
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

impl<T: Element> HostTensor<T> {
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
