//! The statistics of one parameter tensor that a `StepCompleted` carries
//! (story:observatory-parameter-statistics): its L2 norm and its histogram, as
//! approval-record:parameter-statistics-settlements settles them and `spec/domains/observatory.yaml`
//! declares them.
//!
//! Every element is widened to `f64` before anything is computed, so both are taken over the exact
//! values of the `f32` tensor.

use crate::event::{Histogram, ParameterStatistics, StatisticsSubject};

/// The bins of a histogram whose finite elements are not all equal.
pub const BINS: usize = 32;

/// The L2 norm of `values`: the square root of the sum of their squares, each element widened to
/// `f64` and the squares summed in the order given, the row-major order of the tensor. NaN when an
/// element is NaN, and +inf when one is ±inf and none is NaN; JSON writes both as `null`.
pub fn l2_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |sum, &value| {
            let value = f64::from(value);
            sum + value * value
        })
        .sqrt()
}

/// The histogram of `values`.
///
/// With `min` and `max` the smallest and largest finite element, `min < max` gives [`BINS`] bins
/// whose edges are `min + (max - min) * i / 32` for `i` in `0..32` and `max` itself, bin `i`
/// counting the `x` with `edges[i] <= x < edges[i + 1]` and the last bin also `x == max`;
/// `min == max` gives one bin with the edges `[min, max]` that counts every finite element; no
/// finite element gives no edge and no bin. NaN, +inf and −inf are counted beside the bins.
pub fn histogram(values: &[f32]) -> Histogram {
    let mut nan_count = 0_u64;
    let mut positive_infinity_count = 0_u64;
    let mut negative_infinity_count = 0_u64;
    let mut finite = Vec::with_capacity(values.len());
    for &value in values {
        if value.is_nan() {
            nan_count += 1;
        } else if value == f32::INFINITY {
            positive_infinity_count += 1;
        } else if value == f32::NEG_INFINITY {
            negative_infinity_count += 1;
        } else {
            finite.push(f64::from(value));
        }
    }
    let (bin_edges, counts) = bins(&finite);
    Histogram {
        bin_edges,
        counts,
        nan_count,
        positive_infinity_count,
        negative_infinity_count,
    }
}

/// The edges and counts of the finite elements `finite`.
fn bins(finite: &[f64]) -> (Vec<f64>, Vec<u64>) {
    let Some(min) = finite.iter().copied().reduce(f64::min) else {
        return (Vec::new(), Vec::new());
    };
    let max = finite.iter().copied().fold(min, f64::max);
    if min == max {
        return (vec![min, max], vec![finite.len() as u64]);
    }
    let width = max - min;
    let edges: Vec<f64> = (0..BINS)
        .map(|i| min + width * i as f64 / BINS as f64)
        .chain([max])
        .collect();
    let mut counts = vec![0_u64; BINS];
    // The edges ascend, so the inner edges at or below `x` are a prefix, and their number is the
    // bin of `x`: 0 below the second edge, 31 from the last inner edge through `max`.
    let inner = &edges[1..BINS];
    for &x in finite {
        counts[inner.partition_point(|&edge| edge <= x)] += 1;
    }
    (edges, counts)
}

impl ParameterStatistics {
    /// The statistics of the tensor `values` of the parameter `path`, as `subject`.
    pub fn of(path: impl Into<String>, subject: StatisticsSubject, values: &[f32]) -> Self {
        Self {
            path: path.into(),
            subject,
            norm: l2_norm(values),
            histogram: histogram(values),
        }
    }
}
