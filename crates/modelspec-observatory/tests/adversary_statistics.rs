//! Adversary cases for `modelspec_observatory::statistics` (story:observatory-parameter-statistics):
//! every histogram and norm is checked against the definition in `spec/domains/observatory.yaml`
//! (`Histogram`, `ParameterStatistics.norm`) on edge tensors and on seeded random tensors that mix
//! subnormals, values near ±`f32::MAX`, signed zeros, NaN and ±inf.

use modelspec_observatory::Histogram;
use modelspec_observatory::statistics::{histogram, l2_norm};

/// A xorshift64* stream, so every run draws the same tensors.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// One element: a small normal, a subnormal, a value near ±MAX, a signed zero, NaN, ±inf or
    /// any bit pattern.
    fn element(&mut self) -> f32 {
        let sign = if self.below(2) == 0 { 1.0 } else { -1.0 };
        match self.below(10) {
            0 => sign * f32::from_bits(1 + self.below(0x007f_ffff) as u32),
            1 => sign * f32::from_bits(0x7f7f_ffff - self.below(1 << 20) as u32),
            2 => sign * 0.0,
            3 => f32::NAN,
            4 => sign * f32::INFINITY,
            5 => f32::from_bits(self.next() as u32),
            _ => sign * (self.below(1 << 24) as f32 / (1 << 20) as f32),
        }
    }
}

/// The histogram of `values` as the spec defines it, checked bin by bin against `h`.
fn assert_matches_definition(values: &[f32], h: &Histogram) {
    let finite: Vec<f64> = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| f64::from(v))
        .collect();
    let nan = values.iter().filter(|v| v.is_nan()).count() as u64;
    let pos = values.iter().filter(|&&v| v == f32::INFINITY).count() as u64;
    let neg = values.iter().filter(|&&v| v == f32::NEG_INFINITY).count() as u64;
    assert_eq!(
        (
            h.nan_count,
            h.positive_infinity_count,
            h.negative_infinity_count
        ),
        (nan, pos, neg),
        "{values:?}"
    );
    assert_eq!(
        h.counts.iter().sum::<u64>(),
        finite.len() as u64,
        "the bins hold exactly the finite elements: {values:?} {h:?}"
    );
    if finite.is_empty() {
        assert!(h.bin_edges.is_empty() && h.counts.is_empty(), "{h:?}");
        return;
    }
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(h.bin_edges.iter().all(|e| e.is_finite()), "{h:?}");
    if min == max {
        assert_eq!(h.counts, vec![finite.len() as u64], "{values:?}");
        assert_eq!(h.bin_edges.len(), 2);
        assert!(h.bin_edges[0] == min && h.bin_edges[1] == max, "{h:?}");
        return;
    }
    assert_eq!((h.bin_edges.len(), h.counts.len()), (33, 32), "{h:?}");
    assert_eq!(h.bin_edges[0], min);
    assert_eq!(h.bin_edges[32], max, "the last edge is max itself");
    for i in 0..32 {
        let edge = min + (max - min) * i as f64 / 32.0;
        assert_eq!(
            h.bin_edges[i].to_bits(),
            edge.to_bits(),
            "edge {i}: {values:?}"
        );
    }
    for w in h.bin_edges.windows(2) {
        assert!(w[0] < w[1], "the edges ascend strictly: {h:?}");
    }
    let mut expected = vec![0_u64; 32];
    for &x in &finite {
        let bins: Vec<usize> = (0..32)
            .filter(|&i| h.bin_edges[i] <= x && (x < h.bin_edges[i + 1] || (i == 31 && x == max)))
            .collect();
        assert_eq!(bins.len(), 1, "{x} is in exactly one bin: {h:?}");
        expected[bins[0]] += 1;
    }
    assert_eq!(h.counts, expected, "{values:?}");
}

fn assert_norm(values: &[f32]) {
    let sum = values
        .iter()
        .fold(0.0_f64, |s, &v| s + f64::from(v) * f64::from(v));
    let norm = l2_norm(values);
    assert_eq!(norm.to_bits(), sum.sqrt().to_bits(), "{values:?}");
    if values.iter().all(|v| v.is_finite()) {
        assert!(
            norm.is_finite(),
            "an all-finite f32 tensor has a finite norm: {values:?}"
        );
    }
}

#[test]
fn edge_tensors_match_the_definition() {
    let tiny = f32::from_bits(1);
    let cases: Vec<Vec<f32>> = vec![
        vec![],
        vec![f32::NAN; 7],
        vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
        vec![1.5],
        vec![-0.0],
        vec![0.0, -0.0, 0.0],
        vec![tiny, f32::from_bits(2)],
        vec![-tiny, tiny],
        vec![0.0, tiny],
        vec![f32::MIN, f32::MAX],
        vec![f32::MAX, f32::MAX, f32::MIN_POSITIVE],
        vec![f32::MIN, f32::MIN, -f32::MIN_POSITIVE, f32::NAN],
        vec![f32::from_bits(0x7f7f_fffe), f32::MAX],
        vec![1.0, 1.0 + f32::EPSILON],
        vec![f32::MAX; 1000],
        (0..=32).map(|i| i as f32).rev().collect(),
        (0..4096).map(|i| (i as f32 * 0.37).sin()).collect(),
        vec![f32::MAX, f32::MIN, f32::INFINITY, 0.0, f32::NAN, -0.0],
    ];
    for values in &cases {
        let h = histogram(values);
        assert_matches_definition(values, &h);
        assert_norm(values);
    }
}

#[test]
fn seeded_random_tensors_match_the_definition() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for _ in 0..4000 {
        let len = rng.below(97) as usize;
        let values: Vec<f32> = (0..len).map(|_| rng.element()).collect();
        let h = histogram(&values);
        assert_matches_definition(&values, &h);
        assert_norm(&values);
        let json = serde_json::to_string(&h).expect("a histogram serializes");
        let back: Histogram = serde_json::from_str(&json).expect("and reads back");
        assert_eq!(back, h, "{json}");
    }
}

#[test]
fn the_norm_of_the_largest_f32_tensor_a_model_can_hold_is_finite() {
    // 2^20 elements at f32::MAX: the sum of squares is 2^20 * MAX^2, about 1.2e83, far inside f64.
    let values = vec![f32::MAX; 1 << 20];
    let norm = l2_norm(&values);
    assert!(norm.is_finite(), "{norm}");
    let expected = f64::from(f32::MAX) * f64::from(1_u32 << 10);
    assert!(
        ((norm - expected) / expected).abs() < 1e-12,
        "{norm} vs {expected}"
    );
    let h = histogram(&values);
    assert_eq!((h.bin_edges.len(), h.counts), (2, vec![1 << 20]));
}
