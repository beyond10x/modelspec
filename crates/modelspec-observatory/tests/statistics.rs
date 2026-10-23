//! The norm and the histogram of a parameter tensor (story:observatory-parameter-statistics;
//! `spec/domains/observatory.yaml`, `ParameterStatistics` and `Histogram`), as
//! approval-record:parameter-statistics-settlements settles them: 32 bins between the finite
//! min and max, one bin when they are equal, NaN and ±inf counted beside the bins, and the L2
//! norm. Each histogram here is also checked against the generated JSON Schema of `RunEvent`.

use modelspec_observatory::statistics::{BINS, histogram, l2_norm};
use modelspec_observatory::{
    EventSequence, Histogram, ParameterStatistics, RunEventBody, StatisticsSubject, StepCompleted,
    TrainingRunId,
};
use serde_json::Value;

const SCHEMA: &str = include_str!("../schema/entities/modelspec.observatory.RunEvent.schema.json");

/// Checks that `h` holds `total` elements and is valid inside an event against the schema.
fn assert_whole_and_valid(h: &Histogram, total: u64) {
    let in_bins: u64 = h.counts.iter().sum();
    assert_eq!(
        in_bins + h.nan_count + h.positive_infinity_count + h.negative_infinity_count,
        total,
        "{h:?}"
    );
    let event = EventSequence::new(TrainingRunId::new()).event(RunEventBody::StepCompleted(
        StepCompleted {
            step: 0,
            loss: 1.0,
            metrics: Vec::new(),
            parameters: vec![ParameterStatistics {
                path: "M.w".into(),
                subject: StatisticsSubject::Value,
                norm: 1.0,
                histogram: h.clone(),
            }],
        },
    ));
    let schema: Value = serde_json::from_str(SCHEMA).expect("the schema is JSON");
    let validator = jsonschema::validator_for(&schema).expect("the schema compiles");
    let json = serde_json::to_value(&event).expect("an event serializes");
    let errors: Vec<String> = validator
        .iter_errors(&json)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{json}: {errors:?}");
}

#[test]
fn the_norm_is_the_square_root_of_the_sum_of_squares() {
    assert_eq!(l2_norm(&[3.0, 4.0]), 5.0);
    assert_eq!(l2_norm(&[]), 0.0);
    assert_eq!(l2_norm(&[-2.0]), 2.0);
}

#[test]
fn the_norm_is_taken_in_f64_so_f32_max_does_not_overflow() {
    let max = f64::from(f32::MAX);
    assert_eq!(
        l2_norm(&[f32::MAX, f32::MAX]),
        (max * max + max * max).sqrt()
    );
    assert!(l2_norm(&[f32::MAX, f32::MAX]).is_finite());
}

#[test]
fn the_norm_sums_the_squares_in_row_major_order() {
    let values = [1.0e8_f32, 1.0, -1.0e8, 1.0];
    let in_order = values
        .iter()
        .fold(0.0_f64, |sum, &v| sum + f64::from(v) * f64::from(v))
        .sqrt();
    assert_eq!(l2_norm(&values).to_bits(), in_order.to_bits());
}

#[test]
fn a_non_finite_element_makes_the_norm_non_finite() {
    assert!(l2_norm(&[1.0, f32::NAN]).is_nan());
    assert_eq!(l2_norm(&[1.0, f32::INFINITY]), f64::INFINITY);
    assert_eq!(l2_norm(&[1.0, f32::NEG_INFINITY]), f64::INFINITY);
}

#[test]
fn distinct_values_fill_32_bins_between_min_and_max() {
    assert_eq!(BINS, 32);
    let values: Vec<f32> = (0..=32).map(|i| i as f32).collect();
    let h = histogram(&values);
    let edges: Vec<f64> = (0..=32).map(f64::from).collect();
    assert_eq!(h.bin_edges, edges);
    let mut counts = vec![1_u64; 32];
    counts[31] = 2;
    assert_eq!(h.counts, counts, "the last bin also holds max");
    assert_eq!(
        (
            h.nan_count,
            h.positive_infinity_count,
            h.negative_infinity_count
        ),
        (0, 0, 0)
    );
    assert_whole_and_valid(&h, 33);
}

#[test]
fn the_edges_are_min_plus_the_width_times_i_over_32_and_the_last_is_max() {
    let values = [-0.3_f32, 0.7, 0.1, 0.2];
    let h = histogram(&values);
    let (min, max) = (f64::from(-0.3_f32), f64::from(0.7_f32));
    assert_eq!(h.bin_edges.len(), 33);
    for (i, &edge) in h.bin_edges[..32].iter().enumerate() {
        assert_eq!(edge, min + (max - min) * i as f64 / 32.0, "edge {i}");
    }
    assert_eq!(h.bin_edges[32], max);
    for &x in &values {
        let x = f64::from(x);
        let bin = (0..32)
            .find(|&i| h.bin_edges[i] <= x && (x < h.bin_edges[i + 1] || i == 31))
            .expect("every finite element is in a bin");
        assert!(h.counts[bin] >= 1, "{x} in bin {bin}: {h:?}");
    }
    assert_eq!(h.counts[0], 1);
    assert_eq!(h.counts[31], 1);
    assert_whole_and_valid(&h, 4);
}

#[test]
fn an_element_on_an_inner_edge_is_in_the_bin_above_it() {
    let h = histogram(&[0.0, 32.0, 8.0, 16.0, 24.0]);
    assert_eq!(h.bin_edges[8], 8.0);
    assert_eq!(h.counts[7], 0);
    assert_eq!(h.counts[8], 1);
    assert_eq!(h.counts[16], 1);
    assert_eq!(h.counts[24], 1);
    assert_eq!(h.counts[0], 1);
    assert_eq!(h.counts[31], 1);
}

#[test]
fn equal_finite_values_make_one_bin() {
    let h = histogram(&[2.5, 2.5, f32::NAN, 2.5]);
    assert_eq!(h.bin_edges, vec![2.5, 2.5]);
    assert_eq!(h.counts, vec![3]);
    assert_eq!(h.nan_count, 1);
    assert_whole_and_valid(&h, 4);
}

#[test]
fn nan_and_infinities_are_counted_beside_the_bins() {
    let values = [
        1.0,
        f32::NAN,
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        f32::NAN,
        f32::NAN,
    ];
    let h = histogram(&values);
    assert_eq!(h.counts.len(), 32);
    assert_eq!(h.bin_edges.first(), Some(&-1.0));
    assert_eq!(h.bin_edges.last(), Some(&1.0));
    assert_eq!(h.counts.iter().sum::<u64>(), 2);
    assert_eq!(
        (
            h.nan_count,
            h.positive_infinity_count,
            h.negative_infinity_count
        ),
        (3, 2, 1)
    );
    assert_whole_and_valid(&h, 8);
}

#[test]
fn no_finite_element_gives_no_edge_and_no_bin() {
    let h = histogram(&[f32::NAN, f32::NEG_INFINITY]);
    assert!(h.bin_edges.is_empty());
    assert!(h.counts.is_empty());
    assert_eq!((h.nan_count, h.negative_infinity_count), (1, 1));
    assert_whole_and_valid(&h, 2);
    let empty = histogram(&[]);
    assert!(empty.bin_edges.is_empty() && empty.counts.is_empty());
    assert_whole_and_valid(&empty, 0);
}

#[test]
fn the_widest_f32_range_keeps_finite_edges() {
    let h = histogram(&[f32::MIN, f32::MAX, 0.0]);
    assert!(h.bin_edges.iter().all(|edge| edge.is_finite()));
    assert_eq!(h.counts[0], 1);
    assert_eq!(h.counts[16], 1);
    assert_eq!(h.counts[31], 1);
    assert_whole_and_valid(&h, 3);
}

#[test]
fn of_takes_the_norm_and_the_histogram_of_one_tensor() {
    let s = ParameterStatistics::of("M.w", StatisticsSubject::Gradient, &[3.0, 4.0]);
    assert_eq!(s.path, "M.w");
    assert_eq!(s.subject, StatisticsSubject::Gradient);
    assert_eq!(s.norm, 5.0);
    assert_eq!(s.histogram, histogram(&[3.0, 4.0]));
}

#[test]
fn the_three_counts_are_written_beside_the_bins() {
    let h = histogram(&[f32::NAN, 1.0, f32::INFINITY]);
    let json = serde_json::to_string(&h).expect("a histogram serializes");
    assert_eq!(
        json,
        r#"{"bin_edges":[1.0,1.0],"counts":[1],"nan_count":1,"positive_infinity_count":1,"negative_infinity_count":0}"#
    );
}
