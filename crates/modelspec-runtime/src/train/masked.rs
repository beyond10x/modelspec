//! The host side of the masked-row gather (DESIGN.md § 3.3, story:vulkan-mlm-masked-row-gather):
//! the row ids, labels and weights a gathered step reads ([`gather`]), taken at every step from
//! the batch's declared labels and weights.
//!
//! A row is selected when its weight is not zero: `+0.0` and `-0.0` are not selected, a NaN or
//! an infinity is, so a non-finite weight still reaches the loss and its refusal. The selected
//! rows are listed in ascending order, each with its label and weight, then padding rows of row
//! `0`, label `0` and weight `+0.0` up to the capacity, which add exact zeros to the loss's sums
//! and gradients. More selected rows than the capacity is [`E_DATA_012`], never a truncation.

use std::collections::BTreeMap;

use modelspec_core::diag::Diagnostic;
use modelspec_core::passes::masked_rows::{self, MaskedRowsSite};

use crate::cpu::tensor::HostTensor;
use crate::train::data::{DataError, E_DATA_012};

/// The three inputs of a gathered step.
#[derive(Clone, Debug, PartialEq)]
pub struct Gathered {
    /// The selected rows in ascending order, then row `0`.
    pub rows: Vec<i32>,
    /// The label of each selected row, then `0`.
    pub labels: Vec<i32>,
    /// The weight of each selected row, then `+0.0`.
    pub weights: Vec<f32>,
}

impl Gathered {
    /// The inputs by name: the rows and labels `i32`, the weights `f32`.
    pub fn named(
        &self,
        site: &MaskedRowsSite,
    ) -> (
        BTreeMap<String, HostTensor<f32>>,
        BTreeMap<String, HostTensor<i32>>,
    ) {
        let shape = vec![self.rows.len() as u64];
        let tensor_i32 = |values: &[i32]| {
            HostTensor::new(shape.clone(), values.to_vec()).expect("one value per gathered row")
        };
        let weights = HostTensor::new(shape.clone(), self.weights.clone())
            .expect("one weight per gathered row");
        (
            BTreeMap::from([(site.weights_input(), weights)]),
            BTreeMap::from([
                (masked_rows::ROWS_INPUT.to_owned(), tensor_i32(&self.rows)),
                (site.labels_input(), tensor_i32(&self.labels)),
            ]),
        )
    }
}

/// The gathered inputs of step `step` from its batch: the declared labels from `ids` (or, as
/// whole numbers, from `inputs`), the declared weights from `inputs`. [`E_DATA_012`] when more
/// rows are selected than `site.capacity`.
pub fn gather(
    site: &MaskedRowsSite,
    inputs: &BTreeMap<String, HostTensor<f32>>,
    ids: &BTreeMap<String, HostTensor<i32>>,
    step: u64,
) -> Result<Gathered, DataError> {
    let missing = |name: &str| {
        DataError::Refused(Diagnostic::new(
            E_DATA_012,
            format!("the batch of step {step} holds no `{name}` to gather the masked rows from"),
        ))
    };
    let weights = inputs
        .get(&site.weights)
        .ok_or_else(|| missing(&site.weights))?
        .data();
    let labels: Vec<i32> = match ids.get(&site.labels) {
        Some(labels) => labels.data().to_vec(),
        None => inputs
            .get(&site.labels)
            .ok_or_else(|| missing(&site.labels))?
            .data()
            .iter()
            .map(|&v| v as i32)
            .collect(),
    };
    gather_rows(site, &labels, weights, step)
}

/// [`gather`] of the labels and weights of one batch.
pub fn gather_rows(
    site: &MaskedRowsSite,
    labels: &[i32],
    weights: &[f32],
    step: u64,
) -> Result<Gathered, DataError> {
    let capacity = usize::try_from(site.capacity).unwrap_or(usize::MAX);
    let selected = weights.iter().filter(|w| **w != 0.0).count();
    if selected > capacity {
        return Err(DataError::Refused(
            Diagnostic::new(
                E_DATA_012,
                format!(
                    "the batch of step {step} has {selected} rows of non-zero `{}`, more than the \
                     {capacity} rows the masked-row gather holds",
                    site.weights
                ),
            )
            .with("step", step.to_string())
            .with("selected", selected.to_string())
            .with("capacity", capacity.to_string())
            .note("raise `--masked-rows` to at least the count, or give `--masked-rows 0` to train on every row"),
        ));
    }
    let mut gathered = Gathered {
        rows: Vec::with_capacity(capacity),
        labels: Vec::with_capacity(capacity),
        weights: Vec::with_capacity(capacity),
    };
    for (row, &weight) in weights.iter().enumerate() {
        if weight != 0.0 {
            gathered.rows.push(row as i32);
            gathered.labels.push(labels.get(row).copied().unwrap_or(0));
            gathered.weights.push(weight);
        }
    }
    gathered.rows.resize(capacity, 0);
    gathered.labels.resize(capacity, 0);
    gathered.weights.resize(capacity, 0.0);
    Ok(gathered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn site(capacity: u64) -> MaskedRowsSite {
        MaskedRowsSite {
            capacity,
            labels: "labels".into(),
            weights: "weights".into(),
        }
    }

    /// The rows of non-zero weight in ascending order, then padding of row 0, label 0 and weight
    /// `+0.0`.
    #[test]
    fn the_selected_rows_come_first_in_ascending_order() {
        let gathered =
            gather_rows(&site(4), &[7, 8, 9, 10, 11], &[0.0, 1.0, 0.0, 0.5, 0.0], 3).unwrap();
        assert_eq!(gathered.rows, [1, 3, 0, 0]);
        assert_eq!(gathered.labels, [8, 10, 0, 0]);
        assert_eq!(bits(&gathered.weights), bits(&[1.0, 0.5, 0.0, 0.0]));
    }

    /// `-0.0` is a zero weight; a NaN and an infinity are not, so they reach the loss.
    #[test]
    fn a_non_finite_weight_is_selected_and_negative_zero_is_not() {
        let gathered = gather_rows(
            &site(3),
            &[1, 2, 3, 4],
            &[-0.0, f32::NAN, f32::INFINITY, 0.0],
            0,
        )
        .unwrap();
        assert_eq!(gathered.rows, [1, 2, 0]);
        assert!(gathered.weights[0].is_nan());
        assert_eq!(gathered.weights[1], f32::INFINITY);
        assert_eq!(gathered.weights[2].to_bits(), 0.0f32.to_bits());
    }

    /// No selected row: every row is padding, so the loss and its gradient are zero.
    #[test]
    fn an_empty_selection_is_all_padding() {
        let gathered = gather_rows(&site(2), &[5, 6], &[0.0, 0.0], 0).unwrap();
        assert_eq!(gathered.rows, [0, 0]);
        assert_eq!(bits(&gathered.weights), bits(&[0.0, 0.0]));
    }

    /// One row more than the capacity is refused with E-DATA-012 naming the step, the count and
    /// the capacity.
    #[test]
    fn more_rows_than_the_capacity_are_refused() {
        let error = gather_rows(&site(2), &[1, 2, 3], &[1.0, 1.0, 1.0], 5).unwrap_err();
        let DataError::Refused(diagnostic) = error else {
            panic!("a refusal");
        };
        let text = diagnostic.to_string();
        assert!(text.contains("E-DATA-012"), "{text}");
        assert!(text.contains("step 5"), "{text}");
        assert!(text.contains("3 rows"), "{text}");
        assert!(
            text.contains("2 rows the masked-row gather holds"),
            "{text}"
        );
    }
}
