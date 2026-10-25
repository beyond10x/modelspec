//! Adversary cases of story:bert-heldout-evaluation against `train::eval::HeldOut` (coordinator
//! decision D55): the held-out loss is the weighted mean over every row of every batch, bit for
//! bit, not the mean of the batch means; over one batch it is that batch's `cross_entropy`, bit
//! for bit; the batches are `mlm_batch(packed, HeldOut, policy, seed, b, B)` for
//! `b = 0 .. floor(count / B)`; a split with nothing selected gives 0 and 0, not NaN; and
//! parameters holding NaN or an infinity give a value, never a panic.
//!
//! The packed directory is the original-text fixture `tests/fixtures/corpus-tokenizer/`,
//! tokenized at 300 tokens and packed at 16, as `tests/adversary_data_binding.rs` makes it.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::ParamId;
use modelspec_runtime::cpu::{HostTensor, evaluate_all_with_ids};
use modelspec_runtime::dataset::mlm::{self, MlmBatch, mlm_batch};
use modelspec_runtime::dataset::{self, SplitRole, tokenizer};
use modelspec_runtime::train::data::{self, DataSource, PackedData};
use modelspec_runtime::train::eval::HeldOut;
use modelspec_runtime::train::init::{Stream, initialize};

const SEQ_LEN: u64 = 16;
const VOCAB: usize = 300;
const SEED: u64 = 7;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus-tokenizer")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adv_heldout_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// The fixture prepared, tokenized at 300 tokens and packed at 16, under a scratch directory for
/// `case`; returns the packed directory.
fn packed_dir(case: &str) -> PathBuf {
    let dir = scratch(case);
    let data_dir = dir.join("dataset");
    dataset::prepare(&fixture(), &data_dir).expect("the fixture prepares");
    let record = tokenizer::train(&data_dir, VOCAB).expect("the tokenizer trains");
    let tokenizer_file = dir.join("tokenizer.json");
    tokenizer::write(&record, &tokenizer_file).expect("the tokenizer is written");
    let out = dir.join("packed");
    mlm::pack(&data_dir, &tokenizer_file, SEQ_LEN, &out).expect("the dataset packs");
    out
}

/// The fixture `mlm_bag.mspec` of the CLI crate with its 64 rows (`B = 4`) replaced by
/// `b * 16`.
fn mlm_bag(b: usize) -> String {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../modelspec-cli/tests/fixtures/mlm_bag.mspec");
    let source = fs::read_to_string(path).expect("the fixture");
    let rows = b * SEQ_LEN as usize;
    source
        .replace("[64, 300]", &format!("[{rows}, 300]"))
        .replace("[64]", &format!("[{rows}]"))
}

fn compile(source: &str) -> CompiledTraining {
    compile_training("adversary.mspec", source)
        .unwrap_or_else(|diagnostics| panic!("the source compiles: {diagnostics:?}"))
}

/// The packed data of `dir` for `compiled`, its binding's policy replaced by `policy` when given.
fn bind(dir: &Path, compiled: &CompiledTraining, policy: Option<mlm::MaskingPolicy>) -> PackedData {
    let mut binding = match data::source(dir, &compiled.config).expect("a source") {
        DataSource::Packed(binding) => binding,
        DataSource::Dense => panic!("{} holds binding.json", dir.display()),
    };
    if let Some(policy) = policy {
        binding.masking_policy = policy;
    }
    data::packed(dir, binding, &compiled.config, &compiled.forward).expect("the data binds")
}

fn init(compiled: &CompiledTraining) -> BTreeMap<ParamId, HostTensor<f32>> {
    initialize(&compiled.forward.program, SEED, Stream::Init).expect("initial parameters")
}

/// What one held-out batch gives when evaluated directly: its logits rows, labels and weights,
/// and the program's own `cross_entropy` output.
struct Direct {
    logits: Vec<f32>,
    batch: MlmBatch,
    program_loss: f32,
}

fn direct(
    compiled: &CompiledTraining,
    packed: &mlm::Packed,
    policy: &mlm::MaskingPolicy,
    params: &BTreeMap<ParamId, HostTensor<f32>>,
    b: u64,
    batch: usize,
) -> Direct {
    let drawn = mlm_batch(packed, SplitRole::HeldOut, policy, SEED, b, batch).expect("a batch");
    let rows = (batch as u64) * SEQ_LEN;
    let floats = BTreeMap::from([(
        "weights".to_owned(),
        HostTensor::new(vec![rows], drawn.weights.clone()).unwrap(),
    )]);
    let ids = BTreeMap::from([
        (
            "ids".to_owned(),
            HostTensor::new(vec![rows], drawn.ids.clone()).unwrap(),
        ),
        (
            "labels".to_owned(),
            HostTensor::new(vec![rows], drawn.labels.clone()).unwrap(),
        ),
    ]);
    let mut evaluation = evaluate_all_with_ids(&compiled.forward.program, &floats, &ids, params)
        .expect("the forward program runs");
    let logits = evaluation
        .values
        .remove(&compiled.forward.prediction)
        .expect("the prediction")
        .data()
        .to_vec();
    Direct {
        logits,
        batch: drawn,
        program_loss: evaluation.outputs[0].data()[0],
    }
}

/// D55 literally: `max + ln(sum exp(x - max)) - x[label]`, every reduction ascending.
fn nll(row: &[f32], label: usize) -> f32 {
    let mut max = row[0];
    for &v in &row[1..] {
        if v > max {
            max = v;
        }
    }
    let mut total = (row[0] - max).exp();
    for &v in &row[1..] {
        total += (v - max).exp();
    }
    max + total.ln() - row[label]
}

/// H4: on the fixture's two held-out batches of 4 (9 sequences, the ninth never read) the
/// printed loss is the weighted sum of every row's nll over both batches divided by the sum of
/// both batches' weights, bit for bit, and it is not the mean of the two batch means when the
/// two weight sums differ.
#[test]
fn adv_ho_weighted_mean_over_two_batches_bit_for_bit() {
    let dir = packed_dir("two_batches");
    let compiled = compile(&mlm_bag(4));
    let data = bind(&dir, &compiled, None);
    let held = HeldOut::new(&compiled, &data).expect("the held-out split");
    let packed = mlm::load(&dir).unwrap();
    let count = packed.sequence_count(SplitRole::HeldOut);
    assert_eq!(held.batches(), count / 4, "floor(count / B) batches");
    let params = init(&compiled);
    let got = held.evaluate(&compiled, &params).expect("an evaluation");

    let policy = data.binding().masking_policy;
    let (mut num, mut den): (Option<f32>, Option<f32>) = (None, None);
    let mut means = Vec::new();
    let mut sums = Vec::new();
    for b in 0..(count / 4) as u64 {
        let d = direct(&compiled, &packed, &policy, &params, b, 4);
        let mut batch_den = 0.0f32;
        for (n, row) in d.logits.as_chunks::<VOCAB>().0.iter().enumerate() {
            let (w, label) = (d.batch.weights[n], d.batch.labels[n] as usize);
            let term = nll(row, label);
            num = Some(num.map_or(w * term, |acc| w.mul_add(term, acc)));
            den = Some(den.map_or(w, |acc| acc + w));
            batch_den += w;
        }
        means.push(d.program_loss);
        sums.push(batch_den);
    }
    let expected = num.unwrap() / den.unwrap();
    assert_eq!(
        got.loss.to_bits(),
        expected.to_bits(),
        "held-out loss {} against the weighted mean {expected}",
        got.loss
    );
    assert_ne!(
        sums[0], sums[1],
        "the two batches select different numbers of rows"
    );
    let mean_of_means = (means[0] + means[1]) / 2.0;
    assert_ne!(
        got.loss.to_bits(),
        mean_of_means.to_bits(),
        "not the mean of the batch means {means:?}"
    );
}

/// H4: with `B` equal to the split's sequence count there is one batch, and the held-out loss is
/// the bits of that batch's `cross_entropy` output, as DESIGN.md § 8 claims.
#[test]
fn adv_ho_one_batch_is_the_bits_of_cross_entropy() {
    let dir = packed_dir("one_batch");
    let packed = mlm::load(&dir).unwrap();
    let count = packed.sequence_count(SplitRole::HeldOut);
    assert!(packed.sequence_count(SplitRole::Train) >= count);
    let compiled = compile(&mlm_bag(count));
    let data = bind(&dir, &compiled, None);
    let held = HeldOut::new(&compiled, &data).expect("the held-out split");
    assert_eq!(held.batches(), 1);
    let params = init(&compiled);
    let got = held.evaluate(&compiled, &params).expect("an evaluation");
    let d = direct(
        &compiled,
        &packed,
        &data.binding().masking_policy,
        &params,
        0,
        count,
    );
    assert_eq!(
        got.loss.to_bits(),
        d.program_loss.to_bits(),
        "held-out loss {} against the program's cross_entropy {}",
        got.loss,
        d.program_loss
    );
}

/// H3: batch `b` of the held-out pass holds the sequences `b·B .. (b+1)·B` of the split in
/// stored order (its labels are the stored ids), and a batch size larger than the split is
/// refused by `HeldOut::new`, naming both numbers.
#[test]
fn adv_ho_batches_are_stored_order_and_too_large_is_refused() {
    let dir = packed_dir("stored_order");
    let packed = mlm::load(&dir).unwrap();
    let stored = packed.ids(SplitRole::HeldOut);
    let policy = mlm::MaskingPolicy::BERT;
    for b in 0..2u64 {
        let drawn = mlm_batch(&packed, SplitRole::HeldOut, &policy, SEED, b, 4).unwrap();
        let from = b as usize * 4 * SEQ_LEN as usize;
        assert_eq!(
            drawn.labels,
            stored[from..from + 4 * SEQ_LEN as usize],
            "batch {b} in stored order"
        );
    }
    let count = packed.sequence_count(SplitRole::HeldOut);
    let big = count + 1;
    assert!(packed.sequence_count(SplitRole::Train) >= big);
    let compiled = compile(&mlm_bag(big));
    let data = bind(&dir, &compiled, None);
    let error = match HeldOut::new(&compiled, &data) {
        Ok(_) => panic!("a batch of {big} over {count} sequences is refused"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains(&format!(" {big} "))
            || error.ends_with(&format!(" {big}"))
            || error.contains(&format!("{big} that")),
        "{error}"
    );
    assert!(error.contains(&format!(" {count} ")), "{error}");
}

/// H4: a policy that selects nothing gives a weight sum of 0 over every batch; the loss is
/// `+0.0` and the accuracy `0.0`, not NaN, and nothing panics.
#[test]
fn adv_ho_nothing_selected_is_zero_and_zero() {
    let dir = packed_dir("nothing_selected");
    let compiled = compile(&mlm_bag(4));
    let policy = mlm::MaskingPolicy {
        selection_rate: 0.0,
        ..mlm::MaskingPolicy::BERT
    };
    let data = bind(&dir, &compiled, Some(policy));
    let held = HeldOut::new(&compiled, &data).expect("the held-out split");
    let got = held
        .evaluate(&compiled, &init(&compiled))
        .expect("an evaluation");
    assert_eq!(got.loss.to_bits(), 0.0f32.to_bits(), "{got:?}");
    assert_eq!(got.accuracy.to_bits(), 0.0f32.to_bits(), "{got:?}");
}

/// H4: parameters holding a NaN, +inf or -inf give an evaluation, never a panic: the loss is
/// NaN (a non-finite row loss makes the weighted sum NaN whatever the weight, D46), and an
/// accuracy in [0, 1] that never counts a NaN row.
#[test]
fn adv_ho_non_finite_parameters_do_not_panic() {
    let dir = packed_dir("non_finite");
    let compiled = compile(&mlm_bag(4));
    let data = bind(&dir, &compiled, None);
    let held = HeldOut::new(&compiled, &data).expect("the held-out split");
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut params = init(&compiled);
        for tensor in params.values_mut() {
            let shape = tensor.shape().to_vec();
            let mut values = tensor.data().to_vec();
            for v in values.iter_mut().step_by(7) {
                *v = poison;
            }
            *tensor = HostTensor::new(shape, values).unwrap();
        }
        let got = held
            .evaluate(&compiled, &params)
            .unwrap_or_else(|e| panic!("{poison}: an evaluation, not {e}"));
        assert!(got.loss.is_nan(), "{poison}: {got:?}");
        assert!(
            (0.0..=1.0).contains(&got.accuracy),
            "{poison}: an accuracy {got:?}"
        );
    }
}
