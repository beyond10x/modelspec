//! Adversary cases of story:bert-train-data-binding against `train::data::{source, packed}` and
//! `train::runner::run_on` (coordinator decision D48): the tensors a packed run feeds at step `k`
//! are the tensors of `mlm_batch` of step `k`, bound by name; the seed changes the batch; a run
//! resumed at step `s` reads the batch of step `s`; and the shape rules bind and refuse as D48
//! states.
//!
//! The packed directory is the original-text fixture `tests/fixtures/corpus-tokenizer/`,
//! tokenized at 300 tokens and packed at 16, as `tests/mlm.rs` makes it.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_runtime::cpu::{HostTensor, evaluate_with_ids};
use modelspec_runtime::dataset::mlm::{self, MlmBatch, mlm_batch};
use modelspec_runtime::dataset::{self, SplitRole, tokenizer};
use modelspec_runtime::train::data::{self, DataError, DataSource, PackedData};
use modelspec_runtime::train::init::{Stream, initialize};
use modelspec_runtime::train::runner::{self, Data};

const SEQ_LEN: u64 = 16;
const VOCAB: usize = 300;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus-tokenizer")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adv_data_binding_{name}"));
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

/// The fixture `mlm_bag.mspec` of the CLI crate with `edits` applied.
fn mlm_bag(edits: &[(&str, &str)]) -> String {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../modelspec-cli/tests/fixtures/mlm_bag.mspec");
    let mut source = fs::read_to_string(path).expect("the fixture");
    for (from, to) in edits {
        assert!(source.contains(from), "the fixture holds {from:?}");
        source = source.replace(from, to);
    }
    source
}

/// A model that reads all five batch tensors: token and position embeddings, one attention
/// layer under `key_mask` declared `[4, 16]` beside `[64]` ids, a tied decoder, and
/// `cross_entropy` on `labels` and `weights`.
fn all_five(seed: u64) -> String {
    format!(
        "model Mini {{
    param table: Tensor<f32, [300, 16]> = xavier_uniform();
    param pos: Tensor<f32, [16, 16]> = xavier_uniform();
    param w_q: Tensor<f32, [16, 16]> = xavier_uniform();
    param w_k: Tensor<f32, [16, 16]> = xavier_uniform();
    param w_v: Tensor<f32, [16, 16]> = xavier_uniform();

    fn forward(ids: Tensor<i32, [64]>, positions: Tensor<i32, [64]>, key_mask: Tensor<f32, [4, 16]>) -> Tensor<f32, [64, 300]> {{
        return matmul(attention(matmul(add(embedding(table, ids), embedding(pos, positions)), w_q), matmul(add(embedding(table, ids), embedding(pos, positions)), w_k), matmul(add(embedding(table, ids), embedding(pos, positions)), w_v), key_mask, heads = 2), transpose(table));
    }}
}}

train Fit {{
    model = Mini;
    target weights: Tensor<f32, [64]>;
    input key_mask: Tensor<f32, [4, 16]>;
    target labels: Tensor<i32, [64]>;
    input positions: Tensor<i32, [64]>;
    input ids: Tensor<i32, [64]>;
    loss = cross_entropy(model.forward(ids, positions, key_mask), labels, weights);
    optimizer = sgd(lr = 0.5);
    steps = 3;
    seed = {seed};
    device = cpu;
}}
"
    )
}

fn compile(source: &str) -> CompiledTraining {
    compile_training("adversary.mspec", source)
        .unwrap_or_else(|diagnostics| panic!("the source compiles: {diagnostics:?}"))
}

/// The binding and the packed data of `dir` for `compiled`, through the functions `train` calls.
fn bind(dir: &Path, compiled: &CompiledTraining) -> Result<PackedData, DataError> {
    let binding = match data::source(dir, &compiled.config)? {
        DataSource::Packed(binding) => binding,
        DataSource::Dense => panic!("{} holds binding.json", dir.display()),
    };
    data::packed(dir, binding, &compiled.config, &compiled.forward)
}

/// The batch of step `k` drawn directly from the packed files, as story:bert-mlm-batches defines
/// it.
fn direct(dir: &Path, seed: u64, k: u64, b: usize) -> MlmBatch {
    let packed = mlm::load(dir).expect("the packed directory loads");
    mlm_batch(
        &packed,
        SplitRole::Train,
        &mlm::MaskingPolicy::BERT,
        seed,
        k,
        b,
    )
    .expect("a batch")
}

/// The diagnostic code and the declaration a refusal names.
fn refusal(result: Result<PackedData, DataError>) -> (String, String) {
    match result {
        Err(DataError::Refused(d)) => (d.code.to_string(), d.to_string()),
        Err(other) => panic!("not a declaration refusal: {other}"),
        Ok(packed) => panic!("bound with B = {}", packed.batch()),
    }
}

/// H1: for steps 0, 1, 7, the first of the second epoch and the one after, the five tensors the
/// run feeds are `mlm_batch` of that step, element for element, bound by name (the source order
/// of the declarations is not the batch's field order), with the declared shapes.
#[test]
fn adv_db_step_k_feeds_the_mlm_batch_of_step_k_by_name() {
    let dir = packed_dir("step_k");
    let compiled = compile(&all_five(7));
    let packed = bind(&dir, &compiled).expect("all five bind");
    assert_eq!(packed.batch(), 4);
    let sequences = mlm::load(&dir).unwrap().sequence_count(SplitRole::Train) as u64;
    let per_epoch = sequences / 4;
    assert!(
        per_epoch >= 2,
        "the fixture has {sequences} Train sequences"
    );
    for k in [0, 1, 7, per_epoch, per_epoch + 1] {
        let fed = packed.step(k).expect("a batch");
        let want = direct(&dir, 7, k, 4);
        let ids = fed.ids();
        let inputs = fed.inputs();
        assert_eq!(ids["ids"].data(), want.ids.as_slice(), "ids of step {k}");
        assert_eq!(
            ids["positions"].data(),
            want.positions.as_slice(),
            "positions of step {k}"
        );
        assert_eq!(
            ids["labels"].data(),
            want.labels.as_slice(),
            "labels of step {k}"
        );
        assert_eq!(
            inputs["weights"].data(),
            want.weights.as_slice(),
            "weights of step {k}"
        );
        assert_eq!(
            inputs["key_mask"].data(),
            want.key_mask.as_slice(),
            "key_mask of step {k}"
        );
        assert_eq!(ids["ids"].shape(), [64]);
        assert_eq!(inputs["key_mask"].shape(), [4, 16]);
        assert_eq!(ids.len() + inputs.len(), 5);
    }
}

/// H7: another seed draws another batch at the same step (another masking and another epoch
/// order), so the seed reaches the data and not only the initial parameters.
#[test]
fn adv_db_another_seed_draws_another_batch() {
    let dir = packed_dir("seed");
    let seven = bind(&dir, &compile(&mlm_bag(&[]))).expect("binds");
    let eight = bind(&dir, &compile(&mlm_bag(&[("seed = 7;", "seed = 8;")]))).expect("binds");
    for k in [0, 3] {
        let (a, b) = (seven.step(k).unwrap(), eight.step(k).unwrap());
        assert_ne!(
            a.inputs()["weights"].data(),
            b.inputs()["weights"].data(),
            "step {k}: the masked rows do not depend on the seed"
        );
        assert_eq!(a.ids()["ids"].data(), direct(&dir, 7, k, 4).ids.as_slice());
        assert_eq!(b.ids()["ids"].data(), direct(&dir, 8, k, 4).ids.as_slice());
    }
}

/// H1: a run started at step `s` (a resume) reports as the loss of step `s` the loss of its
/// parameters on the batch of step `s`, not of step `s - 1` or `0`, and the loss after its last
/// update on the batch of step `s + steps`.
#[test]
fn adv_db_run_on_reads_the_batch_of_the_absolute_step() {
    let dir = packed_dir("run_on");
    let mut compiled = compile(&mlm_bag(&[]));
    compiled.config.steps = 1;
    let packed = bind(&dir, &compiled).expect("binds");
    let forward = &compiled.forward.program;
    let init = initialize(forward, 7, Stream::Init).expect("initial values");
    let loss_on = |k: u64, params: &BTreeMap<_, HostTensor<f32>>| -> f32 {
        let fed = packed.step(k).unwrap();
        evaluate_with_ids(forward, fed.inputs(), fed.ids(), params).expect("the forward runs")[0]
            .data()[0]
    };
    for start in [0u64, 1, 5] {
        let mut seen = Vec::new();
        let run = runner::run_on(
            &compiled,
            Some(Data::Packed(&packed)),
            &init,
            start,
            |o| seen.push((o.step, o.loss)),
            |_, _| Ok(()),
        )
        .expect("the run finishes");
        assert_eq!(seen[0].0, start);
        assert_eq!(
            seen[0].1.to_bits(),
            loss_on(start, &init).to_bits(),
            "the loss of step {start} is not the loss on the batch of step {start}"
        );
        if start > 0 {
            assert_ne!(
                seen[0].1,
                loss_on(start - 1, &init),
                "batches {start} and {} agree",
                start - 1
            );
        }
        assert_eq!(
            seen[1].1.to_bits(),
            loss_on(start + 1, &run.state.values).to_bits(),
            "the last loss is not on the batch of step {}",
            start + 1
        );
    }
}

/// H2: `[B·S]` ids beside a `[B, S]` key mask bind with one `B` (the run above trains on it); a
/// transposed key mask `[S, B]`, a `[B, S, 1]`, a `[B·S + 1]` and a `B` of 0 are E-DATA-006.
#[test]
fn adv_db_shapes_bind_and_refuse() {
    let dir = packed_dir("shapes");
    let mixed = bind(&dir, &compile(&all_five(7))).expect("mixed forms bind");
    assert_eq!(mixed.batch(), 4);
    let key_mask_as = |shape: &str| {
        mlm_bag(&[(
            "    input ids: Tensor<i32, [64]>;\n",
            &format!(
                "    input ids: Tensor<i32, [64]>;\n    input key_mask: Tensor<f32, {shape}>;\n"
            ),
        )])
    };
    for shape in ["[16, 4]", "[4, 16, 1]", "[65]", "[8, 16]"] {
        let (code, text) = refusal(bind(&dir, &compile(&key_mask_as(shape))));
        assert_eq!(code, "E-DATA-006", "key_mask: Tensor<f32, {shape}>: {text}");
        assert!(text.contains("`key_mask`"), "{text}");
    }
}

/// H2: when two declarations disagree on `B`, `B` comes from the first present in the order
/// ids, positions, key_mask, labels, weights, not from source order, so the refusal names the
/// other one even when it comes first in the file.
#[test]
fn adv_db_disagreeing_b_names_the_later_tensor_in_d48_order() {
    let dir = packed_dir("disagree");
    let source = mlm_bag(&[(
        "    input ids: Tensor<i32, [64]>;\n",
        "    input key_mask: Tensor<f32, [3, 16]>;\n    input ids: Tensor<i32, [64]>;\n",
    )]);
    let (code, text) = refusal(bind(&dir, &compile(&source)));
    assert_eq!(code, "E-DATA-006", "{text}");
    assert!(
        text.contains("`key_mask`") && text.contains("[4, 16]"),
        "B is read from ids (4), not from key_mask, the first in source order: {text}"
    );
}
