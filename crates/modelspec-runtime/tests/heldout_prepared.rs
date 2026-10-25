//! story:bert-heldout-evaluation, coordinator decision D60 (N4): `HeldOut` verifies and plans the
//! forward program once per run, in `HeldOut::new`, and every batch of every evaluation runs the
//! prepared program, each intermediate freed after its last consumer. Before D60 each batch
//! verified and planned the program again and kept the value of every node to the end of its run.
//!
//! The packed directory is the original-text fixture `tests/fixtures/corpus-tokenizer/`,
//! tokenized at 300 tokens and packed at 16, as `tests/adversary_heldout.rs` makes it: its 9
//! held-out sequences are 2 batches of 4.

use std::fs;
use std::path::{Path, PathBuf};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::verify;
use modelspec_runtime::cpu::interpreter;
use modelspec_runtime::dataset::{self, mlm, tokenizer};
use modelspec_runtime::train::data::{self, DataSource, PackedData};
use modelspec_runtime::train::eval::HeldOut;
use modelspec_runtime::train::init::{Stream, initialize};

fn packed_dir(case: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("heldout_prepared_{case}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus-tokenizer");
    let data_dir = dir.join("dataset");
    dataset::prepare(&fixture, &data_dir).expect("the fixture prepares");
    let record = tokenizer::train(&data_dir, 300).expect("the tokenizer trains");
    let tokenizer_file = dir.join("tokenizer.json");
    tokenizer::write(&record, &tokenizer_file).expect("the tokenizer is written");
    let out = dir.join("packed");
    mlm::pack(&data_dir, &tokenizer_file, 16, &out).expect("the dataset packs");
    out
}

fn compiled() -> CompiledTraining {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../modelspec-cli/tests/fixtures/mlm_bag.mspec");
    let source = fs::read_to_string(path).expect("the fixture");
    compile_training("mlm_bag.mspec", &source)
        .unwrap_or_else(|diagnostics| panic!("the fixture compiles: {diagnostics:?}"))
}

fn bind(dir: &Path, compiled: &CompiledTraining) -> PackedData {
    let binding = match data::source(dir, &compiled.config).expect("a source") {
        DataSource::Packed(binding) => binding,
        DataSource::Dense => panic!("{} holds binding.json", dir.display()),
    };
    data::packed(dir, binding, &compiled.config, &compiled.forward).expect("the data binds")
}

/// The forward program is verified once, by `HeldOut::new`, and not again by any batch of two
/// evaluations; the two evaluations give the same bits.
#[test]
fn held_out_verifies_its_forward_program_once_per_run() {
    let dir = packed_dir("verify_once");
    let compiled = compiled();
    let data = bind(&dir, &compiled);
    let params = initialize(&compiled.forward.program, 7, Stream::Init).expect("parameters");

    let before_new = verify::counters::verify();
    let held = HeldOut::new(&compiled, &data).expect("the held-out split");
    assert_eq!(held.batches(), 2, "9 held-out sequences are 2 batches of 4");
    assert_eq!(
        verify::counters::verify() - before_new,
        1,
        "verify calls in HeldOut::new"
    );

    let before_evaluations = verify::counters::verify();
    let first = held.evaluate(&compiled, &params).expect("an evaluation");
    let second = held.evaluate(&compiled, &params).expect("an evaluation");
    assert_eq!(
        verify::counters::verify() - before_evaluations,
        0,
        "verify calls in two evaluations of 2 batches"
    );
    assert_eq!(first.loss.to_bits(), second.loss.to_bits());
    assert_eq!(first.accuracy.to_bits(), second.accuracy.to_bits());
}

/// A batch frees each intermediate after its last consumer: fewer values are live at once than
/// the forward program has nodes, which is the peak of a run that keeps every value.
#[test]
fn a_held_out_batch_does_not_keep_every_intermediate() {
    let dir = packed_dir("frees");
    let compiled = compiled();
    let data = bind(&dir, &compiled);
    let params = initialize(&compiled.forward.program, 7, Stream::Init).expect("parameters");
    let held = HeldOut::new(&compiled, &data).expect("the held-out split");
    held.evaluate(&compiled, &params).expect("an evaluation");
    let nodes = compiled.forward.program.nodes.len();
    let peak = interpreter::counters::peak_live_values();
    assert!(
        peak < nodes,
        "peak live values of a held-out batch {peak}, the forward program has {nodes} nodes"
    );
}
