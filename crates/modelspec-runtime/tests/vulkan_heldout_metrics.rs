//! The held-out evaluation reduced on the device (story:vulkan-eval-metrics-on-device): the
//! `HeldOutRows` and `HeldOutSum` kernels of `spirv::emit::held_out_kernels` and the stepwise
//! Vulkan run that records them after each `HeldOut` batch, one host wait per evaluation.
//!
//! The cases without a device hold the modules to `spirv-val`, the kernels to their shapes, and
//! `HeldOut::metrics_of_sums` to the bits of `HeldOut::metrics_of` when it is given the host's
//! own sums. The device cases (module `device`, `#[ignore]` as "device", run with
//! `--include-ignored`) hold the device reduction to the host tally of the same logits, the
//! accuracy bit for bit and the loss within [`EVAL_LOSS_ULPS`] (coordinator decision D141): the
//! kernels alone on the CPU's logits and on drawn ones with ties, NaN rows and losses past
//! `f32::MAX`, and the stepwise run with `HeldOutReduction::Device` against the same
//! run with `HeldOutReduction::Host` on `mlm_bag`, on the hidden-32 BERT twin and, given
//! [`BERT_TINY_DATA`], on `examples/bert_tiny.mspec` over the packed corpus; and the evaluated
//! evaluations to one host wait each and [`HELD_OUT_STATE`] floats down per evaluation, under the
//! Khronos validation layer's synchronization validation.
//!
//! The packed directory of `mlm_bag` and the twin is the original-text fixture
//! `tests/fixtures/corpus-tokenizer/`, tokenized at 300 tokens and packed at 16, as
//! `tests/adversary_eval_download.rs` makes it: 4 `HeldOut` batches at `B = 2`.
//!
//! Run the device cases with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_heldout_metrics -- --include-ignored`.
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::kernel::{KernelKind, Operand};
use modelspec_core::ir::tensor::ParamId;
use modelspec_runtime::cpu::{HostTensor, evaluate_all_with_ids};
use modelspec_runtime::dataset::mlm::{self, MlmBatch, mlm_batch};
use modelspec_runtime::dataset::{self, SplitRole, tokenizer};
use modelspec_runtime::spirv::emit::{HELD_OUT_STATE, emit, held_out_kernels};
use modelspec_runtime::spirv::validate::validate;
use modelspec_runtime::train::data::{self, DataSource, PackedData};
use modelspec_runtime::train::eval::{HeldOut, HeldOutMetrics};
use modelspec_runtime::train::init::{Stream, initialize};

#[allow(dead_code)]
mod common;

const SEQ_LEN: u64 = 16;
const VOCAB: usize = 300;
const SEED: u64 = 7;
/// The batch of the `mlm_bag` variant: 4 full `HeldOut` batches of the fixture's 9 sequences.
const B: usize = 2;
/// The packed directory `examples/bert_tiny.mspec` is evaluated over, when set.
const BERT_TINY_DATA: &str = "MODELSPEC_BERT_TINY_PACKED";

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("heldout_metrics_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// The fixture prepared, tokenized at 300 tokens and packed at 16 under a scratch directory for
/// `case`; returns the packed directory.
fn packed_dir(case: &str) -> PathBuf {
    let dir = scratch(case);
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus-tokenizer");
    let data_dir = dir.join("dataset");
    dataset::prepare(&fixture, &data_dir).expect("the fixture prepares");
    let record = tokenizer::train(&data_dir, VOCAB).expect("the tokenizer trains");
    let tokenizer_file = dir.join("tokenizer.json");
    tokenizer::write(&record, &tokenizer_file).expect("the tokenizer is written");
    let out = dir.join("packed");
    mlm::pack(&data_dir, &tokenizer_file, SEQ_LEN, &out).expect("the dataset packs");
    out
}

/// `mlm_bag.mspec` of the CLI crate at batch `b` and `steps` steps.
fn mlm_bag(b: usize, steps: u64) -> String {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../modelspec-cli/tests/fixtures/mlm_bag.mspec");
    let source = fs::read_to_string(path).expect("the fixture");
    let rows = b * SEQ_LEN as usize;
    assert!(source.contains("steps = 200;"));
    source
        .replace("[64, 300]", &format!("[{rows}, 300]"))
        .replace("[64]", &format!("[{rows}]"))
        .replace("steps = 200;", &format!("steps = {steps};"))
}

/// `path`, relative to the repository root, with each `from` replaced by its `to`.
fn edited(path: &str, edits: &[(&str, &str)]) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path);
    let mut source = fs::read_to_string(&path).expect("the model file");
    for (from, to) in edits {
        assert!(source.contains(from), "{} holds {from:?}", path.display());
        source = source.replace(from, to);
    }
    source
}

fn compile(source: &str) -> CompiledTraining {
    compile_training("heldout.mspec", source)
        .unwrap_or_else(|diagnostics| panic!("the source compiles: {diagnostics:?}"))
}

fn bind(dir: &Path, compiled: &CompiledTraining) -> PackedData {
    let binding = match data::source(dir, &compiled.config).expect("a source") {
        DataSource::Packed(binding) => binding,
        DataSource::Dense => panic!("{} holds binding.json", dir.display()),
    };
    data::packed(dir, binding, &compiled.config, &compiled.forward).expect("the data binds")
}

fn init(compiled: &CompiledTraining) -> BTreeMap<ParamId, HostTensor<f32>> {
    initialize(&compiled.forward.program, SEED, Stream::Init).expect("initial parameters")
}

/// The logits `[B·S, V]` of `HeldOut` batch `b` on the CPU interpreter, and the batch.
fn logits_of(
    compiled: &CompiledTraining,
    data: &PackedData,
    params: &BTreeMap<ParamId, HostTensor<f32>>,
    b: u64,
) -> (Vec<f32>, MlmBatch) {
    let packed = mlm::load(data.dir()).expect("the packed record");
    let policy = data.binding().masking_policy;
    let drawn = mlm_batch(&packed, SplitRole::HeldOut, &policy, SEED, b, B).expect("a batch");
    let rows = (B as u64) * SEQ_LEN;
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
    (logits, drawn)
}

/// The first largest element, `None` when the row holds a NaN (DESIGN.md § 8 accuracy rule).
fn first_max(row: &[f32]) -> Option<usize> {
    if row.iter().any(|x| x.is_nan()) {
        return None;
    }
    let mut best = 0;
    for (i, &x) in row.iter().enumerate() {
        if x > row[best] {
            best = i;
        }
    }
    Some(best)
}

/// The row loss of the host tally: `max + ln(Σ exp(x - max)) - row[label]`.
fn row_nll(row: &[f32], label: usize) -> f32 {
    let max = row[1..]
        .iter()
        .fold(row[0], |m, &x| if x > m { x } else { m });
    let total = row[1..]
        .iter()
        .fold((row[0] - max).exp(), |sum, &x| sum + (x - max).exp());
    max + total.ln() - row[label]
}

/// The `eval` line's text of `m`, as the CLI prints it.
fn text(m: HeldOutMetrics) -> String {
    format!("loss {} accuracy {}", m.loss, m.accuracy)
}

fn bits(m: HeldOutMetrics) -> (u32, u32) {
    (m.loss.to_bits(), m.accuracy.to_bits())
}

/// The `f32` ulps the device's held-out loss may lie from the host tally's (coordinator
/// decision D141, DESIGN.md § 8): the device's `Exp` and `Log` are not the host libm's `expf`
/// and `logf`, so row losses differ in their last bits.
const EVAL_LOSS_ULPS: i64 = 4;

/// `x` on a line of integers on which adjacent `f32` values are adjacent and both zeros are 0.
fn ordered(x: f32) -> i64 {
    let i = x.to_bits() as i32;
    i64::from(if i < 0 { i32::MIN - i } else { i })
}

/// Why the device's metrics are not the host tally's within the held-out tolerance, both
/// printed, or `None`: the accuracy bit for bit, the loss NaN on both or within
/// [`EVAL_LOSS_ULPS`] of the host's.
fn disagreement(device: HeldOutMetrics, host: HeldOutMetrics) -> Option<String> {
    let both = || format!("device {}, host {}", text(device), text(host));
    if device.accuracy.to_bits() != host.accuracy.to_bits() {
        return Some(format!("accuracy not bit-equal: {}", both()));
    }
    let (d, h) = (device.loss, host.loss);
    if d.is_nan() && h.is_nan() || d.to_bits() == h.to_bits() {
        return None;
    }
    let ulps = (ordered(d) - ordered(h)).abs();
    if d.is_nan() || h.is_nan() || d.is_infinite() || h.is_infinite() || ulps > EVAL_LOSS_ULPS {
        return Some(format!(
            "loss {ulps} ulps apart, above {EVAL_LOSS_ULPS}: {}",
            both()
        ));
    }
    None
}

/// The fixture's 4 batches at `B = 2`, the parameters and the CPU's logits of every batch.
struct Case {
    compiled: CompiledTraining,
    data: PackedData,
    logits: Vec<Vec<f32>>,
    batches: Vec<MlmBatch>,
}

fn case(name: &str) -> Case {
    let dir = packed_dir(name);
    let compiled = compile(&mlm_bag(B, 1));
    let data = bind(&dir, &compiled);
    let params = init(&compiled);
    let n = data.sequences(SplitRole::HeldOut) / B;
    assert_eq!(n, 4, "the fixture's 9 HeldOut sequences are 4 batches of 2");
    let (logits, batches) = (0..n as u64)
        .map(|b| logits_of(&compiled, &data, &params, b))
        .unzip();
    Case {
        compiled,
        data,
        logits,
        batches,
    }
}

/// The sums the host tally takes from `logits`: the `f64` numerator, the first product rounded
/// and each later one fused, and the correct selected rows.
fn host_sums(logits: &[Vec<f32>], batches: &[MlmBatch]) -> (f64, u64) {
    let (mut numerator, mut correct) = (None::<f64>, 0u64);
    for (logits, batch) in logits.iter().zip(batches) {
        for (n, row) in logits.as_chunks::<VOCAB>().0.iter().enumerate() {
            let label = batch.labels[n] as usize;
            let (w, nll) = (f64::from(batch.weights[n]), f64::from(row_nll(row, label)));
            numerator = Some(numerator.map_or(w * nll, |sum| w.mul_add(nll, sum)));
            if batch.weights[n] > 0.0 && first_max(row) == Some(label) {
                correct += 1;
            }
        }
    }
    (numerator.expect("a row"), correct)
}

/// The three modules of batches of `rows` rows of `classes` logits pass `spirv-val`, the two
/// sums differ only in their start, and the rows kernel covers every row.
#[test]
fn held_out_modules_pass_spirv_val_at_the_fixture_and_bert_tiny_shapes() {
    for (rows, classes) in [(32, 300), (4096, 8192), (5, 7), (4_194_305, 1)] {
        let kernels = held_out_kernels(rows, classes).expect("the kernels");
        assert_eq!(kernels.rows.kind, KernelKind::PerRow);
        assert_eq!(kernels.first.kind, KernelKind::LossSum);
        assert_eq!(
            kernels.first.operands.last(),
            Some(&Operand::Immediate(1.0))
        );
        assert_eq!(kernels.next.operands.last(), Some(&Operand::Immediate(0.0)));
        assert_ne!(kernels.first.id, kernels.next.id);
        let covered: u64 = kernels
            .rows
            .workgroups
            .iter()
            .map(|&w| u64::from(w))
            .product::<u64>()
            * u64::from(kernels.rows.workgroup_size);
        assert!(covered >= rows, "{rows} rows, {covered} invocations");
        assert!(kernels.rows.workgroups.iter().all(|&w| w <= 65_535));
        for kernel in [&kernels.rows, &kernels.first, &kernels.next] {
            let words = emit(kernel)
                .unwrap_or_else(|error| panic!("{} [{rows}, {classes}] emits: {error}", kernel.op));
            validate(&words).unwrap_or_else(|error| {
                panic!("{} [{rows}, {classes}]: spirv-val: {error}", kernel.op)
            });
        }
    }
    assert!(held_out_kernels(0, 300).is_err());
    assert!(held_out_kernels(1 << 20, 1 << 13).is_err(), "2^33 logits");
}

/// Given the host tally's own numerator and correct count, `metrics_of_sums` is the bits of
/// `metrics_of` on the logits.
#[test]
fn metrics_of_the_host_sums_are_the_bits_of_metrics_of() {
    let c = case("sums");
    let held = HeldOut::new(&c.compiled, &c.data).expect("the held-out split");
    let (numerator, correct) = host_sums(&c.logits, &c.batches);
    let summed = held.metrics_of_sums(numerator, correct).expect("the sums");
    let tallied = held.metrics_of(&c.logits).expect("metrics_of");
    assert_eq!(
        bits(summed),
        bits(tallied),
        "{summed:?} against {tallied:?}"
    );
    for (b, batch) in c.batches.iter().enumerate() {
        let (labels, weights) = held.targets(b).expect("the batch's targets");
        assert_eq!(labels, batch.labels.as_slice());
        assert_eq!(weights, batch.weights.as_slice());
    }
    assert!(held.targets(4).is_none());
}

/// The device cases.
mod device {
    use std::sync::Once;

    use modelspec_core::artifact::{Backend, CompiledArtifact};
    use modelspec_core::compile::compile_artifact_for;
    use modelspec_core::device::DeviceProfile;
    use modelspec_runtime::spirv::emit::{emit_module_with, held_out_sums};
    use modelspec_runtime::train::runner::{Data, Run};
    use modelspec_runtime::vulkan::negotiate::{Negotiated, negotiate};
    use modelspec_runtime::vulkan::runner::RunStats;
    use modelspec_runtime::vulkan::stepwise::{
        Evaluation, HeldOutReduction, SUBMIT_STEPS, StepSeen, StepwiseOptions, train_stepwise,
    };
    use modelspec_runtime::vulkan::sync::Barrier;
    use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

    use super::common::Draw;
    use super::*;

    static ENVIRONMENT: Once = Once::new();

    /// The validation layer with its synchronization validation on, set once before any case
    /// of this module loads Vulkan.
    fn validated() -> Vulkan {
        ENVIRONMENT.call_once(|| {
            for (name, value) in [
                (VALIDATION_ENV, "1"),
                ("VK_KHRONOS_VALIDATION_VALIDATE_SYNC", "1"),
                ("VK_VALIDATE_SYNC", "1"),
                ("VK_KHRONOS_VALIDATION_SYNCVAL_SUBMIT_TIME_VALIDATION", "1"),
            ] {
                // SAFETY: every case of this module calls `validated` before it reads the
                // environment or loads Vulkan, and `Once` holds the others until this returns.
                unsafe { std::env::set_var(name, value) };
            }
        });
        let vulkan = Vulkan::load().expect("the Vulkan loader opens");
        assert!(vulkan.validation_enabled(), "the validation layer is on");
        vulkan
    }

    fn open(vulkan: &Vulkan) -> (DeviceProfile, Negotiated) {
        let candidates = vulkan.candidates().expect("the devices are probed");
        let profile = candidates[0].profile.clone();
        let negotiated = negotiate(&profile).expect("device 0 negotiates");
        (profile, negotiated)
    }

    fn artifact(source: &str, profile: &DeviceProfile) -> CompiledArtifact {
        compile_artifact_for("heldout.mspec", source, Backend::Vulkan, Some(profile))
            .unwrap_or_else(|diagnostics| panic!("the source compiles for Vulkan: {diagnostics:?}"))
    }

    /// The held-out kernels alone over `logits`, one batch after the other in one command
    /// buffer: the state the last sum wrote, and the row losses of the last batch.
    fn reduce(
        device: &Device<'_>,
        profile: &DeviceProfile,
        held: &HeldOut<'_>,
        logits: &[Vec<f32>],
    ) -> ([f32; 4], Vec<f32>) {
        let rows = held.targets(0).expect("a batch").0.len() as u64;
        let classes = held.classes() as u64;
        let kernels = held_out_kernels(rows, classes).expect("the kernels");
        let pipeline = |kernel| {
            let module = emit_module_with(kernel, profile, false).expect("the module");
            device
                .kernel_pipeline(kernel, &module.spirv)
                .expect("the pipeline")
        };
        let (rows_pipeline, first, next) = (
            pipeline(&kernels.rows),
            pipeline(&kernels.first),
            pipeline(&kernels.next),
        );
        let batch_bytes = 4 * rows * classes;
        let all: Vec<f32> = logits.concat();
        let mut up_logits = device
            .staging_buffer(4 * all.len() as u64)
            .expect("staging");
        up_logits.write_f32(&all).expect("the logits go up");
        let (mut labels, mut weights) = (Vec::new(), Vec::new());
        for b in 0..logits.len() {
            let (l, w) = held.targets(b).expect("a batch");
            labels.extend_from_slice(l);
            weights.extend_from_slice(w);
        }
        let mut up_labels = device.staging_buffer(4 * labels.len() as u64).unwrap();
        up_labels.write_i32(&labels).unwrap();
        let mut up_weights = device.staging_buffer(4 * weights.len() as u64).unwrap();
        up_weights.write_f32(&weights).unwrap();
        let on_logits = device.device_local_buffer(batch_bytes).unwrap();
        let on_labels = device.device_local_buffer(4 * rows).unwrap();
        let on_weights = device.device_local_buffer(4 * rows).unwrap();
        let on_rows = device.device_local_buffer(8 * rows).unwrap();
        let on_state = device.device_local_buffer(4 * HELD_OUT_STATE).unwrap();
        let mut down_state = device.download_buffer(4 * HELD_OUT_STATE).unwrap();
        let mut down_rows = device.download_buffer(8 * rows).unwrap();
        {
            let mut recording = device.record().expect("a recording");
            for b in 0..logits.len() as u64 {
                if b > 0 {
                    recording.barrier(Barrier::ComputeToCompute);
                    recording.barrier(Barrier::ComputeToTransfer);
                    recording.barrier(Barrier::TransferToTransfer);
                }
                recording
                    .copy_region(&up_logits, b * batch_bytes, &on_logits, 0, batch_bytes)
                    .unwrap();
                recording
                    .copy_region(&up_labels, b * 4 * rows, &on_labels, 0, 4 * rows)
                    .unwrap();
                recording
                    .copy_region(&up_weights, b * 4 * rows, &on_weights, 0, 4 * rows)
                    .unwrap();
                recording.barrier(Barrier::TransferToCompute);
                recording
                    .dispatch_kernel(
                        &rows_pipeline,
                        &kernels.rows,
                        &[&on_logits, &on_labels, &on_rows],
                    )
                    .expect("the rows dispatch");
                recording.barrier(Barrier::ComputeToCompute);
                let (pipeline, kernel) = if b == 0 {
                    (&first, &kernels.first)
                } else {
                    (&next, &kernels.next)
                };
                recording
                    .dispatch_kernel(pipeline, kernel, &[&on_rows, &on_weights, &on_state])
                    .expect("the sum dispatch");
            }
            recording.barrier(Barrier::ComputeToTransfer);
            recording
                .copy_region(&on_state, 0, &down_state, 0, 4 * HELD_OUT_STATE)
                .unwrap();
            recording
                .copy_region(&on_rows, 0, &down_rows, 0, 8 * rows)
                .unwrap();
            recording.barrier(Barrier::TransferToHost);
            let recorded = recording.finish().expect("the recording ends");
            device.submit(recorded).unwrap().wait().unwrap();
        }
        let state = down_state.read_f32().unwrap();
        let rows = down_rows.read_f32().unwrap();
        (
            state.try_into().expect("four floats"),
            rows.as_chunks::<2>().0.iter().map(|pair| pair[0]).collect(),
        )
    }

    /// Logits drawn in [-16, 16), ties made by rounding to whole numbers when `ties`.
    fn drawn(seed: u64, count: usize, ties: bool) -> Vec<Vec<f32>> {
        let mut draw = Draw::new(seed);
        (0..count)
            .map(|_| {
                draw.values((2 * SEQ_LEN as usize * VOCAB) as u64)
                    .into_iter()
                    .map(|x| if ties { x.round() } else { x * 4.0 })
                    .collect()
            })
            .collect()
    }

    /// The device reduction of the kernels alone gives the host tally's accuracy and its loss
    /// within `EVAL_LOSS_ULPS`, and the host's correct count exactly, on the CPU's logits of the
    /// fixture's 4 batches and on drawn ones: wide, with ties, a NaN in a selected row, a NaN in a row of weight 0, and
    /// logits of ±3e38 whose losses pass `f32::MAX`.
    #[test]
    #[ignore = "device: needs a Vulkan device"]
    fn device_kernels_give_the_host_tallys_eval() {
        let vulkan = validated();
        let (profile, negotiated) = open(&vulkan);
        let device = vulkan
            .create_device(0, &negotiated)
            .expect("device 0 opens");
        let c = case("device_kernels");
        let held = HeldOut::new(&c.compiled, &c.data).expect("the held-out split");
        let selected_row = |b: usize, positive: bool| {
            c.batches[b]
                .weights
                .iter()
                .position(|&w| (w > 0.0) == positive)
                .expect("such a row")
        };
        let mut cases: Vec<(&str, Vec<Vec<f32>>)> = vec![
            ("cpu logits", c.logits.clone()),
            ("wide", drawn(1, 4, false)),
            ("ties", drawn(2, 4, true)),
        ];
        let mut nan = c.logits.clone();
        nan[1][selected_row(1, true) * VOCAB + 3] = f32::NAN;
        cases.push(("nan in a selected row", nan));
        let mut nan = c.logits.clone();
        nan[2][selected_row(2, false) * VOCAB + 5] = f32::NAN;
        cases.push(("nan in a row of weight 0", nan));
        let mut huge = drawn(3, 4, false);
        for logits in &mut huge {
            for (i, x) in logits.iter_mut().enumerate() {
                *x = if i % 3 == 0 { 3e38 } else { -3e38 };
            }
        }
        cases.push(("losses past f32::MAX", huge));

        let mut wrong = Vec::new();
        for (name, logits) in &cases {
            let (state, last_rows) = reduce(&device, &profile, &held, logits);
            let (numerator, correct) = held_out_sums(state);
            let (_, host_correct) = host_sums(logits, &c.batches);
            let device_metrics = held
                .metrics_of_sums(numerator, correct)
                .expect("the device sums");
            let host_metrics = held.metrics_of(logits).expect("the host tally");
            let last = logits.last().expect("a batch");
            let labels = &c.batches[3].labels;
            let differing = last_rows
                .iter()
                .zip(last.as_chunks::<VOCAB>().0)
                .zip(labels)
                .filter(|((g, row), l)| g.to_bits() != row_nll(&row[..], **l as usize).to_bits())
                .count();
            eprintln!(
                "{name}: device {} host {}; {differing} of {} row losses of the last batch \
                 differ in bits",
                text(device_metrics),
                text(host_metrics),
                last_rows.len()
            );
            if correct != host_correct {
                wrong.push(format!("{name}: correct {correct}, host {host_correct}"));
            }
            if let Some(why) = disagreement(device_metrics, host_metrics) {
                wrong.push(format!("{name}: {why}"));
            }
        }
        let messages = vulkan.take_validation_messages();
        assert!(messages.is_empty(), "{messages:#?}");
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// One stepwise run of `compiled` on `data`, evaluating `held` every `every` steps with
    /// `reduction` when given: the run, its counters and the held-out metrics of each step.
    #[allow(clippy::too_many_arguments)]
    fn run(
        device: &Device<'_>,
        profile: &DeviceProfile,
        negotiated: &Negotiated,
        compiled: &CompiledTraining,
        artifact: &CompiledArtifact,
        data: &PackedData,
        eval: Option<(&HeldOut<'_>, u64, HeldOutReduction)>,
    ) -> (Run, RunStats, Vec<(u64, Option<HeldOutMetrics>)>) {
        let mut seen = Vec::new();
        let options = StepwiseOptions {
            data: Some(Data::Packed(data)),
            eval: eval.map(|(held_out, every, reduction)| Evaluation {
                held_out,
                every,
                reduction,
            }),
            ..StepwiseOptions::default()
        };
        let (run, stats) = train_stepwise(
            device,
            profile,
            negotiated,
            compiled,
            artifact,
            options,
            &mut |step: &StepSeen<'_>| {
                seen.push((step.step, step.held_out));
                Ok(())
            },
        )
        .unwrap_or_else(|error| panic!("the device run finishes: {error}"));
        (run, stats, seen)
    }

    fn loss_bits(run: &Run) -> Vec<u32> {
        run.losses.iter().map(|l| l.to_bits()).collect()
    }

    fn param_bits(run: &Run) -> BTreeMap<ParamId, Vec<u32>> {
        run.state
            .values
            .iter()
            .map(|(p, v)| (*p, v.data().iter().map(|x| x.to_bits()).collect()))
            .collect()
    }

    /// `source` over `dir` for `steps` steps, `--eval-every every`: the device reduction against
    /// the host one and against the run without evaluation, on device 0 under synchronization
    /// validation. Returns the evaluated steps' texts.
    fn device_against_host(source: &str, dir: &Path, steps: u64, every: u64) -> Vec<String> {
        let vulkan = validated();
        let (profile, negotiated) = open(&vulkan);
        let device = vulkan
            .create_device(0, &negotiated)
            .expect("device 0 opens");
        let compiled = compile(source);
        let artifact = artifact(source, &profile);
        let data = bind(dir, &compiled);
        let held = HeldOut::new(&compiled, &data).expect("the held-out split");
        let batches = held.batches() as u64;
        let go = |eval| {
            run(
                &device,
                &profile,
                &negotiated,
                &compiled,
                &artifact,
                &data,
                eval,
            )
        };
        let (plain, plain_stats, _) = go(None);
        let (on_device, stats, device_seen) = go(Some((&held, every, HeldOutReduction::Device)));
        let (on_host, host_stats, host_seen) = go(Some((&held, every, HeldOutReduction::Host)));
        let messages = vulkan.take_validation_messages();
        eprintln!("without eval: {plain_stats}");
        eprintln!("device reduction: {stats}");
        eprintln!("host reduction: {host_stats}");
        let evaluated: Vec<u64> = (0..=steps)
            .filter(|k| k % every == 0 || *k == steps)
            .collect();
        let evaluations = evaluated.len() as u64;
        let pairs: Vec<(u64, HeldOutMetrics, HeldOutMetrics)> = device_seen
            .iter()
            .zip(&host_seen)
            .filter_map(|((k, d), (hk, h))| {
                assert_eq!(k, hk);
                Some((*k, (*d)?, h.expect("the host evaluated it too")))
            })
            .collect();
        for (k, d, h) in &pairs {
            eprintln!("eval {k}: device {}, host {}", text(*d), text(*h));
        }
        assert!(messages.is_empty(), "{messages:#?}");
        assert_eq!(
            pairs.iter().map(|(k, _, _)| *k).collect::<Vec<_>>(),
            evaluated
        );
        let differing: Vec<String> = pairs
            .iter()
            .filter_map(|(k, d, h)| disagreement(*d, *h).map(|why| format!("eval {k}: {why}")))
            .collect();
        assert!(differing.is_empty(), "{differing:#?}");
        // One host wait per segment, the steps from one evaluated step to the next (`every` is
        // at most `SUBMIT_STEPS`) and the last step alone, and one per evaluation reduced on the
        // device, one per batch on the host (story:vulkan-stepwise-multi-step-submit).
        assert!(every <= SUBMIT_STEPS);
        let segments = steps.div_ceil(every) + 1;
        assert_eq!(stats.host_waits, segments + evaluations, "{stats}");
        assert_eq!(stats.submits, segments + evaluations, "{stats}");
        assert_eq!(
            host_stats.host_waits,
            segments + evaluations * batches,
            "{host_stats}"
        );
        let down = stats.bytes_downloaded - plain_stats.bytes_downloaded;
        assert_eq!(
            down,
            evaluations * 4 * HELD_OUT_STATE,
            "{stats}\n{plain_stats}"
        );
        assert!(down <= evaluations * 8 * batches);
        assert_eq!(stats.dispatches_per_step, plain_stats.dispatches_per_step);
        assert_eq!(stats.barriers_per_step, plain_stats.barriers_per_step);
        for (name, run) in [("device", &on_device), ("host", &on_host)] {
            assert_eq!(loss_bits(run), loss_bits(&plain), "{name}: training losses");
            assert_eq!(
                param_bits(run),
                param_bits(&plain),
                "{name}: final parameters"
            );
        }
        pairs.into_iter().map(|(_, d, _)| text(d)).collect()
    }

    /// `mlm_bag` at `B = 2` over 4 `HeldOut` batches, `--eval-every 1` on 3 steps: the device
    /// reduction prints the host reduction's accuracy, and its loss within `EVAL_LOSS_ULPS`, at
    /// every step, one host wait per step and one per evaluation, 16 bytes down per evaluation, and the training of the run without evaluation.
    #[test]
    #[ignore = "device: needs a Vulkan device"]
    fn device_reduction_of_mlm_bag_is_the_host_tallys() {
        let dir = packed_dir("device_bag");
        device_against_host(&mlm_bag(B, 3), &dir, 3, 1);
    }

    /// The hidden-32 BERT twin (dropout, attention, AdamW) on 4 steps, `--eval-every 2`, over
    /// the fixture's 4 batches of 2: as `mlm_bag`.
    #[test]
    #[ignore = "device: needs a Vulkan device"]
    fn device_reduction_of_the_bert_twin_is_the_host_tallys() {
        let dir = packed_dir("device_twin");
        let source = edited(
            "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec",
            &[
                ("steps = 800;", "steps = 4;"),
                ("warmup_steps = 80)", "warmup_steps = 1)"),
            ],
        );
        device_against_host(&source, &dir, 4, 2);
    }

    /// `examples/bert_tiny.mspec` over the packed corpus of [`BERT_TINY_DATA`] on 2 steps,
    /// `--eval-every 1`: the story's acceptance as decided in D141, the host tally's accuracy,
    /// its loss within `EVAL_LOSS_ULPS`, one host wait per evaluation and at most `8 · batches` bytes down per evaluation.
    #[test]
    #[ignore = "device: needs a Vulkan device and MODELSPEC_BERT_TINY_PACKED"]
    fn device_reduction_of_bert_tiny_is_the_host_tallys() {
        let dir = PathBuf::from(
            std::env::var(BERT_TINY_DATA)
                .unwrap_or_else(|_| panic!("{BERT_TINY_DATA} names a packed directory")),
        );
        let source = edited(
            "examples/bert_tiny.mspec",
            &[
                ("steps = 10000;", "steps = 2;"),
                ("warmup_steps = 1000)", "warmup_steps = 1)"),
            ],
        );
        device_against_host(&source, &dir, 2, 1);
    }
}
