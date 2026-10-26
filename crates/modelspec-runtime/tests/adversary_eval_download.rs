//! Adversary pass 1 on story:vulkan-eval-download-per-batch (coordinator decision D76): the
//! held-out reduction fed one batch at a time (`HeldOut::tallying`) and the stepwise Vulkan run
//! that feeds it, one command buffer and one host wait per `HeldOut` batch.
//!
//! The cases without a device hold `Tallying` to the bits of `HeldOut::evaluate`, batch by batch
//! in batch order, and to its refusals: a partial evaluation, a batch past the last and logits
//! of the wrong length are errors, never metrics. The device cases (module `device`, `#[ignore]`
//! as "device", run with `--ignored`) drive `train_stepwise` in this process with the Khronos
//! validation layer's synchronization validation turned on, prove it live with a hazard of their
//! own, and hold the evaluated run to no validation message, the host waits and bytes it counts,
//! the training losses and parameters of the same run without evaluation, bit for bit, and two
//! runs to the same bits.
//!
//! The packed directory is the original-text fixture `tests/fixtures/corpus-tokenizer/`,
//! tokenized at 300 tokens and packed at 16, as `tests/adversary_heldout.rs` makes it: 9
//! `HeldOut` sequences, so `mlm_bag.mspec` at `B = 2` reads 4 batches and never the ninth
//! sequence.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::ParamId;
use modelspec_runtime::cpu::{HostTensor, evaluate_all_with_ids};
use modelspec_runtime::dataset::mlm::{self, MlmBatch, mlm_batch};
use modelspec_runtime::dataset::{self, SplitRole, tokenizer};
use modelspec_runtime::train::data::{self, DataSource, PackedData};
use modelspec_runtime::train::eval::{HeldOut, HeldOutMetrics};
use modelspec_runtime::train::init::{Stream, initialize};

const SEQ_LEN: u64 = 16;
const VOCAB: usize = 300;
const SEED: u64 = 7;
/// The batch of the `mlm_bag` variant: 4 full `HeldOut` batches of the fixture's 9 sequences.
const B: usize = 2;

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adv_eval_download_{name}"));
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

fn compile(source: &str) -> CompiledTraining {
    compile_training("adversary.mspec", source)
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

fn bits(m: HeldOutMetrics) -> (u32, u32) {
    (m.loss.to_bits(), m.accuracy.to_bits())
}

/// The fixture's 4 batches at `B = 2`, the parameters and the logits of every batch.
struct Case {
    compiled: CompiledTraining,
    data: PackedData,
    params: BTreeMap<ParamId, HostTensor<f32>>,
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
        params,
        logits,
        batches,
    }
}

/// The logits of every batch added one at a time, in batch order, give the bits of
/// `HeldOut::evaluate` and of `HeldOut::metrics_of` on all of them at once.
#[test]
fn adv_ed_tallying_batch_by_batch_is_the_bits_of_evaluate() {
    let c = case("tally_bits");
    let held = HeldOut::new(&c.compiled, &c.data).expect("the held-out split");
    assert_eq!(held.batches(), 4);
    let evaluated = held
        .evaluate(&c.compiled, &c.params)
        .expect("an evaluation");
    let mut tally = held.tallying();
    for logits in &c.logits {
        tally.add(logits).expect("a batch of logits");
    }
    let tallied = tally.metrics().expect("every batch was added");
    assert_eq!(
        bits(tallied),
        bits(evaluated),
        "{tallied:?} against {evaluated:?}"
    );
    let at_once = held.metrics_of(&c.logits).expect("metrics_of");
    assert_eq!(
        bits(at_once),
        bits(evaluated),
        "{at_once:?} against {evaluated:?}"
    );
}

/// A tally of 3 of the 4 batches is an error, not the metrics of 3 batches; the fourth batch
/// completes it; a fifth is an error that leaves the completed tally as it was.
#[test]
fn adv_ed_tallying_refuses_a_partial_or_overlong_evaluation() {
    let c = case("tally_partial");
    let held = HeldOut::new(&c.compiled, &c.data).expect("the held-out split");
    let evaluated = held
        .evaluate(&c.compiled, &c.params)
        .expect("an evaluation");
    let mut tally = held.tallying();
    assert!(tally.metrics().is_err(), "no batch added");
    for logits in &c.logits[..3] {
        tally.add(logits).expect("a batch of logits");
    }
    let partial = tally.metrics();
    assert!(partial.is_err(), "3 of 4 batches give metrics: {partial:?}");
    tally.add(&c.logits[3]).expect("the fourth batch");
    let complete = tally.metrics().expect("every batch was added");
    assert_eq!(bits(complete), bits(evaluated));
    assert!(
        tally.add(&c.logits[0]).is_err(),
        "a fifth batch of a split of 4 is refused"
    );
    let after = tally
        .metrics()
        .expect("the refused batch left the tally complete");
    assert_eq!(bits(after), bits(evaluated));
}

/// Logits of the wrong length are refused and do not count as a batch: the same batch's logits
/// added after the refusal, and the rest, give the bits of `evaluate`.
#[test]
fn adv_ed_tallying_refuses_short_logits_without_advancing() {
    let c = case("tally_short");
    let held = HeldOut::new(&c.compiled, &c.data).expect("the held-out split");
    let evaluated = held
        .evaluate(&c.compiled, &c.params)
        .expect("an evaluation");
    let mut tally = held.tallying();
    tally.add(&c.logits[0]).expect("batch 0");
    let short = &c.logits[1][..c.logits[1].len() - VOCAB];
    assert!(tally.add(short).is_err(), "one row short is refused");
    let long: Vec<f32> = c.logits[1]
        .iter()
        .chain(&c.logits[1][..VOCAB])
        .copied()
        .collect();
    assert!(tally.add(&long).is_err(), "one row long is refused");
    for logits in &c.logits[1..] {
        tally.add(logits).expect("a batch of logits");
    }
    let tallied = tally.metrics().expect("every batch was added");
    assert_eq!(bits(tallied), bits(evaluated));
}

/// A NaN in one selected row of batch 1 makes the loss NaN and takes that row out of the
/// correct rows only: the accuracy is the other correct rows over every selected row.
#[test]
fn adv_ed_a_nan_row_makes_the_loss_nan_and_the_row_wrong() {
    let c = case("tally_nan");
    let held = HeldOut::new(&c.compiled, &c.data).expect("the held-out split");
    let (mut correct, mut selected) = (0u64, 0u64);
    for (logits, batch) in c.logits.iter().zip(&c.batches) {
        for (n, row) in logits.as_chunks::<VOCAB>().0.iter().enumerate() {
            if batch.weights[n] > 0.0 {
                selected += 1;
                if first_max(row) == Some(batch.labels[n] as usize) {
                    correct += 1;
                }
            }
        }
    }
    let row = c.batches[1]
        .weights
        .iter()
        .position(|&w| w > 0.0)
        .expect("batch 1 selects a row");
    let was_correct = first_max(&c.logits[1][row * VOCAB..(row + 1) * VOCAB])
        == Some(c.batches[1].labels[row] as usize);
    let mut poisoned = c.logits.clone();
    poisoned[1][row * VOCAB + 3] = f32::NAN;
    let mut tally = held.tallying();
    for logits in &poisoned {
        tally.add(logits).expect("a batch of logits");
    }
    let got = tally.metrics().expect("every batch was added");
    assert!(
        got.loss.is_nan(),
        "a NaN logit of a selected row: loss {}",
        got.loss
    );
    let expected = (correct - u64::from(was_correct)) as f32 / selected as f32;
    assert_eq!(
        got.accuracy.to_bits(),
        expected.to_bits(),
        "{got:?}, expected accuracy {expected}"
    );
}

/// The device cases: `train_stepwise` in this process under synchronization validation.
#[cfg(feature = "vulkan")]
mod device {
    use std::sync::Once;

    use modelspec_core::artifact::{Backend, CompiledArtifact};
    use modelspec_core::compile::compile_artifact_for;
    use modelspec_core::device::DeviceProfile;
    use modelspec_runtime::train::runner::{self, Data, Run};
    use modelspec_runtime::vulkan::negotiate::{Negotiated, negotiate};
    use modelspec_runtime::vulkan::runner::RunStats;
    use modelspec_runtime::vulkan::stepwise::{
        Evaluation, StepSeen, StepwiseOptions, train_stepwise,
    };
    use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

    use super::*;

    const GPU_RUN_ATOL: f32 = 1e-4;
    const GPU_RUN_RTOL: f32 = 1e-3;

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

    /// Records two copies into one region with no barrier between them, a write-after-write
    /// hazard, and asserts the layer reports it: synchronization validation is live.
    fn assert_sync_validation_is_live(vulkan: &Vulkan, device: &Device<'_>) {
        let mut source = device.staging_buffer(256).expect("a staging buffer");
        source
            .write_f32(&[1.0; 64])
            .expect("the staging buffer is written");
        let target = device.device_local_buffer(256).expect("a device buffer");
        {
            let mut recording = device.record().expect("a recording");
            recording
                .copy_region(&source, 0, &target, 0, 256)
                .expect("copy 1");
            recording
                .copy_region(&source, 0, &target, 0, 256)
                .expect("copy 2");
            let recorded = recording.finish().expect("the recording ends");
            device
                .submit(recorded)
                .expect("submit")
                .wait()
                .expect("wait");
        }
        let messages = vulkan.take_validation_messages();
        assert!(
            messages.iter().any(|m| m.contains("hazard detected")),
            "two unsynchronized copies into one region report no hazard, so synchronization \
             validation is off: {messages:#?}"
        );
    }

    fn open(vulkan: &Vulkan) -> (DeviceProfile, Negotiated) {
        let candidates = vulkan.candidates().expect("the devices are probed");
        let profile = candidates[0].profile.clone();
        let negotiated = negotiate(&profile).expect("device 0 negotiates");
        (profile, negotiated)
    }

    fn artifact(source: &str, profile: &DeviceProfile) -> CompiledArtifact {
        compile_artifact_for("adversary.mspec", source, Backend::Vulkan, Some(profile))
            .unwrap_or_else(|diagnostics| panic!("the source compiles for Vulkan: {diagnostics:?}"))
    }

    /// One stepwise run of `compiled` on `data`, evaluating `held` every `every` steps when
    /// given: the run, its counters and the held-out metrics of each step it observed.
    #[allow(clippy::too_many_arguments)]
    fn run(
        device: &Device<'_>,
        profile: &DeviceProfile,
        negotiated: &Negotiated,
        compiled: &CompiledTraining,
        artifact: &CompiledArtifact,
        data: &PackedData,
        eval: Option<(&HeldOut<'_>, u64)>,
    ) -> (Run, RunStats, Vec<(u64, Option<HeldOutMetrics>)>) {
        let mut seen = Vec::new();
        let options = StepwiseOptions {
            data: Some(Data::Packed(data)),
            eval: eval.map(|(held_out, every)| Evaluation { held_out, every }),
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

    /// `mlm_bag` at `B = 2` over 4 `HeldOut` batches, `--eval-every 1` on 3 steps: no
    /// validation message under synchronization validation; eval at steps 0 to 3, each within
    /// the run tolerance of the CPU's; `4 + 4 · 4` host waits and submits; the evaluation adds
    /// `4 · 4` predictions to the bytes downloaded and nothing to the dispatches or barriers per
    /// step; the training losses and final parameters are the bits of the run without
    /// evaluation.
    #[test]
    #[ignore = "device: needs a Vulkan device"]
    fn adv_ed_device_eval_every_step_is_sync_clean_counted_and_inert() {
        let vulkan = validated();
        let (profile, negotiated) = open(&vulkan);
        let device = vulkan
            .create_device(0, &negotiated)
            .expect("device 0 opens");
        assert_sync_validation_is_live(&vulkan, &device);

        let dir = packed_dir("device_bag");
        let source = mlm_bag(B, 3);
        let compiled = compile(&source);
        let artifact = artifact(&source, &profile);
        let data = bind(&dir, &compiled);
        let held = HeldOut::new(&compiled, &data).expect("the held-out split");
        let batches = held.batches() as u64;
        assert_eq!(batches, 4);

        let (plain, plain_stats, _) = run(
            &device,
            &profile,
            &negotiated,
            &compiled,
            &artifact,
            &data,
            None,
        );
        let (evaluated, stats, seen) = run(
            &device,
            &profile,
            &negotiated,
            &compiled,
            &artifact,
            &data,
            Some((&held, 1)),
        );
        let messages = vulkan.take_validation_messages();
        eprintln!("without eval: {plain_stats}");
        eprintln!("with eval: {stats}");

        let mut cpu = Vec::new();
        runner::run_checked(
            &compiled,
            Some(Data::Packed(&data)),
            &BTreeMap::new(),
            0,
            |o| {
                cpu.push((
                    o.step,
                    held.evaluate(&compiled, o.values).map_err(Box::new)?,
                ));
                Ok(())
            },
            |_, _| Ok(()),
        )
        .expect("the CPU run finishes");
        let selected: usize = (0..batches)
            .map(|b| {
                data.draw(SplitRole::HeldOut, b)
                    .expect("a batch")
                    .weights
                    .iter()
                    .filter(|&&w| w > 0.0)
                    .count()
            })
            .sum();

        let mut wrong = Vec::new();
        for ((k, g), (ck, c)) in seen.iter().zip(&cpu) {
            assert_eq!(k, ck);
            let Some(g) = g else {
                wrong.push(format!("step {k}: no held-out metrics"));
                continue;
            };
            let bound = GPU_RUN_ATOL + GPU_RUN_RTOL * c.loss.abs();
            if (g.loss - c.loss).abs() > bound {
                wrong.push(format!("eval {k} loss: gpu {} cpu {}", g.loss, c.loss));
            }
            if f64::from((g.accuracy - c.accuracy).abs()) > 1.0 / selected as f64 {
                wrong.push(format!(
                    "eval {k} accuracy: gpu {} cpu {}",
                    g.accuracy, c.accuracy
                ));
            }
        }
        let prediction = artifact.execution.prediction_view.size_bytes;
        let evaluations = 4u64;
        eprintln!(
            "peak device bytes: {} without eval, {} with, prediction {prediction} bytes",
            plain_stats.peak_device_bytes, stats.peak_device_bytes
        );
        assert!(messages.is_empty(), "{messages:#?}");
        assert_eq!(seen.len(), 4, "{seen:?}");
        assert!(wrong.is_empty(), "{wrong:#?}");
        assert_eq!(stats.host_waits, 4 + evaluations * batches, "{stats}");
        assert_eq!(stats.submits, 4 + evaluations * batches, "{stats}");
        assert_eq!(plain_stats.host_waits, 4, "{plain_stats}");
        assert_eq!(
            stats.bytes_downloaded - plain_stats.bytes_downloaded,
            evaluations * batches * prediction,
            "with {stats}\nwithout {plain_stats}"
        );
        assert_eq!(stats.dispatches_per_step, plain_stats.dispatches_per_step);
        assert_eq!(stats.barriers_per_step, plain_stats.barriers_per_step);
        assert_eq!(loss_bits(&evaluated), loss_bits(&plain), "training losses");
        assert_eq!(
            param_bits(&evaluated),
            param_bits(&plain),
            "final parameters"
        );
    }

    /// The hidden-32 BERT twin (dropout, attention, AdamW) on 4 steps, `--eval-every 2`, over
    /// the fixture's 4 batches of 2: no validation message under synchronization validation,
    /// two evaluated runs give the same held-out bits and training losses, and the run without
    /// evaluation the same training losses and final parameters.
    #[test]
    #[ignore = "device: needs a Vulkan device"]
    fn adv_ed_device_twin_eval_is_sync_clean_repeatable_and_inert() {
        let vulkan = validated();
        let (profile, negotiated) = open(&vulkan);
        let device = vulkan
            .create_device(0, &negotiated)
            .expect("device 0 opens");
        assert_sync_validation_is_live(&vulkan, &device);

        let dir = packed_dir("device_twin");
        let twin = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../modelspec-cli/tests/fixtures/bert_cpu.mspec");
        let mut source = fs::read_to_string(twin).expect("the twin");
        for (from, to) in [
            ("steps = 800;", "steps = 4;"),
            ("warmup_steps = 80)", "warmup_steps = 1)"),
        ] {
            assert!(source.contains(from), "the twin holds {from:?}");
            source = source.replace(from, to);
        }
        let compiled = compile(&source);
        let artifact = artifact(&source, &profile);
        let data = bind(&dir, &compiled);
        let held = HeldOut::new(&compiled, &data).expect("the held-out split");
        assert_eq!(held.batches(), 4);

        let (first, first_stats, first_seen) = run(
            &device,
            &profile,
            &negotiated,
            &compiled,
            &artifact,
            &data,
            Some((&held, 2)),
        );
        let (second, _, second_seen) = run(
            &device,
            &profile,
            &negotiated,
            &compiled,
            &artifact,
            &data,
            Some((&held, 2)),
        );
        let (plain, _, _) = run(
            &device,
            &profile,
            &negotiated,
            &compiled,
            &artifact,
            &data,
            None,
        );
        let messages = vulkan.take_validation_messages();
        eprintln!("twin with eval: {first_stats}");
        let metric_bits =
            |seen: &[(u64, Option<HeldOutMetrics>)]| -> Vec<(u64, Option<(u32, u32)>)> {
                seen.iter().map(|(k, m)| (*k, m.map(bits))).collect()
            };
        assert!(messages.is_empty(), "{messages:#?}");
        let steps: Vec<u64> = first_seen
            .iter()
            .filter(|(_, m)| m.is_some())
            .map(|(k, _)| *k)
            .collect();
        assert_eq!(steps, [0, 2, 4]);
        assert_eq!(metric_bits(&first_seen), metric_bits(&second_seen));
        assert_eq!(loss_bits(&first), loss_bits(&second));
        assert_eq!(loss_bits(&first), loss_bits(&plain), "training losses");
        assert_eq!(param_bits(&first), param_bits(&plain), "final parameters");
        assert_eq!(first_stats.host_waits, 5 + 3 * 4, "{first_stats}");
    }
}
