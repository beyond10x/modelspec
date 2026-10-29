//! Adversary pass 1 on story:vulkan-stepwise-multi-step-submit: a stepwise Vulkan run records
//! up to `submit_steps` steps per command buffer, keeps one segment in flight while it records
//! the next, and must give the bits of one step per command buffer.
//!
//! Every case drives `train_stepwise` in this process with the Khronos validation layer's
//! synchronization validation on (submit-time validation included), proves the layer live with a
//! hazard of its own, and holds the run with `submit_steps` K against the run with K = 1: the
//! losses, the final parameters and moments, every step the observer sees (loss, held-out
//! metrics, values), every checkpoint written on the way, a resumed run, and a diverged run's
//! state. The packed directory is the original-text fixture `tests/fixtures/corpus-tokenizer/`,
//! tokenized at 300 tokens and packed at 16, as `tests/adversary_eval_download.rs` makes it.
//!
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test adversary_multistep -- --ignored --test-threads=1`
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Instant;

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{CompiledTraining, compile_artifact_for, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::tensor::ParamId;
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::dataset::mlm;
use modelspec_runtime::dataset::{self, tokenizer};
use modelspec_runtime::train::data::{self, DataSource, PackedData};
use modelspec_runtime::train::eval::{HeldOut, HeldOutMetrics};
use modelspec_runtime::train::runner::{Data, Run};
use modelspec_runtime::vulkan::negotiate::{Negotiated, negotiate};
use modelspec_runtime::vulkan::runner::{Checkpoints, RunStats, VulkanRunError, WriteCheckpoint};
use modelspec_runtime::vulkan::stepwise::{
    Evaluation, HeldOutReduction, StepSeen, StepwiseOptions, train_stepwise,
};
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

const SEQ_LEN: u64 = 16;
const VOCAB: usize = 300;
const B: usize = 2;

static ENVIRONMENT: Once = Once::new();

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adv_multistep_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

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

fn cli_fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../modelspec-cli/tests/fixtures")
        .join(name);
    fs::read_to_string(path).expect("the fixture")
}

/// `mlm_bag.mspec` at batch `B`, `steps` steps and SGD rate `lr`.
fn mlm_bag(steps: u64, lr: &str) -> String {
    let source = cli_fixture("mlm_bag.mspec");
    let rows = B * SEQ_LEN as usize;
    assert!(source.contains("steps = 200;"));
    assert!(source.contains("optimizer = sgd(lr = 1.0);"));
    source
        .replace("[64, 300]", &format!("[{rows}, 300]"))
        .replace("[64]", &format!("[{rows}]"))
        .replace("steps = 200;", &format!("steps = {steps};"))
        .replace(
            "optimizer = sgd(lr = 1.0);",
            &format!("optimizer = sgd(lr = {lr});"),
        )
}

/// The hidden-32 BERT twin (dropout, `adamw` with warm-up) at `steps` steps.
fn twin(steps: u64) -> String {
    let source = cli_fixture("bert_cpu.mspec");
    assert!(source.contains("steps = 800;"));
    assert!(source.contains("warmup_steps = 80"));
    // A warm-up of 5 steps, so that the rate turns from warm-up to decay inside a segment.
    source
        .replace("steps = 800;", &format!("steps = {steps};"))
        .replace("warmup_steps = 80", "warmup_steps = 5")
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

fn validated() -> Vulkan {
    ENVIRONMENT.call_once(|| {
        for (name, value) in [
            (VALIDATION_ENV, "1"),
            ("VK_KHRONOS_VALIDATION_VALIDATE_SYNC", "1"),
            ("VK_VALIDATE_SYNC", "1"),
            ("VK_KHRONOS_VALIDATION_SYNCVAL_SUBMIT_TIME_VALIDATION", "1"),
        ] {
            // SAFETY: every case calls `validated` before it reads the environment or loads
            // Vulkan, and `Once` holds the others until this returns.
            unsafe { std::env::set_var(name, value) };
        }
    });
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert!(vulkan.validation_enabled(), "the validation layer is on");
    vulkan
}

/// Two unsynchronized copies into one region: the layer must report a hazard.
fn assert_sync_validation_is_live(vulkan: &Vulkan, device: &Device<'_>) {
    let mut source = device.staging_buffer(256).expect("a staging buffer");
    source.write_f32(&[1.0; 64]).expect("written");
    let target = device.device_local_buffer(256).expect("a device buffer");
    {
        let mut recording = device.record().expect("a recording");
        recording.copy_region(&source, 0, &target, 0, 256).unwrap();
        recording.copy_region(&source, 0, &target, 0, 256).unwrap();
        let recorded = recording.finish().expect("the recording ends");
        device.submit(recorded).unwrap().wait().unwrap();
    }
    let messages = vulkan.take_validation_messages();
    assert!(
        messages.iter().any(|m| m.contains("hazard detected")),
        "synchronization validation is off: {messages:#?}"
    );
}

struct Gpu {
    vulkan: Vulkan,
    profile: DeviceProfile,
    negotiated: Negotiated,
}

fn gpu() -> Gpu {
    let vulkan = validated();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    Gpu {
        vulkan,
        profile,
        negotiated,
    }
}

fn artifact(source: &str, profile: &DeviceProfile) -> CompiledArtifact {
    compile_artifact_for("adversary.mspec", source, Backend::Vulkan, Some(profile))
        .unwrap_or_else(|diagnostics| panic!("the source compiles for Vulkan: {diagnostics:?}"))
}

type Bits = BTreeMap<ParamId, Vec<u32>>;

fn bits_of(values: &BTreeMap<ParamId, HostTensor<f32>>) -> Bits {
    values
        .iter()
        .map(|(p, v)| (*p, v.data().iter().map(|x| x.to_bits()).collect()))
        .collect()
}

fn metric_bits(m: HeldOutMetrics) -> (u32, u32) {
    (m.loss.to_bits(), m.accuracy.to_bits())
}

/// What the observer saw at one step, in bits.
#[derive(Debug, PartialEq)]
struct Seen {
    step: u64,
    loss: u32,
    held_out: Option<(u32, u32)>,
    values: Option<Bits>,
}

/// The bits of a finished run and of what it showed on the way.
#[derive(Debug, PartialEq)]
struct Finished {
    start: u64,
    losses: Vec<u32>,
    values: Bits,
    moments: Vec<(String, Vec<u32>)>,
    versions: String,
    seen: Vec<Seen>,
    checkpoints: Vec<(u64, Bits)>,
}

struct Plan<'a> {
    compiled: &'a CompiledTraining,
    artifact: &'a CompiledArtifact,
    data: &'a PackedData,
    eval: Option<(&'a HeldOut<'a>, u64, HeldOutReduction)>,
    checkpoint_every: Option<u64>,
    start: u64,
    init: BTreeMap<ParamId, HostTensor<f32>>,
}

/// One stepwise run of `plan` with `submit_steps`: its bits, or the error it stopped with.
fn run(
    gpu: &Gpu,
    device: &Device<'_>,
    plan: &Plan<'_>,
    submit_steps: Option<u64>,
) -> (Result<(Finished, RunStats), VulkanRunError>, Vec<Seen>) {
    let mut seen = Vec::new();
    let mut written: Vec<(u64, Bits)> = Vec::new();
    let mut writer = |k: u64, values: &BTreeMap<ParamId, HostTensor<f32>>| {
        written.push((k, bits_of(values)));
        Ok(())
    };
    let write: &mut WriteCheckpoint<'_> = &mut writer;
    let options = StepwiseOptions {
        start: plan.start,
        init: plan.init.clone(),
        data: Some(Data::Packed(plan.data)),
        eval: plan.eval.map(|(held_out, every, reduction)| Evaluation {
            held_out,
            every,
            reduction,
        }),
        checkpoints: plan
            .checkpoint_every
            .map(|every| Checkpoints { every, write }),
        submit_steps,
        ..StepwiseOptions::default()
    };
    let result = train_stepwise(
        device,
        &gpu.profile,
        &gpu.negotiated,
        plan.compiled,
        plan.artifact,
        options,
        &mut |step: &StepSeen<'_>| {
            seen.push(Seen {
                step: step.step,
                loss: step.loss.to_bits(),
                held_out: step.held_out.map(metric_bits),
                values: step.values.map(bits_of),
            });
            Ok(())
        },
    );
    match result {
        Ok((run, stats)) => {
            let finished = finished(&run, std::mem::take(&mut seen), written);
            (Ok((finished, stats)), Vec::new())
        }
        Err(error) => (Err(error), seen),
    }
}

fn finished(run: &Run, seen: Vec<Seen>, checkpoints: Vec<(u64, Bits)>) -> Finished {
    Finished {
        start: run.start_step,
        losses: run.losses.iter().map(|l| l.to_bits()).collect(),
        values: bits_of(&run.state.values),
        moments: run
            .state
            .moments
            .iter()
            .map(|(k, v)| {
                (
                    format!("{k:?}"),
                    v.data().iter().map(|x| x.to_bits()).collect(),
                )
            })
            .collect(),
        versions: format!("{:?}", run.state.versions),
        seen,
        checkpoints,
    }
}

fn finish(
    result: Result<(Finished, RunStats), VulkanRunError>,
    k: Option<u64>,
) -> (Finished, RunStats) {
    result.unwrap_or_else(|error| panic!("the run with submit_steps {k:?} finishes: {error}"))
}

/// The first field in which two runs differ, for a readable failure.
fn first_difference(a: &Finished, b: &Finished) -> Option<String> {
    if a.losses != b.losses {
        let at = a.losses.iter().zip(&b.losses).position(|(x, y)| x != y);
        return Some(format!(
            "losses differ first at index {at:?} (lengths {} and {})",
            a.losses.len(),
            b.losses.len()
        ));
    }
    if a.seen != b.seen {
        let at = a.seen.iter().zip(&b.seen).position(|(x, y)| x != y);
        return Some(format!(
            "observed steps differ first at {at:?}: {:?} against {:?}",
            at.map(|i| (&a.seen[i].step, a.seen[i].loss, a.seen[i].held_out)),
            at.map(|i| (&b.seen[i].step, b.seen[i].loss, b.seen[i].held_out))
        ));
    }
    if a.checkpoints != b.checkpoints {
        let steps = |f: &Finished| f.checkpoints.iter().map(|(k, _)| *k).collect::<Vec<_>>();
        return Some(format!(
            "checkpoints differ: steps {:?} against {:?}",
            steps(a),
            steps(b)
        ));
    }
    if a.values != b.values {
        return Some("final parameters differ".into());
    }
    if a.moments != b.moments {
        return Some("final moments differ".into());
    }
    if a.versions != b.versions {
        return Some(format!(
            "versions differ: {} against {}",
            a.versions, b.versions
        ));
    }
    if a.start != b.start {
        return Some("start differs".into());
    }
    None
}

/// MS-1. Without an evaluation, a run keeps one segment in flight while it submits the next:
/// the boundaries between two command buffers that run back to back on the queue, with no host
/// wait between them. `mlm_bag` (SGD, packed data) over 12 steps at K = 1, 2, 3 and the runner's
/// choice: no validation message at any K. Each K runs on an instance of its own, the runner's
/// choice first, since the layer stops repeating a message after its duplicate limit.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn ms_device_back_to_back_segments_are_sync_clean() {
    {
        let gpu = gpu();
        let device = gpu
            .vulkan
            .create_device(0, &gpu.negotiated)
            .expect("device 0 opens");
        assert_sync_validation_is_live(&gpu.vulkan, &device);
    }
    let dir = packed_dir("sync_clean");
    let source = mlm_bag(12, "1.0");
    let compiled = compile(&source);
    let data = bind(&dir, &compiled);
    let mut reported = Vec::new();
    for k in [None, Some(3), Some(2), Some(1)] {
        let gpu = gpu();
        let device = gpu
            .vulkan
            .create_device(0, &gpu.negotiated)
            .expect("device 0 opens");
        let artifact = artifact(&source, &gpu.profile);
        let plan = Plan {
            compiled: &compiled,
            artifact: &artifact,
            data: &data,
            eval: None,
            checkpoint_every: None,
            start: 0,
            init: BTreeMap::new(),
        };
        let (result, _) = run(&gpu, &device, &plan, k);
        let (_, stats) = finish(result, k);
        let messages = gpu.vulkan.take_validation_messages();
        eprintln!(
            "submit_steps {k:?}: {} host waits, {} messages",
            stats.host_waits,
            messages.len()
        );
        if !messages.is_empty() {
            reported.push(format!(
                "submit_steps {k:?}: {} messages, first: {}",
                messages.len(),
                messages[0]
            ));
        }
    }
    assert!(reported.is_empty(), "{reported:#?}");
}

/// The hidden-32 twin (dropout tokens per step, AdamW scalars and warm-up per step, packed MLM
/// batches) over 21 steps without an evaluation: K = 2, 3, 8, 21, 64 and the runner's choice give
/// the bits of K = 1 in every loss, every observed step, the final parameters, moments and
/// versions.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn ms_device_twin_is_the_bits_of_one_step_per_submit() {
    let gpu = gpu();
    let device = gpu
        .vulkan
        .create_device(0, &gpu.negotiated)
        .expect("device 0 opens");
    let dir = packed_dir("twin_bits");
    let source = twin(21);
    let compiled = compile(&source);
    let artifact = artifact(&source, &gpu.profile);
    let data = bind(&dir, &compiled);
    let plan = Plan {
        compiled: &compiled,
        artifact: &artifact,
        data: &data,
        eval: None,
        checkpoint_every: None,
        start: 0,
        init: BTreeMap::new(),
    };
    let (one, one_stats) = finish(run(&gpu, &device, &plan, Some(1)).0, Some(1));
    assert_eq!(one.losses.len(), 22);
    assert_eq!(one_stats.host_waits, 22, "{one_stats}");
    let mut differing = Vec::new();
    for k in [Some(2), Some(3), Some(8), Some(21), Some(64), None] {
        let (other, stats) = finish(run(&gpu, &device, &plan, k).0, k);
        eprintln!("submit_steps {k:?}: {stats}");
        if let Some(why) = first_difference(&one, &other) {
            differing.push(format!("submit_steps {k:?}: {why}"));
        }
    }
    let _ = gpu.vulkan.take_validation_messages();
    assert!(differing.is_empty(), "{differing:#?}");
}

/// `mlm_bag` (SGD) over 23 steps, evaluated every 7 steps (0, 7, 14, 21 and the last, 23) on the
/// device and on the host, with a checkpoint every 5 steps (5, 10, 15, 20), at K = 3, 8, 64:
/// segments of one step, evaluations at step 0 and at the last step, checkpoints that are no
/// multiple of K. Every observed step, held-out metric and checkpoint is the bits of K = 1.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn ms_device_eval_and_checkpoint_schedule_is_the_bits_of_one_step_per_submit() {
    let gpu = gpu();
    let device = gpu
        .vulkan
        .create_device(0, &gpu.negotiated)
        .expect("device 0 opens");
    let dir = packed_dir("schedule");
    let source = mlm_bag(23, "1.0");
    let compiled = compile(&source);
    let artifact = artifact(&source, &gpu.profile);
    let data = bind(&dir, &compiled);
    let held = HeldOut::new(&compiled, &data).expect("the held-out split");
    let mut differing = Vec::new();
    for reduction in [HeldOutReduction::Device, HeldOutReduction::Host] {
        let plan = Plan {
            compiled: &compiled,
            artifact: &artifact,
            data: &data,
            eval: Some((&held, 7, reduction)),
            checkpoint_every: Some(5),
            start: 0,
            init: BTreeMap::new(),
        };
        let (one, _) = finish(run(&gpu, &device, &plan, Some(1)).0, Some(1));
        let evaluated: Vec<u64> = one
            .seen
            .iter()
            .filter(|s| s.held_out.is_some())
            .map(|s| s.step)
            .collect();
        assert_eq!(evaluated, [0, 7, 14, 21, 23], "{reduction:?}");
        let written: Vec<u64> = one.checkpoints.iter().map(|(k, _)| *k).collect();
        assert_eq!(written, [5, 10, 15, 20], "{reduction:?}");
        for k in [Some(3), Some(8), Some(64), None] {
            let (other, stats) = finish(run(&gpu, &device, &plan, k).0, k);
            eprintln!("{reduction:?} submit_steps {k:?}: {stats}");
            if let Some(why) = first_difference(&one, &other) {
                differing.push(format!("{reduction:?} submit_steps {k:?}: {why}"));
            }
            if stats.evaluation.is_zero() {
                differing.push(format!(
                    "{reduction:?} submit_steps {k:?}: five evaluations ran and the evaluation \
                     clock is 0"
                ));
            }
        }
    }
    let _ = gpu.vulkan.take_validation_messages();
    assert!(differing.is_empty(), "{differing:#?}");
}

/// A run resumed at step 10 from the parameters a K = 8 run checkpointed there, over 13 more
/// steps with a checkpoint every 4 (12, 16, 20): K = 8 and K = 3 give the bits of K = 1.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn ms_device_a_resumed_run_is_the_bits_of_one_step_per_submit() {
    let gpu = gpu();
    let device = gpu
        .vulkan
        .create_device(0, &gpu.negotiated)
        .expect("device 0 opens");
    let dir = packed_dir("resume");
    let source = mlm_bag(23, "1.0");
    let compiled = compile(&source);
    let artifact = artifact(&source, &gpu.profile);
    let data = bind(&dir, &compiled);
    let full = Plan {
        compiled: &compiled,
        artifact: &artifact,
        data: &data,
        eval: None,
        checkpoint_every: Some(10),
        start: 0,
        init: BTreeMap::new(),
    };
    let (by_one, _) = finish(run(&gpu, &device, &full, Some(1)).0, Some(1));
    let (by_eight, _) = finish(run(&gpu, &device, &full, Some(8)).0, Some(8));
    assert_eq!(
        first_difference(&by_one, &by_eight),
        None,
        "the full run at K = 8"
    );
    let at_ten = &by_eight
        .checkpoints
        .iter()
        .find(|(k, _)| *k == 10)
        .expect("a checkpoint at 10")
        .1;
    // The checkpoint's values as tensors, shaped as the initial parameters.
    let shapes = modelspec_runtime::train::init::initialize(
        &compiled.forward.program,
        compiled.config.seed,
        modelspec_runtime::train::init::Stream::Init,
    )
    .expect("initial parameters");
    let init: BTreeMap<ParamId, HostTensor<f32>> = shapes
        .iter()
        .map(|(p, t)| {
            let values = at_ten[p].iter().map(|&b| f32::from_bits(b)).collect();
            (
                *p,
                HostTensor::new(t.shape().to_vec(), values).expect("a tensor"),
            )
        })
        .collect();
    let resumed_source = mlm_bag(13, "1.0");
    let resumed_compiled = compile(&resumed_source);
    let resumed_artifact = compile_artifact_for(
        "adversary.mspec",
        &resumed_source,
        Backend::Vulkan,
        Some(&gpu.profile),
    )
    .expect("the resumed source compiles for Vulkan");
    let resumed = Plan {
        compiled: &resumed_compiled,
        artifact: &resumed_artifact,
        data: &data,
        eval: None,
        checkpoint_every: Some(4),
        start: 10,
        init,
    };
    let (one, _) = finish(run(&gpu, &device, &resumed, Some(1)).0, Some(1));
    let written: Vec<u64> = one.checkpoints.iter().map(|(k, _)| *k).collect();
    assert_eq!(written, [12, 16, 20]);
    let mut differing = Vec::new();
    for k in [Some(3), Some(8)] {
        let (other, _) = finish(run(&gpu, &device, &resumed, k).0, k);
        if let Some(why) = first_difference(&one, &other) {
            differing.push(format!("resumed, submit_steps {k:?}: {why}"));
        }
    }
    let _ = gpu.vulkan.take_validation_messages();
    assert!(differing.is_empty(), "{differing:#?}");
}

/// MS-2. `mlm_bag` at `lr = 1e30` diverges a few steps in. `Diverged::run::state` is "the state
/// that step reached" (`train/runner.rs:151`): at K = 8 it must be the state of K = 1, the
/// parameters and versions after the failing step, not after the last step the device ran past
/// it.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn ms_device_a_diverged_run_returns_the_state_of_its_failing_step() {
    let gpu = gpu();
    let device = gpu
        .vulkan
        .create_device(0, &gpu.negotiated)
        .expect("device 0 opens");
    let dir = packed_dir("diverged");
    let source = mlm_bag(40, "1e30");
    let compiled = compile(&source);
    let artifact = artifact(&source, &gpu.profile);
    let data = bind(&dir, &compiled);
    let plan = Plan {
        compiled: &compiled,
        artifact: &artifact,
        data: &data,
        eval: None,
        checkpoint_every: None,
        start: 0,
        init: BTreeMap::new(),
    };
    let diverged = |k: Option<u64>| {
        let (result, seen) = run(&gpu, &device, &plan, k);
        match result {
            Err(VulkanRunError::NonFinite(stopped)) => (*stopped, seen),
            Err(error) => panic!("submit_steps {k:?}: {error}"),
            Ok(_) => panic!("submit_steps {k:?}: the run at lr 1e30 finishes"),
        }
    };
    let (one, one_seen) = diverged(Some(1));
    let (eight, eight_seen) = diverged(Some(8));
    let failing = one.diverged.run.losses.len() as u64 - 1;
    eprintln!(
        "K = 1 fails at step {failing}: {}",
        one.diverged.diagnostic.message
    );
    assert!(
        failing % 8 != 7,
        "the failing step {failing} ends a segment of 8"
    );
    let mut differing = Vec::new();
    if one_seen != eight_seen {
        differing.push("the observed steps differ".to_owned());
    }
    if one.diverged.diagnostic != eight.diverged.diagnostic {
        differing.push(format!(
            "diagnostic: {:?} against {:?}",
            one.diverged.diagnostic, eight.diverged.diagnostic
        ));
    }
    let a = finished(&one.diverged.run, Vec::new(), Vec::new());
    let b = finished(&eight.diverged.run, Vec::new(), Vec::new());
    if let Some(why) = first_difference(&a, &b) {
        differing.push(format!("Diverged::run: {why}"));
    }
    if one.stats.steps != eight.stats.steps {
        differing.push(format!(
            "stats.steps {} against {}",
            one.stats.steps, eight.stats.steps
        ));
    }
    let _ = gpu.vulkan.take_validation_messages();
    assert!(differing.is_empty(), "{differing:#?}");
}

/// The largest `submit_steps` is a refusal or a run, not a panic: the runner sizes two upload
/// buffers of `step_bytes · K`.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn ms_device_the_largest_submit_steps_does_not_panic() {
    let gpu = gpu();
    let device = gpu
        .vulkan
        .create_device(0, &gpu.negotiated)
        .expect("device 0 opens");
    let dir = packed_dir("largest");
    let source = mlm_bag(3, "1.0");
    let compiled = compile(&source);
    let artifact = artifact(&source, &gpu.profile);
    let data = bind(&dir, &compiled);
    let plan = Plan {
        compiled: &compiled,
        artifact: &artifact,
        data: &data,
        eval: None,
        checkpoint_every: None,
        start: 0,
        init: BTreeMap::new(),
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(&gpu, &device, &plan, Some(u64::MAX)).0.map(|_| ())
    }));
    let _ = gpu.vulkan.take_validation_messages();
    match outcome {
        Ok(result) => eprintln!("submit_steps u64::MAX: {result:?}"),
        Err(panic) => {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_default();
            panic!("submit_steps u64::MAX panics: {message}");
        }
    }
}

/// `--clock`'s split on the device: with five evaluations the evaluation clock is above 0, and
/// the training and evaluation clocks together are at most the wall time of the call.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn ms_device_training_and_evaluation_clocks_fit_in_the_wall_time() {
    let gpu = gpu();
    let device = gpu
        .vulkan
        .create_device(0, &gpu.negotiated)
        .expect("device 0 opens");
    let dir = packed_dir("clock");
    let source = mlm_bag(40, "1.0");
    let compiled = compile(&source);
    let artifact = artifact(&source, &gpu.profile);
    let data = bind(&dir, &compiled);
    let held = HeldOut::new(&compiled, &data).expect("the held-out split");
    let plan = Plan {
        compiled: &compiled,
        artifact: &artifact,
        data: &data,
        eval: Some((&held, 10, HeldOutReduction::Device)),
        checkpoint_every: None,
        start: 0,
        init: BTreeMap::new(),
    };
    let wall = Instant::now();
    let (_, stats) = finish(run(&gpu, &device, &plan, None).0, None);
    let wall = wall.elapsed();
    let _ = gpu.vulkan.take_validation_messages();
    eprintln!(
        "training {:?} evaluation {:?} wall {wall:?}",
        stats.training, stats.evaluation
    );
    assert!(!stats.evaluation.is_zero(), "{stats}");
    assert!(!stats.training.is_zero(), "{stats}");
    assert!(stats.training + stats.evaluation <= wall, "{stats}");
}
