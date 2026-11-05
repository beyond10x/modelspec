//! story:vulkan-segment-command-buffer-reuse: a stepwise Vulkan run records the command buffer
//! of a segment of training steps once per upload parity and step count and submits it again,
//! so the driver is asked for a fixed number of command pools however long the run is.
//!
//! The cases run with the Khronos validation layer's synchronization validation on. The
//! hidden-32 BERT twin (dropout tokens, AdamW scalars and MLM batches that differ every step)
//! with segments of 4 steps, most of them submitted again, must give the bits of the same run
//! with one segment for all its training steps, which submits nothing twice. The packed
//! directory is the original-text fixture `tests/fixtures/corpus-tokenizer/`.
//!
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_segment_reuse -- --ignored --test-threads=1`
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Once;

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{CompiledTraining, compile_artifact_for, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::tensor::ParamId;
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::dataset::mlm;
use modelspec_runtime::dataset::{self, tokenizer};
use modelspec_runtime::train::data::{self, DataSource, PackedData};
use modelspec_runtime::train::runner::Data;
use modelspec_runtime::vulkan::negotiate::{Negotiated, negotiate};
use modelspec_runtime::vulkan::stepwise::{StepSeen, StepwiseOptions, train_stepwise};
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, ExecError, VALIDATION_ENV, Vulkan};

const SEQ_LEN: u64 = 16;
const VOCAB: usize = 300;

static ENVIRONMENT: Once = Once::new();

fn packed_dir(case: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("segment_reuse_{case}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
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

/// The hidden-32 BERT twin (dropout, `adamw` with warm-up) at `steps` steps.
fn twin(steps: u64) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../modelspec-cli/tests/fixtures/bert_cpu.mspec");
    let source = fs::read_to_string(path).expect("the fixture");
    assert!(source.contains("steps = 800;"));
    assert!(source.contains("warmup_steps = 80"));
    source
        .replace("steps = 800;", &format!("steps = {steps};"))
        .replace("warmup_steps = 80", "warmup_steps = 5")
}

fn bind(dir: &Path, compiled: &CompiledTraining) -> PackedData {
    let binding = match data::source(dir, &compiled.config).expect("a source") {
        DataSource::Packed(binding) => binding,
        DataSource::Dense => panic!("{} holds binding.json", dir.display()),
    };
    data::packed(dir, binding, &compiled.config, &compiled.forward).expect("the data binds")
}

struct Gpu {
    vulkan: Vulkan,
    profile: DeviceProfile,
    negotiated: Negotiated,
}

fn gpu() -> Gpu {
    ENVIRONMENT.call_once(|| {
        for (name, value) in [
            (VALIDATION_ENV, "1"),
            ("VK_KHRONOS_VALIDATION_VALIDATE_SYNC", "1"),
            ("VK_VALIDATE_SYNC", "1"),
            ("VK_KHRONOS_VALIDATION_SYNCVAL_SUBMIT_TIME_VALIDATION", "1"),
        ] {
            // SAFETY: every case calls `gpu` before it reads the environment or loads Vulkan,
            // and `Once` holds the others until this returns.
            unsafe { std::env::set_var(name, value) };
        }
    });
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert!(vulkan.validation_enabled(), "the validation layer is on");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    Gpu {
        vulkan,
        profile,
        negotiated,
    }
}

type Bits = BTreeMap<ParamId, Vec<u32>>;

fn bits_of(values: &BTreeMap<ParamId, HostTensor<f32>>) -> Bits {
    values
        .iter()
        .map(|(p, v)| (*p, v.data().iter().map(|x| x.to_bits()).collect()))
        .collect()
}

/// What one run leaves: its losses, the losses its observer saw, its final parameters, and the
/// command buffers the device recorded and was given for it.
struct Ran {
    losses: Vec<u32>,
    seen: Vec<(u64, u32)>,
    values: Bits,
    recordings: u64,
    submits: u64,
}

fn run(
    gpu: &Gpu,
    device: &Device<'_>,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    data: &PackedData,
    submit_steps: u64,
) -> Ran {
    let recordings = device.recordings();
    let submits = device.submits();
    let mut seen = Vec::new();
    let options = StepwiseOptions {
        data: Some(Data::Packed(data)),
        submit_steps: Some(submit_steps),
        ..StepwiseOptions::default()
    };
    let (run, stats) = train_stepwise(
        device,
        &gpu.profile,
        &gpu.negotiated,
        compiled,
        artifact,
        options,
        &mut |step: &StepSeen<'_>| {
            seen.push((step.step, step.loss.to_bits()));
            Ok(())
        },
    )
    .unwrap_or_else(|error| panic!("the run with submit_steps {submit_steps} finishes: {error}"));
    assert_eq!(stats.submits, device.submits() - submits);
    Ran {
        losses: run.losses.iter().map(|l| l.to_bits()).collect(),
        seen,
        values: bits_of(&run.state.values),
        recordings: device.recordings() - recordings,
        submits: stats.submits,
    }
}

/// The twin over 41 and over 121 steps in segments of 4: both runs record 5 command buffers
/// (the first segment, one per parity for 4 steps, the last training step alone, the final
/// loss) and submit 12 and 32; and each gives, in every loss, every observed step and the final
/// parameters, the bits of the run whose 40 or 120 training steps before the last are one
/// segment, which submits no command buffer twice. No validation message in any run.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn sr_device_segments_submitted_again_are_the_bits_of_segments_recorded_once() {
    let gpu = gpu();
    let device = gpu
        .vulkan
        .create_device(0, &gpu.negotiated)
        .expect("device 0 opens");
    let dir = packed_dir("bits");
    for (steps, submits) in [(41u64, 12u64), (121, 32)] {
        let source = twin(steps);
        let compiled = compile_training("segment_reuse.mspec", &source)
            .unwrap_or_else(|diagnostics| panic!("the source compiles: {diagnostics:?}"));
        let artifact = compile_artifact_for(
            "segment_reuse.mspec",
            &source,
            Backend::Vulkan,
            Some(&gpu.profile),
        )
        .unwrap_or_else(|diagnostics| panic!("the source compiles for Vulkan: {diagnostics:?}"));
        let data = bind(&dir, &compiled);

        let again = run(&gpu, &device, &compiled, &artifact, &data, 4);
        let once = run(&gpu, &device, &compiled, &artifact, &data, steps - 1);

        assert_eq!(
            (again.recordings, again.submits),
            (5, submits),
            "{steps} steps in segments of 4"
        );
        assert_eq!(
            (once.recordings, once.submits),
            (3, 3),
            "{steps} steps, one segment for the first {}",
            steps - 1
        );
        assert_eq!(again.losses.len() as u64, steps + 1);
        assert_eq!(again.losses, once.losses, "{steps} steps: the losses");
        assert_eq!(again.seen, once.seen, "{steps} steps: the observed steps");
        assert!(
            again.values == once.values,
            "{steps} steps: the final parameters differ"
        );
    }
    drop(device);
    let messages = gpu.vulkan.take_validation_messages();
    assert!(messages.is_empty(), "{messages:#?}");
}

/// A reusable recording that copies a staging buffer to a device buffer and down again runs
/// with what the staging buffer holds at each submission; while one submission of it is alive a
/// second is refused, and a recording begun for one submission cannot be kept.
#[test]
#[ignore = "device: needs a Vulkan device"]
fn sr_device_a_reusable_recording_runs_again_and_is_refused_while_pending() {
    let gpu = gpu();
    let device = gpu
        .vulkan
        .create_device(0, &gpu.negotiated)
        .expect("device 0 opens");
    let mut source = device.staging_buffer(256).expect("a staging buffer");
    source.write_f32(&[1.0; 64]).expect("written");
    let target = device.device_local_buffer(256).expect("a device buffer");
    let download = device.download_buffer(256).expect("a download buffer");
    let recordings = device.recordings();
    let submits = device.submits();

    let mut recording = device.record_reusable().expect("a recording");
    recording.copy_region(&source, 0, &target, 0, 256).unwrap();
    recording.barrier(Barrier::TransferToTransfer);
    recording
        .copy_region(&target, 0, &download, 0, 256)
        .unwrap();
    recording.barrier(Barrier::TransferToHost);
    let reusable = recording
        .finish()
        .expect("the recording ends")
        .reusable()
        .expect("it was recorded to be kept");

    device
        .submit_again(&reusable)
        .expect("the first submission")
        .wait()
        .expect("the first wait");
    assert_eq!(download.mapped_f32().unwrap(), &[1.0; 64]);

    // SAFETY: no submission that reads `source` is pending, and nothing holds a slice of it.
    unsafe { source.write_f32_shared(&[2.0; 64]) }.expect("written again");
    let pending = device
        .submit_again(&reusable)
        .expect("the second submission");
    assert!(matches!(
        device.submit_again(&reusable),
        Err(ExecError::StillPending)
    ));
    pending.wait().expect("the second wait");
    assert_eq!(download.mapped_f32().unwrap(), &[2.0; 64]);

    let once = device
        .record()
        .expect("a recording")
        .finish()
        .expect("the recording ends");
    assert!(matches!(once.reusable(), Err(ExecError::RecordedOnce)));

    assert_eq!(device.recordings() - recordings, 2);
    assert_eq!(device.submits() - submits, 2);
    drop(reusable);
    drop((source, target, download));
    drop(device);
    let messages = gpu.vulkan.take_validation_messages();
    assert!(messages.is_empty(), "{messages:#?}");
}
