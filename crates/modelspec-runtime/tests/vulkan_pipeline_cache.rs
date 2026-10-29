//! Pipeline creation at startup (story:vulkan-pipeline-startup): [`Device::kernel_pipelines`]
//! makes one pipeline per distinct SPIR-V module, and a [`Device::pipeline_cache`] file written
//! by one run is read back by the next, while a corrupt or foreign file is ignored with a warning
//! and the pipelines made through the cache compute the bits of pipelines made without one.
//!
//! The device tests are ignored by default; run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_pipeline_cache -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeSet;
use std::path::PathBuf;

use common::{Case, Draw, ElementwiseOp, Form};
use modelspec_core::ir::kernel::Kernel;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::pipeline::{Pipeline, PipelineRequest, kernel_broadcast};
use modelspec_runtime::vulkan::pipeline_cache::{decode, encode, file_name, identity};
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, ExecError, Vulkan};

const COUNT: u64 = 4_097;

fn add() -> Case {
    Case {
        name: "add".to_owned(),
        op: ElementwiseOp::Add,
        operands: vec![Form::Tensor, Form::Tensor],
    }
}

fn mul_rank0() -> Case {
    Case {
        name: "mul".to_owned(),
        op: ElementwiseOp::Mul,
        operands: vec![Form::Rank0, Form::Tensor],
    }
}

/// A fresh, empty directory under this crate's build directory, named for the test.
fn scratch(name: &str) -> PathBuf {
    let dir =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("vulkan_pipeline_cache_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// Runs the `add` kernel through `pipeline` on `a` and `b` and reads its result back.
fn run_add(
    device: &Device<'_>,
    pipeline: &Pipeline<'_>,
    kernel: &Kernel,
    a: &[f32],
    b: &[f32],
) -> Vec<u32> {
    let bytes = 4 * COUNT;
    let buffers: Vec<Buffer<'_>> = (0..3)
        .map(|_| device.device_local_buffer(bytes).expect("a device buffer"))
        .collect();
    let mut staging: Vec<Buffer<'_>> = (0..3)
        .map(|_| device.staging_buffer(bytes).expect("a staging buffer"))
        .collect();
    staging[0].write_f32(a).expect("a is written");
    staging[1].write_f32(b).expect("b is written");
    let mut recording = device.record().expect("a recording starts");
    for at in 0..2 {
        recording
            .copy(&staging[at], &buffers[at])
            .expect("an upload");
    }
    recording.barrier(Barrier::TransferToCompute);
    let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
    recording
        .dispatch(pipeline, &bound, COUNT as u32, kernel.workgroups)
        .expect("the dispatch is recorded");
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[2], &staging[2])
        .expect("the download");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("the commands are submitted")
        .wait()
        .expect("the device finishes");
    staging[2]
        .read_f32()
        .expect("the result reads")
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

#[test]
#[ignore = "needs a Vulkan device"]
fn kernel_pipelines_make_one_pipeline_per_distinct_module() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let kernels = [
        common::kernel(&add(), 64),
        common::kernel(&add(), 64),
        common::kernel(&add(), 65),
        common::kernel(&mul_rank0(), 64),
        common::kernel(&add(), 65),
    ];
    let words: Vec<Vec<u32>> = kernels
        .iter()
        .map(|kernel| emit(kernel).expect("the kernel emits"))
        .collect();
    let distinct: BTreeSet<(Vec<u32>, Vec<bool>)> = kernels
        .iter()
        .zip(&words)
        .map(|(kernel, words)| (words.clone(), kernel_broadcast(kernel)))
        .collect();
    let requests: Vec<PipelineRequest<'_>> = kernels
        .iter()
        .zip(&words)
        .map(|(kernel, words)| PipelineRequest {
            kernel,
            words,
            seed: None,
        })
        .collect();
    let made = device
        .kernel_pipelines(&requests, None, 4)
        .expect("the pipelines are made");
    assert_eq!(made.distinct(), distinct.len());
    assert!(made.distinct() < requests.len());
    assert!(made.distinct() > 1);
    // Two requests share one pipeline exactly when their words and bindings are equal; the
    // element count is a push constant, so `add` over 64 and over 65 elements is one module.
    for i in 0..kernels.len() {
        for j in 0..kernels.len() {
            let same = words[i] == words[j]
                && kernel_broadcast(&kernels[i]) == kernel_broadcast(&kernels[j]);
            assert_eq!(
                std::ptr::eq(made.get(i), made.get(j)),
                same,
                "requests {i} and {j}"
            );
        }
    }
    assert_eq!(made.each().len(), requests.len());
    for (at, kernel) in kernels.iter().enumerate() {
        assert_eq!(
            made.get(at).broadcast(),
            kernel_broadcast(kernel).as_slice()
        );
    }

    let mut dropout = common::kernel(&add(), 64);
    dropout.op = "Dropout";
    let unseeded = [PipelineRequest {
        kernel: &dropout,
        words: &words[0],
        seed: None,
    }];
    assert!(matches!(
        device.kernel_pipelines(&unseeded, None, 1),
        Err(ExecError::KernelShape { .. })
    ));
    let messages = vulkan.take_validation_messages();
    assert!(messages.is_empty(), "{messages:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn a_cache_file_is_read_back_and_a_corrupt_or_foreign_one_is_ignored_with_the_same_bits() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = &candidates[0].profile;
    let negotiated = negotiate(profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let kernel = common::kernel(&add(), COUNT);
    let words = emit(&kernel).expect("the kernel emits");
    let requests = [PipelineRequest {
        kernel: &kernel,
        words: &words,
        seed: None,
    }];
    let mut draw = Draw::new(0x5049_5045_4c49_4e45);
    let (a, b) = (draw.values(COUNT), draw.values(COUNT));
    let uncached = device
        .kernel_pipeline(&kernel, &words)
        .expect("the pipeline without a cache");
    let expected = run_add(&device, &uncached, &kernel, &a, &b);

    let dir = scratch("roundtrip");
    let file = dir.join(file_name(profile));
    // No file yet: an empty cache, no warning; the run writes the file.
    let mut cache = device
        .pipeline_cache(profile, Some(&dir))
        .expect("the cache opens");
    assert!(cache.warnings().is_empty(), "{:?}", cache.warnings());
    assert_eq!(cache.file(), Some(file.as_path()));
    let made = device
        .kernel_pipelines(&requests, Some(&cache), 2)
        .expect("the pipelines are made");
    assert_eq!(run_add(&device, made.get(0), &kernel, &a, &b), expected);
    assert_eq!(cache.store(), None);
    drop(made);
    drop(cache);
    let written = std::fs::read(&file).expect("the cache file is written");
    let data = decode(&identity(profile), &written).expect("the file is this device's");
    assert!(!data.is_empty(), "the driver's data is kept");

    // Read back: no warning, and the same bits.
    let cache = device
        .pipeline_cache(profile, Some(&dir))
        .expect("the cache opens");
    assert!(cache.warnings().is_empty(), "{:?}", cache.warnings());
    let made = device
        .kernel_pipelines(&requests, Some(&cache), 2)
        .expect("the pipelines are made from the cache");
    assert_eq!(run_add(&device, made.get(0), &kernel, &a, &b), expected);
    drop(made);
    drop(cache);

    let mut flipped = written.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 0x5a;
    let foreign = encode("modelspec-pipeline-cache/1 another device", data);
    let cases: [(&str, Vec<u8>, &str); 4] = [
        (
            "corrupt",
            flipped,
            "its data does not match the hash in its header",
        ),
        (
            "foreign",
            foreign,
            "it was written for another device or driver",
        ),
        ("short", b"abc".to_vec(), "shorter than the 80-byte header"),
        (
            "vulkan-only",
            data.to_vec(),
            "it is not a modelspec pipeline cache file",
        ),
    ];
    for (name, bytes, reason) in cases {
        std::fs::write(&file, &bytes).expect("the file is replaced");
        let mut cache = device
            .pipeline_cache(profile, Some(&dir))
            .expect("a bad file still opens a cache");
        assert_eq!(cache.warnings().len(), 1, "{name}: {:?}", cache.warnings());
        let warning = &cache.warnings()[0];
        assert!(
            warning.starts_with(&format!(
                "warning: pipeline cache {} ignored: ",
                file.display()
            )) && warning.contains(reason),
            "{name}: {warning}"
        );
        let made = device
            .kernel_pipelines(&requests, Some(&cache), 2)
            .expect("the pipelines are made");
        assert_eq!(
            run_add(&device, made.get(0), &kernel, &a, &b),
            expected,
            "{name}: the bits of a run without a cache"
        );
        drop(made);
        assert_eq!(cache.store(), None);
        drop(cache);
        let rewritten = std::fs::read(&file).expect("the file is rewritten");
        assert!(
            decode(&identity(profile), &rewritten).is_ok(),
            "{name}: the run replaces the ignored file with its own"
        );
    }

    // Without a directory the cache lives in memory and writes nothing.
    let mut memory = device
        .pipeline_cache(profile, None)
        .expect("the cache opens");
    assert_eq!(memory.file(), None);
    assert_eq!(memory.store(), None);
    let messages = vulkan.take_validation_messages();
    assert!(messages.is_empty(), "{messages:#?}");
}
