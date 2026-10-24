//! Adversarial cases for the run counters of a Vulkan run (`story:vulkan-step-counters`):
//! the counters are thread-local, so two runs on one thread, and buffers made before a run or
//! kept across it, must not leak into a run's figures.
#![cfg(feature = "vulkan")]

use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_runtime::vulkan::Vulkan;
use modelspec_runtime::vulkan::memory::MemorySpan;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::runner::{train, train_profiled};

const LINEAR_REGRESSION: &str = "examples/linear_regression.mspec";

/// Three runs on one thread and one device, with a large buffer made before the second and
/// dropped after it: each run reports the counts of its own buffers and step, the same three.
#[test]
#[ignore = "needs a Vulkan device"]
fn runs_on_one_thread_count_only_their_own_buffers_and_barriers() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(LINEAR_REGRESSION);
    let source = std::fs::read_to_string(&path).expect("the example reads");
    let compiled = compile_training(LINEAR_REGRESSION, &source).expect("the example compiles");
    let artifact = compile_artifact(LINEAR_REGRESSION, &source, Backend::Vulkan)
        .expect("the example compiles for Vulkan");
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");

    let (_, first) = train(&device, &profile, &negotiated, &compiled, &artifact).expect("run 1");
    let held = device
        .device_local_buffer(1 << 20)
        .expect("a buffer outside any run");
    let (_, second) = train(&device, &profile, &negotiated, &compiled, &artifact).expect("run 2");
    drop(held);
    let (_, third, _) =
        train_profiled(&device, &profile, &negotiated, &compiled, &artifact).expect("run 3");
    drop(device);
    assert!(vulkan.take_validation_messages().is_empty());

    for (name, stats) in [("second", &second), ("third", &third)] {
        assert_eq!(
            (
                stats.barriers_per_step,
                stats.device_allocations,
                stats.peak_device_bytes
            ),
            (
                first.barriers_per_step,
                first.device_allocations,
                first.peak_device_bytes
            ),
            "the {name} run"
        );
    }
    assert_eq!((first.barriers_per_step, first.device_allocations), (7, 12));
}

/// The peak of a span is the largest sum of live allocations, not their total: a real buffer
/// freed inside the span (through `Buffer`'s drop, which no other device test exercises) and
/// re-made leaves the peak at one buffer. A buffer made in an earlier span and freed in a later
/// one does not reduce the later span's live bytes. Catches a `Buffer::drop` that stops calling
/// the free counter, which the run tests miss because a run frees nothing before its end.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_buffer_freed_and_remade_in_a_span_peaks_at_one_buffer() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");

    let span = MemorySpan::start();
    let one = device.device_local_buffer(1 << 20).expect("buffer 1");
    let size = one.allocation_bytes();
    drop(one);
    let two = device.device_local_buffer(1 << 20).expect("buffer 2");
    assert_eq!(span.allocations(), 2);
    assert_eq!(span.peak_bytes(), size, "peak after free and re-make");
    drop(span);

    let later = MemorySpan::start();
    drop(two);
    let three = device.staging_buffer(4096).expect("buffer 3");
    assert_eq!(later.allocations(), 1);
    assert_eq!(later.peak_bytes(), three.allocation_bytes());
    drop(three);
    drop(later);
    drop(device);
    assert!(vulkan.take_validation_messages().is_empty());
}
