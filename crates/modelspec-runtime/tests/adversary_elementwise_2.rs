//! Adversary pass 2 for story:m1-vulkan-elementwise: the fixes of pass 1, attacked.
//!
//! No implementation file is touched; module mutations are applied to SPIR-V words inside the
//! tests. Device cases are ignored by default, like the suite's:
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test adversary_elementwise_2 -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::{Case, ElementwiseOp, Form};
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::{Device, ExecError, Vulkan};

const OP_DECORATE: u32 = 71;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;

fn open() -> Vulkan {
    Vulkan::load().expect("the Vulkan loader opens")
}

fn device(vulkan: &Vulkan) -> Device<'_> {
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens")
}

fn add_words(count: u64) -> Vec<u32> {
    let case = Case {
        name: "add".to_owned(),
        op: ElementwiseOp::Add,
        operands: vec![Form::Tensor, Form::Tensor],
    };
    emit(&common::kernel(&case, count)).expect("the kernel emits")
}

/// `words` with the literal of every `OpDecorate … decoration <literal>` for which `pick`
/// returns a new value replaced by it.
fn redecorate(words: &[u32], decoration: u32, pick: impl Fn(u32) -> Option<u32>) -> Vec<u32> {
    let mut out = words.to_vec();
    let mut at = 5;
    let mut hits = 0;
    while at < out.len() {
        let length = (out[at] >> 16) as usize;
        if out[at] & 0xffff == OP_DECORATE
            && length == 4
            && out[at + 2] == decoration
            && let Some(new) = pick(out[at + 3])
        {
            out[at + 3] = new;
            hits += 1;
        }
        at += length.max(1);
    }
    assert!(hits > 0, "the module has decoration {decoration}");
    out
}

// ---------------------------------------------------------------------------------------------
// Item 5: the one-element exemption of `Recording::dispatch`.

/// Records (never submits) the 64-element add over `a`, `b`, `out` of the given byte sizes.
fn record_add(device: &Device<'_>, a: u64, b: u64, out: u64) -> Result<(), ExecError> {
    let pipeline = device.pipeline(&add_words(64), 3).unwrap();
    let (a, b, out) = (
        device.device_local_buffer(a).unwrap(),
        device.device_local_buffer(b).unwrap(),
        device.device_local_buffer(out).unwrap(),
    );
    let mut recording = device.record().unwrap();
    recording.dispatch(&pipeline, &[&a, &b, &out], 64, [1, 1, 1])
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary2_dispatch_refuses_64_elements_into_a_one_element_tensor_output() {
    let vulkan = open();
    let device = device(&vulkan);
    // The kernel's result is a 64-element tensor (binding 2 is indexed by the invocation), the
    // buffer bound to it holds one f32: 63 of the 64 stores land past its end.
    let recorded = record_add(&device, 256, 256, 4);
    assert!(
        matches!(recorded, Err(ExecError::CountPastBuffer { binding: 2, .. })),
        "a 64-element add into a 1-element output buffer was recorded: {recorded:?}"
    );
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary2_dispatch_refuses_64_elements_from_a_one_element_tensor_input() {
    let vulkan = open();
    let device = device(&vulkan);
    // Operand 0 is a 64-element tensor in the module (not rank 0), bound to one f32.
    let recorded = record_add(&device, 4, 256, 256);
    assert!(
        matches!(recorded, Err(ExecError::CountPastBuffer { binding: 0, .. })),
        "a 64-element add reading a 1-element tensor operand was recorded: {recorded:?}"
    );
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary2_dispatch_refuses_64_elements_into_a_buffer_shorter_than_one_element() {
    let vulkan = open();
    let device = device(&vulkan);
    // Two bytes: `bytes / 4` is 0 elements, which the `elements > 1` exemption also passes.
    let recorded = record_add(&device, 256, 256, 2);
    assert!(
        recorded.is_err(),
        "a 64-element add into a 2-byte output buffer was recorded: {recorded:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// Item 4: `declared_bindings` and `Device::pipeline`.

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary2_pipeline_refuses_a_module_whose_bindings_are_in_descriptor_set_1() {
    let vulkan = open();
    let device = device(&vulkan);
    let _ = vulkan.take_validation_messages();
    // Bindings 0, 1, 2 as before, all moved to descriptor set 1; the layout has only set 0.
    let words = redecorate(&add_words(65), DECORATION_DESCRIPTOR_SET, |_| Some(1));
    let made = device.pipeline(&words, 3).map(|_| ());
    let messages = vulkan.take_validation_messages();
    assert!(
        matches!(made, Err(ExecError::ModuleBindings { .. })),
        "a pipeline with set-0 layout bindings for a module bound in set 1 was created: \
         {made:?}; validation messages: {messages:#?}"
    );
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary2_pipeline_refuses_duplicated_and_accepts_reordered_bindings() {
    let vulkan = open();
    let device = device(&vulkan);
    let words = add_words(65);
    // Binding 1 decorated 0 as well: 0, 0, 2.
    let duplicated = redecorate(&words, DECORATION_BINDING, |b| (b == 1).then_some(0));
    assert!(matches!(
        device.pipeline(&duplicated, 3).map(|_| ()),
        Err(ExecError::ModuleBindings { .. })
    ));
    // Bindings permuted 2, 0, 1: still 0..3.
    let reordered = redecorate(&words, DECORATION_BINDING, |b| Some((b + 2) % 3));
    assert!(device.pipeline(&reordered, 3).is_ok());
    assert!(vulkan.take_validation_messages().is_empty());
}
