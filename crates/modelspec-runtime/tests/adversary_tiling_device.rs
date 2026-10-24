//! Adversary cases for `story:matmul-shape-aware-tiling` on a Vulkan device: shapes at the
//! edges of the block rule that `vulkan_matmul_same_bits.rs` does not run.
//!
//! 1. `edge_matmuls_give_the_cpu_bits`: on normal inputs, and on inputs whose sum is `-0.0` in
//!    every third column (strictly negative `a`, those columns of `b` all `+0.0`, so a sum that
//!    starts from `+0.0` instead of the `k = 0` product shows), every written view is bit for
//!    bit the CPU interpreter's. The input buffers end in NaN words, so a read past the end of
//!    `a` or `b` that reaches an output shows; the output views end in guard words, so a write
//!    past the end shows; each validation message fails the case.
//! 2. `edge_matmuls_write_the_bits_of_the_base_commit`: the same shapes, the sweep sites the
//!    block rule takes and subnormal inputs, against digests recorded at the base commit
//!    (269b557) in `tests/fixtures/adversary_tiling_same_bits.txt` with
//!    `ADVERSARY_TILING_RECORD=1`, re-recorded once at story:deterministic-fma, whose fused
//!    multiply-add changed every view (DESIGN.md § 12 ADR-009); a run on another device or
//!    driver fails rather than compares.
//!
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test adversary_tiling_device -- --ignored`
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use common::{Draw, GUARD_WORDS, guard, guarded};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Epilogue, Kernel, MatMulTile};
use modelspec_core::ir::tensor::{TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::interpreter::evaluate;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

const UNWRITTEN: u32 = 0x7fc0_dead;
/// The words after each input: a quiet NaN, so a read past the end that enters a sum shows.
const INPUT_TAIL: u32 = 0x7fc0_beef;
const INPUT_TAIL_WORDS: usize = 64;
const RECORD_ENV: &str = "ADVERSARY_TILING_RECORD";

const KEEP: Option<Epilogue> = Some(Epilogue::Relu {
    writes_pre_activation: true,
});
const RELU: Option<Epilogue> = Some(Epilogue::Relu {
    writes_pre_activation: false,
});

/// Shapes small enough for the CPU interpreter, at the edges of the block rule.
const CPU_SHAPES: [(u64, u64, u64); 34] = [
    // Square at exactly 128 blocks, `n = 64`; one row of blocks fewer is one column.
    (8192, 64, 64),
    (8128, 64, 64),
    // Square, `m % 64 = 1`, `n = 65`: the second column block holds one column.
    (8129, 65, 65),
    // Square, `k = 63`, `n = 127`.
    (4097, 63, 127),
    // Wide with one row block of 64 rows, and with `m = 1`, `n % 128 = 1`.
    (64, 64, 16_384),
    (1, 64, 16_385),
    // Square with `m = 1`, `k = 1`; `m = 2`, `k = 2`, `n = 8191`.
    (1, 1, 8192),
    (2, 2, 8191),
    // Wide at exactly 128 blocks, whole and partial; square at 112 wide blocks and at `k = 63`.
    (512, 64, 2048),
    (512, 64, 2047),
    (448, 64, 2048),
    (512, 63, 2048),
    // Wide, `k` around its depth of 8 and its loop of 4-tile passes.
    (512, 65, 2048),
    (512, 71, 2048),
    (512, 72, 2048),
    (512, 80, 2048),
    (520, 120, 2049),
    (513, 127, 2040),
    // Square, `k` around its depth of 16 and its loop.
    (8192, 1, 64),
    (8192, 2, 64),
    (8192, 7, 64),
    (8192, 15, 64),
    (8192, 16, 64),
    (8192, 17, 64),
    (8192, 31, 64),
    (8192, 32, 64),
    (8192, 33, 64),
    (8192, 48, 64),
    (8192, 100, 64),
    (8191, 129, 100),
    // The sweep sites the square tile takes: b8192 h64 forward and `[b, 10] x [10, h]`.
    (8192, 784, 64),
    (8192, 10, 64),
    (8192, 10, 256),
    (1000, 10, 1024),
];

/// Sweep sites too large for the CPU interpreter in a test, compared with the base only.
const BASE_ONLY_SHAPES: [(u64, u64, u64); 5] = [
    (784, 1000, 1024),
    (784, 8192, 1024),
    (8192, 784, 256),
    (8192, 784, 1024),
    (784, 8192, 256),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Normal,
    NegativeZero,
    Subnormal,
}

impl Family {
    fn name(self) -> &'static str {
        match self {
            Family::Normal => "normal",
            Family::NegativeZero => "negzero",
            Family::Subnormal => "subnormal",
        }
    }
}

fn inputs(m: u64, k: u64, n: u64, family: Family) -> (Vec<f32>, Vec<f32>) {
    let mut draw = Draw::new(0x6164_7674_696c_6531 ^ (m << 40) ^ (k << 20) ^ n);
    let mut a = draw.values(m * k);
    let mut b = draw.values(k * n);
    match family {
        Family::Normal => {}
        Family::NegativeZero => {
            for value in &mut a {
                *value = -(value.abs() + 0.25);
            }
            for (i, value) in b.iter_mut().enumerate() {
                if (i as u64 % n).is_multiple_of(3) {
                    *value = 0.0;
                }
            }
        }
        Family::Subnormal => {
            for values in [&mut a, &mut b] {
                for (i, value) in values.iter_mut().enumerate().filter(|(i, _)| i % 5 == 2) {
                    let magnitude =
                        f32::from_bits(1 + (i as u32).wrapping_mul(0x2f1d) % 0x007f_ffff);
                    *value = magnitude.copysign(*value);
                }
            }
        }
    }
    (a, b)
}

fn tailed(values: &[f32]) -> Vec<f32> {
    let mut words = values.to_vec();
    words.extend(std::iter::repeat_n(
        f32::from_bits(INPUT_TAIL),
        INPUT_TAIL_WORDS,
    ));
    words
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// Runs `kernel` once over `a` and `b`; every written view whole, guard words included.
fn run(
    device: &Device<'_>,
    kernel: &Kernel,
    spirv: &[u32],
    a: &[f32],
    b: &[f32],
    count: usize,
) -> Vec<Vec<f32>> {
    let outputs = kernel.written();
    let uploads: Vec<Vec<f32>> = [tailed(a), tailed(b)]
        .into_iter()
        .chain((0..outputs).map(|_| guarded(&vec![f32::from_bits(UNWRITTEN); count])))
        .collect();
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|values| {
            device
                .device_local_buffer(bytes_of(values.len()))
                .expect("a device buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|values| {
            let mut buffer = device
                .staging_buffer(bytes_of(values.len()))
                .expect("a staging buffer");
            buffer.write_f32(values).expect("written");
            buffer
        })
        .collect();
    let pipeline = device
        .kernel_pipeline(kernel, spirv)
        .expect("the pipeline is created");
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
    recording
        .dispatch_kernel(&pipeline, kernel, &bound)
        .expect("the dispatch is recorded");
    recording.barrier(Barrier::ComputeToTransfer);
    for at in 2..buffers.len() {
        recording
            .copy(&buffers[at], &staging[at])
            .expect("the download is recorded");
    }
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[2..]
        .iter_mut()
        .map(|buffer| buffer.read_f32().expect("the output reads"))
        .collect()
}

fn changed_guards(view: &[f32], count: usize) -> usize {
    (0..GUARD_WORDS)
        .filter(|&i| view[count + i].to_bits() != guard(i))
        .count()
}

fn cpu(m: u64, k: u64, n: u64, a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut builder = TensorProgramBuilder::new();
    let mut tensors = BTreeMap::new();
    let mut values = Vec::new();
    for (name, shape, data) in [("a", [m, k], a), ("b", [k, n], b)] {
        tensors.insert(
            name.to_owned(),
            HostTensor::new(shape.to_vec(), data.to_vec()).expect("the input has its shape"),
        );
        values.push(
            builder
                .input(name, TensorType::f32(&shape))
                .expect("the input is new"),
        );
    }
    let product = builder.matmul(values[0], values[1]).expect("well typed");
    let activation = builder.relu(product).expect("well typed");
    builder.output(product).expect("a value");
    builder.output(activation).expect("a value");
    let outputs = evaluate(&builder.finish(), &tensors, &BTreeMap::new()).expect("it evaluates");
    (outputs[0].data().to_vec(), outputs[1].data().to_vec())
}

fn open() -> (Vulkan, DeviceProfile) {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
    );
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    (vulkan, profile)
}

fn epilogue_name(epilogue: Option<Epilogue>) -> &'static str {
    match epilogue {
        None => "none",
        Some(Epilogue::Relu {
            writes_pre_activation: true,
        }) => "relu-keep",
        Some(Epilogue::Relu {
            writes_pre_activation: false,
        }) => "relu",
    }
}

fn digest(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/adversary_tiling_same_bits.txt")
}

#[test]
#[ignore = "needs a Vulkan device"]
fn edge_matmuls_give_the_cpu_bits() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut wrong = Vec::new();
    let mut elements = 0u64;
    for (m, k, n) in CPU_SHAPES {
        let tile = MatMulTile::of(m, k, n);
        let count = (m * n) as usize;
        for family in [Family::Normal, Family::NegativeZero] {
            let (a, b) = inputs(m, k, n, family);
            let (sum, activation) = cpu(m, k, n, &a, &b);
            if family == Family::NegativeZero {
                let negative_zeros = sum
                    .iter()
                    .filter(|v| v.to_bits() == (-0.0f32).to_bits())
                    .count();
                assert!(
                    negative_zeros >= count / 4,
                    "[{m}, {k}] x [{k}, {n}]: the CPU sums {negative_zeros} columns to -0"
                );
            }
            for epilogue in [None, RELU, KEEP] {
                let case = format!(
                    "[{m}, {k}] x [{k}, {n}] {} {} ({tile})",
                    epilogue_name(epilogue),
                    family.name()
                );
                let kernel = common::matmul_kernel_with(m, k, n, epilogue);
                let module = emit_module(&kernel, &profile).expect("the kernel emits");
                let views = run(&device, &kernel, &module.spirv, &a, &b, count);
                let expected: Vec<&Vec<f32>> = match epilogue {
                    None => vec![&sum],
                    Some(Epilogue::Relu {
                        writes_pre_activation: false,
                    }) => vec![&activation],
                    Some(Epilogue::Relu {
                        writes_pre_activation: true,
                    }) => vec![&sum, &activation],
                };
                assert_eq!(views.len(), expected.len(), "{case}: one view per write");
                for (at, (got, want)) in views.iter().zip(&expected).enumerate() {
                    elements += count as u64;
                    let differ: Vec<usize> = (0..count)
                        .filter(|&i| got[i].to_bits() != want[i].to_bits())
                        .collect();
                    if let Some(&i) = differ.first() {
                        wrong.push(format!(
                            "{case} view {at}: {} elements differ, first [{}, {}] gpu {:?} cpu {:?}",
                            differ.len(),
                            i as u64 / n,
                            i as u64 % n,
                            got[i],
                            want[i]
                        ));
                    }
                    let changed = changed_guards(got, count);
                    if changed > 0 {
                        wrong.push(format!("{case} view {at}: {changed} guard words changed"));
                    }
                }
                for message in vulkan.take_validation_messages() {
                    wrong.push(format!("{case}: {message}"));
                }
            }
        }
    }
    eprintln!(
        "edge matmuls: {} shapes, {elements} elements compared bit for bit",
        CPU_SHAPES.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn edge_matmuls_write_the_bits_of_the_base_commit() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut lines = String::new();
    writeln!(
        lines,
        "device vulkan {:04x}:{:04x} driver_version {}",
        profile.vendor_id, profile.device_id, profile.driver_version
    )
    .expect("a string");
    let mut wrong = Vec::new();
    for (m, k, n) in CPU_SHAPES.iter().chain(&BASE_ONLY_SHAPES).copied() {
        let count = (m * n) as usize;
        for family in [Family::Normal, Family::NegativeZero, Family::Subnormal] {
            let (a, b) = inputs(m, k, n, family);
            for epilogue in [None, KEEP] {
                let case = format!("{m} {k} {n} {} {}", epilogue_name(epilogue), family.name());
                let kernel = common::matmul_kernel_with(m, k, n, epilogue);
                let module = emit_module(&kernel, &profile).expect("the kernel emits");
                let views = run(&device, &kernel, &module.spirv, &a, &b, count);
                for (at, view) in views.iter().enumerate() {
                    let changed = changed_guards(view, count);
                    if changed > 0 {
                        wrong.push(format!("{case} view {at}: {changed} guard words changed"));
                    }
                    writeln!(lines, "{case} view {at} {:016x}", digest(&view[..count]))
                        .expect("a string");
                }
                for message in vulkan.take_validation_messages() {
                    wrong.push(format!("{case}: {message}"));
                }
            }
        }
    }
    if std::env::var(RECORD_ENV).as_deref() == Ok("1") {
        std::fs::write(fixture_path(), &lines).expect("the fixture is written");
        eprintln!("recorded {}", fixture_path().display());
    } else {
        let recorded = std::fs::read_to_string(fixture_path()).expect("the fixture reads");
        let mut before = recorded.lines();
        let now: Vec<&str> = lines.lines().collect();
        assert_eq!(
            before.next(),
            Some(now[0]),
            "the fixture was recorded on another device or driver"
        );
        let before: Vec<&str> = before.collect();
        assert_eq!(now.len() - 1, before.len(), "one digest per written view");
        for (now, before) in now[1..].iter().zip(&before) {
            if now != before {
                wrong.push(format!("now {now}, base {before}"));
            }
        }
        eprintln!("base bits: {} views compared", before.len());
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
