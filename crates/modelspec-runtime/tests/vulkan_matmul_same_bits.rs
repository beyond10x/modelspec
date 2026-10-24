//! `story:matmul-shape-aware-tiling`, acceptance `same-bits` and `register-blocked`, on a Vulkan
//! device.
//!
//! 1. `same-bits`: the matmul module of every shape of [`SAME_BITS`] (the MNIST step, the
//!    `mlp_sweep` points b1000 h1024 and b8192 h4096, and shapes of at least 1024 by 1024 with a
//!    partial block in each of `m`, `k` and `n`) writes the bits it wrote at the plan before the
//!    tile changed, on the same device. `tests/fixtures/matmul_same_bits.txt` holds those bits
//!    as one FNV-1a digest per written view, recorded at the base commit with
//!    `MODELSPEC_SAME_BITS_RECORD=1`, under a header naming the device and driver it ran on; a
//!    run on another device or driver fails rather than comparing.
//! 2. `register-blocked`: on inputs of normal magnitude each written view of a register-blocked
//!    tile is bit for bit the CPU interpreter's, with and without the `relu` epilogue, and the
//!    guard words after each view stay unchanged.
//!
//! Ignored by default: it creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_matmul_same_bits -- --ignored`.
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

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// Set to `1` to write the fixture from this tree instead of comparing against it.
const RECORD_ENV: &str = "MODELSPEC_SAME_BITS_RECORD";

const KEEP: Option<Epilogue> = Some(Epilogue::Relu {
    writes_pre_activation: true,
});
const RELU: Option<Epilogue> = Some(Epilogue::Relu {
    writes_pre_activation: false,
});

/// `([m, k, n], epilogue, subnormal inputs)`: one same-bits case.
type Case = ((u64, u64, u64), Option<Epilogue>, bool);

/// The same-bits cases.
const SAME_BITS: [Case; 22] = [
    // The MNIST step.
    ((100, 784, 64), KEEP, false),
    ((784, 100, 64), None, false),
    ((100, 64, 10), None, false),
    ((64, 100, 10), None, false),
    ((100, 10, 64), None, false),
    ((100, 784, 64), KEEP, true),
    // mlp_sweep b1000 h1024.
    ((1000, 784, 1024), KEEP, false),
    ((784, 1000, 1024), None, false),
    ((1000, 1024, 10), None, false),
    ((1024, 1000, 10), None, false),
    ((1000, 10, 1024), None, false),
    // mlp_sweep b8192 h4096: the forward matmul and the weight gradient of w1.
    ((8192, 784, 4096), KEEP, false),
    ((784, 8192, 4096), None, false),
    // At least 1024 by 1024, partial blocks, short and long `k`.
    ((1025, 1000, 1031), None, false),
    ((1025, 1000, 1031), KEEP, false),
    ((1025, 1000, 1031), None, true),
    ((1024, 784, 1024), None, false),
    ((1100, 33, 1500), RELU, false),
    ((2049, 129, 1025), None, false),
    ((1024, 1, 1024), None, false),
    ((1031, 17, 1029), None, false),
    ((64, 8192, 64), None, false),
];

/// Shapes of a register-blocked tile small enough for the CPU interpreter, each with every
/// epilogue: whole blocks, a partial block in each of `m`, `k` and `n`, `k = 1`, a `k` shorter
/// than a tile and one whose tile loop runs several passes.
const REGISTER_BLOCKED: [(u64, u64, u64); 6] = [
    (1024, 64, 1024),
    (1025, 100, 1031),
    (1024, 1, 1024),
    (1031, 7, 1029),
    (1088, 300, 1040),
    (1027, 129, 1100),
];

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/matmul_same_bits.txt")
}

/// The device and driver line of the fixture.
fn device_line(profile: &DeviceProfile) -> String {
    format!(
        "device vulkan {:04x}:{:04x} driver_version {}",
        profile.vendor_id, profile.device_id, profile.driver_version
    )
}

/// FNV-1a over the little-endian bytes of each value's bits.
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

/// Makes every seventh value of `values` a subnormal of its sign, varied in magnitude.
fn with_subnormals(values: &mut [f32]) {
    for (i, value) in values.iter_mut().enumerate().filter(|(i, _)| i % 7 == 3) {
        let magnitude = f32::from_bits(1 + (i as u32).wrapping_mul(0x9e37) % 0x007f_ffff);
        assert!(magnitude.is_subnormal());
        *value = magnitude.copysign(*value);
    }
}

/// The inputs of `[m, k] x [k, n]`, drawn from a generator seeded by the shape alone.
fn inputs(m: u64, k: u64, n: u64, subnormal: bool) -> (Vec<f32>, Vec<f32>) {
    let mut draw = Draw::new(0x7361_6d65_6269_7473 ^ (m << 40) ^ (k << 20) ^ n);
    let mut a = draw.values(m * k);
    let mut b = draw.values(k * n);
    if subnormal {
        with_subnormals(&mut a);
        with_subnormals(&mut b);
    }
    (a, b)
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// Runs `kernel` once on the device over `a` and `b`, each written view guarded and filled with
/// [`UNWRITTEN`]; returns every written view whole, its guard words included, in binding order.
fn run(device: &Device<'_>, kernel: &Kernel, spirv: &[u32], a: &[f32], b: &[f32]) -> Vec<Vec<f32>> {
    let outputs = kernel.written();
    let n = (kernel.result.shape.iter())
        .map(|d| match d {
            modelspec_core::ir::tensor::Dim::Static(n) => *n as usize,
            modelspec_core::ir::tensor::Dim::Symbol(_) => unreachable!("static"),
        })
        .product::<usize>();
    let uploads: Vec<Vec<f32>> = [guarded(a), guarded(b)]
        .into_iter()
        .chain((0..outputs).map(|_| guarded(&vec![f32::from_bits(UNWRITTEN); n])))
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

/// The guard words after a written view of `count` values that changed.
fn changed_guards(view: &[f32], count: usize) -> usize {
    (0..GUARD_WORDS)
        .filter(|&i| view[count + i].to_bits() != guard(i))
        .count()
}

/// The CPU interpreter's `matmul(a, b)` and `relu(matmul(a, b))`.
fn cpu(a: (&[u64], &[f32]), b: (&[u64], &[f32])) -> (Vec<f32>, Vec<f32>) {
    let mut builder = TensorProgramBuilder::new();
    let mut tensors = BTreeMap::new();
    let mut values = Vec::new();
    for (name, (shape, data)) in [("a", a), ("b", b)] {
        tensors.insert(
            name.to_owned(),
            HostTensor::new(shape.to_vec(), data.to_vec()).expect("the input has its shape"),
        );
        let ty = TensorType::f32(shape);
        values.push(builder.input(name, ty).expect("the input is new"));
    }
    let product = builder.matmul(values[0], values[1]).expect("well typed");
    let activation = builder.relu(product).expect("well typed");
    builder.output(product).expect("a value");
    builder.output(activation).expect("a value");
    let outputs = evaluate(&builder.finish(), &tensors, &BTreeMap::new()).expect("it evaluates");
    (outputs[0].data().to_vec(), outputs[1].data().to_vec())
}

/// Device 0, its profile and the Vulkan instance.
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

fn case_line(m: u64, k: u64, n: u64, epilogue: Option<Epilogue>, subnormal: bool) -> String {
    let epilogue = match epilogue {
        None => "none",
        Some(Epilogue::Relu {
            writes_pre_activation: true,
        }) => "relu-keep",
        Some(Epilogue::Relu {
            writes_pre_activation: false,
        }) => "relu",
    };
    let family = if subnormal { "subnormal" } else { "normal" };
    format!("{m} {k} {n} {epilogue} {family}")
}

#[test]
#[ignore = "needs a Vulkan device"]
fn every_matmul_writes_the_bits_of_the_plan_before_the_tile_changed() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut lines = String::new();
    writeln!(lines, "{}", device_line(&profile)).expect("a string");
    let mut wrong = Vec::new();
    for ((m, k, n), epilogue, subnormal) in SAME_BITS {
        let (a, b) = inputs(m, k, n, subnormal);
        let kernel = common::matmul_kernel_with(m, k, n, epilogue);
        let module = emit_module(&kernel, &profile).expect("the kernel emits");
        let views = run(&device, &kernel, &module.spirv, &a, &b);
        let count = (m * n) as usize;
        let case = case_line(m, k, n, epilogue, subnormal);
        for (at, view) in views.iter().enumerate() {
            let changed = changed_guards(view, count);
            if changed > 0 {
                wrong.push(format!("{case} view {at}: {changed} guard words changed"));
            }
            writeln!(lines, "{case} view {at} {:016x}", digest(&view[..count])).expect("a string");
        }
        for message in vulkan.take_validation_messages() {
            wrong.push(format!("{case}: {message}"));
        }
    }
    if std::env::var(RECORD_ENV).as_deref() == Ok("1") {
        std::fs::write(fixture_path(), &lines).expect("the fixture is written");
        eprintln!("recorded {}", fixture_path().display());
    } else {
        let recorded = std::fs::read_to_string(fixture_path()).expect("the fixture reads");
        let mut recorded_lines = recorded.lines();
        let header = recorded_lines.next().unwrap_or_default();
        assert_eq!(
            header,
            device_line(&profile),
            "the fixture was recorded on another device or driver; record it again at the \
             base commit with {RECORD_ENV}=1"
        );
        let now: Vec<&str> = lines.lines().skip(1).collect();
        let before: Vec<&str> = recorded_lines.collect();
        assert_eq!(now.len(), before.len(), "one digest per written view");
        for (now, before) in now.iter().zip(&before) {
            if now != before {
                wrong.push(format!("now {now}, before {before}"));
            }
        }
        eprintln!(
            "same bits: {} views compared against {}",
            now.len(),
            fixture_path().display()
        );
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn a_register_blocked_matmul_gives_the_cpu_bits_on_normal_inputs() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut wrong = Vec::new();
    let mut elements = 0u64;
    for (m, k, n) in REGISTER_BLOCKED {
        let tile = MatMulTile::of(m, k, n);
        assert!(
            tile.rows >= 4 && tile.cols_per_invocation >= 4,
            "[{m}, {k}] x [{k}, {n}]: {tile} is register-blocked"
        );
        let (a, b) = inputs(m, k, n, false);
        let (sum, activation) = cpu((&[m, k], &a), (&[k, n], &b));
        for epilogue in [None, RELU, KEEP] {
            let kernel = common::matmul_kernel_with(m, k, n, epilogue);
            let case = case_line(m, k, n, epilogue, false);
            let module = emit_module(&kernel, &profile).expect("the kernel emits");
            let views = run(&device, &kernel, &module.spirv, &a, &b);
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
                let count = want.len();
                elements += count as u64;
                let differ: Vec<usize> = (0..count)
                    .filter(|&i| got[i].to_bits() != want[i].to_bits())
                    .collect();
                if let Some(&i) = differ.first() {
                    wrong.push(format!(
                        "{case} view {at}: {} elements differ, first [{i}] gpu {:?} cpu {:?}",
                        differ.len(),
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
    eprintln!(
        "register blocked: {} shapes, {elements} elements compared bit for bit",
        REGISTER_BLOCKED.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}
