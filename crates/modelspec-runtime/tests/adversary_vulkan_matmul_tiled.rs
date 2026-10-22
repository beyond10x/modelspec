//! Adversary pass 1 on `story:m1-f-forward-matmul-tiled`, on a Vulkan device.
//!
//! 1. Bit identity: inputs of normal magnitude (no subnormal operand or product, so a device
//!    that flushes subnormals cannot differ for that reason) must give the CPU's bits exactly at
//!    every shape of `common::MATMUL_SHAPES`, with every epilogue, whatever class `negotiate`
//!    reports: the unit claims the same `OpFMul`/`OpFAdd` with `NoContraction` in the same
//!    order, and that claim does not depend on the float-control modes for normal values.
//! 2. More than 65,535 workgroups: the schedule spreads them over a 2-D grid (`spread`); the
//!    tiled kernel derives its block from the flat invocation index, and the unit's device test
//!    only ever dispatches a 1-D grid.
//! 3. Large `k`: the module is unrolled, one block of instructions per tile. Pipeline creation
//!    of one matmul module at a `k` a full-batch weight gradient reaches must stay bounded.
//!
//! Ignored by default: it creates a Vulkan instance. Run it with
//! `cargo test -p modelspec-runtime --features vulkan --test adversary_vulkan_matmul_tiled -- --ignored --test-threads 1`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;

use modelspec_core::ir::kernel::{Epilogue, Kernel, MatMulTile};
use modelspec_core::ir::tensor::{Dim, TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::interpreter::evaluate;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, Vulkan};

const UNWRITTEN: u32 = 0x7fc0_dead;

/// Values of magnitude in [0.5, 2), sign random when `signed`.
struct Normal(u64);

impl Normal {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn values(&mut self, count: u64, signed: bool) -> Vec<f32> {
        (0..count)
            .map(|_| {
                let bits = self.next();
                let mantissa = (bits & 0x7f_ffff) as u32;
                let exponent = if bits & (1 << 40) == 0 { 126 } else { 127 };
                let sign = if signed && bits & (1 << 50) != 0 {
                    1u32 << 31
                } else {
                    0
                };
                f32::from_bits(sign | (exponent << 23) | mantissa)
            })
            .collect()
    }
}

fn cpu(m: u64, k: u64, n: u64, a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut builder = TensorProgramBuilder::new();
    let mut tensors = BTreeMap::new();
    let mut values = Vec::new();
    for (name, shape, data) in [("a", [m, k], a), ("b", [k, n], b)] {
        tensors.insert(
            name.to_owned(),
            HostTensor::new(shape.to_vec(), data.to_vec()).expect("shape"),
        );
        values.push(
            builder
                .input(name, TensorType::f32(&shape))
                .expect("new input"),
        );
    }
    let product = builder.matmul(values[0], values[1]).expect("typed");
    let activation = builder.relu(product).expect("typed");
    builder.output(product).expect("a value");
    builder.output(activation).expect("a value");
    let outputs = evaluate(&builder.finish(), &tensors, &BTreeMap::new()).expect("evaluates");
    (outputs[0].data().to_vec(), outputs[1].data().to_vec())
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

fn result_elements(kernel: &Kernel) -> usize {
    kernel
        .result
        .shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n as usize,
            Dim::Symbol(_) => unreachable!("static"),
        })
        .product()
}

fn run(device: &Device<'_>, kernel: &Kernel, spirv: &[u32], a: &[f32], b: &[f32]) -> Vec<Vec<f32>> {
    let count = result_elements(kernel);
    let uploads: Vec<Vec<f32>> = [a.to_vec(), b.to_vec()]
        .into_iter()
        .chain((0..kernel.written()).map(|_| vec![f32::from_bits(UNWRITTEN); count]))
        .collect();
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            device
                .device_local_buffer(bytes_of(v.len()))
                .expect("buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            let mut s = device.staging_buffer(bytes_of(v.len())).expect("staging");
            s.write_f32(v).expect("written");
            s
        })
        .collect();
    let pipeline = device.kernel_pipeline(kernel, spirv).expect("pipeline");
    let mut recording = device.record().expect("recording");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("upload");
    }
    recording.barrier(Barrier::TransferToCompute);
    let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
    recording
        .dispatch_kernel(&pipeline, kernel, &bound)
        .expect("dispatch");
    recording.barrier(Barrier::ComputeToTransfer);
    for at in 2..buffers.len() {
        recording
            .copy(&buffers[at], &staging[at])
            .expect("download");
    }
    recording.barrier(Barrier::TransferToHost);
    device
        .submit(recording.finish().expect("ends"))
        .expect("submitted")
        .wait()
        .expect("finishes");
    staging[2..]
        .iter_mut()
        .map(|s| s.read_f32().expect("reads"))
        .collect()
}

const EPILOGUES: [Option<Epilogue>; 3] = [
    None,
    Some(Epilogue::Relu {
        writes_pre_activation: false,
    }),
    Some(Epilogue::Relu {
        writes_pre_activation: true,
    }),
];

/// Every view of `got` against the CPU's, bit for bit; the first few mismatches.
fn compare(
    name: &str,
    epilogue: Option<Epilogue>,
    got: &[Vec<f32>],
    sum: &[f32],
    activation: &[f32],
    wrong: &mut Vec<String>,
) {
    let expected: Vec<&[f32]> = match epilogue {
        None => vec![sum],
        Some(Epilogue::Relu {
            writes_pre_activation: false,
        }) => vec![activation],
        Some(Epilogue::Relu {
            writes_pre_activation: true,
        }) => vec![sum, activation],
    };
    for (view, (got, want)) in got.iter().zip(&expected).enumerate() {
        let bad: Vec<usize> = (0..want.len())
            .filter(|&i| got[i].to_bits() != want[i].to_bits())
            .collect();
        if !bad.is_empty() {
            let i = bad[0];
            wrong.push(format!(
                "{name} {epilogue:?} view {view}: {} of {} differ, first [{i}] gpu {:?} cpu {:?}",
                bad.len(),
                want.len(),
                got[i],
                want[i]
            ));
        }
    }
}

fn open() -> (Vulkan, modelspec_core::device::DeviceProfile) {
    let vulkan = Vulkan::load().expect("loader");
    let profile = vulkan.candidates().expect("probed")[0].profile.clone();
    (vulkan, profile)
}

#[test]
#[ignore = "needs a Vulkan device"]
fn normal_inputs_give_the_cpu_bits_at_every_tile_shape_and_epilogue() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("negotiates");
    let device = vulkan.create_device(0, &negotiated).expect("opens");
    let mut draw = Normal(0x6164_7665_7273_6172);
    let mut wrong = Vec::new();
    let mut elements = 0;
    for (m, k, n) in common::MATMUL_SHAPES {
        for signed in [false, true] {
            let a = draw.values(m * k, signed);
            let b = draw.values(k * n, signed);
            let (sum, activation) = cpu(m, k, n, &a, &b);
            for epilogue in EPILOGUES {
                let kernel = common::matmul_kernel_with(m, k, n, epilogue);
                let module = emit_module(&kernel, &profile).expect("emits");
                let got = run(&device, &kernel, &module.spirv, &a, &b);
                elements += got.iter().map(Vec::len).sum::<usize>();
                let name = format!("[{m}, {k}] x [{k}, {n}] signed {signed}");
                compare(&name, epilogue, &got, &sum, &activation, &mut wrong);
            }
        }
    }
    eprintln!(
        "adversary bit identity: {elements} elements, {} failing views",
        wrong.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn a_matmul_over_more_than_65535_workgroups_on_a_two_dimensional_grid_gives_the_cpu_bits() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("negotiates");
    let device = vulkan.create_device(0, &negotiated).expect("opens");
    // One row block past 65,536 of the tile this shape gets, whatever its rows per workgroup.
    let block = u64::from(MatMulTile::of(16 * 65_536, 3, 16).rows_per_workgroup());
    let (m, k, n) = (block * 65_536 + 9, 3, 16);
    let mut draw = Normal(0x7370_7265_6164_3264);
    let a = draw.values(m * k, true);
    let b = draw.values(k * n, true);
    let (sum, activation) = cpu(m, k, n, &a, &b);
    let mut wrong = Vec::new();
    for epilogue in EPILOGUES {
        let mut kernel = common::matmul_kernel_with(m, k, n, epilogue);
        let groups = kernel.workgroups[0];
        assert!(
            groups > 65_535,
            "{groups} workgroups need a second dimension"
        );
        kernel.workgroups = [65_535, groups.div_ceil(65_535), 1];
        kernel.id = Kernel::module_id(&kernel.signature());
        let module = emit_module(&kernel, &profile).expect("emits");
        let got = run(&device, &kernel, &module.spirv, &a, &b);
        let name = format!("[{m}, {k}] x [{k}, {n}] on {:?}", kernel.workgroups);
        compare(&name, epilogue, &got, &sum, &activation, &mut wrong);
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn a_large_k_matmul_module_is_emitted_and_its_pipeline_created_within_a_second() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("negotiates");
    let device = vulkan.create_device(0, &negotiated).expect("opens");
    // The forward matmul of MNIST at k = 784 for scale, then k = 4,096, the full-batch MNIST
    // weight gradient `[784, 60000] x [60000, 64]` and the full-batch linear regression weight
    // gradient `[4, 600000] x [600000, 1]` of `adversary_matmul_tiled_spirv.rs`.
    let shapes = [
        (100, 784, 64),
        (64, 4_096, 64),
        (784, 60_000, 64),
        (4, 600_000, 1),
    ];
    let mut slow = Vec::new();
    for (m, k, n) in shapes {
        let kernel = common::matmul_kernel_with(m, k, n, None);
        let started = Instant::now();
        let module = emit_module(&kernel, &profile).expect("emits");
        let emitted = started.elapsed();
        let started = Instant::now();
        let pipeline = device.kernel_pipeline(&kernel, &module.spirv);
        let created = started.elapsed();
        let line = format!(
            "[{m}, {k}] x [{k}, {n}]: {} words, id bound {}, emit {:.3} s, pipeline {:.3} s, {}",
            module.spirv.len(),
            module.spirv[3],
            emitted.as_secs_f64(),
            created.as_secs_f64(),
            if pipeline.is_ok() {
                "created"
            } else {
                "refused"
            },
        );
        eprintln!("{line}");
        std::io::stderr().flush().ok();
        if pipeline.is_err() || (emitted + created).as_secs_f64() > 1.0 {
            slow.push(line);
        }
    }
    assert!(slow.is_empty(), "{slow:#?}");
}
