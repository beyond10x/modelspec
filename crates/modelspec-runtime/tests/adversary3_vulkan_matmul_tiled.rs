//! Adversary pass on `story:m1-f-weight-grad-matmul-tiled`, on a Vulkan device: two-row tiles
//! of up to 16 values of `k` read 2 ahead give the CPU interpreter's bits on signed normal
//! inputs at shapes the unit's suite does not reach:
//!
//! 1. Depth 16, 8 (two columns, memory-limited) and 4 (one column, memory-limited) at every
//!    count of full tiles from 0 to 10, each with no partial last tile, one of 1 and one of
//!    `depth - 1`: the straight tiles before the loop starts (`full < 7`), where the 2 tiles
//!    read ahead exceed the tiles left, and the first passes of the loop.
//! 2. The MNIST weight-gradient rows and columns at `k` around 100.
//!
//! Ignored by default. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test adversary3_vulkan_matmul_tiled -- --ignored --test-threads 1`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

use modelspec_core::ir::kernel::{Epilogue, Kernel, MatMulTile};
use modelspec_core::ir::tensor::{Dim, TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::interpreter::evaluate;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

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

/// Runs each shape with every epilogue on signed normal inputs and returns the failing views.
fn bits(shapes: &[(u64, u64, u64)], seed: u64) -> (usize, Vec<String>) {
    let vulkan = Vulkan::load().expect("loader");
    let profile = vulkan.candidates().expect("probed")[0].profile.clone();
    let negotiated = negotiate(&profile).expect("negotiates");
    let device = vulkan.create_device(0, &negotiated).expect("opens");
    let mut draw = Normal(seed);
    let mut wrong = Vec::new();
    let mut elements = 0;
    for &(m, k, n) in shapes {
        let a = draw.values(m * k, true);
        let b = draw.values(k * n, true);
        let (sum, activation) = cpu(m, k, n, &a, &b);
        for epilogue in EPILOGUES {
            let kernel = common::matmul_kernel_with(m, k, n, epilogue);
            let module = emit_module(&kernel, &profile).expect("emits");
            let got = run(&device, &kernel, &module.spirv, &a, &b);
            let expected: Vec<&[f32]> = match epilogue {
                None => vec![&sum],
                Some(Epilogue::Relu {
                    writes_pre_activation: false,
                }) => vec![&activation],
                Some(Epilogue::Relu {
                    writes_pre_activation: true,
                }) => vec![&sum, &activation],
            };
            assert_eq!(got.len(), expected.len(), "[{m}, {k}] x [{k}, {n}] views");
            for (view, (got, want)) in got.iter().zip(&expected).enumerate() {
                elements += want.len();
                let bad: Vec<usize> = (0..want.len())
                    .filter(|&i| got[i].to_bits() != want[i].to_bits())
                    .collect();
                if let Some(&i) = bad.first() {
                    wrong.push(format!(
                        "[{m}, {k}] x [{k}, {n}] {:?} {epilogue:?} view {view}: {} of {} differ, first [{i}] gpu {:?} cpu {:?}",
                        MatMulTile::of(m, k, n),
                        bad.len(),
                        want.len(),
                        got[i],
                        want[i]
                    ));
                }
            }
        }
    }
    (elements, wrong)
}


#[test]
#[ignore = "needs a Vulkan device"]
fn two_row_tiles_of_16_8_and_4_give_the_cpu_bits_at_0_to_10_full_tiles() {
    assert_eq!(
        Vulkan::load().expect("loader").validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
    );
    let mut shapes = Vec::new();
    // (m, n, depth)
    for (m, n, depth) in [(2045, 16, 16), (16_321, 2, 8), (32_641, 1, 4)] {
        for full in 0..=10u64 {
            for partial in [0, 1, depth - 1] {
                let k = full * depth + partial;
                if k == 0 {
                    continue;
                }
                let tile = MatMulTile::of(m, k, n);
                assert_eq!(tile.rows, 2, "[{m}, {k}] x [{k}, {n}]: {tile:?}");
                assert_eq!(tile.ahead, 2, "[{m}, {k}] x [{k}, {n}]: {tile:?}");
                if k > 16 {
                    assert_eq!(
                        u64::from(tile.depth),
                        depth,
                        "[{m}, {k}] x [{k}, {n}]: {tile:?}"
                    );
                }
                shapes.push((m, k, n));
            }
        }
    }
    let (elements, wrong) = bits(&shapes, 0x7477_6f72_6f77_3136);
    eprintln!(
        "adversary3 two-row depths: {} shapes, {elements} elements, {} failing views",
        shapes.len(),
        wrong.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn the_weight_gradient_rows_give_the_cpu_bits_at_k_around_100() {
    let shapes: Vec<_> = [
        2, 15, 16, 31, 32, 33, 48, 64, 95, 96, 97, 99, 100, 101, 111, 112, 113, 128,
    ]
    .into_iter()
    .map(|k| (784, k, 64))
    .collect();
    for &(m, k, n) in &shapes {
        let tile = MatMulTile::of(m, k, n);
        assert_eq!((tile.rows, tile.ahead), (2, 2), "[{m}, {k}] x [{k}, {n}]");
    }
    let (elements, wrong) = bits(&shapes, 0x6b31_315f_6b31_3030);
    eprintln!(
        "adversary3 weight gradient: {} shapes, {elements} elements, {} failing views",
        shapes.len(),
        wrong.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}
