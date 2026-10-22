//! Adversary pass 2 on `story:m1-f-forward-matmul-tiled`, on a Vulkan device.
//!
//! On inputs of normal magnitude every written view must be the CPU interpreter's bits in any
//! numerics class (the unit's claim, DESIGN.md § 2.2), at shapes the unit's suite does not reach:
//!
//! 1. The two-row threshold (`MATMUL_TWO_ROWS_FROM_WORKGROUPS`): 255, 256 and 257 workgroups of
//!    the one-row tile at 16, 4 and 1 columns per workgroup.
//! 2. A last row block of a two-row tile in which some invocations have a valid second row and
//!    others do not (`m % rows_per_workgroup` above half the block).
//! 3. The tile loop at every remainder `(full - 7) % 4` of its passes, one and two passes, each
//!    with no partial last tile, one of 1 and one of `depth - 1`, for a one-row and a two-row
//!    tile.
//!
//! Ignored by default. Run it with
//! `cargo test -p modelspec-runtime --features vulkan --test adversary2_vulkan_matmul_tiled -- --ignored --test-threads 1`.
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
fn the_two_row_threshold_gives_the_cpu_bits_on_both_sides() {
    // (m, k, n, rows): 255, 256 and 256+ one-row workgroups at 16, 4 and 1 columns.
    let cases: [(u64, u64, u64, u32); 12] = [
        (2040, 773, 16, 1),
        (2041, 773, 16, 2),
        (2048, 773, 16, 2),
        (2049, 773, 16, 2),
        (8160, 300, 3, 1),
        (8161, 300, 3, 2),
        (8193, 300, 3, 2),
        (32640, 100, 1, 1),
        (32641, 100, 1, 2),
        (32769, 100, 1, 2),
        (1016, 500, 17, 1),
        (1017, 500, 17, 2),
    ];
    for (m, k, n, rows) in cases {
        assert_eq!(
            MatMulTile::of(m, k, n).rows,
            rows,
            "[{m}, {k}] x [{k}, {n}]"
        );
    }
    let shapes: Vec<_> = cases.iter().map(|&(m, k, n, _)| (m, k, n)).collect();
    let (elements, wrong) = bits(&shapes, 0x7468_7265_7368_6f6c);
    eprintln!(
        "adversary2 threshold: {elements} elements, {} failing views",
        wrong.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn a_two_row_last_block_with_some_second_rows_inside_gives_the_cpu_bits() {
    let shapes = [
        (2045, 773, 16),
        (2057, 300, 16),
        (1021, 500, 17),
        (8185, 300, 3),
        (33000, 100, 1),
    ];
    for (m, k, n) in shapes {
        let tile = MatMulTile::of(m, k, n);
        let block = u64::from(tile.rows_per_workgroup());
        assert_eq!(tile.rows, 2, "[{m}, {k}] x [{k}, {n}]");
        assert!(m % block > block / 2, "[{m}, {k}] x [{k}, {n}]: {tile:?}");
    }
    let (elements, wrong) = bits(&shapes, 0x6c61_7374_626c_6f63);
    eprintln!(
        "adversary2 last block: {elements} elements, {} failing views",
        wrong.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn the_tile_loop_gives_the_cpu_bits_at_every_remainder_of_its_passes() {
    let mut shapes = Vec::new();
    for (m, n) in [(5, 16), (2045, 16)] {
        let depth = u64::from(MatMulTile::of(m, 1 << 16, n).depth);
        for full in 11..=18 {
            for partial in [0, 1, depth - 1] {
                shapes.push((m, full * depth + partial, n));
            }
        }
    }
    let (elements, wrong) = bits(&shapes, 0x6c6f_6f70_7061_7373);
    eprintln!(
        "adversary2 tile loop: {} shapes, {elements} elements, {} failing views",
        shapes.len(),
        wrong.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}
