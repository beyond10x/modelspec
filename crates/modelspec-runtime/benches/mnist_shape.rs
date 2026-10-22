//! Training throughput of the CPU reference on the shape of the MNIST benchmark of release 0.1.1:
//! 784 -> 64 ReLU -> 10, no bias, `f32`, MSE over the one-hot targets, SGD lr 0.1, batch 100.
//!
//! The data is synthetic and seeded, so the run needs no files. The timed loop includes minibatch
//! assembly and excludes setup, a 20-step warmup and the data generation, as the 0.1.1 benchmark
//! did. It prints images per second and a digest of the final weights' bits, so two builds that
//! must compute the same bits can be compared from their output.
//!
//! Run it with `cargo bench -p modelspec-runtime --bench mnist_shape`; it is not part of
//! `task ci`.

use std::collections::BTreeMap;
use std::time::Instant;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{Initializer, TensorProgramBuilder, TensorType};
use modelspec_core::ir::training::attach_sgd;
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::init::{Stream, initialize};
use modelspec_runtime::train::optimizer::{ParamState, sgd_step};

const INPUTS: usize = 784;
const HIDDEN: usize = 64;
const CLASSES: usize = 10;
const BATCH: usize = 100;
const BATCHES: usize = 60;
const LR: f32 = 0.1;
const SEED: u64 = 42;
const WARMUP: usize = 20;
const STEPS: usize = 200;

/// SplitMix64: the seeded source of the synthetic images and labels.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 24 bits, like a pixel scaled to the unit interval.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn tensor(shape: &[usize], data: &[f32]) -> HostTensor<f32> {
    let shape = shape.iter().map(|&d| d as u64).collect();
    HostTensor::new(shape, data.to_vec()).expect("the shape matches the data")
}

fn main() {
    let mut b = TensorProgramBuilder::new();
    let w1 = b
        .declare_param(
            "Mlp.w1",
            TensorType::f32(&[INPUTS as u64, HIDDEN as u64]),
            Initializer::XavierUniform,
        )
        .unwrap();
    let w2 = b
        .declare_param(
            "Mlp.w2",
            TensorType::f32(&[HIDDEN as u64, CLASSES as u64]),
            Initializer::XavierUniform,
        )
        .unwrap();
    let x = b
        .input("x", TensorType::f32(&[BATCH as u64, INPUTS as u64]))
        .unwrap();
    let y = b
        .input("y", TensorType::f32(&[BATCH as u64, CLASSES as u64]))
        .unwrap();
    let w1 = b.param_read(w1).unwrap();
    let w2 = b.param_read(w2).unwrap();
    let h = b.matmul(x, w1).unwrap();
    let h = b.relu(h).unwrap();
    let o = b.matmul(h, w2).unwrap();
    let loss = b.mse(o, y).unwrap();
    let forward = b.finish();
    let step = attach_sgd(differentiate(&forward, loss).unwrap(), LR).unwrap();
    let mut state = ParamState::new(&step, initialize(&forward, SEED, Stream::Init).unwrap());

    let mut rng = SplitMix(SEED);
    let images: Vec<f32> = (0..BATCHES * BATCH * INPUTS).map(|_| rng.unit()).collect();
    let mut targets = vec![0f32; BATCHES * BATCH * CLASSES];
    for row in targets.chunks_mut(CLASSES) {
        row[(rng.next() % CLASSES as u64) as usize] = 1.0;
    }

    let mut train = |count: usize, start: usize| -> f32 {
        let mut loss = 0.0;
        for s in start..start + count {
            let batch = s % BATCHES;
            let inputs = BTreeMap::from([
                (
                    "x".to_string(),
                    tensor(
                        &[BATCH, INPUTS],
                        &images[batch * BATCH * INPUTS..(batch + 1) * BATCH * INPUTS],
                    ),
                ),
                (
                    "y".to_string(),
                    tensor(
                        &[BATCH, CLASSES],
                        &targets[batch * BATCH * CLASSES..(batch + 1) * BATCH * CLASSES],
                    ),
                ),
            ]);
            loss = sgd_step(&step, &inputs, &mut state).unwrap();
        }
        loss
    };

    train(WARMUP, 0);
    let started = Instant::now();
    let loss = train(STEPS, WARMUP);
    let seconds = started.elapsed().as_secs_f64();

    let mut digest = blake3::Hasher::new();
    for value in state.values.values() {
        for x in value.data() {
            digest.update(&x.to_bits().to_le_bytes());
        }
    }
    println!(
        "mnist_shape: {STEPS} steps of batch {BATCH} after {WARMUP} warmup, {seconds:.3} s, \
         {:.3} ms/step, {:.0} images/s",
        seconds * 1e3 / STEPS as f64,
        (STEPS * BATCH) as f64 / seconds,
    );
    println!(
        "mnist_shape: last loss {loss:e} (bits {:#010x}), weights blake3 {}",
        loss.to_bits(),
        digest.finalize().to_hex()
    );
}
