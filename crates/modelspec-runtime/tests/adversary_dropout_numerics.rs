//! Adversary cases for story:bert-dropout on the CPU interpreter and the runner (DESIGN.md § 2.3,
//! § 4, coordinator decision D47).
//!
//! The draws are derived here from `blake3` and `philox4x32_10` directly. A case named
//! `adv_drop_<n>_...` failed on the unit's commit for its finding and passes after the fix;
//! the others pass and each kills a mutant the unit's own suite lets through.

use std::collections::BTreeMap;

use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact, compile_tensor, compile_training};
use modelspec_core::ir::tensor::{TensorOp, TensorProgram, TensorProgramBuilder, TensorType};
use modelspec_core::ir::training::dropout_sites;
use modelspec_runtime::cpu::{HostTensor, evaluate_seeded};
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::train::init::{
    Stream, dropout_id, dropout_token, initialize, philox4x32_10,
};
use modelspec_runtime::train::runner::run_from;

const SEED: u64 = 0x0123_4567_89ab_cdef;
const SITE: &str = "Reg.forward.dropout#0";

fn id_of(site: &str, step: u64) -> u64 {
    let mut bytes = site.as_bytes().to_vec();
    bytes.extend_from_slice(&step.to_le_bytes());
    let hash = blake3::hash(&bytes);
    let mut first = [0u8; 8];
    first.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(first)
}

fn token_of(site: &str, step: u64) -> HostTensor<i32> {
    let id = id_of(site, step);
    HostTensor::new(vec![2], vec![id as u32 as i32, (id >> 32) as u32 as i32]).unwrap()
}

fn word0(seed: u64, id: u64, index: u32) -> u32 {
    philox4x32_10(
        [index, 4, id as u32, (id >> 32) as u32],
        [seed as u32, (seed >> 32) as u32],
    )[0]
}

fn uniform(w: u32) -> f64 {
    (f64::from(w) + 0.5) / 4_294_967_296.0
}

/// `dropout(x, token, p)` over `x: f32[n]` as the one output.
fn dropout_program(n: u64, p: f32) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[n])).unwrap();
    let token = b.input(SITE, TensorType::i32(&[2])).unwrap();
    let y = b.dropout(x, token, p).unwrap();
    b.output(y).unwrap();
    b.finish()
}

/// `dropout_grad(token, g, p)` over `g: f32[n]` as the one output.
fn dropout_grad_program(n: u64, p: f32) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let g = b.input("g", TensorType::f32(&[n])).unwrap();
    let token = b.input(SITE, TensorType::i32(&[2])).unwrap();
    let y = b.dropout_grad(token, g, p).unwrap();
    b.output(y).unwrap();
    b.finish()
}

fn run_f32(program: &TensorProgram, name: &str, values: Vec<f32>, step: u64) -> Vec<f32> {
    let n = values.len() as u64;
    let inputs = BTreeMap::from([(name.to_owned(), HostTensor::new(vec![n], values).unwrap())]);
    let ids = BTreeMap::from([(SITE.to_owned(), token_of(SITE, step))]);
    evaluate_seeded(program, &inputs, &ids, &BTreeMap::new(), SEED).unwrap()[0]
        .data()
        .to_vec()
}

/// The boundary `u == p` keeps the element (`>=`, D47). `p` is chosen as the exact uniform of an
/// element whose word 0 is below `2^23`, which an `f32` holds exactly. Kills the mutant `u > p`.
#[test]
fn adv_drop_boundary_u_equal_p_keeps_the_element() {
    let step = 3;
    let id = id_of(SITE, step);
    let n = 1u64 << 14;
    let (index, w) = (0..n as u32)
        .map(|i| (i, word0(SEED, id, i)))
        .find(|&(_, w)| w < (1 << 23))
        .expect("an element with word 0 below 2^23 among 2^14");
    let p = uniform(w) as f32;
    assert_eq!(
        f64::from(p),
        uniform(w),
        "p is the element's uniform exactly"
    );
    let out = run_f32(&dropout_program(n, p), "x", vec![1.0; n as usize], step);
    let scale = 1.0f32 / (1.0f32 - p);
    assert_eq!(
        out[index as usize].to_bits(),
        scale.to_bits(),
        "element {index} with u == p = {p:e} must be kept"
    );
    let grad = run_f32(
        &dropout_grad_program(n, p),
        "g",
        vec![1.0; n as usize],
        step,
    );
    assert_eq!(
        grad[index as usize].to_bits(),
        scale.to_bits(),
        "DropoutGrad keeps element {index} at u == p too"
    );
}

/// The step enters the draw id as 8 little-endian bytes: step `2^32 + 1` is not step 1.
#[test]
fn adv_drop_id_takes_the_step_as_8_bytes() {
    let big = (1u64 << 32) + 1;
    assert_eq!(dropout_id(SITE, big), id_of(SITE, big));
    assert_ne!(dropout_id(SITE, big), dropout_id(SITE, 1));
    assert_eq!(dropout_token(SITE, big), token_of(SITE, big));
}

/// A dropped element is `+0.0` in the forward and in the gradient, also for a negative value
/// (DESIGN.md § 2.3, § 4). Kills the mutant `v * 0` for a dropped element, which gives `-0.0`.
#[test]
fn adv_drop_dropped_element_is_positive_zero_for_negative_values() {
    let p = 0.5f32;
    let n = 256u64;
    let id = id_of(SITE, 9);
    let values: Vec<f32> = (0..n).map(|i| -1.0 - i as f32).collect();
    let forward = run_f32(&dropout_program(n, p), "x", values.clone(), 9);
    let grad = run_f32(&dropout_grad_program(n, p), "g", values, 9);
    let mut dropped = 0;
    for i in 0..n as usize {
        if uniform(word0(SEED, id, i as u32)) < f64::from(p) {
            dropped += 1;
            assert_eq!(
                forward[i].to_bits(),
                0.0f32.to_bits(),
                "forward element {i}"
            );
            assert_eq!(grad[i].to_bits(), 0.0f32.to_bits(), "gradient element {i}");
        }
    }
    assert!(dropped > 0);
}

/// The `f64` reference multiplies a kept element by the `f32` scale `1.0f32 / (1.0f32 - p)`
/// widened to `f64` (DESIGN.md § 2.3), not by `1 / (1 - p)` computed in `f64`; at `p = 0.1` the
/// two differ.
#[test]
fn adv_drop_f64_reference_multiplies_by_the_widened_f32_scale() {
    let p = 0.1f32;
    let n = 512u64;
    let id = id_of(SITE, 2);
    let scale32 = f64::from(1.0f32 / (1.0f32 - p));
    assert_ne!(scale32, 1.0 / (1.0 - f64::from(p)));
    let xs: Vec<f64> = (0..n).map(|i| 0.3 + i as f64 * 0.01).collect();
    let program = dropout_program(n, p);
    let inputs = BTreeMap::from([(
        "x".to_owned(),
        HostTensor::new(vec![n], xs.clone()).unwrap(),
    )]);
    let ids = BTreeMap::from([(SITE.to_owned(), token_of(SITE, 2))]);
    let out = evaluate_seeded(&program, &inputs, &ids, &BTreeMap::new(), SEED).unwrap();
    let mut kept = 0;
    for (i, (&y, &x)) in out[0].data().iter().zip(&xs).enumerate() {
        if uniform(word0(SEED, id, i as u32)) >= f64::from(p) {
            kept += 1;
            assert_eq!(y.to_bits(), (x * scale32).to_bits(), "kept element {i}");
        }
    }
    assert!(kept > 0);
}

/// A model function with one dropout, called twice from the loss: `block(block(forward(x)))`.
/// A model function cannot call another (E-NAME-007), so the train block's loss is the one
/// place a function is inlined more than once.
const INLINED_TWICE: &str = "model Reg {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();

    fn block(h: Tensor<f32, [8, 3]>) -> Tensor<f32, [8, 3]> {
        return dropout(h, p = 0.5);
    }

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        return matmul(x, w);
    }
}

train Fit {
    model = Reg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(model.block(model.block(model.forward(x))), y);
    optimizer = sgd(lr = 0.05);
    steps = 2;
    seed = 1;
    device = cpu;
}
";

/// ADV-DROP-1. Two calls of a model function are two dropout draws: the two `Dropout` nodes of
/// `block(block(forward(x)))` read different tokens, so the two applications draw independent
/// masks. The first inlining keeps the name `Reg.block.dropout#0`, the second is
/// `Reg.block@1.dropout#0` (coordinator decision D54, 2026-10-25).
#[test]
fn adv_drop_1_function_inlined_twice_draws_two_masks() {
    let lowered = compile_tensor("inlined_twice.mspec", INLINED_TWICE).expect("it compiles");
    let program = &lowered.program;
    let tokens: Vec<_> = program
        .nodes
        .iter()
        .filter(|n| matches!(n.op, TensorOp::Dropout { .. }))
        .map(|n| n.operands[1])
        .collect();
    assert_eq!(tokens.len(), 2, "two Dropout nodes");
    assert_ne!(
        tokens[0],
        tokens[1],
        "both calls of `block` read one token, sites {:?}",
        dropout_sites(program)
    );
    assert_eq!(
        dropout_sites(program),
        ["Reg.block.dropout#0", "Reg.block@1.dropout#0"]
    );
}

const ONE_STEP: &str = "model Reg {
    param w1: Tensor<f32, [4, 6]> = xavier_uniform();
    param w2: Tensor<f32, [6, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        let h = relu(matmul(x, w1));
        let d = dropout(h, p = 0.3);
        return matmul(d, w2);
    }
}

train Fit {
    model = Reg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 1;
    seed = 11;
    device = cpu;
}
";

/// The runner keys step `k` with the token of the global step `k`, also in a run resumed at
/// `start = k`: the loss it records for step `k` is the training program evaluated on the same
/// parameters with `token(site, k)`. Kills the mutants `step + 1`, `step - 1` and `step - start`.
#[test]
fn adv_drop_runner_keys_each_step_with_the_global_step_token() {
    let compiled = compile_training("one_step.mspec", ONE_STEP).expect("it compiles");
    let forward = &compiled.forward;
    let initial = initialize(&forward.program, compiled.config.seed, Stream::Init).unwrap();
    let data = synthetic(&compiled.config, forward).unwrap();
    let program = &compiled.training.training.program;
    let sites = dropout_sites(program);
    assert_eq!(sites, ["Reg.forward.dropout#0"]);
    for start in [0u64, 1, 7] {
        let run = run_from(
            &compiled,
            None,
            &BTreeMap::new(),
            start,
            |_| {},
            |_, _| Ok(()),
        )
        .unwrap();
        let ids: BTreeMap<String, HostTensor<i32>> = sites
            .iter()
            .map(|s| (s.clone(), token_of(s, start)))
            .collect();
        let want = evaluate_seeded(program, &data.inputs, &ids, &initial, compiled.config.seed)
            .unwrap()[0]
            .data()[0];
        assert_eq!(
            run.losses[0].to_bits(),
            want.to_bits(),
            "start {start}: recorded {} vs token of step {start}: {want}",
            run.losses[0]
        );
    }
}

/// Since story:bert-vulkan-kernels-rowwise a Vulkan artifact of a file with dropout has its
/// `Dropout` and `DropoutGrad` kernels, and the device run is refused with E-TARGET-011 at the
/// `dropout` call, because it supplies no RNG token per step.
#[test]
fn adv_drop_vulkan_artifact_is_e_target_011_at_the_call() {
    let artifact = compile_artifact("one_step.mspec", ONE_STEP, Backend::Vulkan)
        .unwrap_or_else(|errors| panic!("{errors:?}"));
    let ops: Vec<&str> = artifact
        .execution
        .kernels
        .iter()
        .map(|kernel| kernel.op)
        .collect();
    assert!(ops.contains(&"Dropout"), "{ops:?}");
    assert!(ops.contains(&"DropoutGrad"), "{ops:?}");
    #[cfg(feature = "vulkan")]
    {
        let compiled = compile_training("one_step.mspec", ONE_STEP).expect("it compiles");
        let refused = modelspec_runtime::vulkan::runner::dropout_refused(&compiled)
            .expect("the device run is refused");
        assert_eq!(refused.code.to_string(), "E-TARGET-011", "{refused:?}");
        let span = refused.span.as_ref().expect("a span");
        assert_eq!((span.line, span.column), (7, 17), "{refused:?}");
    }
}
