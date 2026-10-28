//! Adversarial device cases for checkpoints of a Vulkan run (`story:m1-vulkan-checkpoint-infer`):
//! the step boundaries of `Checkpoints::every`, the values a mid-run checkpoint reads back, and
//! the prediction of `infer`. Every case fails on a validation message. Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_checkpoint_adversary -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;
use std::path::Path;

use common::run_within;
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{CompiledTraining, compile_artifact, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::tensor::ParamId;
use modelspec_runtime::cpu::{HostTensor, evaluate_all};
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::vulkan::negotiate::{Negotiated, negotiate};
use modelspec_runtime::vulkan::runner::{Checkpoints, RunOptions, infer, train_with};
use modelspec_runtime::vulkan::{Device, Vulkan};

type Values = BTreeMap<ParamId, HostTensor<f32>>;

fn compiled(file: &str, steps: u64) -> (CompiledTraining, CompiledArtifact) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(file);
    let source = std::fs::read_to_string(&path).expect("the example reads");
    let mut compiled = compile_training(file, &source).expect("the example compiles");
    compiled.config.steps = steps;
    let artifact =
        compile_artifact(file, &source, Backend::Vulkan).expect("the example compiles for Vulkan");
    (compiled, artifact)
}

fn on_device<T>(
    label: &str,
    body: impl FnOnce(&Device<'_>, &DeviceProfile, &Negotiated) -> T,
) -> T {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let result = body(&device, &profile, &negotiated);
    drop(device);
    let messages = vulkan.take_validation_messages();
    assert!(messages.is_empty(), "{label}: {messages:#?}");
    eprintln!(
        "{label}: validation {}, 0 messages",
        vulkan.validation_enabled()
    );
    result
}

/// For each `(start, steps, every)`: the steps a checkpoint is written at, the host waits, and
/// each mid-run checkpoint's values against a device run of exactly that many steps from the
/// same start; the final state against a run without checkpoints.
#[test]
#[ignore = "needs a Vulkan device"]
fn checkpoint_boundaries_on_the_device() {
    let cases: [(&str, u64, u64, u64, Vec<u64>); 10] = [
        ("examples/linear_regression.mspec", 0, 10, 3, vec![3, 6, 9]),
        ("examples/linear_regression.mspec", 0, 10, 10, vec![]),
        ("examples/linear_regression.mspec", 0, 10, 11, vec![]),
        (
            "examples/linear_regression.mspec",
            0,
            10,
            1,
            (1..10).collect(),
        ),
        ("examples/linear_regression.mspec", 0, 10, 0, vec![]),
        ("examples/linear_regression.mspec", 4, 10, 3, vec![6, 9, 12]),
        ("examples/linear_regression.mspec", 0, 0, 1, vec![]),
        (
            "examples/linear_regression.mspec",
            0,
            513,
            256,
            vec![256, 512],
        ),
        ("examples/mnist_mlp.mspec", 0, 600, 256, vec![256, 512]),
        (
            "examples/mnist_mlp.mspec",
            0,
            300,
            7,
            (1..300).filter(|k| k % 7 == 0).collect(),
        ),
    ];
    for (file, start, steps, every, expected) in cases {
        let label = format!("{file} start {start} steps {steps} every {every}");
        let (compiled, artifact) = compiled(file, steps);
        // The start state: the parameters of a device run of `start` steps from step 0.
        on_device(&label, |device, profile, negotiated| {
            let init: Values = if start > 0 {
                let (c0, _) = self::compiled(file, start);
                train_with(
                    device,
                    profile,
                    negotiated,
                    &c0,
                    &artifact,
                    RunOptions::default(),
                )
                .expect("the start run finishes")
                .0
                .state
                .values
            } else {
                BTreeMap::new()
            };
            let mut written: Vec<(u64, Values)> = Vec::new();
            let mut write = |k: u64, values: &Values| {
                written.push((k, values.clone()));
                Ok::<(), modelspec_runtime::train::checkpoint::CheckpointError>(())
            };
            let (run, stats) = train_with(
                device,
                profile,
                negotiated,
                &compiled,
                &artifact,
                RunOptions {
                    start,
                    init: init.clone(),
                    checkpoints: Some(Checkpoints {
                        every,
                        write: &mut write,
                    }),
                    data: None,
                    clock: false,
                },
            )
            .expect("the run with checkpoints finishes");
            let steps_written: Vec<u64> = written.iter().map(|(k, _)| *k).collect();
            assert_eq!(steps_written, expected, "{label}");
            assert_eq!(stats.host_waits, 1 + expected.len() as u64, "{label}");
            assert_eq!(run.start_step, start, "{label}");
            assert_eq!(run.losses.len() as u64, steps + 1, "{label}");
            let (plain, _) = train_with(
                device,
                profile,
                negotiated,
                &compiled,
                &artifact,
                RunOptions {
                    start,
                    init: init.clone(),
                    checkpoints: None,
                    data: None,
                    clock: false,
                },
            )
            .expect("the plain run finishes");
            assert_eq!(run.state.versions, plain.state.versions, "{label}");
            for (param, c) in &plain.state.values {
                assert!(
                    run_within(run.state.values[param].data(), c.data()),
                    "{label}: final {param}"
                );
            }
            let mut bit_equal = true;
            for (k, values) in &written {
                let (ck, _) = self::compiled(file, k - start);
                let (exact, _) = train_with(
                    device,
                    profile,
                    negotiated,
                    &ck,
                    &artifact,
                    RunOptions {
                        start,
                        init: init.clone(),
                        checkpoints: None,
                        data: None,
                        clock: false,
                    },
                )
                .expect("the reference run finishes");
                for (param, c) in &exact.state.values {
                    assert!(
                        run_within(values[param].data(), c.data()),
                        "{label}: checkpoint at {k}, {param} differs from a run of {k} steps"
                    );
                    bit_equal &= values[param] == *c;
                }
            }
            eprintln!("{label}: mid-run checkpoints bit-equal to runs of k steps: {bit_equal}");
        });
    }
}

/// `infer` returns the prediction of the CPU forward program within the whole-run bound.
#[test]
#[ignore = "needs a Vulkan device"]
fn device_infer_prediction_matches_the_cpu_forward() {
    for file in [
        "examples/mnist_mlp.mspec",
        "examples/linear_regression.mspec",
    ] {
        let (compiled, artifact) = compiled(file, 200);
        on_device(file, |device, profile, negotiated| {
            let (run, _) = train_with(
                device,
                profile,
                negotiated,
                &compiled,
                &artifact,
                RunOptions::default(),
            )
            .expect("the run finishes");
            let params = run.state.values;
            let (inference, stats) =
                infer(device, profile, negotiated, &compiled, &artifact, &params)
                    .expect("inference finishes");
            assert_eq!(stats.host_waits, 1);
            let batch = synthetic(&compiled.config, &compiled.forward).unwrap();
            let mut cpu = evaluate_all(&compiled.forward.program, &batch.inputs, &params).unwrap();
            let c = cpu.values.remove(&compiled.forward.prediction).unwrap();
            assert_eq!(inference.prediction.shape(), c.shape(), "{file}");
            assert!(
                run_within(inference.prediction.data(), c.data()),
                "{file}: device prediction outside the bound of the CPU's"
            );
            eprintln!(
                "{file}: prediction bit-equal {}",
                inference.prediction.data() == c.data()
            );
        });
    }
}
