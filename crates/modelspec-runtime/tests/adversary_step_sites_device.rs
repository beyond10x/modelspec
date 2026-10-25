//! Adversary pass 1 of story:step-sites-beat-pytorch on a Vulkan device: the `ReluGrad`
//! epilogue and the moved `mse` loss sum on models other than the MNIST example and the
//! `mlp_sweep` point the unit's tests use (the probe models of
//! `crates/modelspec-core/tests/adversary_step_sites.rs`, plus `relu(relu(matmul))`).
//!
//! For each model, a training run on device 0 against the CPU run of the same file:
//!
//! - every final parameter has the CPU run's bits when device 0 negotiates the bit-for-bit
//!   class, and passes the whole-run bound otherwise;
//! - every loss passes the `mse` tolerance against the CPU loss;
//! - the final loss, which the run evaluates with `ExecutionPlan::loss_steps` only, has the bits
//!   of `infer` on the run's final parameters, which records the same steps;
//! - the run dispatches `steps` times the step plan's dispatches plus the dispatches of
//!   `loss_steps`, so the final evaluation records no gradient.
//!
//! Ignored by default: it creates a Vulkan instance. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test adversary_step_sites_device -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::{KernelOp, run_within, term_tolerance};
use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_core::ir::kernel::PlanOp;
use modelspec_runtime::train::runner;
use modelspec_runtime::vulkan::negotiate::{NumericsClass, negotiate};
use modelspec_runtime::vulkan::runner::{infer, train};
use modelspec_runtime::vulkan::{VALIDATION_ENV, Vulkan};

fn model(params: &[&str], (b, i, o): (u64, u64, u64), body: &str, loss: &str) -> String {
    let params: String = params
        .iter()
        .map(|l| format!("    param {l} = xavier_uniform();\n"))
        .collect();
    format!(
        "model Adv {{
{params}
    fn forward(x: Tensor<f32, [{b}, {i}]>)
        -> Tensor<f32, [{b}, {o}]> {{
        {body}
    }}
}}

train Fit {{
    model = Adv;
    input x: Tensor<f32, [{b}, {i}]>;
    target y: Tensor<f32, [{b}, {o}]>;
    loss = {loss};
    optimizer = sgd(lr = 0.05);
    steps = 30;
    seed = 7;
    device = vulkan(0);
}}
"
    )
}

const MSE: &str = "mse(model.forward(x), y)";

fn models() -> Vec<(&'static str, String)> {
    let small = (10, 12, 5);
    let two = ["w1: Tensor<f32, [12, 9]>", "w2: Tensor<f32, [9, 5]>"];
    vec![
        (
            "deep3",
            model(
                &[
                    "w1: Tensor<f32, [12, 9]>",
                    "w2: Tensor<f32, [9, 7]>",
                    "w3: Tensor<f32, [7, 5]>",
                ],
                small,
                "return matmul(relu(matmul(relu(matmul(x, w1)), w2)), w3);",
                MSE,
            ),
        ),
        (
            "relu_of_a_parameter",
            model(&two, small, "return matmul(matmul(x, relu(w1)), w2);", MSE),
        ),
        (
            "bias_before_relu",
            model(
                &[
                    "w1: Tensor<f32, [12, 9]>",
                    "b1: Tensor<f32, [10, 9]>",
                    "w2: Tensor<f32, [9, 5]>",
                ],
                small,
                "return matmul(relu(add(matmul(x, w1), b1)), w2);",
                MSE,
            ),
        ),
        (
            "two_relus_of_one_matmul",
            model(
                &[
                    "w1: Tensor<f32, [12, 9]>",
                    "w2: Tensor<f32, [9, 5]>",
                    "w3: Tensor<f32, [9, 5]>",
                ],
                small,
                "let z = matmul(x, w1);\n        return add(matmul(relu(z), w2), matmul(relu(z), w3));",
                MSE,
            ),
        ),
        (
            "transposed_parameters",
            model(
                &["v1: Tensor<f32, [9, 12]>", "v2: Tensor<f32, [5, 9]>"],
                small,
                "return matmul(relu(matmul(x, transpose(v1))), transpose(v2));",
                MSE,
            ),
        ),
        (
            "wide_second_layer",
            model(
                &["w1: Tensor<f32, [16, 8]>", "w2: Tensor<f32, [8, 512]>"],
                (64, 16, 512),
                "return matmul(relu(matmul(x, w1)), w2);",
                MSE,
            ),
        ),
        (
            "no_relu",
            model(&two, small, "return matmul(matmul(x, w1), w2);", MSE),
        ),
        (
            "scaled_loss",
            model(
                &two,
                small,
                "return matmul(relu(matmul(x, w1)), w2);",
                "mul(mse(model.forward(x), y), 2.0)",
            ),
        ),
        (
            "relu_of_relu",
            model(
                &two,
                small,
                "return matmul(relu(relu(matmul(x, w1))), w2);",
                MSE,
            ),
        ),
    ]
}

#[test]
#[ignore = "needs a Vulkan device"]
fn every_probe_model_trains_to_the_cpu_bits_and_its_final_loss_is_the_infer_loss() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
    );
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let bit_for_bit = negotiated.numerics == NumericsClass::BitForBit;
    eprintln!("numerics class of device 0: {:?}", negotiated.numerics);

    for (name, source) in models() {
        let file = format!("{name}.mspec");
        let compiled = compile_training(&file, &source).expect("compiles");
        let artifact = compile_artifact(&file, &source, Backend::Vulkan).expect("compiles");
        let cpu = runner::run(&compiled).expect("the CPU run finishes");
        let (gpu, stats) = train(&device, &profile, &negotiated, &compiled, &artifact)
            .unwrap_or_else(|e| panic!("{name}: the device run fails: {e}"));

        assert_eq!(gpu.initial, cpu.initial, "{name}");
        for (param, c) in &cpu.state.values {
            let g = &gpu.state.values[param];
            if bit_for_bit {
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
                assert_eq!(
                    bits(g.data()),
                    bits(c.data()),
                    "{name}: {param} differs from the CPU bits"
                );
            } else {
                assert!(run_within(g.data(), c.data()), "{name}: {param}");
            }
        }
        let tolerance = term_tolerance(KernelOp::Mse);
        assert_eq!(gpu.losses.len(), cpu.losses.len(), "{name}");
        for (k, (&g, &c)) in gpu.losses.iter().zip(&cpu.losses).enumerate() {
            let r = tolerance.r(c, f64::from(c), 1);
            assert!(
                tolerance.accepts(g, c, r),
                "{name}: loss {k}: device {g}, cpu {c}"
            );
        }

        let plan = &artifact.execution;
        let per_step = plan
            .step
            .iter()
            .filter(|s| s.op == PlanOp::Dispatch)
            .count() as u64;
        let final_dispatches = plan
            .loss_steps()
            .iter()
            .filter(|&&at| plan.step[at].op == PlanOp::Dispatch)
            .count() as u64;
        assert_eq!(stats.dispatches_per_step, per_step, "{name}");
        assert_eq!(
            stats.dispatches,
            stats.steps * per_step + final_dispatches,
            "{name}: the final loss records the loss steps alone"
        );

        let (inferred, _) = infer(
            &device,
            &profile,
            &negotiated,
            &compiled,
            &artifact,
            &gpu.state.values,
        )
        .unwrap_or_else(|e| panic!("{name}: infer fails: {e}"));
        let last = *gpu.losses.last().expect("steps + 1 losses");
        assert_eq!(
            inferred.loss.to_bits(),
            last.to_bits(),
            "{name}: infer on the final parameters gives the run's final loss"
        );
        eprintln!(
            "{name}: {per_step} dispatches per step, final loss {last:e}, class {:?}",
            stats.class
        );
    }
    drop(device);
    let messages = vulkan.take_validation_messages();
    assert!(messages.is_empty(), "validation messages: {messages:#?}");
}
