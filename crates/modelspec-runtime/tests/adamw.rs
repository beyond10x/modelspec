//! AdamW with decoupled weight decay and the warmup schedule on the CPU reference
//! (DESIGN.md § 2.1, § 6, § 8; story:bert-adamw-warmup).
//!
//! `optimizer = adamw(lr = F, weight_decay = F, beta1 = F, beta2 = F, eps = F, warmup_steps = N)`
//! compiles the update into the training step's `Optimizer` subgraph. At step `t`, counted from
//! 1, every parameter with a gradient `g` takes
//!
//! ```text
//! m = beta1 * m + (1 - beta1) * g
//! v = beta2 * v + (1 - beta2) * g * g
//! p = p - lr_t * (m * c1 / (sqrt(v * c2) + eps) + weight_decay * p)
//! ```
//!
//! with `m` and `v` starting at 0, `c1 = 1 / (1 - beta1^t)`, `c2 = 1 / (1 - beta2^t)`, and `lr_t`
//! linear warmup from 0 over `warmup_steps`, then linear decay to 0 at the run's last step:
//! `lr * t / warmup_steps` for `t <= warmup_steps`, `lr * (steps - t) / (steps - warmup_steps)`
//! after (coordinator decision D50, 2026-10-25, `.engineering/waves/bert-1.md`, settlement 1).
//! The references below compute all of it in `f64` from the gradients the run reports, never
//! through the runtime's own schedule.

use std::collections::BTreeMap;

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::ParamId;
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::runner::run_with;

/// The hyperparameters every source below is written with.
const LR: f32 = 0.01;
const WEIGHT_DECAY: f32 = 0.1;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

/// A linear regression with two parameters, `w` drawn and `b` zero, trained with AdamW.
fn two_parameters(warmup_steps: u64, steps: u64) -> String {
    format!(
        "model Reg {{
    param w: Tensor<f32, [4, 1]> = xavier_uniform();
    param b: Tensor<f32, [1]> = zeros();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {{
        return add_rows(matmul(x, w), b);
    }}
}}

train Fit {{
    model = Reg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = adamw(lr = {LR:?}, weight_decay = {WEIGHT_DECAY:?}, beta1 = {BETA1:?}, beta2 = {BETA2:?}, eps = {EPS:e}, warmup_steps = {warmup_steps});
    steps = {steps};
    seed = 7;
    device = cpu;
}}
"
    )
}

/// Beside `w`, a parameter `u` of ones read only through `relu(u - 2)`, which is negative
/// everywhere, so the gradient of `u` is exactly 0 and only the decay moves it.
fn dead_parameter(warmup_steps: u64, steps: u64) -> String {
    two_parameters(warmup_steps, steps)
        .replace(
            "param b: Tensor<f32, [1]> = zeros();",
            "param u: Tensor<f32, [4, 1]> = ones();",
        )
        .replace(
            "return add_rows(matmul(x, w), b);",
            "return add(matmul(x, w), matmul(x, relu(sub(u, 2.0))));",
        )
}

fn compiled(source: &str) -> CompiledTraining {
    compile_training("adamw.mspec", source).unwrap_or_else(|d| panic!("compiles: {d:?}"))
}

fn param(compiled: &CompiledTraining, path: &str) -> ParamId {
    compiled
        .forward
        .program
        .params
        .iter()
        .find(|(_, decl)| decl.path == path)
        .map(|(&id, _)| id)
        .unwrap_or_else(|| panic!("`{path}` is declared"))
}

/// What the observer saw at one step: the values before its update, their gradients and the
/// learning rate the step's update applies.
struct Seen {
    values: BTreeMap<ParamId, HostTensor<f32>>,
    gradients: Option<BTreeMap<ParamId, HostTensor<f32>>>,
    learning_rate: Option<f32>,
}

fn observe(compiled: &CompiledTraining) -> Vec<Seen> {
    let mut seen = Vec::new();
    run_with(compiled, None, &BTreeMap::new(), |o| {
        seen.push(Seen {
            values: o.values.clone(),
            gradients: o.gradients.cloned(),
            learning_rate: o.learning_rate,
        })
    })
    .expect("the run finishes");
    seen
}

/// The settled schedule in `f64` at step `t`, counted from 1, of a run of `steps` steps.
fn settled(t: u64, warmup_steps: u64, steps: u64) -> f64 {
    let lr = f64::from(LR);
    if t <= warmup_steps {
        lr * t as f64 / warmup_steps as f64
    } else {
        lr * (steps - t) as f64 / (steps - warmup_steps) as f64
    }
}

/// The error of `got` relative to `scale`.
fn relative_to(got: f32, expected: f64, scale: f64) -> f64 {
    (f64::from(got) - expected).abs() / scale
}

fn relative(got: f32, expected: f64) -> f64 {
    relative_to(got, expected, expected.abs())
}

/// Scenario 2. The parameters after steps 1, 2 and 3 equal the AdamW update computed here in
/// `f64` within `1e-6` relative, from the gradient the run reports at each step and moments that
/// start at 0. The error is relative to the larger of the old and the new value: the update
/// subtracts from the old value, so a parameter that passes near 0 keeps the absolute error of
/// the old magnitude.
#[test]
fn one_step() {
    let (warmup_steps, steps) = (2, 10);
    let c = compiled(&two_parameters(warmup_steps, steps));
    let params = [param(&c, "Reg.w"), param(&c, "Reg.b")];
    let seen = observe(&c);
    assert_eq!(seen.len(), steps as usize + 1);
    let (beta1, beta2) = (f64::from(BETA1), f64::from(BETA2));
    let (eps, decay) = (f64::from(EPS), f64::from(WEIGHT_DECAY));
    let mut m: BTreeMap<ParamId, Vec<f64>> = BTreeMap::new();
    let mut v: BTreeMap<ParamId, Vec<f64>> = BTreeMap::new();
    for t in 1..=3u64 {
        let before = &seen[t as usize - 1];
        let after = &seen[t as usize];
        let gradients = before.gradients.as_ref().expect("a step with an update");
        let lr_t = settled(t, warmup_steps, steps);
        let c1 = 1.0 / (1.0 - beta1.powi(t as i32));
        let c2 = 1.0 / (1.0 - beta2.powi(t as i32));
        for &p in &params {
            let g = gradients[&p].data();
            let m = m.entry(p).or_insert_with(|| vec![0.0; g.len()]);
            let v = v.entry(p).or_insert_with(|| vec![0.0; g.len()]);
            let old = before.values[&p].data();
            let new = after.values[&p].data();
            for i in 0..g.len() {
                let g = f64::from(g[i]);
                m[i] = beta1 * m[i] + (1.0 - beta1) * g;
                v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
                let p_old = f64::from(old[i]);
                let expected =
                    p_old - lr_t * (m[i] * c1 / ((v[i] * c2).sqrt() + eps) + decay * p_old);
                assert!(
                    relative_to(new[i], expected, p_old.abs().max(expected.abs())) <= 1e-6,
                    "step {t}, parameter {p}, element {i}: {} against {expected}, from {p_old}",
                    new[i]
                );
                assert_ne!(f64::from(new[i]), p_old, "step {t} moves {p}[{i}]");
            }
        }
    }
}

/// Scenario 3. The learning rate the runner passes at every step from 0 to `warmup_steps + 2`
/// is the settled schedule at `t = step + 1`, evaluated in `f64` and rounded once to `f32`: up
/// from `lr / warmup_steps`, `lr` at the last warmup step, then down to 0 at the run's last step.
#[test]
fn schedule() {
    let warmup_steps = 3;
    let steps = warmup_steps + 3;
    let c = compiled(&two_parameters(warmup_steps, steps));
    let seen = observe(&c);
    for (step, seen) in seen.iter().enumerate().take(warmup_steps as usize + 3) {
        let t = step as u64 + 1;
        let expected = settled(t, warmup_steps, steps) as f32;
        assert_eq!(seen.learning_rate, Some(expected), "step {step}");
    }
    assert_eq!(seen[warmup_steps as usize - 1].learning_rate, Some(LR));
    assert_eq!(seen[steps as usize - 1].learning_rate, Some(0.0));
    assert_eq!(seen[steps as usize].learning_rate, None, "the last loss");

    let no_warmup = compiled(&two_parameters(0, 4));
    let rates: Vec<Option<f32>> = observe(&no_warmup)
        .iter()
        .map(|s| s.learning_rate)
        .collect();
    let expected: Vec<Option<f32>> = (1..=4)
        .map(|t| Some(settled(t, 0, 4) as f32))
        .chain([None])
        .collect();
    assert_eq!(rates, expected);
    assert_eq!(rates[0], Some(LR * 0.75));
}

/// Scenario 4. A parameter whose gradient is exactly 0, read only through `relu` of negative
/// values, is multiplied by `1 - lr_t * weight_decay` in one step: the decay is decoupled from
/// the gradient, and `m = v = 0` adds nothing.
#[test]
fn decoupled_decay() {
    let (warmup_steps, steps) = (0, 5);
    let c = compiled(&dead_parameter(warmup_steps, steps));
    let u = param(&c, "Reg.u");
    let seen = observe(&c);
    let gradient = &seen[0].gradients.as_ref().expect("an update")[&u];
    assert!(
        gradient.data().iter().all(|&g| g == 0.0),
        "{:?}",
        gradient.data()
    );
    let lr_t = settled(1, warmup_steps, steps);
    let factor = 1.0 - lr_t * f64::from(WEIGHT_DECAY);
    assert!(factor < 1.0);
    let before = seen[0].values[&u].data();
    let after = seen[1].values[&u].data();
    for (i, (&old, &new)) in before.iter().zip(after).enumerate() {
        let expected = f64::from(old) * factor;
        assert!(
            relative(new, expected) <= 1e-6,
            "element {i}: {new} against {expected}"
        );
    }
}
