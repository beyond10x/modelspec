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
//! The references below work from the gradients the run reports, never through the runtime's own
//! schedule. `one_step` and `zero_gradient_decays_the_moments` compare bits with an `f32`
//! reference: the host constants `lr_t`, `c1`, `c2` and each `1 - beta` computed in `f64` and
//! rounded once, then every elementwise operation one rounded `f32` operation in the written order
//! (coordinator decision D64). `schedule` and `decoupled_decay` evaluate in `f64`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::ParamId;
use modelspec_core::ir::training::Moment;
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::data::dense;
use modelspec_runtime::train::runner::{RunError, run_from, run_with};

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

/// The error of `got` relative to `expected`.
fn relative(got: f32, expected: f64) -> f64 {
    (f64::from(got) - expected).abs() / expected.abs()
}

/// The host constants of step `t` of a run of `steps` steps, each computed in `f64` and rounded
/// once to `f32`: `(lr_t, c1, c2)`.
fn host_constants(t: u64, warmup_steps: u64, steps: u64) -> (f32, f32, f32) {
    let correction = |beta: f32| (1.0 / (1.0 - f64::from(beta).powi(t as i32))) as f32;
    (
        settled(t, warmup_steps, steps) as f32,
        correction(BETA1),
        correction(BETA2),
    )
}

/// One AdamW update of one element, every operation one rounded `f32` operation in the written
/// order, each `1 - beta` computed in `f64` and rounded once; returns `(p', m', v')`.
fn reference(p: f32, g: f32, m: f32, v: f32, (lr_t, c1, c2): (f32, f32, f32)) -> (f32, f32, f32) {
    let rest1 = (1.0 - f64::from(BETA1)) as f32;
    let rest2 = (1.0 - f64::from(BETA2)) as f32;
    let decayed_m = BETA1 * m;
    let new_m = rest1 * g;
    let m = decayed_m + new_m;
    let decayed_v = BETA2 * v;
    let scaled_g = rest2 * g;
    let new_v = scaled_g * g;
    let v = decayed_v + new_v;
    let m_hat = m * c1;
    let v_hat = v * c2;
    let root = v_hat.sqrt();
    let denominator = root + EPS;
    let direction = m_hat / denominator;
    let shrink = WEIGHT_DECAY * p;
    let step = direction + shrink;
    let scaled = lr_t * step;
    (p - scaled, m, v)
}

/// The moments of the reference, by parameter: `m`, then `v`.
type Moments = BTreeMap<ParamId, (Vec<f32>, Vec<f32>)>;

/// Checks that step `t` took every parameter of `params` from `before` to `after` by the
/// reference update on the gradients `before` reports, bit for bit, and advances `moments`.
fn assert_reference_step(
    t: u64,
    (warmup_steps, steps): (u64, u64),
    params: &[ParamId],
    (before, after): (&Seen, &Seen),
    moments: &mut Moments,
) {
    let gradients = before.gradients.as_ref().expect("a step with an update");
    let constants = host_constants(t, warmup_steps, steps);
    for &p in params {
        let g = gradients[&p].data();
        let (m, v) = moments
            .entry(p)
            .or_insert_with(|| (vec![0.0; g.len()], vec![0.0; g.len()]));
        let old = before.values[&p].data();
        let new = after.values[&p].data();
        for i in 0..g.len() {
            let (expected, m_next, v_next) = reference(old[i], g[i], m[i], v[i], constants);
            (m[i], v[i]) = (m_next, v_next);
            assert_eq!(
                new[i].to_bits(),
                expected.to_bits(),
                "step {t}, parameter {p}, element {i}: {} against {expected}, from {}",
                new[i],
                old[i]
            );
        }
    }
}

/// Scenario 2. The parameters after steps 1, 2 and 3 are bit-equal to the AdamW update computed
/// here in `f32`, every operation rounded in the written order, from the gradient the run
/// reports at each step and moments that start at 0 (coordinator decision D64: an `f64`
/// reference within `1e-6` cannot be met under cancellation, the `f32` reference is the measure).
#[test]
fn one_step() {
    let (warmup_steps, steps) = (2, 10);
    let c = compiled(&two_parameters(warmup_steps, steps));
    let params = [param(&c, "Reg.w"), param(&c, "Reg.b")];
    let seen = observe(&c);
    assert_eq!(seen.len(), steps as usize + 1);
    let mut moments = Moments::new();
    for t in 1..=3u64 {
        let (before, after) = (&seen[t as usize - 1], &seen[t as usize]);
        assert_reference_step(
            t,
            (warmup_steps, steps),
            &params,
            (before, after),
            &mut moments,
        );
        for &p in &params {
            let moved = before.values[&p]
                .data()
                .iter()
                .zip(after.values[&p].data())
                .all(|(old, new)| old != new);
            assert!(moved, "step {t} moves every element of {p}");
        }
    }
}

const ZERO_STEP: u64 = 3;
const DENSE_STEPS: u64 = 6;

fn write_f32(path: &Path, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("the fixture is written");
}

/// Dense data for [`two_parameters`] with batch 8: one batch per step of a run of
/// [`DENSE_STEPS`] steps, the input of step [`ZERO_STEP`] all zero and every other input not.
fn zero_step_data() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adamw")
        .join("zero_step");
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old fixture is removed");
    }
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let samples = 8 * DENSE_STEPS as usize;
    let x: Vec<f32> = (0..samples * 4)
        .map(|k| {
            if (k / 4) / 8 == ZERO_STEP as usize {
                0.0
            } else {
                ((k * 7 % 11) as f32 - 5.0) / 4.0
            }
        })
        .collect();
    let y: Vec<f32> = (0..samples).map(|k| (k % 5) as f32 * 0.5 - 1.0).collect();
    write_f32(&dir.join("x.f32"), &x);
    write_f32(&dir.join("y.f32"), &y);
    dir
}

/// D64, F7. A parameter whose gradient is nonzero for three steps and exactly zero at the fourth
/// keeps moving on its moments: the fourth step decays them, `m = beta1 * m` and
/// `v = beta2 * v`, and the run stays bit-equal to the `f32` reference at every step, its final
/// moments included. A step that left the moments of an all-zero gradient where they were would
/// leave the reference from that step on.
#[test]
fn zero_gradient_decays_the_moments() {
    let (warmup_steps, steps) = (2, DENSE_STEPS);
    let c = compiled(&two_parameters(warmup_steps, steps));
    let data = dense(&zero_step_data(), &c.config, &c.forward).expect("the fixture is dense data");
    let (w, b) = (param(&c, "Reg.w"), param(&c, "Reg.b"));
    let mut seen = Vec::new();
    let run = run_with(&c, Some(&data), &BTreeMap::new(), |o| {
        seen.push(Seen {
            values: o.values.clone(),
            gradients: o.gradients.cloned(),
            learning_rate: o.learning_rate,
        })
    })
    .expect("the run finishes");
    assert_eq!(seen.len(), steps as usize + 1);
    for (step, seen) in seen.iter().enumerate().take(steps as usize) {
        let g = seen.gradients.as_ref().expect("an update")[&w].data();
        let zero = g.iter().all(|&g| g == 0.0);
        assert_eq!(zero, step as u64 == ZERO_STEP, "step {step}: {g:?}");
    }
    let mut moments = Moments::new();
    for t in 1..=steps {
        let (before, after) = (&seen[t as usize - 1], &seen[t as usize]);
        let m_before = moments.get(&w).map(|(m, v)| (m.clone(), v.clone()));
        assert_reference_step(
            t,
            (warmup_steps, steps),
            &[w, b],
            (before, after),
            &mut moments,
        );
        if t == ZERO_STEP + 1 {
            let (m_old, v_old) = m_before.expect("moments after three steps");
            let (m, v) = &moments[&w];
            for i in 0..m.len() {
                assert_ne!(m_old[i], 0.0, "element {i} has a first moment to decay");
                assert_eq!(m[i].to_bits(), (BETA1 * m_old[i]).to_bits(), "m[{i}]");
                assert_eq!(v[i].to_bits(), (BETA2 * v_old[i]).to_bits(), "v[{i}]");
            }
            let moved = before.values[&w]
                .data()
                .iter()
                .zip(after.values[&w].data())
                .all(|(old, new)| old != new);
            assert!(moved, "the zero-gradient step moves every element of w");
        }
    }
    for (p, (m, v)) in &moments {
        for (moment, expected) in [(Moment::First, m), (Moment::Second, v)] {
            let got: Vec<u32> = run.state.moments[&(*p, moment)]
                .data()
                .iter()
                .map(|x| x.to_bits())
                .collect();
            let expected: Vec<u32> = expected.iter().map(|x| x.to_bits()).collect();
            assert_eq!(got, expected, "{moment} of {p}");
        }
    }
}

/// D64, F6. A run of an AdamW file from a step above 0 is refused before its first step: the
/// moments are not checkpointed, and moments restarted at 0 would take the bias corrections of
/// the resumed step. An SGD file resumes from the same step.
#[test]
fn resumed_run_is_refused() {
    let adamw = compiled(&two_parameters(2, 10));
    let refused = run_from(&adamw, None, &BTreeMap::new(), 3, |_| {}, |_, _| Ok(()));
    match refused {
        Err(error @ RunError::MomentsNotCheckpointed { start: 3 }) => {
            assert!(error.to_string().contains("not checkpointed"), "{error}");
        }
        other => panic!("{:?}", other.map(|run| run.losses)),
    }
    assert!(run_from(&adamw, None, &BTreeMap::new(), 0, |_| {}, |_, _| Ok(())).is_ok());
    let source = two_parameters(2, 10);
    let line = source
        .lines()
        .find(|line| line.contains("optimizer = adamw("))
        .expect("the optimizer line");
    let sgd = compiled(&source.replace(line, "    optimizer = sgd(lr = 0.01);"));
    assert!(run_from(&sgd, None, &BTreeMap::new(), 3, |_| {}, |_, _| Ok(())).is_ok());
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
