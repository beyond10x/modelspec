//! Adversary cases for AdamW on the CPU reference (story:bert-adamw-warmup, DESIGN.md § 2.1,
//! § 6, § 8, coordinator decision D50).
//!
//! The update is compared bit for bit with an independent `f32` reference written from D50: the
//! host constants `lr_t`, `c1 = 1 / (1 - beta1^t)` and `c2 = 1 / (1 - beta2^t)` computed in `f64`
//! and rounded once, then each elementwise operation one rounded `f32` operation in the written
//! order, no multiply-add. The schedule is probed at its edges, the moments as state, and the
//! paths the change touched without a test: the checkpoint optimizer record, a step with no
//! parameter to update and a second moment that overflows.
//!
//! A case named `adv_adamw_f<N>_...` is a finding of the adversary's report, ignored until its
//! fix landed in fix pass 1 (coordinator decision D64).

use std::collections::BTreeMap;

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::ParamId;
use modelspec_core::ir::training::{Moment, StateVersion};
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::checkpoint::{Checkpoint, OptimizerConfig, OptimizerKind};
use modelspec_runtime::train::runner::{RunError, run_with};

const LR: f32 = 0.01;
const WEIGHT_DECAY: f32 = 0.1;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

/// The two-parameter regression of the unit's `tests/adamw.rs`, with the optimizer `optimizer`.
fn regression(optimizer: &str, steps: u64) -> String {
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
    optimizer = {optimizer};
    steps = {steps};
    seed = 7;
    device = cpu;
}}
"
    )
}

fn adamw(warmup_steps: u64) -> String {
    format!(
        "adamw(lr = {LR:?}, weight_decay = {WEIGHT_DECAY:?}, beta1 = {BETA1:?}, beta2 = {BETA2:?}, eps = {EPS:e}, warmup_steps = {warmup_steps})"
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

struct Seen {
    values: BTreeMap<ParamId, HostTensor<f32>>,
    gradients: Option<BTreeMap<ParamId, HostTensor<f32>>>,
    learning_rate: Option<f32>,
}

fn observe(compiled: &CompiledTraining) -> (Vec<Seen>, modelspec_runtime::train::runner::Run) {
    let mut seen = Vec::new();
    let run = run_with(compiled, None, &BTreeMap::new(), |o| {
        seen.push(Seen {
            values: o.values.clone(),
            gradients: o.gradients.cloned(),
            learning_rate: o.learning_rate,
        })
    })
    .unwrap_or_else(|e| panic!("the run finishes: {e}"));
    (seen, run)
}

/// The settled schedule at step `t` of `steps`, written from D50 in `f64`, rounded once.
fn lr_t(lr: f32, t: u64, warmup_steps: u64, steps: u64) -> f32 {
    let peak = f64::from(lr);
    let rate = if t <= warmup_steps {
        peak * t as f64 / warmup_steps as f64
    } else {
        peak * (steps - t) as f64 / (steps - warmup_steps) as f64
    };
    rate as f32
}

/// `1 / (1 - beta^t)` in `f64`, rounded once.
fn correction(beta: f32, t: u64) -> f32 {
    (1.0 / (1.0 - f64::from(beta).powi(t as i32))) as f32
}

/// One AdamW update of one element in `f32`, each operation rounded once in D50's written
/// order; returns `(p', m', v')`.
fn update(p: f32, g: f32, m: f32, v: f32, lr_t: f32, c1: f32, c2: f32) -> (f32, f32, f32) {
    update_with(EPS, p, g, m, v, lr_t, c1, c2)
}

/// [`update`] with the denominator term `eps`.
#[allow(clippy::too_many_arguments)]
fn update_with(
    eps: f32,
    p: f32,
    g: f32,
    m: f32,
    v: f32,
    lr_t: f32,
    c1: f32,
    c2: f32,
) -> (f32, f32, f32) {
    let rest1 = 1.0f32 - BETA1;
    let rest2 = 1.0f32 - BETA2;
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
    let denominator = root + eps;
    let direction = m_hat / denominator;
    let shrink = WEIGHT_DECAY * p;
    let step = direction + shrink;
    let scaled = lr_t * step;
    (p - scaled, m, v)
}

/// Hypothesis 1. The runner's parameters after every step of a 12-step run (so steps 1, 2, 3,
/// 10 and the last) are the bits of the `f32` reference propagated from the initial values on
/// the gradients the run reports, and its final moments and versions are the reference's.
#[test]
fn adamw_matches_an_f32_reference_bit_for_bit() {
    let (warmup_steps, steps) = (2, 12);
    let c = compiled(&regression(&adamw(warmup_steps), steps));
    let params = [param(&c, "Reg.w"), param(&c, "Reg.b")];
    let (seen, run) = observe(&c);
    assert_eq!(seen.len(), steps as usize + 1);
    let mut p: BTreeMap<ParamId, Vec<f32>> = params
        .iter()
        .map(|&id| (id, seen[0].values[&id].data().to_vec()))
        .collect();
    let mut m: BTreeMap<ParamId, Vec<f32>> =
        p.iter().map(|(&id, x)| (id, vec![0.0; x.len()])).collect();
    let mut v = m.clone();
    for t in 1..=steps {
        let before = &seen[t as usize - 1];
        let gradients = before.gradients.as_ref().expect("an update");
        let (rate, c1, c2) = (
            lr_t(LR, t, warmup_steps, steps),
            correction(BETA1, t),
            correction(BETA2, t),
        );
        assert_eq!(before.learning_rate, Some(rate), "lr_t at t = {t}");
        for &id in &params {
            let g = gradients[&id].data();
            for i in 0..g.len() {
                let (np, nm, nv) = update(p[&id][i], g[i], m[&id][i], v[&id][i], rate, c1, c2);
                p.get_mut(&id).unwrap()[i] = np;
                m.get_mut(&id).unwrap()[i] = nm;
                v.get_mut(&id).unwrap()[i] = nv;
            }
            let got = seen[t as usize].values[&id].data();
            let got_bits: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
            let want_bits: Vec<u32> = p[&id].iter().map(|x| x.to_bits()).collect();
            assert_eq!(
                got_bits, want_bits,
                "t = {t}, {id}: runner {got:?} against reference {:?}",
                p[&id]
            );
        }
    }
    for &id in &params {
        assert_eq!(run.state.versions[&id], StateVersion(steps), "{id}");
        let bits = |x: &[f32]| x.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(run.state.moments[&(id, Moment::First)].data()),
            bits(&m[&id]),
            "m of {id}"
        );
        assert_eq!(
            bits(run.state.moments[&(id, Moment::Second)].data()),
            bits(&v[&id]),
            "v of {id}"
        );
    }
    assert_eq!(
        run.state.moments.len(),
        4,
        "two moments of each of two parameters"
    );
}

/// Hypothesis 1 and mutant (d). With `eps = 1e-8` the term vanishes in `f32` beside a gradient of
/// order 0.1, so `sqrt(v * c2) + eps` and `sqrt(v * c2 + eps)` give the same bits; with
/// `eps = 0.5` they do not, and the runner must still be the reference's bits over 6 steps.
#[test]
fn adamw_matches_an_f32_reference_with_a_large_eps() {
    let eps = 0.5f32;
    let (warmup_steps, steps) = (2, 6);
    let optimizer = format!(
        "adamw(lr = {LR:?}, weight_decay = {WEIGHT_DECAY:?}, beta1 = {BETA1:?}, beta2 = {BETA2:?}, eps = {eps:?}, warmup_steps = {warmup_steps})"
    );
    let c = compiled(&regression(&optimizer, steps));
    let params = [param(&c, "Reg.w"), param(&c, "Reg.b")];
    let (seen, _) = observe(&c);
    let mut state: BTreeMap<(ParamId, usize), (f32, f32, f32)> = BTreeMap::new();
    for &id in &params {
        for (i, &x) in seen[0].values[&id].data().iter().enumerate() {
            state.insert((id, i), (x, 0.0, 0.0));
        }
    }
    for t in 1..=steps {
        let gradients = seen[t as usize - 1].gradients.as_ref().unwrap();
        let (rate, c1, c2) = (
            lr_t(LR, t, warmup_steps, steps),
            correction(BETA1, t),
            correction(BETA2, t),
        );
        for &id in &params {
            let got = seen[t as usize].values[&id].data();
            for (i, &g) in gradients[&id].data().iter().enumerate() {
                let (p, m, v) = state[&(id, i)];
                let next = update_with(eps, p, g, m, v, rate, c1, c2);
                state.insert((id, i), next);
                assert_eq!(
                    got[i].to_bits(),
                    next.0.to_bits(),
                    "t = {t}, {id}[{i}]: runner {} against reference {}",
                    got[i],
                    next.0
                );
            }
        }
    }
}

/// The element with the largest relative error so far: the error, the step, the parameter, the
/// element, the runner's value, the `f64` reference, the old value and `lr_t * step` in `f64`.
type Worst = (f64, u64, ParamId, usize, f32, f64, f32, f64);

/// Hypothesis 1, the implementor's near-zero element. In the unit's `one_step` setting (warm-up
/// 2 of 10 steps), the element with the largest error against an `f64` reference relative to
/// its own expected value over steps 1 to 3 is still the bits of the `f32` reference, and its
/// error is within 2 ulp of the larger of `|p|` and `|lr_t * step|`: the rounding of a
/// difference of nearly equal values, not a defect of the update.
#[test]
fn adamw_one_step_near_zero_error_is_f32_rounding_of_a_cancellation() {
    let (warmup_steps, steps) = (2, 10);
    let c = compiled(&regression(&adamw(warmup_steps), steps));
    let params = [param(&c, "Reg.w"), param(&c, "Reg.b")];
    let (seen, _) = observe(&c);
    let (b1, b2) = (f64::from(BETA1), f64::from(BETA2));
    let mut m64: BTreeMap<(ParamId, usize), f64> = BTreeMap::new();
    let mut v64: BTreeMap<(ParamId, usize), f64> = BTreeMap::new();
    let mut m32: BTreeMap<(ParamId, usize), f32> = BTreeMap::new();
    let mut v32: BTreeMap<(ParamId, usize), f32> = BTreeMap::new();
    let mut worst: Option<Worst> = None;
    for t in 1..=3u64 {
        let before = &seen[t as usize - 1];
        let gradients = before.gradients.as_ref().unwrap();
        let rate64 = {
            let peak = f64::from(LR);
            if t <= warmup_steps {
                peak * t as f64 / warmup_steps as f64
            } else {
                peak * (steps - t) as f64 / (steps - warmup_steps) as f64
            }
        };
        let (c1_64, c2_64) = (
            1.0 / (1.0 - b1.powi(t as i32)),
            1.0 / (1.0 - b2.powi(t as i32)),
        );
        for &id in &params {
            let g = gradients[&id].data();
            let old = before.values[&id].data();
            let new = seen[t as usize].values[&id].data();
            for i in 0..g.len() {
                let key = (id, i);
                let gd = f64::from(g[i]);
                let m = b1 * m64.get(&key).copied().unwrap_or(0.0) + (1.0 - b1) * gd;
                let v = b2 * v64.get(&key).copied().unwrap_or(0.0) + (1.0 - b2) * gd * gd;
                m64.insert(key, m);
                v64.insert(key, v);
                let p_old = f64::from(old[i]);
                let step = m * c1_64 / ((v * c2_64).sqrt() + f64::from(EPS))
                    + f64::from(WEIGHT_DECAY) * p_old;
                let expected = p_old - rate64 * step;
                let (p32, nm, nv) = update(
                    old[i],
                    g[i],
                    m32.get(&key).copied().unwrap_or(0.0),
                    v32.get(&key).copied().unwrap_or(0.0),
                    lr_t(LR, t, warmup_steps, steps),
                    correction(BETA1, t),
                    correction(BETA2, t),
                );
                m32.insert(key, nm);
                v32.insert(key, nv);
                assert_eq!(new[i].to_bits(), p32.to_bits(), "t = {t}, {id}[{i}]");
                let error = (f64::from(new[i]) - expected).abs() / expected.abs();
                if worst.is_none_or(|w| error > w.0) {
                    worst = Some((error, t, id, i, new[i], expected, old[i], rate64 * step));
                }
            }
        }
    }
    let (error, t, id, i, got, expected, old, moved) = worst.unwrap();
    let scale = f64::from(old).abs().max(moved.abs());
    let ulp = scale * f64::from(f32::EPSILON);
    eprintln!(
        "worst: t = {t}, {id}[{i}]: runner {got:e} (bits = f32 reference), f64 {expected:e}, \
         old {old:e}, lr_t*step {moved:e}, error/|expected| {error:e}, \
         |error| = {:.2} ulp of max(|old|, |lr_t*step|)",
        (f64::from(got) - expected).abs() / ulp
    );
    assert!(
        (f64::from(got) - expected).abs() <= 2.0 * ulp,
        "the error is {} ulp of the operands' scale",
        (f64::from(got) - expected).abs() / ulp
    );
}

/// Hypothesis 2. The rate at `t = 1`, the last warm-up step, the step after it, the
/// second-to-last and the last step; the last update applies `lr_t = 0`, so it leaves every
/// parameter bit-identical (its decay term is scaled by `lr_t` too), while the moments still
/// advance.
#[test]
fn adamw_schedule_edges_and_the_last_update() {
    let (warmup_steps, steps) = (3, 8);
    let c = compiled(&regression(&adamw(warmup_steps), steps));
    let (seen, run) = observe(&c);
    let rate = |t: u64| seen[t as usize - 1].learning_rate.unwrap();
    assert_eq!(
        rate(1),
        (f64::from(LR) / 3.0) as f32,
        "t = 1 is lr / warmup, not 0"
    );
    assert_eq!(rate(3), LR, "the peak at t = warmup_steps");
    assert_eq!(rate(4), (f64::from(LR) * 4.0 / 5.0) as f32);
    assert_eq!(rate(7), (f64::from(LR) / 5.0) as f32);
    assert_eq!(rate(8), 0.0, "0 at the last step");
    assert!(
        seen.iter()
            .filter_map(|s| s.learning_rate)
            .all(|r| r >= 0.0)
    );
    for (id, before) in &seen[steps as usize - 1].values {
        let after = &run.state.values[id];
        assert_eq!(
            before
                .data()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            after.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "{id}: the last update at lr_t = 0 changes nothing"
        );
    }
}

/// Hypothesis 2. `steps = 1, warmup_steps = 0`: no division by zero, and the one update at
/// `lr_t = lr * (1 - 1) / 1 = 0` leaves the parameters as drawn.
#[test]
fn adamw_one_step_without_warmup_trains_nothing() {
    let c = compiled(&regression(&adamw(0), 1));
    let (seen, run) = observe(&c);
    assert_eq!(seen[0].learning_rate, Some(0.0));
    assert_eq!(run.state.values, seen[0].values);
}

/// Hypothesis 2. A run longer than the `train` block (`--steps 20` on `steps = 10`) decays over
/// its own 20 steps and never goes negative; a run shortened to its warm-up is refused before
/// the first step, and one a step longer runs.
#[test]
fn adamw_steps_override_uses_the_runs_total() {
    let mut c = compiled(&regression(&adamw(3), 10));
    c.config.steps = 20;
    let (seen, _) = observe(&c);
    let rates: Vec<f32> = seen.iter().filter_map(|s| s.learning_rate).collect();
    assert_eq!(rates.len(), 20);
    assert!(rates.iter().all(|&r| r >= 0.0), "{rates:?}");
    assert_eq!(rates[3], lr_t(LR, 4, 3, 20));
    assert_eq!(rates[19], 0.0);

    c.config.steps = 3;
    match run_with(&c, None, &BTreeMap::new(), |_| {}) {
        Err(RunError::Schedule {
            warmup_steps: 3,
            steps: 3,
        }) => {}
        other => panic!("{:?}", other.map(|r| r.losses)),
    }
    c.config.steps = 4;
    assert!(run_with(&c, None, &BTreeMap::new(), |_| {}).is_ok());
}

/// Hypothesis 6. Two runs with one seed reach the same state, moments and versions included.
#[test]
fn adamw_two_runs_reach_one_state() {
    let c = compiled(&regression(&adamw(2), 10));
    let (_, first) = observe(&c);
    let (_, second) = observe(&c);
    assert_eq!(first.state, second.state);
    assert_eq!(first.losses, second.losses);
}

/// Hypothesis 3. A checkpoint file whose optimizer breaks the invariant of `OptimizerConfig`
/// (AdamW without a schedule, SGD with one) is refused with a code when read, not a panic.
#[test]
fn adamw_checkpoint_optimizer_invariant_is_refused_on_read() {
    let c = compiled(&regression("sgd(lr = 0.01)", 4));
    let (_, run) = observe(&c);
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("adversary_adamw");
    std::fs::create_dir_all(&dir).unwrap();
    let schedule = modelspec_runtime::train::checkpoint::LearningRateSchedule { warmup_steps: 2 };
    let configs = [
        OptimizerConfig {
            kind: OptimizerKind::AdamW,
            learning_rate: 0.01,
            weight_decay: Some(0.1),
            beta1: Some(0.9),
            beta2: Some(0.999),
            epsilon: Some(1e-8),
            schedule: None,
        },
        OptimizerConfig {
            kind: OptimizerKind::Sgd,
            learning_rate: 0.01,
            weight_decay: None,
            beta1: None,
            beta2: None,
            epsilon: None,
            schedule: Some(schedule),
        },
    ];
    for (k, optimizer) in configs.into_iter().enumerate() {
        let mut checkpoint = Checkpoint::of_state(&c, 4, &run.state.values);
        checkpoint.optimizer = optimizer;
        let path = dir.join(format!("invariant-{k}.msckpt"));
        std::fs::write(&path, checkpoint.encode()).unwrap();
        let refused = modelspec_runtime::train::checkpoint::read(&path)
            .expect_err("an optimizer that breaks the invariant is refused");
        assert!(refused.code().is_some(), "{refused}");
    }
}

/// Finding F2. The optimizer record of a checkpoint of an AdamW file says SGD:
/// `OptimizerConfig::of` was not taught AdamW, so `Checkpoint::of_state` of an AdamW run records
/// `kind: Sgd` without its hyperparameters or schedule, and `resume_into` an AdamW file accepts
/// an SGD checkpoint of the same learning rate, then trains with moments restarted at 0.
#[test]
fn adv_adamw_f2_checkpoint_optimizer_record_of_an_adamw_file() {
    let adamw_file = compiled(&regression(&adamw(2), 10));
    let recorded = OptimizerConfig::of(&adamw_file);
    let sgd_file = compiled(&regression(&format!("sgd(lr = {LR:?})"), 10));
    let (_, sgd_run) = observe(&sgd_file);
    let sgd_checkpoint = Checkpoint::of_state(&sgd_file, 10, &sgd_run.state.values);
    let resumed = sgd_checkpoint.resume_into(&adamw_file);
    assert_eq!(
        (
            recorded.kind,
            recorded.schedule.map(|s| s.warmup_steps),
            resumed
                .as_ref()
                .err()
                .and_then(|e| e.code())
                .map(|c| c.as_str()),
        ),
        (OptimizerKind::AdamW, Some(2), Some("E-CKPT-003")),
        "OptimizerConfig::of an adamw file: {recorded:?}; resume_into it of an SGD checkpoint \
         at lr {LR}: {}",
        if resumed.is_ok() {
            "accepted"
        } else {
            "refused"
        }
    );
}

/// Finding F3. An AdamW file whose loss reads no parameter fails at step 0, while the same file
/// with `sgd` trains: the step has no host inputs (`Optimizer::step_inputs` is `None` without an
/// update), but `PreparedStep::evaluate` supplies `optimizer.lr_t`, `c1` and `c2` anyway and the
/// interpreter refuses a value for an input the program does not have.
#[test]
fn adv_adamw_f3_a_step_with_no_update_runs() {
    let identity = |optimizer: &str| {
        format!(
            "model Id {{
    param u: Tensor<f32, [4, 1]> = ones();

    fn forward(x: Tensor<f32, [8, 1]>) -> Tensor<f32, [8, 1]> {{
        return x;
    }}
}}

train Fit {{
    model = Id;
    input x: Tensor<f32, [8, 1]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = {optimizer};
    steps = 3;
    seed = 42;
    device = cpu;
}}
"
        )
    };
    let sgd = compiled(&identity("sgd(lr = 0.05)"));
    assert!(run_with(&sgd, None, &BTreeMap::new(), |_| {}).is_ok());
    let c = compiled(&identity(&adamw(1)));
    assert!(c.training.optimizer.step_inputs.is_none());
    let result = run_with(&c, None, &BTreeMap::new(), |_| {});
    assert!(
        result.is_ok(),
        "an AdamW run with nothing to update: {}",
        result.err().unwrap()
    );
}

/// Finding F4. A gradient whose scaled square overflows `f32` makes `v` infinite; the update is
/// then `m * c1 / inf = 0` plus the decay, the parameter stays finite, and the run finishes with
/// no diagnostic while its optimizer state is not finite (with `weight_decay = 0` the
/// parameters never move again). E-TRAIN-001 checks the loss and the parameters only.
#[test]
fn adv_adamw_f4_an_infinite_second_moment_is_diagnosed() {
    let source = regression(
        "adamw(lr = 0.01, weight_decay = 0.0, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, warmup_steps = 2)",
        10,
    )
    .replace(
        "return add_rows(matmul(x, w), b);",
        "return mul(add_rows(matmul(x, w), b), 1.0e15);",
    );
    let c = compiled(&source);
    match run_with(&c, None, &BTreeMap::new(), |_| {}) {
        Err(RunError::NonFinite(diverged)) => {
            let text = diverged.diagnostic.to_string();
            assert!(text.contains("Reg.w") || text.contains("Reg.b"), "{text}");
        }
        Ok(run) => {
            let infinite: Vec<String> = run
                .state
                .moments
                .iter()
                .filter(|(_, v)| v.data().iter().any(|x| !x.is_finite()))
                .map(|((id, moment), _)| format!("{moment} of {id}"))
                .collect();
            panic!(
                "the run finished with losses {:?} and non-finite moments {infinite:?}",
                run.losses
            );
        }
        Err(other) => panic!("{other}"),
    }
}
