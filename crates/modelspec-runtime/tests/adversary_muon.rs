//! Adversary cases for `optimizer = muon(...)` on the CPU reference (story:muon-optimizer,
//! adversary pass 1). The unit's `tests/muon.rs` compares steps 1 to 3 of one tall `[4, 3]`
//! weight; these cases drive the runner against an independent `f32` reference written here from
//! the story's § What this story decides and coordinator decision D67 (d): more steps and the
//! momentum buffer `b`, the wide branch (`rows < cols`, no transposition), two Muon parameters of
//! different shapes, degenerate shapes, a zero and a rank-1 gradient, the schedule's edges and a
//! file whose parameters all take AdamW.

use std::collections::BTreeMap;

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::ParamId;
use modelspec_core::ir::training::{Moment, StateVersion};
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::runner::{Run, RunError, run_with};

const LR: f32 = 0.02;
const MOMENTUM: f32 = 0.95;
const WEIGHT_DECAY: f32 = 0.1;
const ADAM_LR: f32 = 0.01;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

fn muon_line(ns_steps: u32, warmup_steps: u64) -> String {
    format!(
        "muon(lr = {LR:?}, momentum = {MOMENTUM:?}, weight_decay = {WEIGHT_DECAY:?}, ns_steps = {ns_steps}, warmup_steps = {warmup_steps}, adam_lr = {ADAM_LR:?}, beta1 = {BETA1:?}, beta2 = {BETA2:?}, eps = {EPS:e})"
    )
}

/// One model `M` with the given parameter declarations and forward body, trained on synthetic
/// data with `optimizer` for `steps` steps.
fn source(
    params: &str,
    x: &str,
    y: &str,
    body: &str,
    optimizer: &str,
    steps: u64,
    seed: u64,
) -> String {
    format!(
        "model M {{
{params}
    fn forward(x: Tensor<f32, {x}>) -> Tensor<f32, {y}> {{
        return {body};
    }}
}}

train Fit {{
    model = M;
    input x: Tensor<f32, {x}>;
    target y: Tensor<f32, {y}>;
    loss = mse(model.forward(x), y);
    optimizer = {optimizer};
    steps = {steps};
    seed = {seed};
    device = cpu;
}}
"
    )
}

fn compiled(source: &str) -> CompiledTraining {
    compile_training("adversary_muon.mspec", source)
        .unwrap_or_else(|d| panic!("compiles: {d:?}\n{source}"))
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

fn observe(
    compiled: &CompiledTraining,
    init: &BTreeMap<ParamId, HostTensor<f32>>,
) -> (Vec<Seen>, Run) {
    let mut seen = Vec::new();
    let run = run_with(compiled, None, init, |o| {
        seen.push(Seen {
            values: o.values.clone(),
            gradients: o.gradients.cloned(),
            learning_rate: o.learning_rate,
        })
    })
    .unwrap_or_else(|e| panic!("the run finishes: {e}"));
    (seen, run)
}

/// D50 at step `t`, in `f64`, rounded once.
fn lr_t(lr: f32, t: u64, warmup_steps: u64, steps: u64) -> f32 {
    let peak = f64::from(lr);
    let rate = if t <= warmup_steps {
        peak * t as f64 / warmup_steps as f64
    } else {
        peak * (steps - t) as f64 / (steps - warmup_steps) as f64
    };
    rate as f32
}

fn correction(beta: f32, t: u64) -> f32 {
    (1.0 / (1.0 - f64::from(beta).powi(t as i32))) as f32
}

/// `[m, k] x [k, n]`, each element `a[i][0] * b[0][j]` then one fused multiply-add per later `k`,
/// ascending (DESIGN.md § 2.2).
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = a[i * k] * b[j];
            for kk in 1..k {
                acc = a[i * k + kk].mul_add(b[kk * n + j], acc);
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn transpose(a: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(m * n);
    for j in 0..n {
        for i in 0..m {
            out.push(a[i * n + j]);
        }
    }
    out
}

/// The orthogonalized `x` of `u` (`[rows, cols]`), before the scale: the story's update, each
/// operation one rounded `f32` operation in the written order, transposed when `rows > cols`;
/// the sum of the squares adds them in `f64` and rounds once (coordinator decision D84).
fn orthogonalize(u: &[f32], rows: usize, cols: usize, ns_steps: u32) -> Vec<f32> {
    let mut sum = f64::from(u[0] * u[0]);
    for &e in &u[1..] {
        sum += f64::from(e * e);
    }
    let denominator = (sum as f32).sqrt() + 1e-7f32;
    let mut x: Vec<f32> = u.iter().map(|&e| e / denominator).collect();
    let tall = rows > cols;
    let (r, c) = if tall {
        x = transpose(&x, rows, cols);
        (cols, rows)
    } else {
        (rows, cols)
    };
    for _ in 0..ns_steps {
        let gram = matmul(&x, &transpose(&x, r, c), r, c, r);
        let square = matmul(&gram, &gram, r, r, r);
        let poly: Vec<f32> = gram
            .iter()
            .zip(&square)
            .map(|(&a, &aa)| -4.7750f32 * a + 2.0315f32 * aa)
            .collect();
        let applied = matmul(&poly, &x, r, r, c);
        x = x
            .iter()
            .zip(&applied)
            .map(|(&e, &f)| 3.4445f32 * e + f)
            .collect();
    }
    if tall {
        x = transpose(&x, r, c);
    }
    x
}

fn scale(rows: usize, cols: usize) -> f32 {
    (0.2 * (rows.max(cols) as f64).sqrt()) as f32
}

/// One Muon step of `p` with gradient `g` and buffer `b`: `(p', b')`.
fn muon_step(
    p: &[f32],
    g: &[f32],
    b: &[f32],
    (rows, cols): (usize, usize),
    rate: f32,
    ns_steps: u32,
) -> (Vec<f32>, Vec<f32>) {
    let b_next: Vec<f32> = b.iter().zip(g).map(|(&b, &g)| MOMENTUM * b + g).collect();
    let u: Vec<f32> = g
        .iter()
        .zip(&b_next)
        .map(|(&g, &b)| g + MOMENTUM * b)
        .collect();
    let x = orthogonalize(&u, rows, cols, ns_steps);
    let s = scale(rows, cols);
    let p_next = p
        .iter()
        .zip(&x)
        .map(|(&p, &x)| p - rate * (s * x + WEIGHT_DECAY * p))
        .collect();
    (p_next, b_next)
}

/// One AdamW step of one element at `adam_lr`'s constants: `(p', m', v')`.
fn adamw_step(p: f32, g: f32, m: f32, v: f32, (rate, c1, c2): (f32, f32, f32)) -> [f32; 3] {
    let rest1 = (1.0 - f64::from(BETA1)) as f32;
    let rest2 = (1.0 - f64::from(BETA2)) as f32;
    let m = BETA1 * m + rest1 * g;
    let v = BETA2 * v + (rest2 * g) * g;
    let direction = (m * c1) / ((v * c2).sqrt() + EPS);
    [p - rate * (direction + WEIGHT_DECAY * p), m, v]
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The state the reference carries for one parameter.
enum Reference {
    Muon { shape: (usize, usize), b: Vec<f32> },
    AdamW { m: Vec<f32>, v: Vec<f32> },
}

/// Propagates the reference over every step of `seen` from the run's own initial values and
/// gradients and asserts each parameter's bits after each step, then the final state of the run
/// (`b`, `m`, `v` and the versions).
fn assert_reference(
    c: &CompiledTraining,
    seen: &[Seen],
    run: &Run,
    plan: &[(&str, Option<(usize, usize)>)],
    (warmup_steps, steps, ns_steps): (u64, u64, u32),
) {
    let mut state: Vec<(ParamId, Reference)> = plan
        .iter()
        .map(|&(path, muon)| {
            let id = param(c, path);
            let n = seen[0].values[&id].len();
            let reference = match muon {
                Some(shape) => Reference::Muon {
                    shape,
                    b: vec![0.0; n],
                },
                None => Reference::AdamW {
                    m: vec![0.0; n],
                    v: vec![0.0; n],
                },
            };
            (id, reference)
        })
        .collect();
    assert_eq!(seen.len() as u64, steps + 1);
    for t in 1..=steps {
        let (before, after) = (&seen[t as usize - 1], &seen[t as usize]);
        let gradients = before.gradients.as_ref().expect("an update");
        let muon_rate = lr_t(LR, t, warmup_steps, steps);
        let adam = (
            lr_t(ADAM_LR, t, warmup_steps, steps),
            correction(BETA1, t),
            correction(BETA2, t),
        );
        for (id, reference) in &mut state {
            let old = before.values[id].data();
            let g = gradients[id].data();
            let expected = match reference {
                Reference::Muon { shape, b } => {
                    let (p, b_next) = muon_step(old, g, b, *shape, muon_rate, ns_steps);
                    *b = b_next;
                    p
                }
                Reference::AdamW { m, v } => (0..old.len())
                    .map(|i| {
                        let [p, m_next, v_next] = adamw_step(old[i], g[i], m[i], v[i], adam);
                        (m[i], v[i]) = (m_next, v_next);
                        p
                    })
                    .collect(),
            };
            assert_eq!(
                bits(after.values[id].data()),
                bits(&expected),
                "step {t}, parameter {id}: runner {:?} against the reference {expected:?}",
                after.values[id].data()
            );
        }
    }
    for (id, reference) in &state {
        assert_eq!(
            run.state.versions[id],
            StateVersion(steps),
            "version of {id}"
        );
        match reference {
            Reference::Muon { b, .. } => {
                assert_eq!(
                    bits(run.state.moments[&(*id, Moment::Momentum)].data()),
                    bits(b),
                    "the final buffer b of {id}"
                );
                assert!(!run.state.moments.contains_key(&(*id, Moment::First)));
            }
            Reference::AdamW { m, v } => {
                assert_eq!(
                    bits(run.state.moments[&(*id, Moment::First)].data()),
                    bits(m)
                );
                assert_eq!(
                    bits(run.state.moments[&(*id, Moment::Second)].data()),
                    bits(v)
                );
                assert!(!run.state.moments.contains_key(&(*id, Moment::Momentum)));
            }
        }
    }
}

/// Hypothesis 1. The tall `[4, 3]` weight (transposed branch) and the `[3]` bias over all 12
/// steps of a run (steps 1, 2, 3, 10 and the last, whose rate is 0), and the final buffer `b`
/// of the weight and moments of the bias, are the bits of the reference.
#[test]
fn adv_muon_tall_weight_twelve_steps_and_buffer_are_the_reference() {
    let (warmup, steps) = (2, 12);
    let c = compiled(&source(
        "    param w: Tensor<f32, [4, 3]> = xavier_uniform();\n    param b: Tensor<f32, [3]> = zeros();",
        "[8, 4]",
        "[8, 3]",
        "add_rows(matmul(x, w), b)",
        &muon_line(5, warmup),
        steps,
        7,
    ));
    let (seen, run) = observe(&c, &BTreeMap::new());
    assert_reference(
        &c,
        &seen,
        &run,
        &[("M.w", Some((4, 3))), ("M.b", None)],
        (warmup, steps, 5),
    );
}

/// Hypothesis 1. The wide `[3, 4]` weight (no transposition, `x xᵀ` already the smaller Gram
/// matrix), which no unit test runs, and a `[4]` bias, over 12 steps.
#[test]
fn adv_muon_wide_weight_twelve_steps_and_buffer_are_the_reference() {
    let (warmup, steps) = (3, 12);
    let c = compiled(&source(
        "    param w: Tensor<f32, [3, 4]> = xavier_uniform();\n    param b: Tensor<f32, [4]> = zeros();",
        "[8, 3]",
        "[8, 4]",
        "add_rows(matmul(x, w), b)",
        &muon_line(5, warmup),
        steps,
        3,
    ));
    let (seen, run) = observe(&c, &BTreeMap::new());
    assert_reference(
        &c,
        &seen,
        &run,
        &[("M.w", Some((3, 4))), ("M.b", None)],
        (warmup, steps, 5),
    );
}

/// Hypothesis 2. Two Muon parameters of different shapes, a tall `[6, 4]` and a wide `[4, 9]`,
/// each with its own scale (`0.2 * sqrt(6)` and `0.6`) and buffer, over 6 steps.
#[test]
fn adv_muon_two_matrices_of_different_shapes_are_the_reference() {
    let (warmup, steps) = (1, 6);
    let c = compiled(&source(
        "    param w1: Tensor<f32, [6, 4]> = xavier_uniform();\n    param w2: Tensor<f32, [4, 9]> = xavier_uniform();",
        "[8, 6]",
        "[8, 9]",
        "matmul(matmul(x, w1), w2)",
        &muon_line(5, warmup),
        steps,
        5,
    ));
    let (seen, run) = observe(&c, &BTreeMap::new());
    assert_reference(
        &c,
        &seen,
        &run,
        &[("M.w1", Some((6, 4))), ("M.w2", Some((4, 9)))],
        (warmup, steps, 5),
    );
}

/// Hypothesis 3. Degenerate shapes: a `[1, 1]` weight (Gram matrix `[1, 1]`), a `[1, 16]` row
/// and a `[16, 1]` column, at `ns_steps = 1` and `10`, stay finite and are the reference.
#[test]
fn adv_muon_degenerate_shapes_and_ns_step_bounds_are_the_reference() {
    for (rows, cols) in [(1usize, 1usize), (1, 16), (16, 1)] {
        for ns_steps in [1u32, 10] {
            let (warmup, steps) = (1, 4);
            let c = compiled(&source(
                &format!("    param w: Tensor<f32, [{rows}, {cols}]> = xavier_uniform();"),
                &format!("[8, {rows}]"),
                &format!("[8, {cols}]"),
                "matmul(x, w)",
                &muon_line(ns_steps, warmup),
                steps,
                9,
            ));
            let (seen, run) = observe(&c, &BTreeMap::new());
            for s in &seen {
                assert!(
                    s.values
                        .values()
                        .all(|v| v.data().iter().all(|e| e.is_finite()))
                );
            }
            assert_reference(
                &c,
                &seen,
                &run,
                &[("M.w", Some((rows, cols)))],
                (warmup, steps, ns_steps),
            );
        }
    }
}

/// Hypothesis 3. A gradient that is exactly zero (the prediction is multiplied by `0.0`): the
/// norm is 0, `x = 0 / 1e-7 = 0`, Newton-Schulz keeps 0, and the update is the decay alone,
/// `p - lr_t * (0 + weight_decay * p)`, finite, with `b` still 0.
#[test]
fn adv_muon_zero_gradient_is_the_decay_alone() {
    let (warmup, steps) = (1, 4);
    let c = compiled(&source(
        "    param w: Tensor<f32, [4, 3]> = xavier_uniform();",
        "[8, 4]",
        "[8, 3]",
        "mul(matmul(x, w), 0.0)",
        &muon_line(5, warmup),
        steps,
        2,
    ));
    let w = param(&c, "M.w");
    let (seen, run) = observe(&c, &BTreeMap::new());
    for t in 1..=steps {
        let (before, after) = (&seen[t as usize - 1], &seen[t as usize]);
        let g = before.gradients.as_ref().expect("an update")[&w].data();
        assert!(g.iter().all(|&e| e == 0.0), "a zero gradient: {g:?}");
        let rate = lr_t(LR, t, warmup, steps);
        let expected: Vec<f32> = before.values[&w]
            .data()
            .iter()
            .map(|&p| p - rate * (0.0 + WEIGHT_DECAY * p))
            .collect();
        assert_eq!(bits(after.values[&w].data()), bits(&expected), "step {t}");
    }
    let b = run.state.moments[&(w, Moment::Momentum)].data();
    assert!(b.iter().all(|&e| e == 0.0), "{b:?}");
}

/// Hypothesis 3. A batch of one sample gives a rank-1 gradient (an outer product: one nonzero
/// singular value): no NaN, no inf, and the bits of the reference over 5 steps.
#[test]
fn adv_muon_rank_one_gradient_is_the_reference() {
    let (warmup, steps) = (1, 5);
    let c = compiled(&source(
        "    param w: Tensor<f32, [4, 3]> = xavier_uniform();",
        "[1, 4]",
        "[1, 3]",
        "matmul(x, w)",
        &muon_line(5, warmup),
        steps,
        4,
    ));
    let (seen, run) = observe(&c, &BTreeMap::new());
    assert_reference(
        &c,
        &seen,
        &run,
        &[("M.w", Some((4, 3)))],
        (warmup, steps, 5),
    );
}

/// Hypothesis 1. A `[64, 32]` weight whose values span `1e-3` to `1e3`, so its gradient's
/// squares span many orders and the order of `sum(u * u)` decides its bits: three steps are
/// still the reference's, which sums from the first element in ascending order in `f64`.
#[test]
fn adv_muon_wide_dynamic_range_sum_order_is_the_reference() {
    let (warmup, steps) = (1, 3);
    let c = compiled(&source(
        "    param w: Tensor<f32, [64, 32]> = xavier_uniform();",
        "[16, 64]",
        "[16, 32]",
        "matmul(x, w)",
        &muon_line(5, warmup),
        steps,
        13,
    ));
    let w = param(&c, "M.w");
    let magnitudes = [1e-3f32, 1e3, 0.37, 1e-2, 250.0, 1e1, 3e-3, 1e2];
    let data: Vec<f32> = (0..64 * 32)
        .map(|i| {
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            sign * magnitudes[i % magnitudes.len()] * (1.0 + (i % 7) as f32 / 7.0)
        })
        .collect();
    let init = BTreeMap::from([(w, HostTensor::new(vec![64, 32], data).expect("a tensor"))]);
    let (seen, run) = observe(&c, &init);
    let g = seen[0].gradients.as_ref().expect("an update")[&w].data();
    let (low, high) = g
        .iter()
        .filter(|e| **e != 0.0)
        .fold((f32::MAX, 0.0f32), |(l, h), e| {
            (l.min(e.abs()), h.max(e.abs()))
        });
    eprintln!("gradient magnitudes in [{low:e}, {high:e}]");
    assert!(
        high / low > 1e4,
        "a gradient of wide range: [{low:e}, {high:e}]"
    );
    assert_reference(
        &c,
        &seen,
        &run,
        &[("M.w", Some((64, 32)))],
        (warmup, steps, 5),
    );
}

/// Hypothesis 4. The rates at `t = 1`, the last warm-up step, the step after it and the last
/// step; `warmup_steps = 0`; the last update applies `lr_t = 0` and leaves the weight
/// bit-identical while its buffer still advances.
#[test]
fn adv_muon_schedule_edges_and_the_last_update() {
    for (warmup, steps) in [(3u64, 8u64), (0, 5)] {
        let c = compiled(&source(
            "    param w: Tensor<f32, [4, 3]> = xavier_uniform();",
            "[8, 4]",
            "[8, 3]",
            "matmul(x, w)",
            &muon_line(5, warmup),
            steps,
            7,
        ));
        let w = param(&c, "M.w");
        let (seen, run) = observe(&c, &BTreeMap::new());
        let rates: Vec<f32> = seen.iter().filter_map(|s| s.learning_rate).collect();
        let expected: Vec<f32> = (1..=steps).map(|t| lr_t(LR, t, warmup, steps)).collect();
        assert_eq!(rates, expected, "warmup {warmup} of {steps}");
        assert_eq!(*rates.last().unwrap(), 0.0);
        let (last, end) = (&seen[steps as usize - 1], &seen[steps as usize]);
        assert_eq!(bits(last.values[&w].data()), bits(end.values[&w].data()));
        let b = run.state.moments[&(w, Moment::Momentum)].data();
        assert!(b.iter().any(|&e| e != 0.0));
    }
}

/// Hypothesis 4. A run longer than its `train` block decays over its own steps; one shortened
/// to its warm-up is refused before the first step (`RunError::Schedule`), as for AdamW.
#[test]
fn adv_muon_steps_override_uses_the_runs_total() {
    let mut c = compiled(&source(
        "    param w: Tensor<f32, [4, 3]> = xavier_uniform();",
        "[8, 4]",
        "[8, 3]",
        "matmul(x, w)",
        &muon_line(5, 3),
        10,
        7,
    ));
    c.config.steps = 20;
    let (seen, _) = observe(&c, &BTreeMap::new());
    let rates: Vec<f32> = seen.iter().filter_map(|s| s.learning_rate).collect();
    assert_eq!(rates.len(), 20);
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
}

/// Hypothesis 5. Two runs with one seed reach the same state, buffers included.
#[test]
fn adv_muon_two_runs_reach_one_state() {
    let c = compiled(&source(
        "    param w: Tensor<f32, [4, 3]> = xavier_uniform();\n    param b: Tensor<f32, [3]> = zeros();",
        "[8, 4]",
        "[8, 3]",
        "add_rows(matmul(x, w), b)",
        &muon_line(5, 2),
        10,
        7,
    ));
    let (_, first) = observe(&c, &BTreeMap::new());
    let (_, second) = observe(&c, &BTreeMap::new());
    assert_eq!(first.state, second.state);
    assert_eq!(first.losses, second.losses);
}

fn bias_only(optimizer: &str) -> String {
    source(
        "    param b: Tensor<f32, [4]> = zeros();",
        "[8, 4]",
        "[8, 4]",
        "add_rows(x, b)",
        optimizer,
        10,
        7,
    )
}

/// Hypothesis 5. A file with no matrix parameter trains the same under `muon(...)` as under
/// `adamw(...)` at `lr = adam_lr`: the AdamW side is not changed by the presence of the Muon
/// machinery, bit for bit, moments included.
#[test]
fn adv_muon_without_a_matrix_is_adamw_bit_for_bit() {
    let adamw = compiled(&bias_only(&format!(
        "adamw(lr = {ADAM_LR:?}, weight_decay = {WEIGHT_DECAY:?}, beta1 = {BETA1:?}, beta2 = {BETA2:?}, eps = {EPS:e}, warmup_steps = 2)"
    )));
    let muon = compiled(&bias_only(&muon_line(5, 2)));
    assert!(muon.training.optimizer.buffers.is_empty());
    assert!(muon.training.optimizer.muon_lr_t.is_none());
    let (seen_a, run_a) = observe(&adamw, &BTreeMap::new());
    let (seen_m, run_m) = observe(&muon, &BTreeMap::new());
    for (a, m) in seen_a.iter().zip(&seen_m) {
        assert_eq!(a.values, m.values);
    }
    assert_eq!(run_a.state, run_m.state);
}

/// Finding ADV-MUON-3 (note). With no parameter taking Muon, a `muon(...)` run reports the
/// schedule of its Muon `lr` as the run's learning rate, a rate no parameter applies; the
/// parameters move at `adam_lr`'s schedule. Since coordinator decision D69 the run reports the
/// `adam_lr` rate when no parameter takes Muon.
#[test]
fn adv_muon_3_reported_rate_without_a_matrix_is_the_applied_one() {
    let muon = compiled(&bias_only(&muon_line(5, 2)));
    let (seen, _) = observe(&muon, &BTreeMap::new());
    let rates: Vec<f32> = seen.iter().filter_map(|s| s.learning_rate).collect();
    let applied: Vec<f32> = (1..=10).map(|t| lr_t(ADAM_LR, t, 2, 10)).collect();
    assert_eq!(
        rates, applied,
        "reported rates (Muon lr = {LR}) against the rates the AdamW parameters apply \
         (adam_lr = {ADAM_LR})"
    );
}

/// Finding ADV-MUON-4 (note, constructed). A gradient whose square overflows `f32` (the
/// prediction scaled by `1e15`, so the loss stays finite near `1e30` while the gradient is near
/// `1e30`) makes `sum(u * u)` infinite, so `x = u / inf = 0`: the weight moves by its decay
/// alone at every step and the run finishes with no diagnostic, its buffer finite. The AdamW
/// side of the same file stops with E-TRAIN-001 on its infinite second moment (adversary_adamw
/// F4), so Muon was silent where AdamW is diagnosed. Since coordinator decision D69 the norm is
/// E-TRAIN-001 at that step.
#[test]
fn adv_muon_4_an_overflowing_norm_is_not_silent() {
    let (warmup, steps) = (1, 4);
    let c = compiled(&source(
        "    param w: Tensor<f32, [4, 3]> = xavier_uniform();",
        "[8, 4]",
        "[8, 3]",
        "mul(matmul(x, w), 1.0e15)",
        &muon_line(5, warmup),
        steps,
        2,
    ));
    let w = param(&c, "M.w");
    let mut seen = Vec::new();
    let result = run_with(&c, None, &BTreeMap::new(), |o| {
        seen.push((
            o.values[&w].data().to_vec(),
            o.gradients.map(|g| g[&w].data().to_vec()),
            o.learning_rate,
        ))
    });
    match result {
        Err(RunError::NonFinite(_)) => {}
        Err(other) => panic!("{other}"),
        Ok(run) => {
            let (before, gradient, rate) = &seen[0];
            let gradient = gradient.as_ref().expect("an update");
            let largest = gradient.iter().fold(0.0f32, |m, g| m.max(g.abs()));
            let rate = rate.expect("a rate");
            let decay_only: Vec<f32> = before
                .iter()
                .map(|&p| p - rate * (0.0 + WEIGHT_DECAY * p))
                .collect();
            assert_ne!(
                bits(&seen[1].0),
                bits(&decay_only),
                "step 1 is the decay alone: the largest gradient element is {largest:e}, the \
                 losses {:?}, the final buffer finite: {}",
                run.losses,
                run.state.moments[&(w, Moment::Momentum)]
                    .data()
                    .iter()
                    .all(|e| e.is_finite())
            );
        }
    }
}
