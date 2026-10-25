//! Muon, orthogonalized momentum for the matrix parameters, on the CPU reference (DESIGN.md
//! § 2.1, § 6, § 8; story:muon-optimizer, scenarios 1 and 2).
//!
//! `optimizer = muon(lr = F, momentum = F, weight_decay = F, ns_steps = N, warmup_steps = N,
//! adam_lr = F, beta1 = F, beta2 = F, eps = F)` compiles the update into the training step's
//! `Optimizer` subgraph. A parameter of rank 2 that no `embedding` reads takes, at step `t`
//! counted from 1, with gradient `g` and momentum buffer `b` starting at 0:
//!
//! ```text
//! b = momentum * b + g
//! u = g + momentum * b
//! x = u / (sqrt(sum(u * u)) + 1e-7), transposed when d_out > d_in
//! ns_steps times: a = x xᵀ; x = 3.4445 x + (-4.7750 a + 2.0315 a a) x
//! x transposed back
//! p = p - lr_t * (0.2 * sqrt(max(d_out, d_in)) * x + weight_decay * p)
//! ```
//!
//! every other parameter with a gradient the AdamW update of story:bert-adamw-warmup with
//! `adam_lr`, `beta1`, `beta2`, `eps` and the same `weight_decay`. `lr_t` is the warm-up schedule
//! of coordinator decision D50 applied to `lr` for the Muon parameters and to `adam_lr` for the
//! AdamW ones (coordinator decision D67 (d), 2026-10-25, `.engineering/waves/bert-1.md`). The
//! reference below rounds every elementwise operation once in `f32` in the written order, sums
//! `u * u` from the first element in order, computes each matmul element as the interpreter's
//! fused chain in ascending `k`, and takes the host constants (`lr_t`, `c1`, `c2`, each
//! `1 - beta` and the scale) in `f64` rounded once.

use std::collections::BTreeMap;

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::ParamId;
use modelspec_core::ir::training::Moment;
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::runner::run_with;

const LR: f32 = 0.02;
const MOMENTUM: f32 = 0.95;
const WEIGHT_DECAY: f32 = 0.1;
const ADAM_LR: f32 = 0.01;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

/// `[4, 3]` weight `w` and `[3]` bias `b`, trained with Muon.
fn two_parameters(warmup_steps: u64, steps: u64) -> String {
    format!(
        "model Reg {{
    param w: Tensor<f32, [4, 3]> = xavier_uniform();
    param b: Tensor<f32, [3]> = zeros();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {{
        return add_rows(matmul(x, w), b);
    }}
}}

train Fit {{
    model = Reg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = muon(lr = {LR:?}, momentum = {MOMENTUM:?}, weight_decay = {WEIGHT_DECAY:?}, ns_steps = 5, warmup_steps = {warmup_steps}, adam_lr = {ADAM_LR:?}, beta1 = {BETA1:?}, beta2 = {BETA2:?}, eps = {EPS:e});
    steps = {steps};
    seed = 7;
    device = cpu;
}}
"
    )
}

fn compiled(source: &str) -> CompiledTraining {
    compile_training("muon.mspec", source).unwrap_or_else(|d| panic!("compiles: {d:?}"))
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

/// The settled schedule of D50 at step `t` for the peak rate `lr`, in `f64`.
fn settled(lr: f32, t: u64, warmup_steps: u64, steps: u64) -> f64 {
    let lr = f64::from(lr);
    if t <= warmup_steps {
        lr * t as f64 / warmup_steps as f64
    } else {
        lr * (steps - t) as f64 / (steps - warmup_steps) as f64
    }
}

/// `[m, k] x [k, n]` as the interpreter computes it: each element the `k = 0` product, then each
/// later product fused into the sum, in ascending `k`.
fn matmul(a: &[f32], b: &[f32], (m, k, n): (usize, usize, usize)) -> Vec<f32> {
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

fn transpose(a: &[f32], (m, n): (usize, usize)) -> Vec<f32> {
    let mut out = Vec::with_capacity(m * n);
    for j in 0..n {
        for i in 0..m {
            out.push(a[i * n + j]);
        }
    }
    out
}

/// The orthogonalized update `x` of `u` of shape `[rows, cols]`, before the scale, every
/// operation one rounded `f32` operation in the written order.
fn newton_schulz(u: &[f32], (rows, cols): (usize, usize), ns_steps: usize) -> Vec<f32> {
    let mut squares = u.iter().map(|&e| e * e);
    let first = squares.next().expect("a matrix");
    let sum = squares.fold(first, |acc, s| acc + s);
    let norm = sum.sqrt();
    let denominator = norm + 1e-7f32;
    let mut x: Vec<f32> = u.iter().map(|&e| e / denominator).collect();
    let transposed = rows > cols;
    let (r, c) = if transposed {
        x = transpose(&x, (rows, cols));
        (cols, rows)
    } else {
        (rows, cols)
    };
    for _ in 0..ns_steps {
        let xt = transpose(&x, (r, c));
        let a = matmul(&x, &xt, (r, c, r));
        let aa = matmul(&a, &a, (r, r, r));
        let ca: Vec<f32> = aa.iter().map(|&e| 2.0315f32 * e).collect();
        let ba: Vec<f32> = a.iter().map(|&e| -4.7750f32 * e).collect();
        let s: Vec<f32> = ba.iter().zip(&ca).map(|(&p, &q)| p + q).collect();
        let sx = matmul(&s, &x, (r, r, c));
        x = x
            .iter()
            .zip(&sx)
            .map(|(&e, &f)| 3.4445f32 * e + f)
            .collect();
    }
    if transposed {
        x = transpose(&x, (r, c));
    }
    x
}

/// `0.2 * sqrt(max(d_out, d_in))` in `f64`, rounded once.
fn scale(rows: usize, cols: usize) -> f32 {
    (0.2 * (rows.max(cols) as f64).sqrt()) as f32
}

/// One Muon update of `p` of shape `[rows, cols]` with gradient `g` and buffer `b`; returns
/// `(p', b')`.
fn muon_reference(
    p: &[f32],
    g: &[f32],
    b: &[f32],
    shape: (usize, usize),
    lr_t: f32,
) -> (Vec<f32>, Vec<f32>) {
    let b_next: Vec<f32> = b.iter().zip(g).map(|(&b, &g)| MOMENTUM * b + g).collect();
    let u: Vec<f32> = g
        .iter()
        .zip(&b_next)
        .map(|(&g, &b)| g + MOMENTUM * b)
        .collect();
    let x = newton_schulz(&u, shape, 5);
    let scale = scale(shape.0, shape.1);
    let p_next = p
        .iter()
        .zip(&x)
        .map(|(&p, &x)| {
            let scaled = scale * x;
            let shrink = WEIGHT_DECAY * p;
            let step = scaled + shrink;
            p - lr_t * step
        })
        .collect();
    (p_next, b_next)
}

/// One AdamW update of one element with `adam_lr`'s host constants; returns `(p', m', v')`.
fn adamw_reference(p: f32, g: f32, m: f32, v: f32, (lr_t, c1, c2): (f32, f32, f32)) -> [f32; 3] {
    let rest1 = (1.0 - f64::from(BETA1)) as f32;
    let rest2 = (1.0 - f64::from(BETA2)) as f32;
    let m = BETA1 * m + rest1 * g;
    let v = BETA2 * v + (rest2 * g) * g;
    let direction = (m * c1) / ((v * c2).sqrt() + EPS);
    let step = direction + WEIGHT_DECAY * p;
    [p - lr_t * step, m, v]
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Scenario 1. Steps 1, 2 and 3 of `muon(...)` on a `[4, 3]` weight and a `[3]` bias are
/// bit-equal to the `f32` reference: the weight took the Muon update with `lr`'s schedule and
/// its momentum buffer, the bias the AdamW update with `adam_lr`'s; the observer sees the Muon
/// rate, and the run ends with the buffer of the reference as `b` of the weight.
#[test]
fn one_step() {
    let (warmup_steps, steps) = (2, 10);
    let c = compiled(&two_parameters(warmup_steps, steps));
    let (w, bias) = (param(&c, "Reg.w"), param(&c, "Reg.b"));
    let mut seen = Vec::new();
    let run = run_with(&c, None, &BTreeMap::new(), |o| {
        seen.push(Seen {
            values: o.values.clone(),
            gradients: o.gradients.cloned(),
            learning_rate: o.learning_rate,
        })
    })
    .expect("the run finishes");
    assert_eq!(seen.len(), steps as usize + 1);
    let mut buffer = vec![0.0f32; 12];
    let (mut m, mut v) = (vec![0.0f32; 3], vec![0.0f32; 3]);
    for t in 1..=3u64 {
        let (before, after) = (&seen[t as usize - 1], &seen[t as usize]);
        let gradients = before.gradients.as_ref().expect("an update");
        let lr_t = settled(LR, t, warmup_steps, steps) as f32;
        assert_eq!(before.learning_rate, Some(lr_t), "step {t}: the Muon rate");

        let (expected, b_next) = muon_reference(
            before.values[&w].data(),
            gradients[&w].data(),
            &buffer,
            (4, 3),
            lr_t,
        );
        buffer = b_next;
        assert_eq!(
            bits(after.values[&w].data()),
            bits(&expected),
            "step {t}: w {:?} against {expected:?}",
            after.values[&w].data()
        );

        let correction = |beta: f32| (1.0 / (1.0 - f64::from(beta).powi(t as i32))) as f32;
        let constants = (
            settled(ADAM_LR, t, warmup_steps, steps) as f32,
            correction(BETA1),
            correction(BETA2),
        );
        let old = before.values[&bias].data();
        let g = gradients[&bias].data();
        for i in 0..3 {
            let [p, m_next, v_next] = adamw_reference(old[i], g[i], m[i], v[i], constants);
            (m[i], v[i]) = (m_next, v_next);
            assert_eq!(
                after.values[&bias].data()[i].to_bits(),
                p.to_bits(),
                "step {t}: bias element {i}"
            );
        }
        for p in [w, bias] {
            let moved = before.values[&p]
                .data()
                .iter()
                .zip(after.values[&p].data())
                .all(|(old, new)| old != new);
            assert!(moved, "step {t} moves every element of {p}");
        }
    }
    let mut state: Vec<(ParamId, Moment)> = run.state.moments.keys().copied().collect();
    state.sort();
    assert_eq!(
        state,
        [
            (w, Moment::Momentum),
            (bias, Moment::First),
            (bias, Moment::Second)
        ]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>(),
        "the weight keeps a momentum buffer, the bias its two moments"
    );
}

/// The singular values of a symmetric eigenproblem: the square roots of the eigenvalues of
/// `xᵀ x`, by cyclic Jacobi rotations in `f64`.
fn singular_values(x: &[f64], (rows, cols): (usize, usize)) -> Vec<f64> {
    let mut a = vec![0.0f64; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            a[i * cols + j] = (0..rows).map(|r| x[r * cols + i] * x[r * cols + j]).sum();
        }
    }
    for _ in 0..100 {
        let off: f64 = (0..cols)
            .flat_map(|i| (0..cols).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * cols + j] * a[i * cols + j])
            .sum();
        if off < 1e-24 {
            break;
        }
        for p in 0..cols {
            for q in p + 1..cols {
                let apq = a[p * cols + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * cols + q] - a[p * cols + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..cols {
                    let akp = a[k * cols + p];
                    let akq = a[k * cols + q];
                    a[k * cols + p] = c * akp - s * akq;
                    a[k * cols + q] = s * akp + c * akq;
                }
                for k in 0..cols {
                    let apk = a[p * cols + k];
                    let aqk = a[q * cols + k];
                    a[p * cols + k] = c * apk - s * aqk;
                    a[q * cols + k] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..cols).map(|i| a[i * cols + i].max(0.0).sqrt()).collect()
}

/// Scenario 2. For the gradient of a `[64, 32]` weight on random data, every singular value of
/// the orthogonalized update, computed in `f64` from the `f32` result, is within `[0.68, 1.3]` at
/// `ns_steps = 5` (the quintic with the fixed coefficients pulls them into about `[0.68, 1.13]`; the
/// story asked for `[0.7, 1.3]`, coordinator decision D68 takes the measured band), and the
/// update's RMS after the scale is within `[0.16, 0.22]` (measured 0.176; the scale assumes an
/// orthogonal update, and the band below 1 lowers it). The weight
/// starts at 0 and `lr_t` is exactly 1, so the parameter after the step is the negated scaled
/// update, bit for bit.
#[test]
fn newton_schulz_singular_values() {
    let source = "model Wide {
    param w: Tensor<f32, [64, 32]> = xavier_uniform();

    fn forward(x: Tensor<f32, [128, 64]>) -> Tensor<f32, [128, 32]> {
        return matmul(x, w);
    }
}

train Fit {
    model = Wide;
    input x: Tensor<f32, [128, 64]>;
    target y: Tensor<f32, [128, 32]>;
    loss = mse(model.forward(x), y);
    optimizer = muon(lr = 2.0, momentum = 0.95, weight_decay = 0.1, ns_steps = 5, warmup_steps = 0, adam_lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8);
    steps = 2;
    seed = 11;
    device = cpu;
}
";
    let c = compiled(source);
    let w = param(&c, "Wide.w");
    let zeros = HostTensor::new(vec![64, 32], vec![0.0f32; 64 * 32]).expect("a [64, 32] tensor");
    let init = BTreeMap::from([(w, zeros)]);
    let mut seen = Vec::new();
    run_with(&c, None, &init, |o| {
        seen.push((
            o.values[&w].data().to_vec(),
            o.gradients.map(|g| g[&w].data().to_vec()),
            o.learning_rate,
        ))
    })
    .expect("the run finishes");
    let (before, gradient, rate) = &seen[0];
    assert!(before.iter().all(|&e| e == 0.0));
    assert_eq!(*rate, Some(1.0));
    let gradient = gradient.as_ref().expect("an update");
    let after = &seen[1].0;

    let u: Vec<f32> = gradient.iter().map(|&g| g + MOMENTUM * g).collect();
    let x = newton_schulz(&u, (64, 32), 5);
    let scale = scale(64, 32);
    let expected: Vec<f32> = x.iter().map(|&e| -(scale * e)).collect();
    assert_eq!(bits(after), bits(&expected), "the step is the reference");

    let update: Vec<f64> = after.iter().map(|&p| -f64::from(p)).collect();
    let unscaled: Vec<f64> = update.iter().map(|&e| e / f64::from(scale)).collect();
    let sigma = singular_values(&unscaled, (64, 32));
    assert_eq!(sigma.len(), 32);
    let (low, high) = sigma
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), &s| (l.min(s), h.max(s)));
    eprintln!("singular values in [{low}, {high}]");
    let rms = (update.iter().map(|e| e * e).sum::<f64>() / update.len() as f64).sqrt();
    eprintln!("update RMS {rms}");
    assert!((0.16..=0.22).contains(&rms), "update RMS {rms}");
    assert!(
        sigma.iter().all(|s| (0.68..=1.3).contains(s)),
        "singular values {sigma:?}"
    );
}
