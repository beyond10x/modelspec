//! story:jepa-sigreg-vulkan on a Vulkan device, DESIGN.md § 9 gate 5: `SigregColumns` and
//! `SigregColumnsGrad` run once per shape `P: [N, M]`, `P = z · proj` with `z: [N, 8]` drawn and
//! `proj: [8, M]` a site's projection, for `N ∈ {1, 63, 64, 65, 256}` and
//! `M ∈ {1, 63, 64, 65, 1024}`; every element of each result is within `GPU_ATOL + GPU_RTOL · r`
//! of the CPU interpreter's (`tests/common/mod.rs`), with the guard words after the output
//! unchanged and no validation message. The test prints, per op, the largest differences and
//! whether every element was the CPU's bits; bit-equality is recorded, not required.
//!
//! `r` is the sum of the absolute values of the element's terms, each weighted by how much the
//! element moves with it (the reduction rule of gate 5): for `T_m = N Σ_j c_j (S_j² + d_j²)` the
//! `N c_j (S_j² + d_j²)` of its final sum and, through the means `S_j = Σ_n s_nj / N` and
//! `C_j = Σ_n c_nj / N` (`d_j = C_j - phi_j`), `2 c_j (|S_j| Σ_n |s_nj| + |d_j| Σ_n |c_nj|)`; for
//! the gradient element `2 g Σ_j c_j t_j (S_j c - d_j s)`, `|2 g| Σ_j c_j t_j (|S_j| |c| + |d_j| |s|)`.
//!
//! Ignored by default: it creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_sigreg -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

use common::{Draw, GPU_ATOL, GPU_RTOL, GUARD_WORDS, guard, guarded, within};
use modelspec_core::ir::kernel::{Kernel, Operand};
use modelspec_core::ir::tensor::{Dim, TensorOp, TensorProgramBuilder, TensorType};
use modelspec_core::schedule::rowwise_kernel;
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::{HostTensor, sin_cos};
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::train::init::sigreg_projection;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// The site whose projections the cases draw.
const SITE: &str = "Spread.train.sigreg#0";

/// The run seed of the projections.
const SEED: u64 = 5;

/// The width `K` of `z`, the rows of a projection.
const K: u64 = 8;

const ROWS: [u64; 5] = [1, 63, 64, 65, 256];
const COLUMNS: [u64; 5] = [1, 63, 64, 65, 1024];

/// The knots of the quadrature in `f64`: `t_j`, `phi_j` and the weight `c_j`, `j = 1..=8`
/// (`crate::cpu::interpreter`'s `sigreg_knot`).
fn knots() -> [(f64, f64, f64); 8] {
    std::array::from_fn(|k| {
        let t = 0.625 * (k + 1) as f64;
        let phi = (-0.5 * t * t).exp();
        let share = if k == 7 { 1.0 } else { 2.0 };
        (t, phi, share * 0.625 * phi)
    })
}

/// `P = z · proj` of `z: [n, 8]` and `proj: [8, m]` in `f32`, each element's sum of products
/// from its first product, each later one fused.
fn project(z: &[f32], proj: &[f32], n: usize, m: usize) -> Vec<f32> {
    let k = K as usize;
    let mut p = Vec::with_capacity(n * m);
    for row in 0..n {
        for column in 0..m {
            let mut sum = z[row * k] * proj[column];
            for i in 1..k {
                sum = z[row * k + i].mul_add(proj[i * m + column], sum);
            }
            p.push(sum);
        }
    }
    p
}

/// Per knot of column `column` of `P: [n, m]`: the means `(C_j, S_j)` and the sums of
/// `|cos(t_j x)|` and `|sin(t_j x)|`, in `f64` from the `f32` `sin_cos` of the CPU.
fn column_moments(p: &[f32], n: usize, m: usize, column: usize) -> [(f64, f64, f64, f64); 8] {
    let knots = knots();
    std::array::from_fn(|k| {
        let t = knots[k].0 as f32;
        let (mut cos, mut sin, mut abs_cos, mut abs_sin) = (0.0, 0.0, 0.0, 0.0);
        for row in 0..n {
            let (s, c) = sin_cos(t * p[row * m + column]);
            sin += f64::from(s);
            cos += f64::from(c);
            abs_sin += f64::from(s).abs();
            abs_cos += f64::from(c).abs();
        }
        (cos / n as f64, sin / n as f64, abs_cos, abs_sin)
    })
}

/// The reference `r` of each `SigregColumns` element (module documentation).
fn columns_reference(p: &[f32], n: usize, m: usize) -> Vec<f32> {
    let knots = knots();
    (0..m)
        .map(|column| {
            let moments = column_moments(p, n, m, column);
            let mut r = 0.0;
            for (&(_, phi, weight), &(cos, sin, abs_cos, abs_sin)) in knots.iter().zip(&moments) {
                let d = cos - phi;
                r += n as f64 * weight * (sin * sin + d * d);
                r += 2.0 * weight * (sin.abs() * abs_sin + d.abs() * abs_cos);
            }
            r as f32
        })
        .collect()
}

/// The reference `r` of each `SigregColumnsGrad` element (module documentation).
fn grad_reference(p: &[f32], g: &[f32], n: usize, m: usize) -> Vec<f32> {
    let knots = knots();
    let moments: Vec<_> = (0..m)
        .map(|column| column_moments(p, n, m, column))
        .collect();
    p.iter()
        .enumerate()
        .map(|(i, &x)| {
            let column = i % m;
            let mut r = 0.0;
            for (&(t, phi, weight), &(cos, sin, _, _)) in knots.iter().zip(&moments[column]) {
                let (s, c) = sin_cos(t as f32 * x);
                let d = cos - phi;
                r += weight * t * (sin.abs() * f64::from(c).abs() + d.abs() * f64::from(s).abs());
            }
            (2.0 * f64::from(g[column]).abs() * r) as f32
        })
        .collect()
}

/// One kernel run: the op, its `f32` operands with their names and types, and the result type.
struct Case {
    op: TensorOp,
    operands: Vec<(String, TensorType, Vec<f32>)>,
    result: TensorType,
}

/// The CPU interpreter's result of `case`: a program of its one op over its operands as inputs.
fn cpu(case: &Case) -> Vec<f32> {
    let mut builder = TensorProgramBuilder::new();
    let mut inputs = BTreeMap::new();
    let mut values = Vec::new();
    for (name, ty, data) in &case.operands {
        let shape = ty
            .shape
            .iter()
            .map(|dim| match dim {
                Dim::Static(n) => *n,
                Dim::Symbol(_) => unreachable!("a test type is static"),
            })
            .collect();
        inputs.insert(name.clone(), HostTensor::new(shape, data.clone()).unwrap());
        values.push(
            builder
                .input(name.clone(), ty.clone())
                .expect("a new input"),
        );
    }
    let result = match case.op {
        TensorOp::SigregColumns => builder.sigreg_columns(values[0]),
        TensorOp::SigregColumnsGrad => builder.sigreg_columns_grad(values[0], values[1]),
        ref other => unreachable!("{} is no sigreg case", other.name()),
    }
    .expect("the case is well typed");
    builder.output(result).expect("the result is a value");
    let program = builder.finish();
    evaluate_seeded(&program, &inputs, &BTreeMap::new(), &BTreeMap::new(), SEED)
        .expect("the CPU evaluates")[0]
        .data()
        .to_vec()
}

/// The kernel of `case`, as the schedule builds it.
fn kernel_of(case: &Case) -> Kernel {
    let operands = case
        .operands
        .iter()
        .map(|(_, ty, _)| Operand::Buffer(ty.clone()))
        .collect();
    rowwise_kernel(&case.op, operands, case.result.clone())
        .unwrap_or_else(|| panic!("{} has a per-column kernel", case.op.name()))
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// Runs the kernel of `case` once on `device`: uploads its operands and the guarded output,
/// dispatches, reads back, in one command buffer with one host wait. Returns the output with
/// its guard words.
fn gpu(device: &Device<'_>, case: &Case, elements: usize) -> Vec<f32> {
    let kernel = kernel_of(case);
    let words = emit(&kernel).expect("the kernel emits");
    let pipeline = device
        .kernel_pipeline(&kernel, &words)
        .expect("the pipeline is created");
    let mut uploads: Vec<Vec<f32>> = case.operands.iter().map(|(_, _, d)| d.clone()).collect();
    uploads.push(guarded(&vec![f32::from_bits(UNWRITTEN); elements]));
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|data| {
            device
                .device_local_buffer(bytes_of(data.len()))
                .expect("a buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|data| {
            let mut buffer = device
                .staging_buffer(bytes_of(data.len()))
                .expect("a staging buffer");
            buffer
                .write_f32(data)
                .expect("the staging buffer is written");
            buffer
        })
        .collect();
    let output = buffers.len() - 1;
    let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    recording
        .dispatch_kernel(&pipeline, &kernel, &bound)
        .expect("the dispatch is recorded");
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[output], &staging[output])
        .expect("the download is recorded");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[output].read_f32().expect("the output reads")
}

fn open() -> Vulkan {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1"),
        "the validation layer is on exactly when {VALIDATION_ENV}=1"
    );
    vulkan
}

/// What one op showed over every shape.
struct Observed {
    op: &'static str,
    elements: usize,
    max_abs: f64,
    max_rel: f64,
    max_bound_share: f64,
    bits_equal: bool,
    outside: Vec<String>,
}

impl Observed {
    fn new(op: &'static str) -> Self {
        Self {
            op,
            elements: 0,
            max_abs: 0.0,
            max_rel: 0.0,
            max_bound_share: 0.0,
            bits_equal: true,
            outside: Vec::new(),
        }
    }

    /// Compares `got`, the output with its guard words, with the CPU's `expected` under the
    /// references `r`.
    fn compare(&mut self, shape: (u64, u64), got: &[f32], expected: &[f32], r: &[f32]) {
        let (n, m) = shape;
        let count = expected.len();
        self.elements += count;
        for (i, ((&g, &c), &r)) in got.iter().zip(expected).zip(r).enumerate() {
            let error = (f64::from(g) - f64::from(c)).abs();
            if g.to_bits() != c.to_bits() {
                self.bits_equal = false;
            }
            self.max_abs = self.max_abs.max(error);
            if c != 0.0 {
                self.max_rel = self.max_rel.max(error / f64::from(c).abs());
            }
            let bound = f64::from(GPU_ATOL) + f64::from(GPU_RTOL) * f64::from(r);
            self.max_bound_share = self.max_bound_share.max(error / bound);
            if !within(g, c, r, GPU_ATOL, GPU_RTOL) && self.outside.len() < 4 {
                self.outside.push(format!(
                    "{} [{n}, {m}] [{i}]: gpu {g:?} cpu {c:?} r {r:?}",
                    self.op
                ));
            }
        }
        for i in 0..GUARD_WORDS {
            if got[count + i].to_bits() != guard(i) {
                self.outside
                    .push(format!("{} [{n}, {m}]: guard word {i} changed", self.op));
            }
        }
    }
}

#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_sigreg_kernels_match_the_cpu_interpreter_within_tolerance_with_guard_words_unchanged() {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut draw = Draw::new(0x7369_6772_6567_2133);
    let mut columns = Observed::new("SigregColumns");
    let mut grads = Observed::new("SigregColumnsGrad");
    let mut messages = Vec::new();
    let mut shapes = 0;
    for n in ROWS {
        for m in COLUMNS {
            let z: Vec<f32> = draw.values(n * K).iter().map(|v| v * 0.5).collect();
            let projection = sigreg_projection(SEED, SITE, n + m, &TensorType::f32(&[K, m]))
                .expect("the projection draws");
            let p = project(&z, projection.data(), n as usize, m as usize);
            let g = draw.values(m);
            let p_type = TensorType::f32(&[n, m]);
            let g_type = TensorType::f32(&[m]);
            let forward = Case {
                op: TensorOp::SigregColumns,
                operands: vec![("p".to_owned(), p_type.clone(), p.clone())],
                result: g_type.clone(),
            };
            let backward = Case {
                op: TensorOp::SigregColumnsGrad,
                operands: vec![
                    ("p".to_owned(), p_type.clone(), p.clone()),
                    ("g".to_owned(), g_type, g.clone()),
                ],
                result: p_type,
            };
            let (nu, mu) = (n as usize, m as usize);
            let expected = cpu(&forward);
            let got = gpu(&device, &forward, mu);
            columns.compare((n, m), &got, &expected, &columns_reference(&p, nu, mu));
            for message in vulkan.take_validation_messages() {
                messages.push(format!("SigregColumns [{n}, {m}]: {message}"));
            }
            let expected = cpu(&backward);
            let got = gpu(&device, &backward, nu * mu);
            grads.compare((n, m), &got, &expected, &grad_reference(&p, &g, nu, mu));
            for message in vulkan.take_validation_messages() {
                messages.push(format!("SigregColumnsGrad [{n}, {m}]: {message}"));
            }
            shapes += 1;
        }
    }
    eprintln!("op | shapes | elements | max |g-c| | max rel | max |g-c| / bound | bits equal");
    for seen in [&columns, &grads] {
        eprintln!(
            "{} | {shapes} | {} | {:.3e} | {:.3e} | {:.3e} | {}",
            seen.op,
            seen.elements,
            seen.max_abs,
            seen.max_rel,
            seen.max_bound_share,
            if seen.bits_equal { "yes" } else { "no" }
        );
    }
    assert_eq!(shapes, ROWS.len() * COLUMNS.len());
    let outside: Vec<&String> = columns.outside.iter().chain(&grads.outside).collect();
    assert!(outside.is_empty(), "{outside:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
}
