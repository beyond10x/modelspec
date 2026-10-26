//! The modules of the attention and cross-entropy kernels (DESIGN.md § 5.2, coordinator
//! decision D73): `Attention`, `AttentionGradQ`, `AttentionGradK` and `AttentionGradV` one
//! invocation per row and head, `CrossEntropyGrad` one per row, and the `CrossEntropy` loss sum
//! one workgroup of 64; and the corrected square root and quotient of the `Sqrt` and `Div`
//! elementwise kernels.
//!
//! Each computes its op as the CPU interpreter does, operation for operation, so the device
//! differs from the CPU only by its `exp`, `log`, division and square root, which Vulkan does
//! not round correctly. Every sum runs in ascending index order: it starts at `-0.0` and adds
//! each term, so its first term is its value exactly, and a sum of products fuses each product
//! into it by one `Fma`, so its first product is rounded once, as the CPU's `sum_of_products`.
//! Every float add, subtract, multiply, division and `Fma` is decorated `NoContraction` outside
//! `fast_math`.
//!
//! Invocation `r · heads + a` of an attention kernel computes head `a` of row `r`, the head's
//! `H / heads` columns of the row: a query row for `Attention` and `AttentionGradQ`, a key row
//! for `AttentionGradK` and `AttentionGradV`. Row `r` is token `r % S` of sequence `r / S`, with
//! `S` at most `modelspec_core::schedule::MAX_ATTENTION_KEYS`, and a key is masked where its
//! mask entry is 0. The scores, the softmax maximum and sum and the row dot of the gradient are
//! recomputed where they are used, from the same operands by the same instructions, so they
//! hold the same value at every use; nothing is staged between invocations. The softmax
//! maximum starts at the first unmasked key's score and is replaced only by a greater one. The
//! output rows are summed in place: `Attention` and `AttentionGradQ` over the unmasked keys of
//! their query, `AttentionGradK` and `AttentionGradV` over every query of their key's sequence,
//! with no atomics; a masked key's row of `AttentionGradK` and `AttentionGradV` is `+0`.
//!
//! The `CrossEntropy` workgroup computes the row losses `max + ln(sum) - logits[label]` of 64
//! rows at a time into workgroup memory, invocation `i` row `i` of the block, and after a
//! barrier invocation 0 adds the block's weighted losses and weights to the numerator and
//! denominator in ascending row order; the loss is `+0` when the denominator is 0, else their
//! quotient. A label outside the classes reads class 0; the CPU refuses one before it runs.
//! `CrossEntropyGrad` row `n` is `(g · w[n] / sum(w)) · (p - [class == label])` per class, `+0`
//! where the weight sum is 0.

use super::{EmitError, Kernel, KernelKind, Operand, Shell, Word, operands_error, static_shape};
use modelspec_core::ir::tensor::DType;
use modelspec_core::schedule::MAX_ATTENTION_KEYS;
use rspirv::spirv::GlslStd450Op;

/// The workgroup size of every attention and cross-entropy kernel.
const GROUP: u32 = 64;

/// Which attention op a module computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AttentionOp {
    Forward,
    GradQ,
    GradK,
    GradV,
}

/// The shapes of an attention kernel: `S` keys per sequence, `H` columns per row, `d = H /
/// heads` per head, and the scale `1 / sqrt(d)` as the CPU computes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Attend {
    op: AttentionOp,
    s: u32,
    h: u32,
    d: u32,
    heads: u32,
    scale: f32,
}

/// The gradient seed of `CrossEntropyGrad`: a constant of the module or the rank-0 binding 3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Seed {
    Immediate(f32),
    Buffer,
}

/// What an attention or cross-entropy module computes, its constants taken from the kernel's
/// shapes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum AttentionLoss {
    Attention(Attend),
    CrossEntropy { rows: u32, classes: u32 },
    CrossEntropyGrad { rows: u32, classes: u32, seed: Seed },
}

impl AttentionLoss {
    /// The module of `kernel`, or why its kind, op, operands or workgroups are not one: each op
    /// has the kind, operand types and shapes `modelspec_core::schedule::attention_loss_kernel`
    /// gives it, every element count fits `u32`, and the workgroups of 64 cover its invocations.
    pub(super) fn of(kernel: &Kernel) -> Result<Self, EmitError> {
        let wrong = |what: &str| operands_error(kernel, what.to_owned());
        if kernel.workgroup_size != GROUP {
            return Err(wrong(
                "an attention or cross-entropy kernel runs workgroups of 64",
            ));
        }
        if kernel.dtype != DType::F32 || kernel.result.dtype != DType::F32 {
            return Err(wrong("an attention or cross-entropy kernel writes f32"));
        }
        let mut buffers = Vec::new();
        let mut immediates = Vec::new();
        for operand in &kernel.operands {
            match operand {
                Operand::Buffer(ty) => buffers.push((ty.dtype, static_shape(kernel, ty)?)),
                Operand::Immediate(value) => immediates.push(*value),
            }
        }
        let result = static_shape(kernel, &kernel.result)?;
        let elements = |shape: &[u64]| shape.iter().product::<u64>();
        let fits = |n: u64| u32::try_from(n).ok().filter(|&n| n > 0);
        for (_, shape) in &buffers {
            if !shape.is_empty() {
                fits(elements(shape)).ok_or_else(|| wrong("an operand of 0 or 2^32 elements"))?;
            }
        }
        let types: Vec<DType> = buffers.iter().map(|(dtype, _)| *dtype).collect();
        let shapes: Vec<&[u64]> = buffers.iter().map(|(_, shape)| shape.as_slice()).collect();
        let u = |n: u64| n as u32;
        use DType::{F32, I32};
        let attention = match kernel.op {
            "Attention" => Some(AttentionOp::Forward),
            "AttentionGradQ" => Some(AttentionOp::GradQ),
            "AttentionGradK" => Some(AttentionOp::GradK),
            "AttentionGradV" => Some(AttentionOp::GradV),
            _ => None,
        };
        let unsupported = || EmitError::Unsupported {
            kind: kernel.kind,
            op: kernel.op,
        };
        let (kind, invocations, body) = if let Some(op) = attention {
            let arity = if op == AttentionOp::Forward { 4 } else { 5 };
            let heads = match immediates[..] {
                [heads] if heads >= 1.0 && heads.fract() == 0.0 => heads as u64,
                _ => return Err(wrong("an attention kernel carries its head count last")),
            };
            if buffers.len() != arity || types.iter().any(|&t| t != F32) {
                return Err(wrong(
                    "attention reads f32 q, k, v, a key mask and its gradient",
                ));
            }
            let (n, h) = match shapes[0] {
                [n, h] => (*n, *h),
                _ => return Err(wrong("attention reads [n, h] rows")),
            };
            let rows_match = [1, 2]
                .into_iter()
                .chain((arity == 5).then_some(4))
                .all(|at| shapes[at] == [n, h]);
            let (b, s) = match shapes[3] {
                [b, s] => (*b, *s),
                _ => return Err(wrong("attention reads a [b, s] key mask")),
            };
            if !rows_match
                || result != [n, h]
                || h % heads != 0
                || b.checked_mul(s) != Some(n)
                || s > MAX_ATTENTION_KEYS
            {
                return Err(wrong(
                    "attention reads [b s, h] rows, h a multiple of the heads, at most 512 keys",
                ));
            }
            let invocations = fits(n.saturating_mul(heads))
                .ok_or_else(|| wrong("more than 2^32 rows and heads"))?;
            let d = h / heads;
            (
                KernelKind::PerRow,
                invocations,
                AttentionLoss::Attention(Attend {
                    op,
                    s: u(s),
                    h: u(h),
                    d: u(d),
                    heads: u(heads),
                    scale: 1.0f32 / (d as f32).sqrt(),
                }),
            )
        } else {
            match (
                kernel.op,
                types.as_slice(),
                shapes.as_slice(),
                result.as_slice(),
                immediates.as_slice(),
            ) {
                ("CrossEntropy", [F32, I32, F32], [[n, v], [ln], [wn]], [], [])
                    if n == ln && n == wn =>
                {
                    (
                        KernelKind::LossSum,
                        1,
                        AttentionLoss::CrossEntropy {
                            rows: u(*n),
                            classes: u(*v),
                        },
                    )
                }
                ("CrossEntropyGrad", [F32, I32, F32], [[n, v], [ln], [wn]], [rn, rv], [g])
                    if n == ln && n == wn && (n, v) == (rn, rv) =>
                {
                    (
                        KernelKind::PerRow,
                        u(*n),
                        AttentionLoss::CrossEntropyGrad {
                            rows: u(*n),
                            classes: u(*v),
                            seed: Seed::Immediate(*g),
                        },
                    )
                }
                (
                    "CrossEntropyGrad",
                    [F32, I32, F32, F32],
                    [[n, v], [ln], [wn], []],
                    [rn, rv],
                    [],
                ) if n == ln && n == wn && (n, v) == (rn, rv) => (
                    KernelKind::PerRow,
                    u(*n),
                    AttentionLoss::CrossEntropyGrad {
                        rows: u(*n),
                        classes: u(*v),
                        seed: Seed::Buffer,
                    },
                ),
                _ => return Err(unsupported()),
            }
        };
        if kernel.kind != kind {
            return Err(unsupported());
        }
        let covered = kernel
            .workgroups
            .iter()
            .map(|&w| u64::from(w))
            .product::<u64>()
            * u64::from(GROUP);
        if covered < u64::from(invocations) {
            return Err(operands_error(
                kernel,
                format!("{covered} invocations cover fewer than {invocations}"),
            ));
        }
        Ok(body)
    }

    /// The `f32` values of workgroup memory the module stages: the row losses of one block.
    pub(super) fn shared(self) -> Option<u32> {
        match self {
            AttentionLoss::CrossEntropy { .. } => Some(GROUP),
            _ => None,
        }
    }

    /// The `f32` function variables the module keeps beside the shell's one: the softmax
    /// maximum and sum and the row dot of an attention kernel; the row sum, numerator and
    /// denominator of the loss; the row sum of its gradient.
    pub(super) fn f32_vars(self) -> usize {
        match self {
            AttentionLoss::Attention(_) | AttentionLoss::CrossEntropy { .. } => 3,
            AttentionLoss::CrossEntropyGrad { .. } => 1,
        }
    }

    /// The `u32` function variables the module keeps beside the shell's two: the column counter
    /// and the found flag of the softmax maximum of an attention kernel; the term counter of the
    /// loss.
    pub(super) fn u32_vars(self) -> usize {
        match self {
            AttentionLoss::Attention(_) => 2,
            AttentionLoss::CrossEntropy { .. } => 1,
            AttentionLoss::CrossEntropyGrad { .. } => 0,
        }
    }
}

/// The place of one row's head: element `row · H + head · d` of a `[n, H]` binding.
fn head_at(m: &mut Shell, at: &Attend, row: Word, column: Word) -> Result<Word, EmitError> {
    let h = m.uint(at.h);
    let base = m.mul(row, h)?;
    m.add(base, column)
}

impl Shell {
    /// `sqrt(x)` within 1 ulp of the CPU's correctly rounded root: the device's `s = sqrt(x)`
    /// corrected to `s + fma(0 - s, s, x) * ((1 / s) * 0.5)`, one Newton step; `s` stands where
    /// the residual is 0 or NaN: an exact root, a zero, an infinite, negative or NaN `x`.
    pub(super) fn root(&mut self, x: Word) -> Result<Word, EmitError> {
        let (zero, half, one) = (self.float(0.0), self.float(0.5), self.float(1.0));
        let s = self.sqrt(x)?;
        let q = self.f_div(one, s)?;
        let neg_s = self.f_sub(zero, s)?;
        let residual = self.f_fma(neg_s, s, x)?;
        let half_q = self.f_mul(q, half)?;
        let corrected = self.f_fma(residual, half_q, s)?;
        self.where_residual(residual, corrected, s)
    }

    /// `a / b` within 1 ulp of the CPU's correctly rounded quotient: the device's `q = a / b`
    /// corrected to `q + fma(0 - q, b, a) / b`, one residual step; `q` stands where the
    /// residual is 0 or NaN: an exact quotient, a zero or infinite `b`, an infinite or NaN `a`.
    /// `q` also stands where it is not finite, so a quotient that overflows is the infinity the
    /// CPU gives, not the NaN of `inf - inf` its residual step would make (coordinator decision
    /// D74, F1).
    pub(super) fn quotient(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        let zero = self.float(0.0);
        let q = self.f_div(a, b)?;
        let neg_q = self.f_sub(zero, q)?;
        let residual = self.f_fma(neg_q, b, a)?;
        let step = self.f_div(residual, b)?;
        let corrected = self.f_add(q, step)?;
        let refined = self.where_residual(residual, corrected, q)?;
        let infinite = self.b.is_inf(self.bool_ty, None, q)?;
        let nan = self.b.is_nan(self.bool_ty, None, q)?;
        let not_finite = self.b.logical_or(self.bool_ty, None, infinite, nan)?;
        Ok(self.b.select(self.f32_ty, None, not_finite, q, refined)?)
    }

    /// `GLSL.std.450 Log` of `x`.
    fn ln(&mut self, x: Word) -> Result<Word, EmitError> {
        self.glsl_f32(GlslStd450Op::Log, &[x])
    }

    /// `if x > max { x } else { max }`, the CPU's maximum step.
    fn greater_of(&mut self, x: Word, max: Word) -> Result<Word, EmitError> {
        let greater = self.b.f_ord_greater_than(self.bool_ty, None, x, max)?;
        Ok(self.b.select(self.f32_ty, None, greater, x, max)?)
    }

    /// The module body of `body`.
    pub(super) fn attention_loss(&mut self, body: AttentionLoss) -> Result<(), EmitError> {
        if let AttentionLoss::CrossEntropy { rows, classes } = body {
            return self.cross_entropy(rows, classes);
        }
        let (index, count) = (self.index, self.count);
        let inside = self.less(index, count)?;
        self.when(inside, |m| match body {
            AttentionLoss::Attention(at) => m.attention(at),
            AttentionLoss::CrossEntropyGrad {
                rows,
                classes,
                seed,
            } => m.cross_entropy_grad(rows, classes, seed),
            AttentionLoss::CrossEntropy { .. } => unreachable!("handled above"),
        })
    }

    /// The sum of products `a[a_at + c] * b[b_at + c]` over the `width` columns of one head, in
    /// ascending order, the first product rounded and each later one fused into the sum. Uses
    /// the shell's `f32` variable and the column counter.
    fn head_dot(
        &mut self,
        a: usize,
        a_at: Word,
        b: usize,
        b_at: Word,
        width: u32,
    ) -> Result<Word, EmitError> {
        let (acc, var) = (self.f32_var, self.extra_u32_vars[0]);
        let start = self.float(-0.0);
        self.b.store(acc, start, None, [])?;
        let (zero, one, end) = (self.uint(0), self.uint(1), self.uint(width));
        self.counted(var, zero, end, one, |m, c| {
            let x_at = m.add(a_at, c)?;
            let x = m.read(a, x_at)?;
            let y_at = m.add(b_at, c)?;
            let y = m.read(b, y_at)?;
            let sum = m.b.load(m.f32_ty, None, acc, None, [])?;
            let fused = m.f_fma(x, y, sum)?;
            m.b.store(acc, fused, None, [])?;
            Ok(())
        })?;
        Ok(self.b.load(self.f32_ty, None, acc, None, [])?)
    }

    /// The score of the query head at `q_at` and the key head at `k_at`: their dot product
    /// times the scale.
    fn score(&mut self, at: &Attend, q_at: Word, k_at: Word) -> Result<Word, EmitError> {
        let dot = self.head_dot(0, q_at, 1, k_at, at.d)?;
        let scale = self.float(at.scale);
        self.f_mul(dot, scale)
    }

    /// Whether the key mask keeps the key of row `j`: its entry is not 0.
    fn unmasked(&mut self, j: Word) -> Result<Word, EmitError> {
        let entry = self.read(3, j)?;
        let zero = self.float(0.0);
        Ok(self.b.f_unord_not_equal(self.bool_ty, None, entry, zero)?)
    }

    /// `body(j, k_at)` for each unmasked key row `j` of the sequence from row `start`, in
    /// ascending order, `k_at` its head's place. The key counter is the shell's second `u32`
    /// variable.
    fn each_key(
        &mut self,
        at: &Attend,
        start: Word,
        column: Word,
        mut body: impl FnMut(&mut Self, Word, Word) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        let var = self.u32_vars[1];
        let (one, s) = (self.uint(1), self.uint(at.s));
        let end = self.add(start, s)?;
        self.counted(var, start, end, one, |m, j| {
            let keep = m.unmasked(j)?;
            m.when(keep, |m| {
                let k_at = head_at(m, at, j, column)?;
                body(m, j, k_at)
            })
        })
    }

    /// The softmax of the query head at `q_at` over the unmasked keys of its sequence from row
    /// `start`: the maximum score, the first unmasked one replaced by each later greater one,
    /// then the sum of `exp(score - max)` in ascending key order.
    fn attention_softmax(
        &mut self,
        at: &Attend,
        q_at: Word,
        start: Word,
        column: Word,
    ) -> Result<(Word, Word), EmitError> {
        let flag = self.extra_u32_vars[1];
        let (max_var, total_var) = (self.matmul_vars[0], self.matmul_vars[1]);
        let (zero, one) = (self.uint(0), self.uint(1));
        let f_zero = self.float(0.0);
        self.b.store(flag, zero, None, [])?;
        self.b.store(max_var, f_zero, None, [])?;
        self.each_key(at, start, column, |m, _, k_at| {
            let x = m.score(at, q_at, k_at)?;
            let max = m.b.load(m.f32_ty, None, max_var, None, [])?;
            let kept = m.greater_of(x, max)?;
            let found = m.b.load(m.u32_ty, None, flag, None, [])?;
            let found = m.b.i_not_equal(m.bool_ty, None, found, zero)?;
            let next = m.b.select(m.f32_ty, None, found, kept, x)?;
            m.b.store(max_var, next, None, [])?;
            m.b.store(flag, one, None, [])?;
            Ok(())
        })?;
        let max = self.b.load(self.f32_ty, None, max_var, None, [])?;
        let start_sum = self.float(-0.0);
        self.b.store(total_var, start_sum, None, [])?;
        self.each_key(at, start, column, |m, _, k_at| {
            let x = m.score(at, q_at, k_at)?;
            let d = m.f_sub(x, max)?;
            let e = m.exp(d)?;
            let total = m.b.load(m.f32_ty, None, total_var, None, [])?;
            let sum = m.f_add(total, e)?;
            m.b.store(total_var, sum, None, [])?;
            Ok(())
        })?;
        let total = self.b.load(self.f32_ty, None, total_var, None, [])?;
        Ok((max, total))
    }

    /// The probability of the key head at `k_at` for the query head at `q_at`:
    /// `exp(score - max) / total`.
    fn probability(
        &mut self,
        at: &Attend,
        q_at: Word,
        k_at: Word,
        (max, total): (Word, Word),
    ) -> Result<Word, EmitError> {
        let x = self.score(at, q_at, k_at)?;
        let d = self.f_sub(x, max)?;
        let e = self.exp(d)?;
        self.f_div(e, total)
    }

    /// `dP`, the dot product of the gradient head at `q_at` and the value head at `k_at`.
    fn probability_grad(&mut self, at: &Attend, q_at: Word, k_at: Word) -> Result<Word, EmitError> {
        self.head_dot(4, q_at, 2, k_at, at.d)
    }

    /// The sum of products `P · dP` over the unmasked keys of the query head at `q_at`, in
    /// ascending key order.
    fn row_dot(
        &mut self,
        at: &Attend,
        q_at: Word,
        start: Word,
        column: Word,
        softmax: (Word, Word),
    ) -> Result<Word, EmitError> {
        let acc = self.matmul_vars[2];
        let start_sum = self.float(-0.0);
        self.b.store(acc, start_sum, None, [])?;
        self.each_key(at, start, column, |m, _, k_at| {
            let p = m.probability(at, q_at, k_at, softmax)?;
            let dp = m.probability_grad(at, q_at, k_at)?;
            let sum = m.b.load(m.f32_ty, None, acc, None, [])?;
            let fused = m.f_fma(p, dp, sum)?;
            m.b.store(acc, fused, None, [])?;
            Ok(())
        })?;
        Ok(self.b.load(self.f32_ty, None, acc, None, [])?)
    }

    /// `dS = P · (dP - d)` of the query head at `q_at` and the key head at `k_at`.
    fn score_grad(
        &mut self,
        at: &Attend,
        q_at: Word,
        k_at: Word,
        softmax: (Word, Word),
        d: Word,
    ) -> Result<Word, EmitError> {
        let p = self.probability(at, q_at, k_at, softmax)?;
        let dp = self.probability_grad(at, q_at, k_at)?;
        let diff = self.f_sub(dp, d)?;
        self.f_mul(p, diff)
    }

    /// `body(c)` for each column `c` of one head, in ascending order, on the column counter.
    fn each_column(
        &mut self,
        at: &Attend,
        mut body: impl FnMut(&mut Self, Word) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        let var = self.extra_u32_vars[0];
        let (zero, one, d) = (self.uint(0), self.uint(1), self.uint(at.d));
        self.counted(var, zero, d, one, |m, c| body(m, c))
    }

    /// Writes `value` to every column of the output head at `out_at`.
    fn fill_head(&mut self, at: &Attend, out_at: Word, value: f32) -> Result<(), EmitError> {
        let value = self.float(value);
        self.each_column(at, |m, c| {
            let place = m.add(out_at, c)?;
            m.write(place, value)
        })
    }

    /// `out[out_at + c] = fma(a, x[x_at + c], out[out_at + c])` for each column of one head:
    /// one more term of the head's sums in place.
    fn fuse_head(
        &mut self,
        at: &Attend,
        out_at: Word,
        a: Word,
        x: usize,
        x_at: Word,
    ) -> Result<(), EmitError> {
        self.each_column(at, |m, c| {
            let place = m.add(out_at, c)?;
            let current = m.read_result(place)?;
            let from = m.add(x_at, c)?;
            let value = m.read(x, from)?;
            let fused = m.f_fma(a, value, current)?;
            m.write(place, fused)
        })
    }

    /// Multiplies every column of the output head at `out_at` by the scale.
    fn scale_head(&mut self, at: &Attend, out_at: Word) -> Result<(), EmitError> {
        let scale = self.float(at.scale);
        self.each_column(at, |m, c| {
            let place = m.add(out_at, c)?;
            let current = m.read_result(place)?;
            let scaled = m.f_mul(current, scale)?;
            m.write(place, scaled)
        })
    }

    /// One row and head of an attention kernel, invocation `row · heads + head`.
    fn attention(&mut self, at: Attend) -> Result<(), EmitError> {
        let index = self.index;
        let heads = self.uint(at.heads);
        let row = self.div(index, heads)?;
        let head = self.rem(index, heads)?;
        let d = self.uint(at.d);
        let column = self.mul(head, d)?;
        let s = self.uint(at.s);
        let sequence = self.div(row, s)?;
        let start = self.mul(sequence, s)?;
        let row_at = head_at(self, &at, row, column)?;
        match at.op {
            AttentionOp::Forward => {
                let softmax = self.attention_softmax(&at, row_at, start, column)?;
                self.fill_head(&at, row_at, -0.0)?;
                self.each_key(&at, start, column, |m, _, k_at| {
                    let p = m.probability(&at, row_at, k_at, softmax)?;
                    m.fuse_head(&at, row_at, p, 2, k_at)
                })
            }
            AttentionOp::GradQ => {
                let softmax = self.attention_softmax(&at, row_at, start, column)?;
                let rd = self.row_dot(&at, row_at, start, column, softmax)?;
                self.fill_head(&at, row_at, -0.0)?;
                self.each_key(&at, start, column, |m, _, k_at| {
                    let ds = m.score_grad(&at, row_at, k_at, softmax, rd)?;
                    m.fuse_head(&at, row_at, ds, 1, k_at)
                })?;
                self.scale_head(&at, row_at)
            }
            AttentionOp::GradK | AttentionOp::GradV => {
                let keep = self.unmasked(row)?;
                let masked = self.b.logical_not(self.bool_ty, None, keep)?;
                self.when(masked, |m| m.fill_head(&at, row_at, 0.0))?;
                self.when(keep, |m| {
                    m.fill_head(&at, row_at, -0.0)?;
                    let var = m.u32_vars[0];
                    let (one, end) = (m.uint(1), m.add(start, s)?);
                    m.counted(var, start, end, one, |m, i| {
                        let q_at = head_at(m, &at, i, column)?;
                        let softmax = m.attention_softmax(&at, q_at, start, column)?;
                        if at.op == AttentionOp::GradK {
                            let rd = m.row_dot(&at, q_at, start, column, softmax)?;
                            let ds = m.score_grad(&at, q_at, row_at, softmax, rd)?;
                            m.fuse_head(&at, row_at, ds, 0, q_at)
                        } else {
                            let p = m.probability(&at, q_at, row_at, softmax)?;
                            m.fuse_head(&at, row_at, p, 4, q_at)
                        }
                    })?;
                    if at.op == AttentionOp::GradK {
                        m.scale_head(&at, row_at)?;
                    }
                    Ok(())
                })
            }
        }
    }

    /// The maximum of the `classes` logits of the row at `base`, the first replaced by each
    /// later greater one, into the shell's `f32` variable, and the sum of `exp(x - max)` in
    /// ascending order into `total_var`. Uses the shell's second `u32` variable.
    fn logit_softmax(
        &mut self,
        base: Word,
        classes: u32,
        total_var: Word,
    ) -> Result<(Word, Word), EmitError> {
        let (var, max_var) = (self.u32_vars[1], self.f32_var);
        let (zero, one, end) = (self.uint(0), self.uint(1), self.uint(classes));
        let first = self.read(0, base)?;
        self.b.store(max_var, first, None, [])?;
        self.counted(var, one, end, one, |m, c| {
            let at = m.add(base, c)?;
            let x = m.read(0, at)?;
            let max = m.b.load(m.f32_ty, None, max_var, None, [])?;
            let kept = m.greater_of(x, max)?;
            m.b.store(max_var, kept, None, [])?;
            Ok(())
        })?;
        let max = self.b.load(self.f32_ty, None, max_var, None, [])?;
        let start = self.float(-0.0);
        self.b.store(total_var, start, None, [])?;
        self.counted(var, zero, end, one, |m, c| {
            let at = m.add(base, c)?;
            let x = m.read(0, at)?;
            let d = m.f_sub(x, max)?;
            let e = m.exp(d)?;
            let total = m.b.load(m.f32_ty, None, total_var, None, [])?;
            let sum = m.f_add(total, e)?;
            m.b.store(total_var, sum, None, [])?;
            Ok(())
        })?;
        let total = self.b.load(self.f32_ty, None, total_var, None, [])?;
        Ok((max, total))
    }

    /// The sum of the `rows` weights of binding 2 in ascending order, on the shell's `f32`
    /// variable and first `u32` variable.
    fn weight_sum(&mut self, rows: u32) -> Result<Word, EmitError> {
        let (var, acc) = (self.u32_vars[0], self.f32_var);
        let start = self.float(-0.0);
        self.b.store(acc, start, None, [])?;
        let (zero, one, end) = (self.uint(0), self.uint(1), self.uint(rows));
        self.counted(var, zero, end, one, |m, n| {
            let w = m.read(2, n)?;
            m.accumulate(w)
        })?;
        Ok(self.b.load(self.f32_ty, None, acc, None, [])?)
    }

    /// The `CrossEntropy` loss of one workgroup over `rows` rows of `classes` logits.
    fn cross_entropy(&mut self, rows: u32, classes: u32) -> Result<(), EmitError> {
        let local = self.local.expect("a module with workgroup memory");
        let [block_var, _] = self.u32_vars;
        let term_var = self.extra_u32_vars[0];
        let [total_var, num_var, den_var] = [
            self.matmul_vars[0],
            self.matmul_vars[1],
            self.matmul_vars[2],
        ];
        let (zero, one) = (self.uint(0), self.uint(1));
        let (n_rows, v, group) = (self.uint(rows), self.uint(classes), self.uint(GROUP));
        let blocks = self.uint(rows.div_ceil(GROUP));
        let start = self.float(-0.0);
        self.b.store(num_var, start, None, [])?;
        self.b.store(den_var, start, None, [])?;
        let first_invocation = self.equal(local, zero)?;
        self.counted(block_var, zero, blocks, one, |m, block| {
            let base = m.mul(block, group)?;
            let row = m.add(base, local)?;
            let staged = m.less(row, n_rows)?;
            m.when(staged, |m| {
                let at = m.mul(row, v)?;
                let (max, total) = m.logit_softmax(at, classes, total_var)?;
                let label = m.read_u32(1, row)?;
                let valid = m.less(label, v)?;
                let label = m.b.select(m.u32_ty, None, valid, label, zero)?;
                let target_at = m.add(at, label)?;
                let target = m.read(0, target_at)?;
                let log = m.ln(total)?;
                let shifted = m.f_add(max, log)?;
                let loss = m.f_sub(shifted, target)?;
                m.store_shared(local, loss)
            })?;
            m.barrier()?;
            m.when(first_invocation, |m| {
                let left = m.sub(n_rows, base)?;
                let len = m.min(left, group)?;
                m.counted(term_var, zero, len, one, |m, t| {
                    let n = m.add(base, t)?;
                    let w = m.read(2, n)?;
                    let loss = m.load_shared(t)?;
                    let num = m.b.load(m.f32_ty, None, num_var, None, [])?;
                    let fused = m.f_fma(w, loss, num)?;
                    m.b.store(num_var, fused, None, [])?;
                    let den = m.b.load(m.f32_ty, None, den_var, None, [])?;
                    let sum = m.f_add(den, w)?;
                    m.b.store(den_var, sum, None, [])?;
                    Ok(())
                })
            })?;
            m.barrier()
        })?;
        self.when(first_invocation, |m| {
            let num = m.b.load(m.f32_ty, None, num_var, None, [])?;
            let den = m.b.load(m.f32_ty, None, den_var, None, [])?;
            let f_zero = m.float(0.0);
            let empty = m.b.f_ord_equal(m.bool_ty, None, den, f_zero)?;
            let q = m.quotient(num, den)?;
            let loss = m.b.select(m.f32_ty, None, empty, f_zero, q)?;
            m.write(zero, loss)
        })
    }

    /// Row `index` of the `CrossEntropyGrad` of `rows` rows of `classes` logits.
    fn cross_entropy_grad(&mut self, rows: u32, classes: u32, seed: Seed) -> Result<(), EmitError> {
        let index = self.index;
        let (var, total_var) = (self.u32_vars[1], self.matmul_vars[0]);
        let den = self.weight_sum(rows)?;
        let f_zero = self.float(0.0);
        let empty = self.b.f_ord_equal(self.bool_ty, None, den, f_zero)?;
        let (zero, one, v) = (self.uint(0), self.uint(1), self.uint(classes));
        let base = self.mul(index, v)?;
        let (max, total) = self.logit_softmax(base, classes, total_var)?;
        let g = match seed {
            Seed::Immediate(g) => self.float(g),
            Seed::Buffer => self.read(3, zero)?,
        };
        let w = self.read(2, index)?;
        let gw = self.f_mul(g, w)?;
        let scale = self.quotient(gw, den)?;
        let label = self.read_u32(1, index)?;
        let f_one = self.float(1.0);
        self.counted(var, zero, v, one, |m, c| {
            let at = m.add(base, c)?;
            let x = m.read(0, at)?;
            let d = m.f_sub(x, max)?;
            let e = m.exp(d)?;
            let p = m.f_div(e, total)?;
            let hit = m.equal(c, label)?;
            let target = m.b.select(m.f32_ty, None, hit, f_one, f_zero)?;
            let diff = m.f_sub(p, target)?;
            let y = m.f_mul(scale, diff)?;
            let out = m.b.select(m.f32_ty, None, empty, f_zero, y)?;
            m.write(at, out)
        })
    }
}
