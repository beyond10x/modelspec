//! The modules of the attention and cross-entropy kernels (DESIGN.md § 5.2, coordinator
//! decision D73): `Attention`, `AttentionGradQ`, `AttentionGradK` and `AttentionGradV` one
//! workgroup per sequence, head and tile of rows, `CrossEntropyGrad` one invocation per row,
//! and the `CrossEntropy` loss sum one workgroup of 64; and the corrected square root and
//! quotient of the `Sqrt` and `Div` elementwise kernels.
//!
//! Each computes its op as the CPU interpreter does, operation for operation, so the device
//! differs from the CPU only by its `exp`, `log`, division and square root, which Vulkan does
//! not round correctly. Every sum runs in ascending index order: it starts at `-0.0` and adds
//! each term, so its first term is its value exactly, and a sum of products fuses each product
//! into it by one `Fma`, so its first product is rounded once, as the CPU's `sum_of_products`.
//! Every float add, subtract, multiply, division and `Fma` is decorated `NoContraction` outside
//! `fast_math`.
//!
//! Workgroup `(b · heads + a) · tiles + t` of an attention kernel, of
//! `modelspec_core::schedule::ATTENTION_WORKGROUP` invocations, computes head `a` of the `T`
//! rows of tile `t` of sequence `b`, `T` from `modelspec_core::schedule::attention_tile`: query
//! rows for `Attention` and `AttentionGradQ`, key rows for `AttentionGradK` and
//! `AttentionGradV` (story:vulkan-attention-workgroup-per-head). Row `r` is token `r % S` of
//! sequence `r / S`, with `S` at most `modelspec_core::schedule::MAX_ATTENTION_KEYS`, and a key
//! is masked where its mask entry is 0. The workgroup computes the weights of its rows, scores,
//! probabilities and their gradients, into workgroup memory between barriers, one invocation
//! per key or query and holding that row's head, so each weight is computed once; the scores
//! are computed from the same operands by the same instructions wherever they are used, so they
//! hold the same value at every use. Then the `ATTENTION_WORKGROUP / T` invocations of each row
//! each sum their columns of it, every column `lane + u · lanes` in its own variable, so a
//! weight is loaded once for all of them. `Attention` and `AttentionGradQ` sum without a mask
//! check, several keys loaded at a time, when the mask keeps every key of the sequence, which
//! the workgroup finds before it sums. The softmax maximum and sum
//! and the row dot of the gradient are computed once per query head when the kernels share
//! softmax statistics (`Kernel::softmax_stats`, story:vulkan-attention-softmax-stats-reused):
//! `Attention` writes the maximum and sum of query head `r · heads + a` to elements `3 (r · heads
//! + a)` and `+ 1` of its first written binding, `AttentionGradQ` reads them from its first
//! written binding and writes its row dot to `+ 2`, and `AttentionGradK` and `AttentionGradV`
//! read the three from binding 5 for every query of their key's sequence, by the instructions
//! that computed them, so each use holds the value the recomputation held. Without shared
//! statistics each kernel recomputes them where it uses them. The softmax
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
use modelspec_core::ir::kernel::SOFTMAX_STATS_PER_QUERY;
use modelspec_core::ir::tensor::DType;
use modelspec_core::schedule::{
    ATTENTION_WORKGROUP, MAX_ATTENTION_KEYS, attention_tile, attention_workgroups,
};
use rspirv::spirv::GlslStd450Op;

/// The workgroup size of every attention and cross-entropy kernel.
const GROUP: u32 = 64;

/// The values per query head of the softmax statistics: maximum, sum and row dot.
const SOFTMAX_STATS: u32 = SOFTMAX_STATS_PER_QUERY as u32;

/// The read binding of the softmax statistics `AttentionGradK` and `AttentionGradV` read: the one
/// after `q`, `k`, `v`, the key mask and the gradient.
const STATS_BINDING: usize = 5;

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
    /// Whether the kernel shares its softmax statistics through the binding after its operands
    /// (`modelspec_core::ir::kernel::Kernel::softmax_stats`) instead of recomputing them.
    stats: bool,
    /// The rows of one sequence's head a workgroup computes
    /// (`modelspec_core::schedule::attention_tile`), the tiles of a head, the invocations of one
    /// row (`ATTENTION_WORKGROUP / tile`), and the workgroups that compute.
    tile: u32,
    tiles: u32,
    lanes: u32,
    groups: u32,
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
    /// gives it, every element count fits `u32`, and its workgroups, of
    /// `modelspec_core::schedule::ATTENTION_WORKGROUP` for an attention op and 64 for another,
    /// cover its invocations.
    pub(super) fn of(kernel: &Kernel) -> Result<Self, EmitError> {
        let wrong = |what: &str| operands_error(kernel, what.to_owned());
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
        if attention.is_none() && kernel.softmax_stats.is_some() {
            return Err(wrong("only an attention kernel shares softmax statistics"));
        }
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
            let stats = match &kernel.softmax_stats {
                None => false,
                Some(ty) => {
                    let rows = n.saturating_mul(heads);
                    if ty.dtype != F32
                        || static_shape(kernel, ty)? != [rows, SOFTMAX_STATS_PER_QUERY]
                    {
                        return Err(wrong(
                            "attention shares [n · heads, 3] f32 softmax statistics",
                        ));
                    }
                    true
                }
            };
            let groups = attention_workgroups(b, s, heads)
                .and_then(fits)
                .filter(|&groups| {
                    u64::from(groups) * u64::from(ATTENTION_WORKGROUP) <= u32::MAX.into()
                })
                .ok_or_else(|| wrong("more than 2^32 invocations"))?;
            let tile = attention_tile(s);
            let d = h / heads;
            (
                KernelKind::PerRow,
                groups * ATTENTION_WORKGROUP,
                AttentionLoss::Attention(Attend {
                    op,
                    s: u(s),
                    h: u(h),
                    d: u(d),
                    heads: u(heads),
                    scale: 1.0f32 / (d as f32).sqrt(),
                    stats,
                    tile: u(tile),
                    tiles: u(s.div_ceil(tile)),
                    lanes: ATTENTION_WORKGROUP / u(tile),
                    groups,
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
        let size = if attention.is_some() {
            ATTENTION_WORKGROUP
        } else {
            GROUP
        };
        if kernel.workgroup_size != size {
            return Err(operands_error(
                kernel,
                format!(
                    "an attention kernel runs workgroups of {ATTENTION_WORKGROUP}, any other of {GROUP}"
                ),
            ));
        }
        let covered = kernel
            .workgroups
            .iter()
            .map(|&w| u64::from(w))
            .product::<u64>()
            * u64::from(size);
        if covered < u64::from(invocations) {
            return Err(operands_error(
                kernel,
                format!("{covered} invocations cover fewer than {invocations}"),
            ));
        }
        Ok(body)
    }

    /// The `f32` values of workgroup memory the module stages: the row losses of one block; the
    /// [`Attend::staged`] values of an attention workgroup.
    pub(super) fn shared(self) -> Option<u32> {
        match self {
            AttentionLoss::CrossEntropy { .. } => Some(GROUP),
            AttentionLoss::Attention(at) => Some(at.staged()),
            AttentionLoss::CrossEntropyGrad { .. } => None,
        }
    }

    /// The `f32` function variables the module keeps beside the shell's one: the softmax
    /// maximum and sum, the row dot and the output sums of an attention kernel; the row sum,
    /// numerator and denominator of the loss; the row sum of its gradient.
    pub(super) fn f32_vars(self) -> usize {
        match self {
            AttentionLoss::Attention(at) => OUTPUT_SUMS + at.columns() as usize,
            AttentionLoss::CrossEntropy { .. } => 3,
            AttentionLoss::CrossEntropyGrad { .. } => 1,
        }
    }

    /// The `u32` function variables the module keeps beside the shell's two: the column counter
    /// of a head's dot product, the found flag of the softmax maximum, and the counters of the
    /// keys, the rows of a tile and the terms of a sum of an attention kernel; the term counter of
    /// the loss.
    pub(super) fn u32_vars(self) -> usize {
        match self {
            AttentionLoss::Attention(_) => 5,
            AttentionLoss::CrossEntropy { .. } => 1,
            AttentionLoss::CrossEntropyGrad { .. } => 0,
        }
    }
}

impl Attend {
    /// The `f32` values of one row's weights of a workgroup: one per key of its sequence.
    fn weights(self) -> u32 {
        self.tile * self.s
    }

    /// Where a workgroup's statistics start in its workgroup memory: after the weights, and for
    /// `AttentionGradQ` after the probability gradients.
    fn stats_base(self) -> u32 {
        match self.op {
            AttentionOp::GradQ => 2 * self.weights(),
            _ => self.weights(),
        }
    }

    /// The `f32` values of workgroup memory an attention workgroup stages: the weights of its
    /// rows, and for `Attention` and `AttentionGradQ` the softmax statistics of its rows after
    /// them, for `AttentionGradQ` after the probability gradients; within
    /// `modelspec_core::schedule::attention_workgroup_values`.
    fn staged(self) -> u32 {
        match self.op {
            AttentionOp::Forward | AttentionOp::GradQ => self.all_kept() + 1,
            AttentionOp::GradK | AttentionOp::GradV => self.weights(),
        }
    }

    /// Where an `Attention` or `AttentionGradQ` workgroup keeps whether the mask keeps every
    /// key of its sequence, `1.0` if it does, else `0.0`: after its rows' statistics.
    fn all_kept(self) -> u32 {
        self.stats_base() + SOFTMAX_STATS * self.tile
    }

    /// The output columns an invocation owns: `d / lanes`, rounded up.
    fn columns(self) -> u32 {
        self.d.div_ceil(self.lanes)
    }

    /// The rows an output sum without a mask check takes at a time: [`SUM_VALUES`] values over
    /// the owned columns, at least 1 and at most [`SUM_ROWS`].
    fn sum_rows(self) -> u32 {
        (SUM_VALUES / self.columns()).clamp(1, SUM_ROWS)
    }
}

/// The values an output sum without a mask check loads ahead of their products, and the most
/// rows it takes at a time ([`Attend::sum_rows`]).
const SUM_VALUES: u32 = 32;
const SUM_ROWS: u32 = 8;

/// The first of an attention kernel's `f32` variables that hold its output sums, one per owned
/// column ([`Attend::columns`]), after the softmax maximum and sum and the row dot.
const OUTPUT_SUMS: usize = 3;

/// The widest head whose columns an invocation holds as values while it computes many dot
/// products with it, and whose dot products are emitted without a loop, every load before the
/// first product.
const HELD_HEAD: u32 = 128;

/// One operand of a head's dot product: the head at a place of a binding, or the values of a
/// head an invocation holds.
#[derive(Clone, Copy)]
enum Head<'a> {
    Global(usize, Word),
    Values(&'a [Word]),
}

impl<'a> Head<'a> {
    /// The held `values` of the head at `at` of `binding`, or the head there.
    fn of(values: Option<&'a [Word]>, binding: usize, at: Word) -> Self {
        match values {
            Some(values) => Head::Values(values),
            None => Head::Global(binding, at),
        }
    }
}

/// The words of one attention workgroup and invocation: the first row of its sequence and of
/// its tile, the rows of the tile in the sequence, the first column of its head, the head, the
/// invocation's row of the tile and its lane among the row's invocations, whether the row is
/// one of the sequence, the row, its head's place, and where its weights start in workgroup
/// memory.
#[derive(Clone, Copy)]
struct Tile {
    start: Word,
    first: Word,
    rows: Word,
    column: Word,
    head: Word,
    lane: Word,
    active: Word,
    own: Word,
    own_at: Word,
    base: Word,
    /// Where the row's statistics start in workgroup memory (`Attention`, `AttentionGradQ`).
    row_stats: Word,
}

/// A word an attention module computes from a row of a sequence: a weight, or whether the mask
/// keeps it.
type OfRow<'a> = &'a dyn Fn(&mut Shell, Word) -> Result<Word, EmitError>;

/// The head of a key an invocation holds, if it holds one ([`Shell::held_head`]).
type HeldOf<'a> = &'a dyn Fn(&mut Shell, Word) -> Result<Option<Vec<Word>>, EmitError>;

/// The work of one key and one row of a query tile: the key's row, its held head, the tile's
/// row and the place of their weight.
type KeyRow<'a> =
    &'a dyn Fn(&mut Shell, Word, Option<&[Word]>, Word, Word) -> Result<(), EmitError>;

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
        match body {
            AttentionLoss::CrossEntropy { rows, classes } => self.cross_entropy(rows, classes),
            AttentionLoss::Attention(at) => self.attention(at),
            AttentionLoss::CrossEntropyGrad {
                rows,
                classes,
                seed,
            } => {
                let (index, count) = (self.index, self.count);
                let inside = self.less(index, count)?;
                self.when(inside, |m| m.cross_entropy_grad(rows, classes, seed))
            }
        }
    }

    /// The sum of products `a[a_at + c] * b[b_at + c]` over the `width` columns of one head
    /// ([`Shell::dot`]).
    fn head_dot(
        &mut self,
        a: usize,
        a_at: Word,
        b: usize,
        b_at: Word,
        width: u32,
    ) -> Result<Word, EmitError> {
        self.dot(Head::Global(a, a_at), Head::Global(b, b_at), width)
    }

    /// The columns of the head at `at` of `binding` as values the invocation holds, every load
    /// emitted at once; `None` past [`HELD_HEAD`] columns.
    fn held_head(
        &mut self,
        at: &Attend,
        binding: usize,
        head_at: Word,
    ) -> Result<Option<Vec<Word>>, EmitError> {
        if at.d > HELD_HEAD {
            return Ok(None);
        }
        (0..at.d)
            .map(|c| {
                let offset = self.uint(c);
                let place = self.add(head_at, offset)?;
                self.read(binding, place)
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }

    /// The sum of products `a[c] * b[c]` over the `width` columns of one head, in ascending
    /// order, the first product rounded and each later one fused into the sum. Up to
    /// [`HELD_HEAD`] columns the products are emitted one after the other with every load of
    /// the head before the first; wider heads of two bindings run a loop on the shell's `f32`
    /// variable and the column counter.
    fn dot(&mut self, a: Head<'_>, b: Head<'_>, width: u32) -> Result<Word, EmitError> {
        if width <= HELD_HEAD {
            let columns = |m: &mut Self, head: Head<'_>| -> Result<Vec<Word>, EmitError> {
                match head {
                    Head::Values(values) => Ok(values.to_vec()),
                    Head::Global(binding, at) => (0..width)
                        .map(|c| {
                            let offset = m.uint(c);
                            let place = m.add(at, offset)?;
                            m.read(binding, place)
                        })
                        .collect(),
                }
            };
            let xs = columns(self, a)?;
            let ys = columns(self, b)?;
            let mut sum = self.float(-0.0);
            for (x, y) in xs.into_iter().zip(ys) {
                sum = self.f_fma(x, y, sum)?;
            }
            return Ok(sum);
        }
        let (Head::Global(a, a_at), Head::Global(b, b_at)) = (a, b) else {
            return Err(EmitError::Build(format!(
                "a held head has at most {HELD_HEAD} columns"
            )));
        };
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
        self.softmax_of(at, start, column, &|m, _, k_at| m.score(at, q_at, k_at))
    }

    /// The softmax of [`Shell::attention_softmax`] over the scores `score(j, k_at)` of the
    /// unmasked key rows `j` of the sequence from row `start`, `k_at` their head's place.
    fn softmax_of(
        &mut self,
        at: &Attend,
        start: Word,
        column: Word,
        score: &dyn Fn(&mut Self, Word, Word) -> Result<Word, EmitError>,
    ) -> Result<(Word, Word), EmitError> {
        let flag = self.extra_u32_vars[1];
        let (max_var, total_var) = (self.matmul_vars[0], self.matmul_vars[1]);
        let (zero, one) = (self.uint(0), self.uint(1));
        let f_zero = self.float(0.0);
        self.b.store(flag, zero, None, [])?;
        self.b.store(max_var, f_zero, None, [])?;
        self.each_key(at, start, column, |m, j, k_at| {
            let x = score(m, j, k_at)?;
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
        self.each_key(at, start, column, |m, j, k_at| {
            let x = score(m, j, k_at)?;
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

    /// The sums of products `weight(i) * x[i_at + c]` for each owned column `c` of `columns`
    /// ([`Shell::owned_columns`]), in ascending order of the rows `i` of the sequence from row
    /// `start` that `kept(i)` keeps, every row without `kept`, `i_at` their head's place at
    /// `column`: each starts at `-0.0` and fuses each product into it, as the CPU's sums of
    /// `P · V`, `dS · K`, `dS · Q` and `P · G`. A row's weight is loaded once for all the
    /// columns. Without `kept` the rows are taken [`Attend::sum_rows`] at a time, every value
    /// loaded before the first product is fused. Uses the term counter and the output sums.
    #[allow(clippy::too_many_arguments)]
    fn row_sums(
        &mut self,
        at: &Attend,
        start: Word,
        column: Word,
        x: usize,
        columns: &[(Word, Option<Word>)],
        weight: OfRow<'_>,
        kept: Option<OfRow<'_>>,
    ) -> Result<Vec<Word>, EmitError> {
        let var = self.extra_u32_vars[4];
        let sums = self.matmul_vars[OUTPUT_SUMS..OUTPUT_SUMS + columns.len()].to_vec();
        let start_sum = self.float(-0.0);
        for &sum in &sums {
            self.b.store(sum, start_sum, None, [])?;
        }
        let load = |m: &mut Self, i: Word| -> Result<(Word, Vec<Word>), EmitError> {
            let w = weight(m, i)?;
            let row = m.add(start, i)?;
            let row_at = head_at(m, at, row, column)?;
            let xs = columns
                .iter()
                .map(|&(c, _)| {
                    let place = m.add(row_at, c)?;
                    m.read(x, place)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok((w, xs))
        };
        let fuse = |m: &mut Self, terms: &[(Word, Vec<Word>)]| -> Result<(), EmitError> {
            for (u, &var) in sums.iter().enumerate() {
                let mut sum = m.b.load(m.f32_ty, None, var, None, [])?;
                for (w, xs) in terms {
                    sum = m.f_fma(*w, xs[u], sum)?;
                }
                m.b.store(var, sum, None, [])?;
            }
            Ok(())
        };
        let (zero, one, s) = (self.uint(0), self.uint(1), self.uint(at.s));
        match kept {
            Some(kept) => self.counted(var, zero, s, one, |m, i| {
                let keep = kept(m, i)?;
                m.when(keep, |m| {
                    let term = load(m, i)?;
                    fuse(m, &[term])
                })
            })?,
            None => {
                let rows = at.sum_rows();
                let whole = at.s / rows * rows;
                if whole > 0 {
                    let (end, step) = (self.uint(whole), self.uint(rows));
                    self.counted(var, zero, end, step, |m, first| {
                        let terms = (0..rows)
                            .map(|u| {
                                let offset = m.uint(u);
                                let i = m.add(first, offset)?;
                                load(m, i)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        fuse(m, &terms)
                    })?;
                }
                let terms = (whole..at.s)
                    .map(|i| {
                        let i = self.uint(i);
                        load(self, i)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if !terms.is_empty() {
                    fuse(self, &terms)?;
                }
            }
        }
        sums.iter()
            .map(|&sum| Ok(self.b.load(self.f32_ty, None, sum, None, [])?))
            .collect()
    }

    /// Writes each of `sums` to its owned column of `columns` of the output head at `own_at`,
    /// times the scale for `scaled`.
    fn write_sums(
        &mut self,
        at: &Attend,
        own_at: Word,
        columns: &[(Word, Option<Word>)],
        sums: &[Word],
        scaled: bool,
    ) -> Result<(), EmitError> {
        for (&(c, inside), &sum) in columns.iter().zip(sums) {
            let value = if scaled {
                let scale = self.float(at.scale);
                self.f_mul(sum, scale)?
            } else {
                sum
            };
            let out_at = self.add(own_at, c)?;
            match inside {
                None => self.write(out_at, value)?,
                Some(inside) => self.when(inside, |m| m.write(out_at, value))?,
            }
        }
        Ok(())
    }

    /// The columns of its row's output head the invocation at `lane` owns: `lane + u · lanes`
    /// for each `u` below [`Attend::columns`], and, where `lanes` does not divide `d`, whether
    /// the column is one of the head's, a column past it read as column 0 and not written.
    fn owned_columns(
        &mut self,
        at: &Attend,
        lane: Word,
    ) -> Result<Vec<(Word, Option<Word>)>, EmitError> {
        let (lanes, zero, d) = (self.uint(at.lanes), self.uint(0), self.uint(at.d));
        (0..at.columns())
            .map(|u| {
                let offset = self.uint(u);
                let step = self.mul(offset, lanes)?;
                let c = self.add(lane, step)?;
                if at.d.is_multiple_of(at.lanes) {
                    Ok((c, None))
                } else {
                    let inside = self.less(c, d)?;
                    let read = self.b.select(self.u32_ty, None, inside, c, zero)?;
                    Ok((read, Some(inside)))
                }
            })
            .collect()
    }

    /// The words of the invocation's workgroup `group` and its index `local` in it.
    fn tile(&mut self, at: &Attend, group: Word, local: Word) -> Result<Tile, EmitError> {
        let (tiles, heads) = (self.uint(at.tiles), self.uint(at.heads));
        let (s, d, rows_per_tile) = (self.uint(at.s), self.uint(at.d), self.uint(at.tile));
        let lanes = self.uint(at.lanes);
        let tile = self.rem(group, tiles)?;
        let pair = self.div(group, tiles)?;
        let head = self.rem(pair, heads)?;
        let sequence = self.div(pair, heads)?;
        let start = self.mul(sequence, s)?;
        let column = self.mul(head, d)?;
        let tile_start = self.mul(tile, rows_per_tile)?;
        let first = self.add(start, tile_start)?;
        let left = self.sub(s, tile_start)?;
        let rows = self.min(rows_per_tile, left)?;
        let r = self.div(local, lanes)?;
        let lane = self.rem(local, lanes)?;
        let active = self.less(r, rows)?;
        let own = self.add(first, r)?;
        let own_at = head_at(self, at, own, column)?;
        let base = self.mul(r, s)?;
        let three = self.uint(SOFTMAX_STATS);
        let stats_base = self.uint(at.stats_base());
        let row_stats = self.mul(r, three)?;
        let row_stats = self.add(stats_base, row_stats)?;
        Ok(Tile {
            start,
            first,
            rows,
            column,
            head,
            lane,
            active,
            own,
            own_at,
            base,
            row_stats,
        })
    }

    /// Element `3 (row · heads + head) + offset` of the softmax statistics: the maximum, sum
    /// (`offset` 1) or row dot (2) of query head `row · heads + head`.
    fn stats_at(
        &mut self,
        at: &Attend,
        row: Word,
        head: Word,
        offset: u32,
    ) -> Result<Word, EmitError> {
        let (heads, three, offset) = (
            self.uint(at.heads),
            self.uint(SOFTMAX_STATS),
            self.uint(offset),
        );
        let query = self.mul(row, heads)?;
        let query = self.add(query, head)?;
        let query = self.mul(query, three)?;
        self.add(query, offset)
    }

    /// The workgroup of an attention kernel (story:vulkan-attention-workgroup-per-head):
    /// workgroup `(sequence · heads + head) · tiles + tile` computes the head's rows of the
    /// tile, every phase in workgroup memory between barriers; the workgroups past the kernel's
    /// do nothing.
    fn attention(&mut self, at: Attend) -> Result<(), EmitError> {
        let local = self
            .local
            .ok_or_else(|| EmitError::Build("an attention module stages its rows".to_owned()))?;
        let index = self.index;
        let size = self.uint(ATTENTION_WORKGROUP);
        let group = self.div(index, size)?;
        let groups = self.uint(at.groups);
        let inside = self.less(group, groups)?;
        self.when(inside, |m| {
            let tile = m.tile(&at, group, local)?;
            match at.op {
                AttentionOp::Forward | AttentionOp::GradQ => m.query_tile(&at, tile, local),
                AttentionOp::GradK | AttentionOp::GradV => m.key_tile(&at, tile, local),
            }
        })
    }

    /// The query rows of an `Attention` or `AttentionGradQ` workgroup. One invocation per key
    /// computes the key's scores with every row of the tile into the rows' weights, holding the
    /// key's head; the first invocation of each row its softmax maximum and sum from them, in
    /// ascending key order; one invocation per key the probabilities in place, and for
    /// `AttentionGradQ` the probability gradients `dP` beside them, holding the value head, then
    /// the first invocation of each row the row dot from both and one invocation per key `dS`
    /// in place; then each invocation sums the columns of the row it owns over the keys.
    fn query_tile(&mut self, at: &Attend, t: Tile, local: Word) -> Result<(), EmitError> {
        let grad = at.op == AttentionOp::GradQ;
        let (s, zero, one, two) = (self.uint(at.s), self.uint(0), self.uint(1), self.uint(2));
        let size = self.uint(ATTENTION_WORKGROUP);
        let gradients = self.uint(at.weights());
        let first_lane = self.equal(t.lane, zero)?;
        let serial = self.and(t.active, first_lane)?;
        let (keys, rows) = (self.extra_u32_vars[2], self.extra_u32_vars[3]);
        // `body(j, key row, weight place of row r)` for each row `r` of the tile and each key
        // `j` the mask keeps, one invocation per key, `head(j)` computed once per key.
        let each_kept =
            |m: &mut Self, head: HeldOf<'_>, body: KeyRow<'_>| -> Result<(), EmitError> {
                m.counted(keys, local, s, size, |m, j| {
                    let row = m.add(t.start, j)?;
                    let keep = m.unmasked(row)?;
                    m.when(keep, |m| {
                        let held = head(m, row)?;
                        m.counted(rows, zero, t.rows, one, |m, r| {
                            let weight_at = m.mul(r, s)?;
                            let weight_at = m.add(weight_at, j)?;
                            body(m, row, held.as_deref(), r, weight_at)
                        })
                    })
                })
            };
        // Whether the mask keeps every key of the sequence: set by the first invocation, then
        // cleared by the invocation of each masked key.
        let all_kept = self.uint(at.all_kept());
        let first = self.equal(local, zero)?;
        self.when(first, |m| {
            let set = m.float(1.0);
            m.store_shared(all_kept, set)
        })?;
        self.barrier()?;
        self.counted(keys, local, s, size, |m, j| {
            let row = m.add(t.start, j)?;
            let keep = m.unmasked(row)?;
            let masked = m.b.logical_not(m.bool_ty, None, keep)?;
            m.when(masked, |m| {
                let clear = m.float(0.0);
                m.store_shared(all_kept, clear)
            })
        })?;
        each_kept(
            self,
            &|m, row| {
                let k_at = head_at(m, at, row, t.column)?;
                m.held_head(at, 1, k_at)
            },
            &|m, row, k, r, weight_at| {
                let q_row = m.add(t.first, r)?;
                let q_at = head_at(m, at, q_row, t.column)?;
                let k_at = head_at(m, at, row, t.column)?;
                let dot = m.dot(Head::Global(0, q_at), Head::of(k, 1, k_at), at.d)?;
                let scale = m.float(at.scale);
                let x = m.f_mul(dot, scale)?;
                m.store_shared(weight_at, x)
            },
        )?;
        self.barrier()?;
        self.when(serial, |m| {
            let total_at = m.add(t.row_stats, one)?;
            let softmax = if grad && at.stats {
                let max_at = m.stats_at(at, t.own, t.head, 0)?;
                let sum_at = m.stats_at(at, t.own, t.head, 1)?;
                (m.read_out(0, max_at)?, m.read_out(0, sum_at)?)
            } else {
                let (base, start) = (t.base, t.start);
                m.softmax_of(at, t.start, t.column, &|m, row, _| {
                    let j = m.sub(row, start)?;
                    let weight_at = m.add(base, j)?;
                    m.load_shared(weight_at)
                })?
            };
            m.store_shared(t.row_stats, softmax.0)?;
            m.store_shared(total_at, softmax.1)?;
            if at.stats && !grad {
                let max_at = m.stats_at(at, t.own, t.head, 0)?;
                let sum_at = m.stats_at(at, t.own, t.head, 1)?;
                m.write_to(0, max_at, softmax.0)?;
                m.write_to(0, sum_at, softmax.1)?;
            }
            Ok(())
        })?;
        self.barrier()?;
        let stats_base = self.uint(at.stats_base());
        let three = self.uint(SOFTMAX_STATS);
        // Where row `r`'s statistics start in workgroup memory.
        let row_stats = |m: &mut Self, r: Word| -> Result<Word, EmitError> {
            let slot = m.mul(r, three)?;
            m.add(stats_base, slot)
        };
        each_kept(
            self,
            &|m, row| {
                if grad {
                    let v_at = head_at(m, at, row, t.column)?;
                    m.held_head(at, 2, v_at)
                } else {
                    Ok(None)
                }
            },
            &|m, row, v, r, weight_at| {
                let slot = row_stats(m, r)?;
                let total_at = m.add(slot, one)?;
                let max = m.load_shared(slot)?;
                let total = m.load_shared(total_at)?;
                let x = m.load_shared(weight_at)?;
                let d = m.f_sub(x, max)?;
                let e = m.exp(d)?;
                let p = m.f_div(e, total)?;
                m.store_shared(weight_at, p)?;
                if grad {
                    let g_row = m.add(t.first, r)?;
                    let g_at = head_at(m, at, g_row, t.column)?;
                    let v_at = head_at(m, at, row, t.column)?;
                    let dp = m.dot(Head::Global(4, g_at), Head::of(v, 2, v_at), at.d)?;
                    let dp_at = m.add(gradients, weight_at)?;
                    m.store_shared(dp_at, dp)?;
                }
                Ok(())
            },
        )?;
        self.barrier()?;
        if grad {
            self.when(serial, |m| {
                let acc = m.matmul_vars[2];
                let start_sum = m.float(-0.0);
                m.b.store(acc, start_sum, None, [])?;
                m.counted(m.extra_u32_vars[4], zero, s, one, |m, j| {
                    let weight_at = m.add(t.base, j)?;
                    let row = m.add(t.start, j)?;
                    let keep = m.unmasked(row)?;
                    m.when(keep, |m| {
                        let p = m.load_shared(weight_at)?;
                        let dp_at = m.add(gradients, weight_at)?;
                        let dp = m.load_shared(dp_at)?;
                        let sum = m.b.load(m.f32_ty, None, acc, None, [])?;
                        let fused = m.f_fma(p, dp, sum)?;
                        m.b.store(acc, fused, None, [])?;
                        Ok(())
                    })
                })?;
                let rd = m.b.load(m.f32_ty, None, acc, None, [])?;
                let rd_at = m.add(t.row_stats, two)?;
                m.store_shared(rd_at, rd)?;
                if at.stats {
                    let out_at = m.stats_at(at, t.own, t.head, 2)?;
                    m.write_to(0, out_at, rd)?;
                }
                Ok(())
            })?;
            self.barrier()?;
            each_kept(self, &|_, _| Ok(None), &|m, _, _, r, weight_at| {
                let slot = row_stats(m, r)?;
                let rd_at = m.add(slot, two)?;
                let rd = m.load_shared(rd_at)?;
                let p = m.load_shared(weight_at)?;
                let dp_at = m.add(gradients, weight_at)?;
                let dp = m.load_shared(dp_at)?;
                let diff = m.f_sub(dp, rd)?;
                let ds = m.f_mul(p, diff)?;
                m.store_shared(weight_at, ds)
            })?;
            self.barrier()?;
        }
        let x = if grad { 1 } else { 2 };
        let weight = |m: &mut Self, j: Word| -> Result<Word, EmitError> {
            let weight_at = m.add(t.base, j)?;
            m.load_shared(weight_at)
        };
        let flag = self.load_shared(all_kept)?;
        let set = self.float(1.0);
        let every = self.b.f_ord_equal(self.bool_ty, None, flag, set)?;
        let some = self.b.logical_not(self.bool_ty, None, every)?;
        self.when(t.active, |m| {
            let columns = m.owned_columns(at, t.lane)?;
            m.when(every, |m| {
                let sums = m.row_sums(at, t.start, t.column, x, &columns, &weight, None)?;
                m.write_sums(at, t.own_at, &columns, &sums, grad)
            })?;
            m.when(some, |m| {
                let kept = |m: &mut Self, j: Word| -> Result<Word, EmitError> {
                    let row = m.add(t.start, j)?;
                    m.unmasked(row)
                };
                let sums = m.row_sums(at, t.start, t.column, x, &columns, &weight, Some(&kept))?;
                m.write_sums(at, t.own_at, &columns, &sums, grad)
            })
        })
    }

    /// The key rows of an `AttentionGradK` or `AttentionGradV` workgroup. One invocation per
    /// query of the sequence reads or computes the query's softmax statistics and, holding the
    /// query head (and for `AttentionGradK` the gradient head), computes its weight with every
    /// unmasked key row of the tile, `dS` for `AttentionGradK` and `P` for `AttentionGradV`;
    /// then each invocation sums the columns of the row it owns over the queries, `+0` for a
    /// masked key.
    fn key_tile(&mut self, at: &Attend, t: Tile, local: Word) -> Result<(), EmitError> {
        let grad_k = at.op == AttentionOp::GradK;
        let (s, zero, one) = (self.uint(at.s), self.uint(0), self.uint(1));
        let size = self.uint(ATTENTION_WORKGROUP);
        let (queries, rows) = (self.u32_vars[0], self.extra_u32_vars[3]);
        self.counted(queries, local, s, size, |m, i| {
            let row = m.add(t.start, i)?;
            let q_at = head_at(m, at, row, t.column)?;
            let (softmax, rd) = if at.stats {
                let max_at = m.stats_at(at, row, t.head, 0)?;
                let sum_at = m.stats_at(at, row, t.head, 1)?;
                let softmax = (
                    m.read(STATS_BINDING, max_at)?,
                    m.read(STATS_BINDING, sum_at)?,
                );
                let rd = if grad_k {
                    let rd_at = m.stats_at(at, row, t.head, 2)?;
                    Some(m.read(STATS_BINDING, rd_at)?)
                } else {
                    None
                };
                (softmax, rd)
            } else {
                let softmax = m.attention_softmax(at, q_at, t.start, t.column)?;
                let rd = if grad_k {
                    Some(m.row_dot(at, q_at, t.start, t.column, softmax)?)
                } else {
                    None
                };
                (softmax, rd)
            };
            let q = m.held_head(at, 0, q_at)?;
            let g = if grad_k {
                m.held_head(at, 4, q_at)?
            } else {
                None
            };
            m.counted(rows, zero, t.rows, one, |m, r| {
                let key_row = m.add(t.first, r)?;
                let keep = m.unmasked(key_row)?;
                m.when(keep, |m| {
                    let k_at = head_at(m, at, key_row, t.column)?;
                    let dot =
                        m.dot(Head::of(q.as_deref(), 0, q_at), Head::Global(1, k_at), at.d)?;
                    let scale = m.float(at.scale);
                    let x = m.f_mul(dot, scale)?;
                    let d = m.f_sub(x, softmax.0)?;
                    let e = m.exp(d)?;
                    let p = m.f_div(e, softmax.1)?;
                    let weight = match rd {
                        Some(rd) => {
                            let dp = m.dot(
                                Head::of(g.as_deref(), 4, q_at),
                                Head::Global(2, k_at),
                                at.d,
                            )?;
                            let diff = m.f_sub(dp, rd)?;
                            m.f_mul(p, diff)?
                        }
                        None => p,
                    };
                    let weight_at = m.mul(r, s)?;
                    let weight_at = m.add(weight_at, i)?;
                    m.store_shared(weight_at, weight)
                })
            })
        })?;
        self.barrier()?;
        let x = if grad_k { 0 } else { 4 };
        self.when(t.active, |m| {
            let keep = m.unmasked(t.own)?;
            let masked = m.b.logical_not(m.bool_ty, None, keep)?;
            let columns = m.owned_columns(at, t.lane)?;
            m.when(masked, |m| {
                let zero = m.float(0.0);
                let zeros = vec![zero; columns.len()];
                m.write_sums(at, t.own_at, &columns, &zeros, false)
            })?;
            m.when(keep, |m| {
                let weight = |m: &mut Self, i: Word| -> Result<Word, EmitError> {
                    let weight_at = m.add(t.base, i)?;
                    m.load_shared(weight_at)
                };
                let sums = m.row_sums(at, t.start, t.column, x, &columns, &weight, None)?;
                m.write_sums(at, t.own_at, &columns, &sums, grad_k)
            })
        })
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
