//! The modules of the attention and cross-entropy kernels (DESIGN.md § 5.2, coordinator
//! decision D73): `Attention`, `AttentionGradQ`, `AttentionGradK` and `AttentionGradV` one
//! workgroup per sequence, head and tile of rows, the `CrossEntropy` loss one invocation per
//! row, then a loss sum of one workgroup of 64, `CrossEntropyGrad` one invocation per row and one
//! more, then one per gradient element; and the corrected square root and quotient of the `Sqrt`
//! and `Div` elementwise kernels.
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
//! `AttentionGradV` (story:vulkan-attention-workgroup-per-head). Key row `r` is key `r % S` of
//! sequence `r / S`, with `S` at most `modelspec_core::schedule::MAX_ATTENTION_KEYS`, query row
//! `r` query `r % L` of sequence `r / L`, `1 <= L <= S` (story:attention-fewer-query-rows), and
//! a key is masked where its mask entry is 0. The workgroup computes the weights of its rows,
//! scores, probabilities and their gradients, into workgroup memory between barriers, one
//! invocation per key or query and holding that row's head, so each weight is computed once;
//! the scores are computed from the same operands by the same instructions wherever they are
//! used, so they hold the same value at every use. Then the `ATTENTION_WORKGROUP / T`
//! invocations of each row each sum their columns of it, every column `lane + u · lanes` in its
//! own variable, so a weight is loaded once for all of them. `Attention` and `AttentionGradQ` sum
//! without a mask check, several keys loaded at a time, when the mask keeps every key of the
//! sequence, which the workgroup finds before it sums. The softmax maximum and sum
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
//! `CrossEntropy` invocation `n` writes the loss `max + ln(sum) - logits[label]` of row `n`.
//! The loss sum's workgroup stages those row losses and their weights 64 rows at a time in
//! workgroup memory, invocation `i` row `i` of the block, and after a barrier invocation 0 adds
//! the block's weighted losses and weights to the numerator and denominator in ascending row
//! order; the loss is `+0` when the denominator is 0, else their quotient. A label outside the
//! classes reads class 0; the CPU refuses one before it runs.
//! `CrossEntropyGrad` row `n` is `(g · w[n] / sum(w)) · (p - [class == label])` per class, `+0`
//! where the weight sum is 0. Its first stage's invocation `n` writes the maximum and the sum of
//! `exp(x - max)` of row `n` by the instructions of the row loss, and invocation `N` the weight
//! sum in ascending row order; its second stage's invocation `n · V + c` reads them and computes
//! element `c` of row `n` by the instructions the row's loop ran for it before the split.

use super::{EmitError, Kernel, KernelKind, Operand, Shell, Word, operands_error, static_shape};
use modelspec_core::ir::kernel::SOFTMAX_STATS_PER_QUERY;
use modelspec_core::ir::tensor::DType;
use modelspec_core::schedule::{
    ATTENTION_STAGE_VALUES, ATTENTION_WORKGROUP, MAX_ATTENTION_KEYS, MAX_SPAN_KEYS, attention_tile,
    attention_workgroups,
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

/// The shapes of an attention kernel: `S` keys and `L` queries per sequence, `H` columns per
/// row, `d = H / heads` per head, and the scale `1 / sqrt(d)` as the CPU computes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Attend {
    op: AttentionOp,
    s: u32,
    l: u32,
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
    /// The columns of a head a workgroup stages per row of a chunk for its dot products
    /// ([`Shell::staged_dots`]): `d`, at most `8 · tile`.
    stage_columns: u32,
    /// The spans the kernel reads in place of a key mask, `None` for a key mask
    /// (story:attention-event-spans).
    spans: Option<Spans>,
}

/// The spans of an attention kernel over spans (story:attention-event-spans): its fourth
/// binding holds `U` pairs `(start, length)` as `u32`; span `u` is `S` "sequence" `u` of the
/// workgroup layout, its keys rows `start .. start + length` of the `n` rows of `k` and `v`,
/// `start` clamped to `n` and `length` to the rows left and to `S`
/// (`modelspec_core::schedule::MAX_SPAN_KEYS`). Its queries are the same rows of `q`, or row `u`
/// alone with `per_span`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Spans {
    n: u32,
    u: u32,
    per_span: bool,
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
    /// Stage 1 of the loss: the unweighted loss of each row.
    CrossEntropyRows {
        classes: u32,
    },
    /// Stage 2: the weighted mean of the row losses.
    CrossEntropy {
        rows: u32,
    },
    /// Stage 1 of the gradient: the maximum and exponential sum of each row, then the weight
    /// sum.
    CrossEntropyGradRows {
        rows: u32,
        classes: u32,
    },
    /// Stage 2: the gradient of each element.
    CrossEntropyGrad {
        rows: u32,
        classes: u32,
        seed: Seed,
    },
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
            'attention: {
                let arity = if op == AttentionOp::Forward { 4 } else { 5 };
                let heads = match immediates[..] {
                    [heads] if heads >= 1.0 && heads.fract() == 0.0 => heads as u64,
                    _ => return Err(wrong("an attention kernel carries its head count last")),
                };
                let spans = buffers.len() == arity && types[3] == I32;
                if buffers.len() != arity
                    || types
                        .iter()
                        .enumerate()
                        .any(|(at, &t)| t != F32 && !(spans && at == 3))
                {
                    return Err(wrong(
                        "attention reads f32 q, k, v, a key mask and its gradient",
                    ));
                }
                if spans {
                    break 'attention Attend::over_spans(kernel, op, heads, &shapes, &result)?;
                }
                let (n, h) = match shapes[0] {
                    [n, h] => (*n, *h),
                    _ => return Err(wrong("attention reads [n, h] rows")),
                };
                let m = match shapes[1] {
                    [m, _] => *m,
                    _ => return Err(wrong("attention reads [m, h] keys")),
                };
                let rows_match = [1, 2].into_iter().all(|at| shapes[at] == [m, h])
                    && (arity == 4 || shapes[4] == [n, h]);
                let (b, s) = match shapes[3] {
                    [b, s] => (*b, *s),
                    _ => return Err(wrong("attention reads a [b, s] key mask")),
                };
                let written = match op {
                    AttentionOp::GradK | AttentionOp::GradV => [m, h],
                    AttentionOp::Forward | AttentionOp::GradQ => [n, h],
                };
                if !rows_match
                    || result != written
                    || h % heads != 0
                    || b.checked_mul(s) != Some(m)
                    || n % b != 0
                    || !(1..=s).contains(&(n / b))
                    || s > MAX_ATTENTION_KEYS
                {
                    return Err(wrong(
                        "attention reads [b l, h] queries and [b s, h] keys, 1 <= l <= s, h a \
                     multiple of the heads, at most 512 keys",
                    ));
                }
                let l = n / b;
                let rows = match op {
                    AttentionOp::GradK | AttentionOp::GradV => s,
                    AttentionOp::Forward | AttentionOp::GradQ => l,
                };
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
                let groups = attention_workgroups(b, rows, s, heads)
                    .and_then(fits)
                    .filter(|&groups| {
                        u64::from(groups) * u64::from(ATTENTION_WORKGROUP) <= u32::MAX.into()
                    })
                    .ok_or_else(|| wrong("more than 2^32 invocations"))?;
                let tile = attention_tile(rows, s);
                let d = h / heads;
                (
                    KernelKind::PerRow,
                    groups * ATTENTION_WORKGROUP,
                    AttentionLoss::Attention(Attend {
                        op,
                        s: u(s),
                        l: u(l),
                        h: u(h),
                        d: u(d),
                        heads: u(heads),
                        scale: 1.0f32 / (d as f32).sqrt(),
                        stats,
                        tile: u(tile),
                        tiles: u(rows.div_ceil(tile)),
                        lanes: ATTENTION_WORKGROUP / u(tile),
                        groups,
                        stage_columns: u(
                            d.min(ATTENTION_STAGE_VALUES * tile / u64::from(ATTENTION_WORKGROUP))
                        ),
                        spans: None,
                    }),
                )
            }
        } else {
            match (
                kernel.op,
                types.as_slice(),
                shapes.as_slice(),
                result.as_slice(),
                immediates.as_slice(),
            ) {
                ("CrossEntropy", [F32, I32], [[n, v], [ln]], [rn], []) if n == ln && n == rn => (
                    KernelKind::PerRow,
                    u(*n),
                    AttentionLoss::CrossEntropyRows { classes: u(*v) },
                ),
                ("CrossEntropy", [F32, F32], [[n], [wn]], [], []) if n == wn => (
                    KernelKind::LossSum,
                    1,
                    AttentionLoss::CrossEntropy { rows: u(*n) },
                ),
                ("CrossEntropyGrad", [F32, I32, F32], [[n, v], [ln], [wn]], [sn, 2], [])
                    if n == ln && n == wn && Some(*sn) == n.checked_add(1) =>
                {
                    fits(sn.saturating_mul(2)).ok_or_else(|| wrong("2^32 row statistics"))?;
                    let invocations = u(*sn);
                    (
                        KernelKind::PerRow,
                        invocations,
                        AttentionLoss::CrossEntropyGradRows {
                            rows: u(*n),
                            classes: u(*v),
                        },
                    )
                }
                (
                    "CrossEntropyGrad",
                    [F32, F32, I32, F32, rest @ ..],
                    [[sn, 2], [n, v], [ln], [wn], seed_shape @ ..],
                    [rn, rv],
                    immediate,
                ) if n == ln && n == wn && Some(*sn) == n.checked_add(1) && (n, v) == (rn, rv) => {
                    let seed = match (rest, seed_shape, immediate) {
                        ([], [], [g]) => Seed::Immediate(*g),
                        ([F32], [[]], []) => Seed::Buffer,
                        _ => return Err(unsupported()),
                    };
                    let invocations = fits(n.saturating_mul(*v))
                        .ok_or_else(|| wrong("more than 2^32 gradient elements"))?;
                    (
                        KernelKind::PerElement,
                        invocations,
                        AttentionLoss::CrossEntropyGrad {
                            rows: u(*n),
                            classes: u(*v),
                            seed,
                        },
                    )
                }
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

    /// The `f32` values of workgroup memory the module stages: the row losses of one block and
    /// their weights; the [`Attend::staged`] values of an attention workgroup.
    pub(super) fn shared(self) -> Option<u32> {
        match self {
            AttentionLoss::CrossEntropy { .. } => Some(2 * GROUP),
            AttentionLoss::Attention(at) => Some(at.staged()),
            _ => None,
        }
    }

    /// The `f32` function variables the module keeps beside the shell's one: the softmax
    /// maximum and sum, the row dot and the output sums of an attention kernel; the row sum of a
    /// row loss; the numerator and denominator of the loss sum; the row sum of the gradient.
    pub(super) fn f32_vars(self) -> usize {
        match self {
            AttentionLoss::Attention(at) => OUTPUT_SUMS + at.columns() as usize,
            AttentionLoss::CrossEntropy { .. } => 2,
            AttentionLoss::CrossEntropyRows { .. }
            | AttentionLoss::CrossEntropyGradRows { .. }
            | AttentionLoss::CrossEntropyGrad { .. } => 1,
        }
    }

    /// The `u32` function variables the module keeps beside the shell's two: the column counter
    /// of a head's dot product, the found flag of the softmax maximum, and the counters of the
    /// keys, the rows of a tile and the terms of a sum of an attention kernel; the term counter of
    /// the loss sum.
    pub(super) fn u32_vars(self) -> usize {
        match self {
            AttentionLoss::Attention(_) => 5,
            AttentionLoss::CrossEntropy { .. } => 1,
            AttentionLoss::CrossEntropyRows { .. }
            | AttentionLoss::CrossEntropyGradRows { .. }
            | AttentionLoss::CrossEntropyGrad { .. } => 0,
        }
    }
}

impl Attend {
    /// The module of an attention kernel over spans (story:attention-event-spans), its
    /// operands' types checked: `q` (and `g`) `[n, h]`, `k` and `v` `[m, h]`, spans `[U, 2]`,
    /// `n = m` or `n = U`, `h` a multiple of the heads; its kind, invocations and body.
    fn over_spans(
        kernel: &Kernel,
        op: AttentionOp,
        heads: u64,
        shapes: &[&[u64]],
        result: &[u64],
    ) -> Result<(KernelKind, u32, AttentionLoss), EmitError> {
        let wrong = |what: &str| operands_error(kernel, what.to_owned());
        let (n, h, m, u) = match (shapes[0], shapes[1], shapes[3]) {
            ([n, h], [m, _], [u, 2]) => (*n, *h, *m, *u),
            _ => {
                return Err(wrong(
                    "attention reads [n, h] rows, [m, h] keys and [u, 2] spans",
                ));
            }
        };
        let rows_match = [1, 2].into_iter().all(|at| shapes[at] == [m, h])
            && (shapes.len() == 4 || shapes[4] == [n, h]);
        let written = match op {
            AttentionOp::GradK | AttentionOp::GradV => [m, h],
            AttentionOp::Forward | AttentionOp::GradQ => [n, h],
        };
        if !rows_match || result != written || h % heads != 0 || (n != m && n != u) {
            return Err(wrong(
                "attention over spans reads [n, h] queries, one per key row or one per span, \
                 and [m, h] keys, h a multiple of the heads",
            ));
        }
        let s = MAX_SPAN_KEYS;
        let per_span = n != m;
        let rows = match op {
            AttentionOp::GradK | AttentionOp::GradV => s,
            AttentionOp::Forward | AttentionOp::GradQ if per_span => 1,
            AttentionOp::Forward | AttentionOp::GradQ => s,
        };
        let stats = match &kernel.softmax_stats {
            None => false,
            Some(ty) => {
                if ty.dtype != DType::F32
                    || static_shape(kernel, ty)?
                        != [n.saturating_mul(heads), SOFTMAX_STATS_PER_QUERY]
                {
                    return Err(wrong(
                        "attention shares [n · heads, 3] f32 softmax statistics",
                    ));
                }
                true
            }
        };
        let groups = attention_workgroups(u, rows, s, heads)
            .and_then(|groups| u32::try_from(groups).ok())
            .filter(|&groups| {
                groups > 0 && u64::from(groups) * u64::from(ATTENTION_WORKGROUP) <= u32::MAX.into()
            })
            .ok_or_else(|| wrong("more than 2^32 invocations"))?;
        let tile = attention_tile(rows, s);
        let d = h / heads;
        let at = Attend {
            op,
            s: s as u32,
            l: if per_span { 1 } else { s as u32 },
            h: h as u32,
            d: d as u32,
            heads: heads as u32,
            scale: 1.0f32 / (d as f32).sqrt(),
            stats,
            tile: tile as u32,
            tiles: rows.div_ceil(tile) as u32,
            lanes: ATTENTION_WORKGROUP / tile as u32,
            groups,
            stage_columns: d.min(ATTENTION_STAGE_VALUES * tile / u64::from(ATTENTION_WORKGROUP))
                as u32,
            spans: Some(Spans {
                n: m as u32,
                u: u as u32,
                per_span,
            }),
        };
        Ok((
            KernelKind::PerRow,
            groups * ATTENTION_WORKGROUP,
            AttentionLoss::Attention(at),
        ))
    }

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
        let operands = match self.op {
            AttentionOp::GradK => 2,
            _ => 1,
        };
        self.stage_base() + operands * self.stage()
    }

    /// Where a workgroup's staged operands start in its workgroup memory: after the prefix
    /// length of `Attention` and `AttentionGradQ`, after the weights of `AttentionGradK` and
    /// `AttentionGradV`.
    fn stage_base(self) -> u32 {
        match self.op {
            AttentionOp::Forward | AttentionOp::GradQ => self.prefix_length() + 1,
            AttentionOp::GradK | AttentionOp::GradV => self.weights(),
        }
    }

    /// The stride of a staged column: one value per row of a chunk and one against bank
    /// conflicts.
    fn stage_stride(self) -> u32 {
        self.lanes + 1
    }

    /// The `f32` values of one staged operand: [`Attend::stage_columns`] columns of
    /// [`Attend::stage_stride`], within `modelspec_core::schedule::attention_stage_values`.
    fn stage(self) -> u32 {
        self.stage_columns * self.stage_stride()
    }

    /// Where an `Attention` or `AttentionGradQ` workgroup keeps whether the mask keeps a prefix
    /// of the keys of its sequence, keys `0..L` and no other, `1.0` if it does, else `0.0`:
    /// after its rows' statistics.
    fn prefix_flag(self) -> u32 {
        self.stats_base() + SOFTMAX_STATS * self.tile
    }

    /// Where an `Attention` or `AttentionGradQ` workgroup keeps the `L` of a prefix mask, as an
    /// `f32`: after [`Attend::prefix_flag`].
    fn prefix_length(self) -> u32 {
        self.prefix_flag() + 1
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

/// The words of one attention workgroup and invocation: the first key row and the first query
/// row of its sequence, the first row of its tile (a query row of `Attention` and
/// `AttentionGradQ`, a key row of the other two), the rows of the tile in the sequence, the first
/// column of its head, the head, the invocation's row of the tile and its lane among the row's
/// invocations, whether the row is one of the sequence, the row, its head's place, and where its
/// weights start in workgroup memory.
#[derive(Clone, Copy)]
struct Tile {
    start: Word,
    queries: Word,
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
    /// The keys and the queries of the sequence: `S` and `L`, or a span's length and its
    /// queries, the length or 1 (story:attention-event-spans).
    keys: Word,
    queries_count: Word,
}

/// A word an attention module computes from a row of a sequence: a weight, or whether the mask
/// keeps it.
type OfRow<'a> = &'a dyn Fn(&mut Shell, Word) -> Result<Word, EmitError>;

/// The rows of a sequence an output sum takes ([`Shell::row_sums`]).
#[derive(Clone, Copy)]
enum Terms<'a> {
    /// The first rows, as many as the constant, every one without a mask check.
    All(u32),
    /// The first rows, as many as the word holds, without a mask check: the keys of a mask that
    /// keeps a prefix of them.
    First(Word),
    /// Every row the function keeps.
    Kept(OfRow<'a>),
}

/// The head of a key an invocation holds, if it holds one ([`Shell::held_head`]).
type HeldOf<'a> = &'a dyn Fn(&mut Shell, Word) -> Result<Option<Vec<Word>>, EmitError>;

/// The end of [`Shell::staged_dots`] for one row: the shell, the row, whether it is below the
/// count and its sums.
type Finish<'a> = &'a dyn Fn(&mut Shell, Word, Word, &[Word]) -> Result<(), EmitError>;

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

    /// `then` under `cond`, or unconditionally without one.
    fn when_some(
        &mut self,
        cond: Option<Word>,
        then: impl FnOnce(&mut Self) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        match cond {
            Some(cond) => self.when(cond, then),
            None => then(self),
        }
    }

    /// The module body of `body`.
    pub(super) fn attention_loss(&mut self, body: AttentionLoss) -> Result<(), EmitError> {
        let guarded = match body {
            AttentionLoss::CrossEntropy { rows } => return self.cross_entropy_sum(rows),
            AttentionLoss::Attention(at) => return self.attention(at),
            guarded => guarded,
        };
        let (index, count) = (self.index, self.count);
        let inside = self.less(index, count)?;
        self.when(inside, |m| match guarded {
            AttentionLoss::CrossEntropyRows { classes } => m.cross_entropy_row(classes),
            AttentionLoss::CrossEntropyGradRows { rows, classes } => {
                m.cross_entropy_grad_rows(rows, classes)
            }
            AttentionLoss::CrossEntropyGrad {
                rows,
                classes,
                seed,
            } => m.cross_entropy_grad(rows, classes, seed),
            AttentionLoss::CrossEntropy { .. } | AttentionLoss::Attention(_) => {
                unreachable!("handled above")
            }
        })
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

    /// `body(j, k_at)` for each unmasked key row `j` of the first `keys` rows of the sequence from
    /// row `start`, in ascending order, `k_at` its head's place; every one of them without
    /// `checked`, where the caller knows the mask keeps them. The key counter is the shell's
    /// second `u32` variable.
    fn each_key(
        &mut self,
        at: &Attend,
        start: Word,
        keys: Word,
        column: Word,
        checked: bool,
        mut body: impl FnMut(&mut Self, Word, Word) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        let var = self.u32_vars[1];
        let one = self.uint(1);
        let end = self.add(start, keys)?;
        self.counted(var, start, end, one, |m, j| {
            if !checked {
                let k_at = head_at(m, at, j, column)?;
                return body(m, j, k_at);
            }
            let keep = m.unmasked(j)?;
            m.when(keep, |m| {
                let k_at = head_at(m, at, j, column)?;
                body(m, j, k_at)
            })
        })
    }

    /// The softmax of the query head at `q_at` over the unmasked keys of its sequence from row
    /// `start`: the maximum score, the first unmasked one replaced by each later greater one,
    /// then the sum of `exp(score - max)` in ascending key order. Over spans, `span_keys` the
    /// span's length, its keys without a mask check (story:attention-event-spans).
    fn attention_softmax(
        &mut self,
        at: &Attend,
        q_at: Word,
        start: Word,
        column: Word,
        span_keys: Option<Word>,
    ) -> Result<(Word, Word), EmitError> {
        let keys = match span_keys {
            Some(keys) => keys,
            None => self.uint(at.s),
        };
        let checked = span_keys.is_none();
        self.softmax_of(at, start, keys, column, checked, &|m, _, k_at| {
            m.score(at, q_at, k_at)
        })
    }

    /// The softmax of [`Shell::attention_softmax`] over the scores `score(j, k_at)` of the
    /// unmasked key rows `j` of the first `keys` rows of the sequence from row `start`, `k_at`
    /// their head's place: [`Shell::softmax_max`], then [`Shell::exp_sum`] of `exp(score - max)`.
    fn softmax_of(
        &mut self,
        at: &Attend,
        start: Word,
        keys: Word,
        column: Word,
        checked: bool,
        score: &dyn Fn(&mut Self, Word, Word) -> Result<Word, EmitError>,
    ) -> Result<(Word, Word), EmitError> {
        let max = self.softmax_max(at, start, keys, column, checked, score)?;
        let total = self.exp_sum(at, start, keys, column, checked, &|m, j, k_at| {
            let x = score(m, j, k_at)?;
            let d = m.f_sub(x, max)?;
            m.exp(d)
        })?;
        Ok((max, total))
    }

    /// The softmax maximum of the scores `score(j, k_at)` of the key rows of
    /// [`Shell::each_key`]: the first one, replaced by each later greater one; `0.0` without
    /// one.
    fn softmax_max(
        &mut self,
        at: &Attend,
        start: Word,
        keys: Word,
        column: Word,
        checked: bool,
        score: &dyn Fn(&mut Self, Word, Word) -> Result<Word, EmitError>,
    ) -> Result<Word, EmitError> {
        let flag = self.extra_u32_vars[1];
        let max_var = self.matmul_vars[0];
        let (zero, one) = (self.uint(0), self.uint(1));
        let f_zero = self.float(0.0);
        self.b.store(flag, zero, None, [])?;
        self.b.store(max_var, f_zero, None, [])?;
        self.each_key(at, start, keys, column, checked, |m, j, k_at| {
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
        Ok(self.b.load(self.f32_ty, None, max_var, None, [])?)
    }

    /// The softmax sum of the terms `term(j, k_at)` of the key rows of [`Shell::each_key`], each
    /// `exp(score - max)`: from `-0.0`, each added in ascending key order.
    fn exp_sum(
        &mut self,
        at: &Attend,
        start: Word,
        keys: Word,
        column: Word,
        checked: bool,
        term: &dyn Fn(&mut Self, Word, Word) -> Result<Word, EmitError>,
    ) -> Result<Word, EmitError> {
        let total_var = self.matmul_vars[1];
        let start_sum = self.float(-0.0);
        self.b.store(total_var, start_sum, None, [])?;
        self.each_key(at, start, keys, column, checked, |m, j, k_at| {
            let e = term(m, j, k_at)?;
            let total = m.b.load(m.f32_ty, None, total_var, None, [])?;
            let sum = m.f_add(total, e)?;
            m.b.store(total_var, sum, None, [])?;
            Ok(())
        })?;
        Ok(self.b.load(self.f32_ty, None, total_var, None, [])?)
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
    /// ascending key order; over spans the `span_keys` keys of the span, without a mask check.
    fn row_dot(
        &mut self,
        at: &Attend,
        q_at: Word,
        start: Word,
        column: Word,
        softmax: (Word, Word),
        span_keys: Option<Word>,
    ) -> Result<Word, EmitError> {
        let acc = self.matmul_vars[2];
        let start_sum = self.float(-0.0);
        self.b.store(acc, start_sum, None, [])?;
        let keys = match span_keys {
            Some(keys) => keys,
            None => self.uint(at.s),
        };
        let checked = span_keys.is_none();
        self.each_key(at, start, keys, column, checked, |m, _, k_at| {
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
    /// `start` that `terms` names ([`Terms`]), `i_at` their head's place at `column`: each starts
    /// at `-0.0` and fuses each product into it, as the CPU's sums of `P · V`, `dS · K`, `dS · Q`
    /// and `P · G`. A row's weight is loaded once for all the columns. Without a mask check the
    /// rows are taken [`Attend::sum_rows`] at a time, every value loaded before the first product
    /// is fused, and the rows past the last whole group one at a time. Uses the term counter and
    /// the output sums.
    #[allow(clippy::too_many_arguments)]
    fn row_sums(
        &mut self,
        at: &Attend,
        start: Word,
        column: Word,
        x: usize,
        columns: &[(Word, Option<Word>)],
        weight: OfRow<'_>,
        terms: Terms<'_>,
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
        match terms {
            Terms::Kept(kept) => self.counted(var, zero, s, one, |m, i| {
                let keep = kept(m, i)?;
                m.when(keep, |m| {
                    let term = load(m, i)?;
                    fuse(m, &[term])
                })
            })?,
            Terms::First(count) => {
                let rows = at.sum_rows();
                let step = self.uint(rows);
                let left = self.rem(count, step)?;
                let whole = self.sub(count, left)?;
                self.counted(var, zero, whole, step, |m, first| {
                    let terms = (0..rows)
                        .map(|u| {
                            let offset = m.uint(u);
                            let i = m.add(first, offset)?;
                            load(m, i)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    fuse(m, &terms)
                })?;
                self.counted(var, whole, count, one, |m, i| {
                    let term = load(m, i)?;
                    fuse(m, &[term])
                })?;
            }
            Terms::All(count) => {
                let rows = at.sum_rows();
                let whole = count / rows * rows;
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
                let terms = (whole..count)
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

    /// The dot products of the invocation's own heads with the rows `origin + i` for `i` below
    /// `count` (story:vulkan-attention-tiled-decider-shapes). The rows are taken in chunks of
    /// [`Attend::lanes`], on the chunk counter `var`, and each `(own, binding)` of `dots` is a dot
    /// product of the own head `own` with the head of row `chunk + lane` of `binding`. The
    /// workgroup stages the chunk's heads of each binding in workgroup memory,
    /// [`Attend::stage_columns`] columns at a time, column `c` of row `i` at `c · (lanes + 1) +
    /// i`, every invocation reading consecutive columns of a row from the binding; between
    /// barriers each invocation then fuses the products of the staged columns into its sums.
    /// Each sum starts at `-0.0` and fuses the products of the head's columns in ascending
    /// order, `own` first in each product when `own_first`, else second, as
    /// [`Shell::dot`] computes it. Then `finish(i, valid, sums)` runs on every invocation, `i`
    /// the row of the sequence its sums belong to and `valid` whether `i` is below `count`.
    #[allow(clippy::too_many_arguments)]
    fn staged_dots(
        &mut self,
        at: &Attend,
        t: &Tile,
        local: Word,
        var: Word,
        origin: Word,
        count: Word,
        dots: &[(Head<'_>, usize)],
        own_first: bool,
        finish: Finish<'_>,
    ) -> Result<(), EmitError> {
        let (zero, lanes) = (self.uint(0), self.uint(at.lanes));
        let stride = at.stage_stride();
        self.counted(var, zero, count, lanes, |m, chunk| {
            let start_sum = m.float(-0.0);
            let mut sums = vec![start_sum; dots.len()];
            let mut from = 0;
            while from < at.d {
                let width = at.stage_columns.min(at.d - from);
                let elements = at.lanes * width;
                m.barrier()?;
                for u in 0..elements.div_ceil(ATTENTION_WORKGROUP) {
                    let offset = m.uint(u * ATTENTION_WORKGROUP);
                    let e = m.add(local, offset)?;
                    let columns = m.uint(width);
                    let i = m.div(e, columns)?;
                    let c = m.rem(e, columns)?;
                    let index = m.add(chunk, i)?;
                    let below = m.less(index, count)?;
                    let inside = if (u + 1) * ATTENTION_WORKGROUP <= elements {
                        None
                    } else {
                        let n = m.uint(elements);
                        Some(m.less(e, n)?)
                    };
                    let valid = match inside {
                        Some(inside) => m.and(below, inside)?,
                        None => below,
                    };
                    let safe = m.b.select(m.u32_ty, None, valid, index, zero)?;
                    let row = m.add(origin, safe)?;
                    let row_at = head_at(m, at, row, t.column)?;
                    let first = m.uint(from);
                    let row_at = m.add(row_at, first)?;
                    let place = m.add(row_at, c)?;
                    let step = m.uint(stride);
                    let slot = m.mul(c, step)?;
                    let slot = m.add(slot, i)?;
                    let stores = dots
                        .iter()
                        .enumerate()
                        .map(|(k, &(_, binding))| {
                            let value = m.read(binding, place)?;
                            let f_zero = m.float(0.0);
                            let value = m.b.select(m.f32_ty, None, valid, value, f_zero)?;
                            let base = m.uint(at.stage_base() + k as u32 * at.stage());
                            Ok((m.add(base, slot)?, value))
                        })
                        .collect::<Result<Vec<_>, EmitError>>()?;
                    let store = |m: &mut Self| -> Result<(), EmitError> {
                        for &(slot, value) in &stores {
                            m.store_shared(slot, value)?;
                        }
                        Ok(())
                    };
                    match inside {
                        Some(inside) => m.when(inside, store)?,
                        None => store(m)?,
                    }
                }
                m.barrier()?;
                for c in 0..width {
                    for (k, &(own, _)) in dots.iter().enumerate() {
                        let base = at.stage_base() + k as u32 * at.stage() + c * stride;
                        let base = m.uint(base);
                        let slot = m.add(base, t.lane)?;
                        let staged = m.load_shared(slot)?;
                        let column = from + c;
                        let mine = match own {
                            Head::Values(values) => values[column as usize],
                            Head::Global(binding, own_at) => {
                                let offset = m.uint(column);
                                let place = m.add(own_at, offset)?;
                                m.read(binding, place)?
                            }
                        };
                        let (x, y) = if own_first {
                            (mine, staged)
                        } else {
                            (staged, mine)
                        };
                        sums[k] = m.f_fma(x, y, sums[k])?;
                    }
                }
                from += width;
            }
            let index = m.add(chunk, t.lane)?;
            let valid = m.less(index, count)?;
            finish(m, index, valid, &sums)
        })
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
        if let Some(spans) = at.spans {
            return self.span_tile(at, spans, group, local);
        }
        let (tiles, heads) = (self.uint(at.tiles), self.uint(at.heads));
        let (s, d, rows_per_tile) = (self.uint(at.s), self.uint(at.d), self.uint(at.tile));
        let lanes = self.uint(at.lanes);
        let tile = self.rem(group, tiles)?;
        let pair = self.div(group, tiles)?;
        let head = self.rem(pair, heads)?;
        let sequence = self.div(pair, heads)?;
        let start = self.mul(sequence, s)?;
        let l = self.uint(at.l);
        let queries = self.mul(sequence, l)?;
        let column = self.mul(head, d)?;
        let tile_start = self.mul(tile, rows_per_tile)?;
        let (origin, count) = match at.op {
            AttentionOp::Forward | AttentionOp::GradQ => (queries, l),
            AttentionOp::GradK | AttentionOp::GradV => (start, s),
        };
        let first = self.add(origin, tile_start)?;
        let left = self.sub(count, tile_start)?;
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
            queries,
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
            keys: s,
            queries_count: l,
        })
    }

    /// The first row and the length of span `u`, `start` clamped to the `n` rows of the keys and
    /// the length to the rows left and to `S`, so a span the CPU refuses reads no row outside the
    /// keys (story:attention-event-spans).
    fn span_at(&mut self, at: &Attend, spans: Spans, u: Word) -> Result<(Word, Word), EmitError> {
        let (one, two, n, s) = (
            self.uint(1),
            self.uint(2),
            self.uint(spans.n),
            self.uint(at.s),
        );
        let start_at = self.mul(u, two)?;
        let length_at = self.add(start_at, one)?;
        let start = self.read_u32(3, start_at)?;
        let length = self.read_u32(3, length_at)?;
        let start = self.min(start, n)?;
        let room = self.sub(n, start)?;
        let length = self.min(length, room)?;
        let length = self.min(length, s)?;
        Ok((start, length))
    }

    /// The words of the invocation's workgroup `group` over spans (story:attention-event-spans):
    /// workgroup `(u · heads + head) · tiles + tile` takes rows `tile · T ..` of span `u`'s query
    /// rows (`Attention`, `AttentionGradQ`) or key rows (`AttentionGradK`, `AttentionGradV`),
    /// none where the span has fewer.
    fn span_tile(
        &mut self,
        at: &Attend,
        spans: Spans,
        group: Word,
        local: Word,
    ) -> Result<Tile, EmitError> {
        let (tiles, heads) = (self.uint(at.tiles), self.uint(at.heads));
        let (s, d, rows_per_tile) = (self.uint(at.s), self.uint(at.d), self.uint(at.tile));
        let (lanes, zero, one) = (self.uint(at.lanes), self.uint(0), self.uint(1));
        let tile = self.rem(group, tiles)?;
        let pair = self.div(group, tiles)?;
        let head = self.rem(pair, heads)?;
        let sequence = self.div(pair, heads)?;
        let (start, keys) = self.span_at(at, spans, sequence)?;
        let (queries, queries_count) = if spans.per_span {
            (sequence, one)
        } else {
            (start, keys)
        };
        let column = self.mul(head, d)?;
        let tile_start = self.mul(tile, rows_per_tile)?;
        let (origin, count) = match at.op {
            AttentionOp::Forward | AttentionOp::GradQ => (queries, queries_count),
            AttentionOp::GradK | AttentionOp::GradV => (start, keys),
        };
        let first = self.add(origin, tile_start)?;
        let some = self.less(tile_start, count)?;
        let left = self.sub(count, tile_start)?;
        let left = self.b.select(self.u32_ty, None, some, left, zero)?;
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
            queries,
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
            keys,
            queries_count,
        })
    }

    /// Writes `+0` to the head's columns of the rows of the result no span computes
    /// (story:attention-event-spans): the workgroups of span `u` the rows from its end to the
    /// start of span `u + 1`, or to the last row after the last span, and those of span 0 also
    /// the rows before it; the `T` tiles of a head share them, `ATTENTION_WORKGROUP` elements
    /// a workgroup at a time. A result of one row per span has no such rows.
    fn span_gaps(
        &mut self,
        at: &Attend,
        spans: Spans,
        t: &Tile,
        group: Word,
        local: Word,
    ) -> Result<(), EmitError> {
        let (zero, one, d, tiles) = (
            self.uint(0),
            self.uint(1),
            self.uint(at.d),
            self.uint(at.tiles),
        );
        let (n, last) = (self.uint(spans.n), self.uint(spans.u - 1));
        let size = self.uint(ATTENTION_WORKGROUP);
        let tile = self.rem(group, tiles)?;
        let pair = self.div(group, tiles)?;
        let heads = self.uint(at.heads);
        let u = self.div(pair, heads)?;
        let end = self.add(t.start, t.keys)?;
        let after = self.add(u, one)?;
        let more = self.less(u, last)?;
        let next_u = self.b.select(self.u32_ty, None, more, after, u)?;
        let (next, _) = self.span_at(at, spans, next_u)?;
        let next = self.b.select(self.u32_ty, None, more, next, n)?;
        let opens = self.equal(u, zero)?;
        let stride = self.mul(tiles, size)?;
        let offset = self.mul(tile, size)?;
        let first = self.add(offset, local)?;
        let var = self.extra_u32_vars[0];
        let gaps = [(zero, t.start, Some(opens)), (end, next, None)];
        for (from, to, when) in gaps {
            let fill = |m: &mut Self| -> Result<(), EmitError> {
                let wide = m.less(from, to)?;
                let width = m.sub(to, from)?;
                let width = m.b.select(m.u32_ty, None, wide, width, zero)?;
                let elements = m.mul(width, d)?;
                m.counted(var, first, elements, stride, |m, e| {
                    let row = m.div(e, d)?;
                    let c = m.rem(e, d)?;
                    let row = m.add(from, row)?;
                    let place = head_at(m, at, row, t.column)?;
                    let place = m.add(place, c)?;
                    let f_zero = m.float(0.0);
                    m.write(place, f_zero)
                })
            };
            match when {
                Some(when) => self.when(when, fill)?,
                None => fill(self)?,
            }
        }
        Ok(())
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
            let body = |m: &mut Self| match at.op {
                AttentionOp::Forward | AttentionOp::GradQ => m.query_tile(&at, tile, local),
                AttentionOp::GradK | AttentionOp::GradV => m.key_tile(&at, tile, local),
            };
            let Some(spans) = at.spans else {
                return body(m);
            };
            let per_query = matches!(at.op, AttentionOp::Forward | AttentionOp::GradQ);
            if !(spans.per_span && per_query) {
                m.span_gaps(&at, spans, &tile, group, local)?;
            }
            let zero = m.uint(0);
            let busy = m.less(zero, tile.rows)?;
            m.when(busy, body)
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
        // The keys every loop runs over and the forms of the loops: over spans the span's
        // keys, every one kept (story:attention-event-spans); under a key mask as below.
        let (end, forms) = if at.spans.is_some() {
            (t.keys, vec![(None, false)])
        } else {
            let (end, prefix, some) = self.prefix_keys(at, &t, local, keys)?;
            (end, vec![(Some(prefix), false), (Some(some), true)])
        };
        // `body(j, key row, weight place of row r)` for each row `r` of the tile and each key
        // `j` the mask keeps, one invocation per key, `head(j)` computed once per key.
        let each_kept =
            |m: &mut Self, head: HeldOf<'_>, body: KeyRow<'_>| -> Result<(), EmitError> {
                m.counted(keys, local, end, size, |m, j| {
                    let row = m.add(t.start, j)?;
                    let key = |m: &mut Self| {
                        let held = head(m, row)?;
                        m.counted(rows, zero, t.rows, one, |m, r| {
                            let weight_at = m.mul(r, s)?;
                            let weight_at = m.add(weight_at, j)?;
                            body(m, row, held.as_deref(), r, weight_at)
                        })
                    };
                    if at.spans.is_some() {
                        return key(m);
                    }
                    let keep = m.unmasked(row)?;
                    m.when(keep, key)
                })
            };
        // The scores of the invocation's row with each key below `end` the mask keeps, the
        // query head held and the key heads staged, a chunk of keys at a time.
        let own = self.b.select(self.u32_ty, None, t.active, t.own, t.first)?;
        let own_at = head_at(self, at, own, t.column)?;
        let (start, base, active) = (t.start, t.base, t.active);
        // `body(j, sums)` where the invocation's row is one of the tile's and key `j` is below
        // `end` and kept.
        let kept_key = |m: &mut Self,
                        j: Word,
                        valid: Word,
                        body: &dyn Fn(&mut Self, Word) -> Result<(), EmitError>|
         -> Result<(), EmitError> {
            let keep = if at.spans.is_some() {
                m.and(valid, active)?
            } else {
                let zero = m.uint(0);
                let safe = m.b.select(m.u32_ty, None, valid, j, zero)?;
                let row = m.add(start, safe)?;
                let kept = m.unmasked(row)?;
                let keep = m.and(valid, kept)?;
                m.and(keep, active)?
            };
            m.when(keep, |m| {
                let weight_at = m.add(base, j)?;
                body(m, weight_at)
            })
        };
        let q = self.held_head(at, 0, own_at)?;
        self.staged_dots(
            at,
            &t,
            local,
            keys,
            t.start,
            end,
            &[(Head::of(q.as_deref(), 0, own_at), 1)],
            true,
            &|m, j, valid, sums| {
                kept_key(m, j, valid, &|m, weight_at| {
                    let scale = m.float(at.scale);
                    let x = m.f_mul(sums[0], scale)?;
                    m.store_shared(weight_at, x)
                })
            },
        )?;
        self.barrier()?;
        let stats_base = self.uint(at.stats_base());
        let three = self.uint(SOFTMAX_STATS);
        // Where row `r`'s statistics start in workgroup memory.
        let row_stats = |m: &mut Self, r: Word| -> Result<Word, EmitError> {
            let slot = m.mul(r, three)?;
            m.add(stats_base, slot)
        };
        let total_at = self.add(t.row_stats, one)?;
        if grad && at.stats {
            self.when(serial, |m| {
                let max_at = m.stats_at(at, t.own, t.head, 0)?;
                let sum_at = m.stats_at(at, t.own, t.head, 1)?;
                let max = m.read_out(0, max_at)?;
                let total = m.read_out(0, sum_at)?;
                m.store_shared(t.row_stats, max)?;
                m.store_shared(total_at, total)
            })?;
            self.barrier()?;
            each_kept(self, &|_, _| Ok(None), &|m, _, _, r, weight_at| {
                let slot = row_stats(m, r)?;
                let total_at = m.add(slot, one)?;
                let max = m.load_shared(slot)?;
                let total = m.load_shared(total_at)?;
                let x = m.load_shared(weight_at)?;
                let d = m.f_sub(x, max)?;
                let e = m.exp(d)?;
                let p = m.f_div(e, total)?;
                m.store_shared(weight_at, p)
            })?;
        } else {
            // The softmax of each row from its scores: the first invocation of the row finds the
            // maximum, one invocation per key computes `exp(score - max)` in place, the first
            // invocation of the row sums them in ascending key order, one invocation per key
            // divides each by the sum in place. Under a prefix mask the row's loops read no mask.
            let (base, start) = (t.base, t.start);
            let weight = |m: &mut Self, row: Word, _: Word| -> Result<Word, EmitError> {
                let j = m.sub(row, start)?;
                let weight_at = m.add(base, j)?;
                m.load_shared(weight_at)
            };
            self.when(serial, |m| {
                for &(when, checked) in &forms {
                    m.when_some(when, |m| {
                        let max = m.softmax_max(at, start, end, t.column, checked, &weight)?;
                        m.store_shared(t.row_stats, max)
                    })?;
                }
                Ok(())
            })?;
            self.barrier()?;
            each_kept(self, &|_, _| Ok(None), &|m, _, _, r, weight_at| {
                let slot = row_stats(m, r)?;
                let max = m.load_shared(slot)?;
                let x = m.load_shared(weight_at)?;
                let d = m.f_sub(x, max)?;
                let e = m.exp(d)?;
                m.store_shared(weight_at, e)
            })?;
            self.barrier()?;
            self.when(serial, |m| {
                for &(when, checked) in &forms {
                    m.when_some(when, |m| {
                        let total = m.exp_sum(at, start, end, t.column, checked, &weight)?;
                        m.store_shared(total_at, total)
                    })?;
                }
                if at.stats {
                    let max = m.load_shared(t.row_stats)?;
                    let total = m.load_shared(total_at)?;
                    let max_at = m.stats_at(at, t.own, t.head, 0)?;
                    let sum_at = m.stats_at(at, t.own, t.head, 1)?;
                    m.write_to(0, max_at, max)?;
                    m.write_to(0, sum_at, total)?;
                }
                Ok(())
            })?;
            self.barrier()?;
            each_kept(self, &|_, _| Ok(None), &|m, _, _, r, weight_at| {
                let slot = row_stats(m, r)?;
                let total_at = m.add(slot, one)?;
                let total = m.load_shared(total_at)?;
                let e = m.load_shared(weight_at)?;
                let p = m.f_div(e, total)?;
                m.store_shared(weight_at, p)
            })?;
        }
        if grad {
            // `dP`, the gradient head of the invocation's row with each value head below `end`
            // the mask keeps, beside the probabilities.
            let g = self.held_head(at, 4, own_at)?;
            self.staged_dots(
                at,
                &t,
                local,
                keys,
                t.start,
                end,
                &[(Head::of(g.as_deref(), 4, own_at), 2)],
                true,
                &|m, j, valid, sums| {
                    kept_key(m, j, valid, &|m, weight_at| {
                        let dp_at = m.add(gradients, weight_at)?;
                        m.store_shared(dp_at, sums[0])
                    })
                },
            )?;
        }
        self.barrier()?;
        if grad {
            self.when(serial, |m| {
                let acc = m.matmul_vars[2];
                let start_sum = m.float(-0.0);
                m.b.store(acc, start_sum, None, [])?;
                // Under a prefix mask every key before `end` is kept, so the loop reads no mask.
                for &(when, checked) in &forms {
                    m.when_some(when, |m| {
                        m.counted(m.extra_u32_vars[4], zero, end, one, |m, j| {
                            let weight_at = m.add(t.base, j)?;
                            let term = |m: &mut Self| -> Result<(), EmitError> {
                                let p = m.load_shared(weight_at)?;
                                let dp_at = m.add(gradients, weight_at)?;
                                let dp = m.load_shared(dp_at)?;
                                let sum = m.b.load(m.f32_ty, None, acc, None, [])?;
                                let fused = m.f_fma(p, dp, sum)?;
                                m.b.store(acc, fused, None, [])?;
                                Ok(())
                            };
                            if checked {
                                let row = m.add(t.start, j)?;
                                let keep = m.unmasked(row)?;
                                m.when(keep, term)
                            } else {
                                term(m)
                            }
                        })
                    })?;
                }
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
        self.when(t.active, |m| {
            let columns = m.owned_columns(at, t.lane)?;
            let kept = |m: &mut Self, j: Word| -> Result<Word, EmitError> {
                let row = m.add(t.start, j)?;
                m.unmasked(row)
            };
            for &(when, checked) in &forms {
                m.when_some(when, |m| {
                    let terms = if checked {
                        Terms::Kept(&kept)
                    } else {
                        Terms::First(end)
                    };
                    let sums = m.row_sums(at, t.start, t.column, x, &columns, &weight, terms)?;
                    m.write_sums(at, t.own_at, &columns, &sums, grad)
                })?;
            }
            Ok(())
        })
    }

    /// Whether the key mask keeps a prefix of the keys of the tile's sequence, keys `0..L` and no
    /// other, whether it does not, and the keys the loops of [`Shell::query_tile`] run over:
    /// `L` under a prefix mask, else `S`.
    fn prefix_keys(
        &mut self,
        at: &Attend,
        t: &Tile,
        local: Word,
        keys: Word,
    ) -> Result<(Word, Word, Word), EmitError> {
        let (s, zero, one) = (self.uint(at.s), self.uint(0), self.uint(1));
        let size = self.uint(ATTENTION_WORKGROUP);
        // Whether the mask keeps a prefix of the sequence's keys, keys `0..L` and no other, and
        // that `L`: the first invocation sets the flag and `L = S`, then the invocation of each
        // key clears the flag where a kept key follows a masked one and writes `L = j` where
        // key `j` is masked and is the first key or follows a kept one. Under a prefix mask at
        // most one key writes `L`; under another the flag is cleared and `L` is not read.
        let (flag_at, length_at) = (self.uint(at.prefix_flag()), self.uint(at.prefix_length()));
        let first = self.equal(local, zero)?;
        self.when(first, |m| {
            let set = m.float(1.0);
            m.store_shared(flag_at, set)?;
            let all = m.float(at.s as f32);
            m.store_shared(length_at, all)
        })?;
        self.barrier()?;
        self.counted(keys, local, s, size, |m, j| {
            let row = m.add(t.start, j)?;
            let keep = m.unmasked(row)?;
            let opens = m.equal(j, zero)?;
            let before = m.sub(row, one)?;
            let before = m.b.select(m.u32_ty, None, opens, row, before)?;
            let kept_before = m.unmasked(before)?;
            let masked_before = m.b.logical_not(m.bool_ty, None, kept_before)?;
            let follows_masked = m.b.logical_and(m.bool_ty, None, masked_before, keep)?;
            let late = m.b.logical_not(m.bool_ty, None, opens)?;
            let breaks = m.and(follows_masked, late)?;
            m.when(breaks, |m| {
                let clear = m.float(0.0);
                m.store_shared(flag_at, clear)
            })?;
            let masked = m.b.logical_not(m.bool_ty, None, keep)?;
            let edge = m.b.logical_or(m.bool_ty, None, opens, kept_before)?;
            let ends = m.and(masked, edge)?;
            m.when(ends, |m| {
                let length = m.b.convert_u_to_f(m.f32_ty, None, j)?;
                m.store_shared(length_at, length)
            })
        })?;
        self.barrier()?;
        let flag = self.load_shared(flag_at)?;
        let set = self.float(1.0);
        let prefix = self.b.f_ord_equal(self.bool_ty, None, flag, set)?;
        let some = self.b.logical_not(self.bool_ty, None, prefix)?;
        let length = self.load_shared(length_at)?;
        let length = self.b.convert_f_to_u(self.u32_ty, None, length)?;
        // The keys every loop runs over: `0..L` under a prefix mask, else every key.
        let end = self.b.select(self.u32_ty, None, prefix, length, s)?;
        Ok((end, prefix, some))
    }

    /// The weights of an `AttentionGradK` or `AttentionGradV` workgroup that reads the shared
    /// softmax statistics (story:vulkan-attention-tiled-decider-shapes): each invocation computes
    /// the weight of its key row of the tile with the queries `chunk + lane` of the sequence, the
    /// query heads (and for `AttentionGradK` the gradient heads) staged a chunk at a time
    /// ([`Shell::staged_dots`]) and the key head (and value head) read in place, by the
    /// instructions of the per-query loop of [`Shell::key_tile`]: `dS` for `AttentionGradK` and
    /// `P` for `AttentionGradV`, where the key row is one of the tile's and the mask keeps it.
    fn staged_key_weights(
        &mut self,
        at: &Attend,
        t: &Tile,
        local: Word,
        var: Word,
    ) -> Result<(), EmitError> {
        let grad_k = at.op == AttentionOp::GradK;
        let l = t.queries_count;
        let own = self.b.select(self.u32_ty, None, t.active, t.own, t.first)?;
        let own_at = head_at(self, at, own, t.column)?;
        let keep_own = if at.spans.is_some() {
            t.active
        } else {
            let kept = self.unmasked(own)?;
            self.and(t.active, kept)?
        };
        let mut dots = vec![(Head::Global(1, own_at), 0)];
        if grad_k {
            dots.push((Head::Global(2, own_at), 4));
        }
        let (start, head, base) = (t.queries, t.head, t.base);
        self.staged_dots(
            at,
            t,
            local,
            var,
            start,
            l,
            &dots,
            false,
            &|m, i, valid, sums| {
                let keep = m.and(valid, keep_own)?;
                m.when(keep, |m| {
                    let row = m.add(start, i)?;
                    let max_at = m.stats_at(at, row, head, 0)?;
                    let sum_at = m.stats_at(at, row, head, 1)?;
                    let max = m.read(STATS_BINDING, max_at)?;
                    let total = m.read(STATS_BINDING, sum_at)?;
                    let scale = m.float(at.scale);
                    let x = m.f_mul(sums[0], scale)?;
                    let d = m.f_sub(x, max)?;
                    let e = m.exp(d)?;
                    let p = m.f_div(e, total)?;
                    let weight = if grad_k {
                        let rd_at = m.stats_at(at, row, head, 2)?;
                        let rd = m.read(STATS_BINDING, rd_at)?;
                        let diff = m.f_sub(sums[1], rd)?;
                        m.f_mul(p, diff)?
                    } else {
                        p
                    };
                    let weight_at = m.add(base, i)?;
                    m.store_shared(weight_at, weight)
                })
            },
        )
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
        let spans = at.spans.is_some();
        let weights = |m: &mut Self| -> Result<(), EmitError> {
            if at.stats {
                m.staged_key_weights(at, &t, local, queries)?;
                return m.barrier();
            }
            let l = t.queries_count;
            m.counted(queries, local, l, size, |m, i| {
                let row = m.add(t.queries, i)?;
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
                    let keys = spans.then_some(t.keys);
                    let softmax = m.attention_softmax(at, q_at, t.start, t.column, keys)?;
                    let rd = if grad_k {
                        Some(m.row_dot(at, q_at, t.start, t.column, softmax, keys)?)
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
                    let weight = |m: &mut Self| {
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
                    };
                    if spans {
                        return weight(m);
                    }
                    let keep = m.unmasked(key_row)?;
                    m.when(keep, weight)
                })
            })?;
            m.barrier()
        };
        if spans {
            // Over spans a workgroup computes only where its tile has rows, every one a key of
            // the span (story:attention-event-spans).
            weights(self)?;
        } else {
            // Whether the mask keeps a key row of the tile: a workgroup of masked key rows only
            // writes their `+0` rows, computing no weight.
            let mut any = self.unmasked(t.first)?;
            for u in 1..at.tile {
                let u = self.uint(u);
                let inside = self.less(u, t.rows)?;
                let offset = self.b.select(self.u32_ty, None, inside, u, zero)?;
                let key_row = self.add(t.first, offset)?;
                let keep = self.unmasked(key_row)?;
                let kept = self.and(inside, keep)?;
                any = self.b.logical_or(self.bool_ty, None, any, kept)?;
            }
            self.when(any, weights)?;
        }
        let x = if grad_k { 0 } else { 4 };
        self.when(t.active, |m| {
            let weight = |m: &mut Self, i: Word| -> Result<Word, EmitError> {
                let weight_at = m.add(t.base, i)?;
                m.load_shared(weight_at)
            };
            if spans {
                let columns = m.owned_columns(at, t.lane)?;
                let first = Terms::First(t.queries_count);
                let sums = m.row_sums(at, t.queries, t.column, x, &columns, &weight, first)?;
                return m.write_sums(at, t.own_at, &columns, &sums, grad_k);
            }
            let keep = m.unmasked(t.own)?;
            let masked = m.b.logical_not(m.bool_ty, None, keep)?;
            let columns = m.owned_columns(at, t.lane)?;
            m.when(masked, |m| {
                let zero = m.float(0.0);
                let zeros = vec![zero; columns.len()];
                m.write_sums(at, t.own_at, &columns, &zeros, false)
            })?;
            m.when(keep, |m| {
                let all = Terms::All(at.l);
                let sums = m.row_sums(at, t.queries, t.column, x, &columns, &weight, all)?;
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

    /// Row `index` of the `CrossEntropy` row losses over rows of `classes` logits:
    /// `max + ln(sum) - logits[label]`.
    fn cross_entropy_row(&mut self, classes: u32) -> Result<(), EmitError> {
        let (index, total_var) = (self.index, self.matmul_vars[0]);
        let (zero, v) = (self.uint(0), self.uint(classes));
        let at = self.mul(index, v)?;
        let (max, total) = self.logit_softmax(at, classes, total_var)?;
        let label = self.read_u32(1, index)?;
        let valid = self.less(label, v)?;
        let label = self.b.select(self.u32_ty, None, valid, label, zero)?;
        let target_at = self.add(at, label)?;
        let target = self.read(0, target_at)?;
        let log = self.ln(total)?;
        let shifted = self.f_add(max, log)?;
        let loss = self.f_sub(shifted, target)?;
        self.write(index, loss)
    }

    /// The `CrossEntropy` loss of one workgroup over the `rows` row losses of binding 0 and
    /// their weights in binding 1.
    fn cross_entropy_sum(&mut self, rows: u32) -> Result<(), EmitError> {
        let local = self.local.expect("a module with workgroup memory");
        let [block_var, _] = self.u32_vars;
        let term_var = self.extra_u32_vars[0];
        let [num_var, den_var] = [self.matmul_vars[0], self.matmul_vars[1]];
        let (zero, one) = (self.uint(0), self.uint(1));
        let (n_rows, group) = (self.uint(rows), self.uint(GROUP));
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
                let loss = m.read(0, row)?;
                m.store_shared(local, loss)?;
                let w = m.read(1, row)?;
                let w_at = m.add(group, local)?;
                m.store_shared(w_at, w)
            })?;
            m.barrier()?;
            m.when(first_invocation, |m| {
                let left = m.sub(n_rows, base)?;
                let len = m.min(left, group)?;
                m.counted(term_var, zero, len, one, |m, t| {
                    let w_at = m.add(group, t)?;
                    let w = m.load_shared(w_at)?;
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

    /// Invocation `index` of the `CrossEntropyGrad` row statistics of `rows` rows of `classes`
    /// logits: for a row, its maximum and exponential sum at elements `2 · index` and
    /// `2 · index + 1`; for invocation `rows`, the weight sum at elements `2 · rows` and
    /// `2 · rows + 1`.
    fn cross_entropy_grad_rows(&mut self, rows: u32, classes: u32) -> Result<(), EmitError> {
        let (index, total_var) = (self.index, self.matmul_vars[0]);
        let (one, two, n, v) = (
            self.uint(1),
            self.uint(2),
            self.uint(rows),
            self.uint(classes),
        );
        let at = self.mul(index, two)?;
        let next = self.add(at, one)?;
        let row = self.less(index, n)?;
        self.when(row, |m| {
            let base = m.mul(index, v)?;
            let (max, total) = m.logit_softmax(base, classes, total_var)?;
            m.write(at, max)?;
            m.write(next, total)
        })?;
        let weights = self.equal(index, n)?;
        self.when(weights, |m| {
            let den = m.weight_sum(rows)?;
            m.write(at, den)?;
            m.write(next, den)
        })
    }

    /// Element `index` of the `CrossEntropyGrad` of `rows` rows of `classes` logits, from the
    /// row statistics of binding 0.
    fn cross_entropy_grad(&mut self, rows: u32, classes: u32, seed: Seed) -> Result<(), EmitError> {
        let index = self.index;
        let (zero, one, two, v) = (self.uint(0), self.uint(1), self.uint(2), self.uint(classes));
        let row = self.div(index, v)?;
        let c = self.rem(index, v)?;
        let den_at = self.uint(2 * rows);
        let den = self.read(0, den_at)?;
        let f_zero = self.float(0.0);
        let empty = self.b.f_ord_equal(self.bool_ty, None, den, f_zero)?;
        let max_at = self.mul(row, two)?;
        let total_at = self.add(max_at, one)?;
        let max = self.read(0, max_at)?;
        let total = self.read(0, total_at)?;
        let g = match seed {
            Seed::Immediate(g) => self.float(g),
            Seed::Buffer => self.read(4, zero)?,
        };
        let w = self.read(3, row)?;
        let gw = self.f_mul(g, w)?;
        let scale = self.quotient(gw, den)?;
        let label = self.read_u32(2, row)?;
        let f_one = self.float(1.0);
        let x = self.read(1, index)?;
        let d = self.f_sub(x, max)?;
        let e = self.exp(d)?;
        let p = self.f_div(e, total)?;
        let hit = self.equal(c, label)?;
        let target = self.b.select(self.f32_ty, None, hit, f_one, f_zero)?;
        let diff = self.f_sub(p, target)?;
        let y = self.f_mul(scale, diff)?;
        let out = self.b.select(self.f32_ty, None, empty, f_zero, y)?;
        self.write(index, out)
    }
}
