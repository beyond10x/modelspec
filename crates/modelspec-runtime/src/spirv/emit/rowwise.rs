//! The modules of the fifteen row-wise kernels (DESIGN.md § 5.2): `EmbeddingSort` one invocation
//! per id; `Embedding`, `EmbeddingGrad`, `AddRows`, `Gelu`, `GeluGrad`, `Dropout` and `DropoutGrad`
//! one invocation per output element;
//! `LayerNorm`, `LayerNormGradX`, `Softmax`, `SoftmaxGrad` and the row moments of
//! `LayerNormGradGamma` one per output row; `SumRows` and the column sum of
//! `LayerNormGradGamma` one workgroup per few columns ([`ColumnTile`]). The two `sigreg` kernels
//! (story:jepa-sigreg-vulkan) are per column too: `SigregColumns` writes its column's statistic
//! and `SigregColumnsGrad` every row of its column, each recomputing the column's means in one
//! pass over the rows and computing `sin` and `cos` by the CPU's written-out routine
//! ([`crate::cpu::tensor::sin_cos`]), never `GLSL.std.450 Sin` or `Cos`.
//!
//! Each computes its op as the CPU interpreter does, operation for operation: every sum in
//! ascending index order from its first term, each sum of products from its first product,
//! rounded, with each later product fused into it by one `Fma`, and every float add, subtract,
//! multiply, division and `Fma` decorated `NoContraction` outside `fast_math`. So a kernel whose
//! element is a copy, one rounded operation or a sequence of rounded adds gives the CPU's bits
//! (`Embedding`, `EmbeddingGrad`, `AddRows`, `SumRows`, `Dropout`, `DropoutGrad`) on a device
//! that preserves subnormals, and the others differ only by the `exp`, division and square root
//! of the device, which Vulkan does not round correctly. `LayerNorm` and its two gradients
//! correct each quotient by the row width, and the square root and reciprocal that form `rstd`,
//! by one `Fma` residual step each, so their means, variance and `rstd` are within 1 ulp of the
//! CPU's values.
//!
//! `Embedding` reads an id outside the table as row 0 and writes `0.0` for it, so no invocation
//! reads past the table; the CPU refuses such an id before it runs. `EmbeddingGrad` is two
//! kernels (story:vulkan-embedding-grad-sorted-ids). Its sort, `per_row` and `i32`, gives
//! position `k` of the `n` ids its rank, the number of ids below it, compared as `u32`, plus the
//! number of equal ids at earlier positions, and writes the id and `k` into row `rank` of an
//! `i32[n, 2]`: the ids sorted by id, then by position, with no float operation. For each block
//! of 64 ids, invocation `i` of the workgroup copies id `i` of the block into workgroup memory,
//! and after a barrier each invocation compares its id with the block's ids. Its gradient,
//! `per_element`, finds the sorted rows of its table row `r` by two binary searches of a fixed
//! number of steps, so element `(r, j)` starts at `0.0` and adds column `j` of the gradient row
//! of every id equal to `r` in ascending id position, as the CPU does, then writes once, without
//! atomics; a row no id names writes `0.0`.
//! `SumRows` and `LayerNormGradGamma` stage tiles of rows of their columns in workgroup memory,
//! so every read is coalesced, and add each column in one invocation in ascending row order
//! ([`Shell::sum_rows`], [`Shell::layer_norm_grad_gamma`]); `LayerNormGradGamma` reads the mean
//! and `rstd` of each row from the `[rows, 2]` its moments kernel wrote
//! (story:vulkan-layer-norm-grad-gamma-row-moments). `Dropout` and `DropoutGrad` compute
//! the Philox4x32-10 block of each element from the site's RNG token and the run key, the
//! specialization constants `crate::vulkan::pipeline::SEED_SPEC_IDS`, and keep the element when
//! word 0 is at least [`keep_threshold`] of `p`.

use super::{EmitError, Kernel, KernelKind, Operand, Shell, Word, operands_error, static_shape};
use crate::cpu::interpreter::{SIGREG_KNOTS, sigreg_knot};
use crate::cpu::tensor::{COS_P, FRAC_2_PI, PI_2_P1, PI_2_P2, PI_2_P3, SIN_P};
use modelspec_core::ir::tensor::{DType, TensorOp};
use modelspec_core::schedule::column_sum_columns;
use rspirv::spirv::GlslStd450Op;

/// `sqrt(2 / pi)`, the scale of the tanh form of GELU, as the CPU interpreter holds it.
const GELU_SCALE: f64 = 0.797_884_560_802_865_4;
/// `0.044715`, the cubic coefficient of the tanh form of GELU.
const GELU_CUBIC: f64 = 0.044_715;
/// The `epsilon` of `LayerNorm` and its gradients.
const LAYER_NORM_EPS: f32 = 1e-12;
/// The Philox4x32-10 multipliers and Weyl constants (`crate::train::init::philox4x32_10`).
const PHILOX_M: [u32; 2] = [0xD251_1F53, 0xCD9E_8D57];
const PHILOX_W: [u32; 2] = [0x9E37_79B9, 0xBB67_AE85];
/// The counter word of the dropout stream (`crate::train::init::Stream::Dropout`).
const DROPOUT_STREAM: u32 = 4;
/// The workgroup size of every row-wise kernel.
const GROUP: u32 = 64;

/// The tile of a column sum (story:vulkan-column-sums-coalesced): the columns one workgroup
/// adds ([`column_sum_columns`]), the invocations that stage each of them, a power of two, and
/// the rows of one tile. Workgroup memory holds a tile column by column, each column's values
/// at a pitch of the tile's rows plus 4, so a `vec4` read of four rows of one column by each
/// adding invocation is free of bank conflicts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ColumnTile {
    columns: u32,
    slots: u32,
    rows: u32,
}

impl ColumnTile {
    /// The column sum `op` over rows of `width`: tiles of 8 to 16 values per invocation, 256
    /// rows where that is within them.
    fn of(op: &TensorOp, width: u32) -> Self {
        let columns = column_sum_columns(op, u64::from(width)).map_or(GROUP, |c| c as u32);
        let slots = GROUP / columns;
        let per = (256 / slots).clamp(8, 16);
        Self {
            columns,
            slots,
            rows: per * slots,
        }
    }

    /// `SumRows` over rows of `width`.
    fn sum_rows(width: u32) -> Self {
        Self::of(&TensorOp::SumRows, width)
    }

    /// The column kernel of `LayerNormGradGamma` over rows of `width`. Its workgroup memory
    /// holds the `xhat` columns of a tile, then its `g` columns: at most 2,560 values, within
    /// the 16 KiB every device has (`modelspec_core::ir::kernel::MIN_SHARED_MEMORY_BYTES`).
    fn layer_norm_grad_gamma(width: u32) -> Self {
        Self::of(&TensorOp::LayerNormGradGamma, width)
    }

    /// The `f32` values of a tile's columns in workgroup memory.
    fn values(self) -> u32 {
        self.columns * self.pitch()
    }

    /// The invocations of a dispatch over `width` columns: 64 per workgroup of `columns`.
    fn invocations(self, width: u32) -> Option<u32> {
        width.div_ceil(self.columns).checked_mul(GROUP)
    }

    /// The values each invocation stages per tile, one per `slots` rows.
    fn per_invocation(self) -> u32 {
        self.rows / self.slots
    }

    /// The distance between two columns of a tile in workgroup memory.
    fn pitch(self) -> u32 {
        self.rows + 4
    }
}

/// `value` as the CPU interpreter's `f32` constant: the sum of its two `f32` parts, rounded.
fn constant(value: f64) -> f32 {
    let high = value as f32;
    let low = (value - f64::from(high)) as f32;
    high + low
}

/// The smallest word `w` whose uniform `(w + 0.5) / 2^32` is at least `p`, for `p` in
/// `[0, 1)`: the CPU keeps an element exactly when its word 0 is at least this
/// (`crate::train::init::dropout_keeps`).
pub(super) fn keep_threshold(p: f32) -> u32 {
    let uniform = |w: u64| (w as f64 + 0.5) / 4_294_967_296.0;
    let p = f64::from(p);
    let mut t = (p * 4_294_967_296.0 - 0.5)
        .ceil()
        .clamp(0.0, 4_294_967_295.0) as u64;
    while t > 0 && uniform(t - 1) >= p {
        t -= 1;
    }
    while t < u64::from(u32::MAX) && uniform(t) < p {
        t += 1;
    }
    t as u32
}

/// What a row-wise module computes, its constants taken from the kernel's shapes: `rows` and
/// `width` of its `[rows, width]` operand, `ids` of an id operand, `table` rows of an
/// embedding table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Rowwise {
    Embedding {
        table: u32,
        width: u32,
    },
    /// The sort of `EmbeddingGrad`: `ids` ids into `[ids, 2]` rows of id and position.
    EmbeddingSort {
        ids: u32,
    },
    /// The gradient of `EmbeddingGrad` over the `[ids, 2]` sorted rows.
    EmbeddingGrad {
        ids: u32,
        width: u32,
    },
    AddRows {
        width: u32,
    },
    SumRows {
        rows: u32,
        width: u32,
    },
    Gelu,
    GeluGrad,
    /// `Dropout` (values at binding 0, token at 1) or `DropoutGrad` (token at 0, values at 1).
    Dropout {
        values: usize,
        token: usize,
        threshold: u32,
        scale: f32,
    },
    LayerNorm {
        width: u32,
    },
    LayerNormGradX {
        width: u32,
    },
    /// The row moments of `LayerNormGradGamma`: `[rows, width]` into `[rows, 2]`.
    LayerNormMoments {
        width: u32,
    },
    /// The column sum of `LayerNormGradGamma` over the moments, `x` and `g`.
    LayerNormGradGamma {
        rows: u32,
        width: u32,
    },
    Softmax {
        width: u32,
    },
    SoftmaxGrad {
        width: u32,
    },
    SigregColumns {
        rows: u32,
        width: u32,
    },
    SigregColumnsGrad {
        rows: u32,
        width: u32,
    },
}

impl Rowwise {
    /// The row-wise module of `kernel`, or why its kind, op, operands or workgroups are not
    /// one: each op has the kind, operand types and shapes `modelspec_core::schedule::
    /// rowwise_kernel` gives it, every element count fits `u32`, and the workgroups of 64 cover
    /// its invocations.
    pub(super) fn of(kernel: &Kernel) -> Result<Self, EmitError> {
        let wrong = |what: &str| operands_error(kernel, what.to_owned());
        if kernel.workgroup_size != GROUP {
            return Err(wrong("a row-wise kernel runs workgroups of 64"));
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
            fits(elements(shape)).ok_or_else(|| wrong("an operand of 0 or 2^32 elements"))?;
        }
        let out = fits(elements(&result)).ok_or_else(|| wrong("a result of 0 or 2^32 elements"))?;
        // The sort of `EmbeddingGrad` writes ids and positions; every other kernel writes f32.
        let writes = match (kernel.op, kernel.kind) {
            ("EmbeddingGrad", KernelKind::PerRow) => DType::I32,
            _ => DType::F32,
        };
        if kernel.dtype != writes || kernel.result.dtype != writes {
            return Err(wrong(match writes {
                DType::I32 => "the sort of EmbeddingGrad writes i32",
                _ => "a row-wise kernel writes f32",
            }));
        }
        let types: Vec<DType> = buffers.iter().map(|(dtype, _)| *dtype).collect();
        let shapes: Vec<&[u64]> = buffers.iter().map(|(_, shape)| shape.as_slice()).collect();
        let u = |n: u64| n as u32;
        use DType::{F32, I32};
        use KernelKind::{PerColumn, PerElement, PerRow};
        let (kind, invocations, body) = match (
            kernel.op,
            types.as_slice(),
            shapes.as_slice(),
            result.as_slice(),
            immediates.as_slice(),
        ) {
            ("Embedding", [F32, I32], [[v, h], [n]], [rn, rh], []) if (n, h) == (rn, rh) => (
                PerElement,
                out,
                Rowwise::Embedding {
                    table: u(*v),
                    width: u(*h),
                },
            ),
            ("EmbeddingGrad", [I32], [[n]], [rn, 2], []) if n == rn => {
                (PerRow, u(*n), Rowwise::EmbeddingSort { ids: u(*n) })
            }
            ("EmbeddingGrad", [I32, F32], [[n, 2], [gn, h]], [_, rh], []) if n == gn && h == rh => {
                (
                    PerElement,
                    out,
                    Rowwise::EmbeddingGrad {
                        ids: u(*n),
                        width: u(*h),
                    },
                )
            }
            ("AddRows", [F32, F32], [[n, h], [bh]], [rn, rh], [])
                if h == bh && (n, h) == (rn, rh) =>
            {
                (PerElement, out, Rowwise::AddRows { width: u(*h) })
            }
            ("SumRows", [F32], [[n, h]], [rh], []) if h == rh && *n > 0 && *h > 0 => (
                PerColumn,
                ColumnTile::sum_rows(u(*h))
                    .invocations(u(*h))
                    .ok_or_else(|| wrong("too many columns"))?,
                Rowwise::SumRows {
                    rows: u(*n),
                    width: u(*h),
                },
            ),
            ("Gelu", [F32], [x], r, []) if *x == r => (PerElement, out, Rowwise::Gelu),
            ("GeluGrad", [F32, F32], [x, g], r, []) if *x == r && *g == r => {
                (PerElement, out, Rowwise::GeluGrad)
            }
            ("Dropout", [F32, I32], [x, [2]], r, [p]) if *x == r => {
                (PerElement, out, dropout(kernel, 0, 1, *p)?)
            }
            ("DropoutGrad", [I32, F32], [[2], g], r, [p]) if *g == r => {
                (PerElement, out, dropout(kernel, 1, 0, *p)?)
            }
            ("LayerNorm", [F32, F32, F32], [[n, h], [gh], [bh]], [rn, rh], [])
                if h == gh && h == bh && (n, h) == (rn, rh) =>
            {
                (PerRow, u(*n), Rowwise::LayerNorm { width: u(*h) })
            }
            ("LayerNormGradX", [F32, F32, F32], [[n, h], [gh], g], [rn, rh], [])
                if h == gh && *g == [*n, *h] && (n, h) == (rn, rh) =>
            {
                (PerRow, u(*n), Rowwise::LayerNormGradX { width: u(*h) })
            }
            ("LayerNormGradGamma", [F32], [[n, h]], [rn, 2], []) if n == rn && *n > 0 && *h > 0 => {
                (PerRow, u(*n), Rowwise::LayerNormMoments { width: u(*h) })
            }
            ("LayerNormGradGamma", [F32, F32, F32], [[mn, 2], [n, h], g], [rh], [])
                if mn == n && h == rh && *g == [*n, *h] && *n > 0 && *h > 0 =>
            {
                (
                    PerColumn,
                    ColumnTile::layer_norm_grad_gamma(u(*h))
                        .invocations(u(*h))
                        .ok_or_else(|| wrong("too many columns"))?,
                    Rowwise::LayerNormGradGamma {
                        rows: u(*n),
                        width: u(*h),
                    },
                )
            }
            ("Softmax", [F32], [[n, h]], [rn, rh], []) if (n, h) == (rn, rh) => {
                (PerRow, u(*n), Rowwise::Softmax { width: u(*h) })
            }
            ("SoftmaxGrad", [F32, F32], [[n, h], g], [rn, rh], [])
                if *g == [*n, *h] && (n, h) == (rn, rh) =>
            {
                (PerRow, u(*n), Rowwise::SoftmaxGrad { width: u(*h) })
            }
            ("SigregColumns", [F32], [[n, m]], [rm], []) if m == rm => (
                PerColumn,
                u(*m),
                Rowwise::SigregColumns {
                    rows: u(*n),
                    width: u(*m),
                },
            ),
            ("SigregColumnsGrad", [F32, F32], [[n, m], [gm]], [rn, rm], [])
                if m == gm && (n, m) == (rn, rm) =>
            {
                (
                    PerColumn,
                    u(*m),
                    Rowwise::SigregColumnsGrad {
                        rows: u(*n),
                        width: u(*m),
                    },
                )
            }
            _ => {
                return Err(EmitError::Unsupported {
                    kind: kernel.kind,
                    op: kernel.op,
                });
            }
        };
        if kernel.kind != kind {
            return Err(EmitError::Unsupported {
                kind: kernel.kind,
                op: kernel.op,
            });
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

    /// The `f32` values of workgroup memory the module stages.
    pub(super) fn shared(self) -> Option<u32> {
        match self {
            Rowwise::LayerNormGradGamma { width, .. } => {
                Some(2 * ColumnTile::layer_norm_grad_gamma(width).values())
            }
            Rowwise::EmbeddingSort { .. } => Some(GROUP),
            Rowwise::SumRows { width, .. } => Some(ColumnTile::sum_rows(width).values()),
            _ => None,
        }
    }

    /// The `u32` function variables the module keeps beside the shell's two.
    pub(super) fn u32_vars(self) -> usize {
        match self {
            Rowwise::EmbeddingSort { .. } => 1,
            Rowwise::EmbeddingGrad { .. } => 2,
            _ => 0,
        }
    }

    /// Whether the module reads its workgroup memory as `vec4`s: the column sums do.
    pub(super) fn shared_vectors(self) -> bool {
        matches!(
            self,
            Rowwise::SumRows { .. } | Rowwise::LayerNormGradGamma { .. }
        )
    }

    /// The `f32` function variables the module keeps beside the shell's one.
    pub(super) fn f32_vars(self) -> usize {
        match self {
            Rowwise::SumRows { width, .. } => ColumnTile::sum_rows(width).per_invocation() as usize,
            Rowwise::LayerNormGradGamma { width, .. } => {
                4 * ColumnTile::layer_norm_grad_gamma(width).per_invocation() as usize
            }
            Rowwise::SigregColumns { .. } | Rowwise::SigregColumnsGrad { .. } => {
                2 * SIGREG_KNOTS - 1
            }
            _ => 0,
        }
    }

    /// Whether the module reads the run key.
    pub(super) fn seeded(self) -> bool {
        matches!(self, Rowwise::Dropout { .. })
    }
}

/// The dropout module at drop probability `p`, its values at binding `values` and its token at
/// `token`: the threshold of `p` and the scale `1 / (1 - p)`, rounded once as the CPU does.
fn dropout(kernel: &Kernel, values: usize, token: usize, p: f32) -> Result<Rowwise, EmitError> {
    if !(0.0..1.0).contains(&p) {
        return Err(operands_error(
            kernel,
            format!("a drop probability {p} outside [0, 1)"),
        ));
    }
    Ok(Rowwise::Dropout {
        values,
        token,
        threshold: keep_threshold(p),
        scale: 1.0f32 / (1.0f32 - p),
    })
}

impl Shell {
    pub(super) fn sub(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        Ok(self.b.i_sub(self.u32_ty, None, a, b)?)
    }

    pub(super) fn div(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        Ok(self.b.u_div(self.u32_ty, None, a, b)?)
    }

    pub(super) fn rem(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        Ok(self.b.u_mod(self.u32_ty, None, a, b)?)
    }

    fn xor(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        Ok(self.b.bitwise_xor(self.u32_ty, None, a, b)?)
    }

    /// The high and low 32 bits of the 64-bit product `a * b`.
    fn mul_hi_lo(&mut self, a: Word, b: Word) -> Result<(Word, Word), EmitError> {
        let pair = self.b.type_struct([self.u32_ty, self.u32_ty]);
        let product = self.b.u_mul_extended(pair, None, a, b)?;
        let lo = self.b.composite_extract(self.u32_ty, None, product, [0])?;
        let hi = self.b.composite_extract(self.u32_ty, None, product, [1])?;
        Ok((hi, lo))
    }

    /// Philox4x32-10 of `counter` under `key`, as `crate::train::init::philox4x32_10`: ten
    /// rounds, the key bumped by the Weyl constants before each round after the first.
    fn philox(&mut self, counter: [Word; 4], key: [Word; 2]) -> Result<[Word; 4], EmitError> {
        let m = PHILOX_M.map(|value| self.uint(value));
        let w = PHILOX_W.map(|value| self.uint(value));
        let (mut c, mut k) = (counter, key);
        for round in 0..10 {
            if round > 0 {
                k = [self.add(k[0], w[0])?, self.add(k[1], w[1])?];
            }
            let (hi0, lo0) = self.mul_hi_lo(m[0], c[0])?;
            let (hi1, lo1) = self.mul_hi_lo(m[1], c[2])?;
            let a = self.xor(hi1, c[1])?;
            let a = self.xor(a, k[0])?;
            let b = self.xor(hi0, c[3])?;
            let b = self.xor(b, k[1])?;
            c = [a, lo1, b, lo0];
        }
        Ok(c)
    }

    /// `2 u` of GELU at `x`: `(2 sqrt(2/pi)) * (x + ((0.044715 * x) * x) * x)`.
    fn gelu_twice_inner(&mut self, x: Word) -> Result<Word, EmitError> {
        let cubic = self.float(constant(GELU_CUBIC));
        let twice_scale = self.float(2.0f32 * constant(GELU_SCALE));
        let a = self.f_mul(cubic, x)?;
        let a = self.f_mul(a, x)?;
        let a = self.f_mul(a, x)?;
        let s = self.f_add(x, a)?;
        self.f_mul(twice_scale, s)
    }

    /// `1 / (1 + exp(0 - v))`.
    fn logistic(&mut self, v: Word) -> Result<Word, EmitError> {
        let (zero, one) = (self.float(0.0), self.float(1.0));
        let negated = self.f_sub(zero, v)?;
        let e = self.exp(negated)?;
        let d = self.f_add(one, e)?;
        self.f_div(one, d)
    }

    fn gelu(&mut self, x: Word) -> Result<Word, EmitError> {
        let v = self.gelu_twice_inner(x)?;
        let s = self.logistic(v)?;
        self.f_mul(x, s)
    }

    /// `(s + ((2 (s x)) r) slope) g`, with `s` the logistic of `2u`, `r` that of `0 - 2u` and
    /// `slope = sqrt(2/pi) (1 + ((3 * 0.044715) x) x)`.
    fn gelu_grad(&mut self, x: Word, g: Word) -> Result<Word, EmitError> {
        let (zero, one, two) = (self.float(0.0), self.float(1.0), self.float(2.0));
        let scale = self.float(constant(GELU_SCALE));
        let three_cubic = self.float(3.0f32 * constant(GELU_CUBIC));
        let v = self.gelu_twice_inner(x)?;
        let s = self.logistic(v)?;
        let negated = self.f_sub(zero, v)?;
        let r = self.logistic(negated)?;
        let a = self.f_mul(three_cubic, x)?;
        let a = self.f_mul(a, x)?;
        let a = self.f_add(one, a)?;
        let slope = self.f_mul(scale, a)?;
        let t = self.f_mul(s, x)?;
        let t = self.f_mul(two, t)?;
        let t = self.f_mul(t, r)?;
        let t = self.f_mul(t, slope)?;
        let t = self.f_add(s, t)?;
        self.f_mul(t, g)
    }

    /// The mean and `1 / sqrt(variance + epsilon)` of the `width` values of read binding
    /// `binding` from `base`, as the CPU's `row_moments`: the sum from the first value in
    /// ascending order, divided by the width; the squares of the differences from the mean, the
    /// first rounded and each later one fused into the sum, divided by the width. Uses the
    /// shell's `f32` variable and its second `u32` variable.
    fn row_moments(
        &mut self,
        binding: usize,
        base: Word,
        width: u32,
    ) -> Result<(Word, Word), EmitError> {
        let (var, acc) = (self.u32_vars[1], self.f32_var);
        let (one, end) = (self.uint(1), self.uint(width));
        let first = self.read(binding, base)?;
        self.b.store(acc, first, None, [])?;
        self.counted(var, one, end, one, |m, j| {
            let at = m.add(base, j)?;
            let x = m.read(binding, at)?;
            m.accumulate(x)
        })?;
        let total = self.b.load(self.f32_ty, None, acc, None, [])?;
        let mean = self.over_width(total, width)?;
        let d = self.f_sub(first, mean)?;
        let square = self.f_mul(d, d)?;
        self.b.store(acc, square, None, [])?;
        self.counted(var, one, end, one, |m, j| {
            let at = m.add(base, j)?;
            let x = m.read(binding, at)?;
            let d = m.f_sub(x, mean)?;
            m.accumulate_square(d)
        })?;
        let squares = self.b.load(self.f32_ty, None, acc, None, [])?;
        let variance = self.over_width(squares, width)?;
        let eps = self.float(LAYER_NORM_EPS);
        let shifted = self.f_add(variance, eps)?;
        let rstd = self.reciprocal_sqrt(shifted)?;
        Ok((mean, rstd))
    }

    /// `a / width` within 1 ulp of the CPU's quotient: the device's quotient `q` (`OpFDiv`, 2.5 ulp
    /// on Vulkan) corrected by one residual step, `q + fma(q, -width, a) * (1 / width)`, the
    /// residual one `Fma` and the step another, with `1 / width` the `f32` nearest to it. `q`
    /// stands where the residual is 0 or NaN: an exact quotient, a zero of either sign, an
    /// infinite or NaN `a`.
    fn over_width(&mut self, a: Word, width: u32) -> Result<Word, EmitError> {
        let count = self.float(width as f32);
        let q = self.f_div(a, count)?;
        let negated = self.float(-(width as f32));
        let inverse = self.float(1.0 / width as f32);
        let residual = self.f_fma(q, negated, a)?;
        let corrected = self.f_fma(residual, inverse, q)?;
        self.where_residual(residual, corrected, q)
    }

    /// `1 / sqrt(v)` with the CPU's bits, which rounds the square root to `f32` and then its
    /// reciprocal: [`Shell::rounded_root`] of `v`, then [`Shell::rounded_reciprocal`] of that
    /// root, for every `v` from `2^-64` up, so for every `variance + 1e-12`, an infinite `v`
    /// giving `0` and a NaN a NaN.
    fn reciprocal_sqrt(&mut self, v: Word) -> Result<Word, EmitError> {
        let root = self.rounded_root(v)?;
        self.rounded_reciprocal(root)
    }

    /// The `f32` values just below and just above a positive finite `x`: its bits minus one and
    /// plus one.
    fn neighbours(&mut self, x: Word) -> Result<(Word, Word), EmitError> {
        let one = self.uint(1);
        let bits = self.b.bitcast(self.u32_ty, None, x)?;
        let below = self.sub(bits, one)?;
        let above = self.add(bits, one)?;
        let below = self.b.bitcast(self.f32_ty, None, below)?;
        let above = self.b.bitcast(self.f32_ty, None, above)?;
        Ok((below, above))
    }

    /// `sqrt(v)` correctly rounded, the CPU's root, for every `v` from `2^-64` up: the
    /// corrected device root `s` ([`Shell::root`], within 1 ulp of `sqrt(v)`), then whichever of
    /// `s` and its neighbours `d` below and `u` above is nearest to `sqrt(v)`: `u` where
    /// `fma(0 - s, u, v) > 0`, `d` where `fma(0 - d, s, v) <= 0`, else `s`. The midpoint of `s`
    /// and `u` squared is `s u + ulp^2 / 4`, and `v - s u` is a multiple of `ulp^2`, so `v` lies
    /// above that square exactly when `v - s u > 0`; below, `v - d s <= 0` likewise, with the ulp
    /// of `d`. Each `Fma` keeps the sign of its exact difference, which is 0 or at least `2^-112`
    /// in magnitude, a normal value no device flushes. An infinite `v` gives its infinity and a
    /// NaN a NaN: every comparison with the NaN their differences make is false.
    fn rounded_root(&mut self, v: Word) -> Result<Word, EmitError> {
        let zero = self.float(0.0);
        let s = self.root(v)?;
        let (below, above) = self.neighbours(s)?;
        let neg_s = self.f_sub(zero, s)?;
        let over = self.f_fma(neg_s, above, v)?;
        let neg_below = self.f_sub(zero, below)?;
        let under = self.f_fma(neg_below, s, v)?;
        let up = self.b.f_ord_greater_than(self.bool_ty, None, over, zero)?;
        let down = self
            .b
            .f_ord_less_than_equal(self.bool_ty, None, under, zero)?;
        let lower = self.b.select(self.f32_ty, None, down, below, s)?;
        Ok(self.b.select(self.f32_ty, None, up, above, lower)?)
    }

    /// `1 / s` correctly rounded, the CPU's quotient, for every `s` from `2^-32` to `2^64`: the
    /// device's `p = 1 / s` corrected by one Newton step to `p + p * fma(0 - p, s, 1)` (`p` where
    /// that residual is 0 or NaN), within 1 ulp of `1 / s`, so its residual
    /// `r = fma(0 - p, s, 1) = 1 - p s` is exact; then `u`, the neighbour above, where
    /// `r > s * ((u - p) * 0.5)`, `d`, the neighbour below, where `r < 0 - s * ((p - d) * 0.5)`,
    /// else `p`. `1 / s - p = r / s`, the midpoints lie half a gap from `p`, and `1 / s` is never
    /// a midpoint; the gaps, halves and products by `s` are exact. An infinite `s` gives `0` and a
    /// NaN a NaN: their residuals are NaN, and every comparison with it is false.
    fn rounded_reciprocal(&mut self, s: Word) -> Result<Word, EmitError> {
        let (zero, half, one) = (self.float(0.0), self.float(0.5), self.float(1.0));
        let p = self.f_div(one, s)?;
        let neg_p = self.f_sub(zero, p)?;
        let residual = self.f_fma(neg_p, s, one)?;
        let stepped = self.f_fma(residual, p, p)?;
        let p = self.where_residual(residual, stepped, p)?;
        let neg_p = self.f_sub(zero, p)?;
        let r = self.f_fma(neg_p, s, one)?;
        let (below, above) = self.neighbours(p)?;
        let gap = self.f_sub(above, p)?;
        let gap = self.f_mul(gap, half)?;
        let up_bound = self.f_mul(s, gap)?;
        let gap = self.f_sub(p, below)?;
        let gap = self.f_mul(gap, half)?;
        let down_bound = self.f_mul(s, gap)?;
        let down_bound = self.f_sub(zero, down_bound)?;
        let up = self.b.f_ord_greater_than(self.bool_ty, None, r, up_bound)?;
        let down = self.b.f_ord_less_than(self.bool_ty, None, r, down_bound)?;
        let lower = self.b.select(self.f32_ty, None, down, below, p)?;
        Ok(self.b.select(self.f32_ty, None, up, above, lower)?)
    }

    /// `corrected` where `residual` is ordered and not zero, else `raw`.
    pub(super) fn where_residual(
        &mut self,
        residual: Word,
        corrected: Word,
        raw: Word,
    ) -> Result<Word, EmitError> {
        let zero = self.float(0.0);
        let step = self.b.f_ord_not_equal(self.bool_ty, None, residual, zero)?;
        Ok(self.b.select(self.f32_ty, None, step, corrected, raw)?)
    }

    /// The module body of `rowwise`.
    pub(super) fn rowwise(&mut self, rowwise: Rowwise) -> Result<(), EmitError> {
        if let Rowwise::LayerNormGradGamma { rows, width } = rowwise {
            return self.layer_norm_grad_gamma(rows, width);
        }
        if let Rowwise::EmbeddingGrad { ids, width } = rowwise {
            return self.embedding_grad(ids, width);
        }
        if let Rowwise::EmbeddingSort { ids } = rowwise {
            return self.embedding_sort(ids);
        }
        if let Rowwise::SumRows { rows, width } = rowwise {
            return self.sum_rows(rows, width);
        }
        let (index, count) = (self.index, self.count);
        let inside = self.less(index, count)?;
        self.when(inside, |m| match rowwise {
            Rowwise::Embedding { table, width } => m.embedding(table, width),
            Rowwise::AddRows { width } => {
                let w = m.uint(width);
                let j = m.rem(index, w)?;
                let x = m.read(0, index)?;
                let b = m.read(1, j)?;
                let sum = m.f_add(x, b)?;
                m.write(index, sum)
            }
            Rowwise::Gelu => {
                let x = m.read(0, index)?;
                let y = m.gelu(x)?;
                m.write(index, y)
            }
            Rowwise::GeluGrad => {
                let x = m.read(0, index)?;
                let g = m.read(1, index)?;
                let y = m.gelu_grad(x, g)?;
                m.write(index, y)
            }
            Rowwise::Dropout {
                values,
                token,
                threshold,
                scale,
            } => m.dropout(values, token, threshold, scale),
            Rowwise::LayerNorm { width } => m.layer_norm(width),
            Rowwise::LayerNormMoments { width } => m.layer_norm_moments(width),
            Rowwise::LayerNormGradX { width } => m.layer_norm_grad_x(width),
            Rowwise::Softmax { width } => m.softmax(width),
            Rowwise::SoftmaxGrad { width } => m.softmax_grad(width),
            Rowwise::SigregColumns { rows, width } => m.sigreg_columns(rows, width),
            Rowwise::SigregColumnsGrad { rows, width } => m.sigreg_columns_grad(rows, width),
            Rowwise::LayerNormGradGamma { .. }
            | Rowwise::EmbeddingSort { .. }
            | Rowwise::EmbeddingGrad { .. }
            | Rowwise::SumRows { .. } => {
                unreachable!("handled above")
            }
        })
    }

    /// `(sin(x), cos(x))` by the CPU's written-out routine ([`crate::cpu::tensor::sin_cos`]),
    /// instruction for instruction: `q = RoundEven(x * (2/pi))`, the reduction
    /// `r = fma(-q, P1, x)`, then `P2`, then `P3`, `z = r * r`, the two polynomials by `Fma`
    /// and the products `s * z` and `c * z` rounded before their last `Fma`, and the quadrant
    /// `q mod 4` of `q` converted to a signed integer selecting `(s, c)`, `(c, -s)`, `(-s, -c)`
    /// or `(-c, s)`, each negation an `OpFNegate`.
    fn sin_cos(&mut self, x: Word) -> Result<(Word, Word), EmitError> {
        let scale = self.float(FRAC_2_PI);
        let scaled = self.f_mul(x, scale)?;
        let q = self.glsl_f32(GlslStd450Op::RoundEven, &[scaled])?;
        let neg_q = self.b.f_negate(self.f32_ty, None, q)?;
        let mut r = x;
        for part in [PI_2_P1, PI_2_P2, PI_2_P3] {
            let part = self.float(part);
            r = self.f_fma(neg_q, part, r)?;
        }
        let z = self.f_mul(r, r)?;
        let [s0, s1, s2] = SIN_P.map(|value| self.float(value));
        let s = self.f_fma(s0, z, s1)?;
        let s = self.f_fma(s, z, s2)?;
        let sz = self.f_mul(s, z)?;
        let s = self.f_fma(sz, r, r)?;
        let [c0, c1, c2] = COS_P.map(|value| self.float(value));
        let c = self.f_fma(c0, z, c1)?;
        let c = self.f_fma(c, z, c2)?;
        let cz = self.f_mul(c, z)?;
        let (minus_half, one) = (self.float(-0.5), self.float(1.0));
        let half = self.f_fma(minus_half, z, one)?;
        let c = self.f_fma(cz, z, half)?;
        let i32_ty = self.b.type_int(32, 1);
        let signed = self.b.convert_f_to_s(i32_ty, None, q)?;
        let quadrant = self.b.bitcast(self.u32_ty, None, signed)?;
        let (zero, one, two) = (self.uint(0), self.uint(1), self.uint(2));
        let bit = |m: &mut Self, value: Word, mask: Word| -> Result<Word, EmitError> {
            let masked = m.b.bitwise_and(m.u32_ty, None, value, mask)?;
            Ok(m.b.i_not_equal(m.bool_ty, None, masked, zero)?)
        };
        let swap = bit(self, quadrant, one)?;
        let negate_sin = bit(self, quadrant, two)?;
        let next = self.add(quadrant, one)?;
        let negate_cos = bit(self, next, two)?;
        let base_sin = self.b.select(self.f32_ty, None, swap, c, s)?;
        let base_cos = self.b.select(self.f32_ty, None, swap, s, c)?;
        let negated_sin = self.b.f_negate(self.f32_ty, None, base_sin)?;
        let negated_cos = self.b.f_negate(self.f32_ty, None, base_cos)?;
        let sin = self
            .b
            .select(self.f32_ty, None, negate_sin, negated_sin, base_sin)?;
        let cos = self
            .b
            .select(self.f32_ty, None, negate_cos, negated_cos, base_cos)?;
        Ok((sin, cos))
    }

    /// The knots `t_j` of the `sigreg` quadrature as the CPU rounds them, `j = 1..=8`.
    fn sigreg_knots(&mut self) -> [Word; SIGREG_KNOTS] {
        std::array::from_fn(|k| self.float(constant(sigreg_knot(k + 1).0)))
    }

    /// The means `(C_j, S_j)` of `cos(t_j x)` and `sin(t_j x)` over column `index` of the
    /// `[rows, width]` binding 0, as the CPU's `sigreg_means`: each argument `t_j * x` rounded,
    /// each sum from row 0 in ascending row order, one accumulator per knot and function (the
    /// shell's `f32` variable and the module's 15 beside it), and each sum divided by `rows` by
    /// [`Shell::over_width`]. One pass over the rows computes all 16 sums; each is the CPU's
    /// sequence of adds. Uses the shell's first `u32` variable.
    fn sigreg_means(
        &mut self,
        rows: u32,
        width: u32,
    ) -> Result<[(Word, Word); SIGREG_KNOTS], EmitError> {
        let index = self.index;
        let knots = self.sigreg_knots();
        let accumulators: Vec<Word> = std::iter::once(self.f32_var)
            .chain(self.matmul_vars.iter().copied())
            .collect();
        if accumulators.len() < 2 * SIGREG_KNOTS {
            return Err(EmitError::Build(format!(
                "a sigreg module keeps {} accumulators, not {}",
                2 * SIGREG_KNOTS,
                accumulators.len()
            )));
        }
        let first = self.read(0, index)?;
        for (j, &t) in knots.iter().enumerate() {
            let argument = self.f_mul(t, first)?;
            let (sin, cos) = self.sin_cos(argument)?;
            self.b.store(accumulators[2 * j], cos, None, [])?;
            self.b.store(accumulators[2 * j + 1], sin, None, [])?;
        }
        let (one, end, w) = (self.uint(1), self.uint(rows), self.uint(width));
        let var = self.u32_vars[0];
        self.counted(var, one, end, one, |m, n| {
            let row = m.mul(n, w)?;
            let at = m.add(row, index)?;
            let x = m.read(0, at)?;
            for (j, &t) in knots.iter().enumerate() {
                let argument = m.f_mul(t, x)?;
                let (sin, cos) = m.sin_cos(argument)?;
                for (acc, value) in [(accumulators[2 * j], cos), (accumulators[2 * j + 1], sin)] {
                    let sum = m.b.load(m.f32_ty, None, acc, None, [])?;
                    let sum = m.f_add(sum, value)?;
                    m.b.store(acc, sum, None, [])?;
                }
            }
            Ok(())
        })?;
        let mut means = [(0, 0); SIGREG_KNOTS];
        for (j, mean) in means.iter_mut().enumerate() {
            let cos = self
                .b
                .load(self.f32_ty, None, accumulators[2 * j], None, [])?;
            let sin = self
                .b
                .load(self.f32_ty, None, accumulators[2 * j + 1], None, [])?;
            *mean = (self.over_width(cos, rows)?, self.over_width(sin, rows)?);
        }
        Ok(means)
    }

    /// `SigregColumns`, column `index`: the CPU's `sigreg_columns`, with `d_j = C_j - phi_j` and
    /// `q_j = fma(S_j, S_j, d_j * d_j)`, the sum `c_1 * q_1` and then `fma(c_j, q_j, sum)`, and
    /// the result `rows * sum`.
    fn sigreg_columns(&mut self, rows: u32, width: u32) -> Result<(), EmitError> {
        let index = self.index;
        let means = self.sigreg_means(rows, width)?;
        let mut sum = None;
        for (j, &(cos, sin)) in means.iter().enumerate() {
            let (_, phi, weight) = sigreg_knot(j + 1);
            let (phi, weight) = (self.float(constant(phi)), self.float(constant(weight)));
            let d = self.f_sub(cos, phi)?;
            let square = self.f_mul(d, d)?;
            let q = self.f_fma(sin, sin, square)?;
            sum = Some(match sum {
                None => self.f_mul(weight, q)?,
                Some(sum) => self.f_fma(weight, q, sum)?,
            });
        }
        let count = self.float(rows as f32);
        let sum = sum.expect("eight knots");
        let statistic = self.f_mul(count, sum)?;
        self.write(index, statistic)
    }

    /// `SigregColumnsGrad`, column `index` of every row: the CPU's `sigreg_columns_grad`, the
    /// means recomputed by [`Shell::sigreg_means`], then for row `n` in ascending order, with
    /// `(s_j, c_j)` the [`Shell::sin_cos`] of `t_j * P[n, index]`, the term
    /// `fma(0 - (C_j - phi_j), s_j, S_j * c_j)`, the sum `(c_j t_j) * term_1` and then
    /// `fma(c_j t_j, term_j, sum)`, and the element `(2 * g[index]) * sum`.
    fn sigreg_columns_grad(&mut self, rows: u32, width: u32) -> Result<(), EmitError> {
        let index = self.index;
        let means = self.sigreg_means(rows, width)?;
        let zero = self.float(0.0);
        let mut knots = Vec::with_capacity(SIGREG_KNOTS);
        for (j, &(cos, sin)) in means.iter().enumerate() {
            let (t, phi, weight) = sigreg_knot(j + 1);
            let slope = self.float(constant(weight * t));
            let (t, phi) = (self.float(constant(t)), self.float(constant(phi)));
            let d = self.f_sub(cos, phi)?;
            let negated = self.f_sub(zero, d)?;
            knots.push((t, negated, sin, slope));
        }
        let g = self.read(1, index)?;
        let two = self.float(2.0);
        let twice = self.f_mul(two, g)?;
        let (start, one, end, w) = (
            self.uint(0),
            self.uint(1),
            self.uint(rows),
            self.uint(width),
        );
        let var = self.u32_vars[0];
        self.counted(var, start, end, one, |m, n| {
            let row = m.mul(n, w)?;
            let at = m.add(row, index)?;
            let x = m.read(0, at)?;
            let mut sum = None;
            for &(t, negated, sin_mean, slope) in &knots {
                let argument = m.f_mul(t, x)?;
                let (s, c) = m.sin_cos(argument)?;
                let product = m.f_mul(sin_mean, c)?;
                let term = m.f_fma(negated, s, product)?;
                sum = Some(match sum {
                    None => m.f_mul(slope, term)?,
                    Some(sum) => m.f_fma(slope, term, sum)?,
                });
            }
            let element = m.f_mul(twice, sum.expect("eight knots"))?;
            m.write(at, element)
        })
    }

    /// Output element `index` is element `index % width` of table row `ids[index / width]`, or
    /// `0.0` for an id at or past `table` rows, which reads row 0 instead.
    fn embedding(&mut self, table: u32, width: u32) -> Result<(), EmitError> {
        let index = self.index;
        let w = self.uint(width);
        let n = self.div(index, w)?;
        let j = self.rem(index, w)?;
        let id = self.read_u32(1, n)?;
        let rows = self.uint(table);
        let valid = self.less(id, rows)?;
        let row = self.mul(id, w)?;
        let at = self.add(row, j)?;
        let zero = self.uint(0);
        let safe = self.b.select(self.u32_ty, None, valid, at, zero)?;
        let value = self.read(0, safe)?;
        let f_zero = self.float(0.0);
        let out = self.b.select(self.f32_ty, None, valid, value, f_zero)?;
        self.write(index, out)
    }

    /// The sort of `EmbeddingGrad`, invocation `index` for id `index`: its rank, the number of ids
    /// below it and of equal ids at earlier positions, counted over every id in ascending
    /// position, then its id and position written, as their bits, to row `rank` of the
    /// `[ids, 2]` result. Each block of 64 ids is staged in workgroup memory, id `i` of the
    /// block read by invocation `i` and kept as the bits of an `f32`, between two barriers that
    /// every invocation of the workgroup reaches. Integer compares only, so the rows are the
    /// stable sort by id on every device.
    fn embedding_sort(&mut self, ids: u32) -> Result<(), EmitError> {
        let (index, count) = (self.index, self.count);
        let local = self.local.expect("a module with workgroup memory");
        let [block_var, t_var] = self.u32_vars;
        let rank_var = self.extra_u32_vars[0];
        let (zero, one, two) = (self.uint(0), self.uint(1), self.uint(2));
        let (n_ids, group) = (self.uint(ids), self.uint(GROUP));
        let inside = self.less(index, count)?;
        let at = self.b.select(self.u32_ty, None, inside, index, zero)?;
        let own = self.read_u32(0, at)?;
        self.b.store(rank_var, zero, None, [])?;
        self.counted(block_var, zero, n_ids, group, |m, block| {
            let n = m.add(block, local)?;
            let staged = m.less(n, n_ids)?;
            m.when(staged, |m| {
                let id = m.read_u32(0, n)?;
                let bits = m.b.bitcast(m.f32_ty, None, id)?;
                m.store_shared(local, bits)
            })?;
            m.barrier()?;
            m.when(inside, |m| {
                let left = m.sub(n_ids, block)?;
                let end = m.min(left, group)?;
                m.counted(t_var, zero, end, one, |m, t| {
                    let bits = m.load_shared(t)?;
                    let id = m.b.bitcast(m.u32_ty, None, bits)?;
                    let position = m.add(block, t)?;
                    let below = m.less(id, own)?;
                    let equal = m.equal(id, own)?;
                    let earlier = m.less(position, index)?;
                    let tie = m.and(equal, earlier)?;
                    let before = m.b.logical_or(m.bool_ty, None, below, tie)?;
                    let step = m.b.select(m.u32_ty, None, before, one, zero)?;
                    let rank = m.b.load(m.u32_ty, None, rank_var, None, [])?;
                    let rank = m.add(rank, step)?;
                    m.b.store(rank_var, rank, None, [])?;
                    Ok(())
                })
            })?;
            m.barrier()
        })?;
        self.when(inside, |m| {
            let rank = m.b.load(m.u32_ty, None, rank_var, None, [])?;
            let row = m.mul(rank, two)?;
            let id = m.b.bitcast(m.f32_ty, None, own)?;
            m.write(row, id)?;
            let next = m.add(row, one)?;
            let position = m.b.bitcast(m.f32_ty, None, index)?;
            m.write(next, position)
        })
    }

    /// The first row of the `[ids, 2]` sorted ids at binding 0 whose id is not below `key`, or
    /// `ids`: a binary search of fixed length, the bit length of `ids`, in the module's two extra
    /// `u32` variables and the shell's first.
    fn lower_bound(&mut self, ids: u32, key: Word) -> Result<Word, EmitError> {
        let [low_var, length_var] = [self.extra_u32_vars[0], self.extra_u32_vars[1]];
        let (zero, one, two) = (self.uint(0), self.uint(1), self.uint(2));
        let (n_ids, steps) = (self.uint(ids), self.uint(u32::BITS - ids.leading_zeros()));
        self.b.store(low_var, zero, None, [])?;
        self.b.store(length_var, n_ids, None, [])?;
        self.counted(self.u32_vars[0], zero, steps, one, |m, _| {
            let low = m.b.load(m.u32_ty, None, low_var, None, [])?;
            let length = m.b.load(m.u32_ty, None, length_var, None, [])?;
            let half = m.b.shift_right_logical(m.u32_ty, None, length, one)?;
            let middle = m.add(low, half)?;
            let open = m.less(zero, length)?;
            let probe = m.b.select(m.u32_ty, None, open, middle, zero)?;
            let row = m.mul(probe, two)?;
            let id = m.read_u32(0, row)?;
            let below = m.less(id, key)?;
            let right = m.and(open, below)?;
            let past = m.add(middle, one)?;
            let low = m.b.select(m.u32_ty, None, right, past, low)?;
            let rest = m.sub(length, half)?;
            let rest = m.sub(rest, one)?;
            let length = m.b.select(m.u32_ty, None, right, rest, half)?;
            m.b.store(low_var, low, None, [])?;
            m.b.store(length_var, length, None, [])?;
            Ok(())
        })?;
        Ok(self.b.load(self.u32_ty, None, low_var, None, [])?)
    }

    /// Element `index` of the table gradient, column `j = index % width` of table row
    /// `r = index / width`: `0.0`, then plus element `j` of gradient row `n` for each row
    /// `(r, n)` of the sorted ids at binding 0 in order, so for every position `n` whose id is
    /// `r` in ascending order, in the shell's `f32` variable, written once. The rows of `r` are
    /// those from the first whose id is not below `r` to the first whose id is not below
    /// `r + 1` ([`Shell::lower_bound`]).
    fn embedding_grad(&mut self, ids: u32, width: u32) -> Result<(), EmitError> {
        let (index, count, acc) = (self.index, self.count, self.f32_var);
        let inside = self.less(index, count)?;
        self.when(inside, |m| {
            let (one, two, w) = (m.uint(1), m.uint(2), m.uint(width));
            let row = m.div(index, w)?;
            let j = m.rem(index, w)?;
            let start = m.lower_bound(ids, row)?;
            let next = m.add(row, one)?;
            let end = m.lower_bound(ids, next)?;
            let f_zero = m.float(0.0);
            m.b.store(acc, f_zero, None, [])?;
            m.counted(m.u32_vars[1], start, end, one, |m, k| {
                let at = m.mul(k, two)?;
                let at = m.add(at, one)?;
                let n = m.read_u32(0, at)?;
                let g_base = m.mul(n, w)?;
                let g_at = m.add(g_base, j)?;
                let g = m.read(1, g_at)?;
                m.accumulate(g)
            })?;
            let sum = m.b.load(m.f32_ty, None, acc, None, [])?;
            m.write(index, sum)
        })
    }

    /// `values[index] * scale` where word 0 of the Philox block of `[index, 4, token]` under
    /// the run key is at least `threshold`, else `+0`.
    fn dropout(
        &mut self,
        values: usize,
        token: usize,
        threshold: u32,
        scale: f32,
    ) -> Result<(), EmitError> {
        let index = self.index;
        let (zero, one) = (self.uint(0), self.uint(1));
        let low = self.read_u32(token, zero)?;
        let high = self.read_u32(token, one)?;
        let stream = self.uint(DROPOUT_STREAM);
        let key = self.seed.expect("a dropout module declares the run key");
        let block = self.philox([index, stream, low, high], key)?;
        let t = self.uint(threshold);
        let keep = self
            .b
            .u_greater_than_equal(self.bool_ty, None, block[0], t)?;
        let x = self.read(values, index)?;
        let s = self.float(scale);
        let scaled = self.f_mul(x, s)?;
        let f_zero = self.float(0.0);
        let out = self.b.select(self.f32_ty, None, keep, scaled, f_zero)?;
        self.write(index, out)
    }

    /// Row `index`: `((x - mean) * rstd) * gamma + beta` per element.
    fn layer_norm(&mut self, width: u32) -> Result<(), EmitError> {
        let index = self.index;
        let w = self.uint(width);
        let base = self.mul(index, w)?;
        let (mean, rstd) = self.row_moments(0, base, width)?;
        let (zero, one, var) = (self.uint(0), self.uint(1), self.u32_vars[0]);
        self.counted(var, zero, w, one, |m, j| {
            let at = m.add(base, j)?;
            let x = m.read(0, at)?;
            let gamma = m.read(1, j)?;
            let beta = m.read(2, j)?;
            let d = m.f_sub(x, mean)?;
            let xhat = m.f_mul(d, rstd)?;
            let scaled = m.f_mul(xhat, gamma)?;
            let y = m.f_add(scaled, beta)?;
            m.write(at, y)
        })
    }

    /// The moments of row `index` for `LayerNormGradGamma`
    /// (story:vulkan-layer-norm-grad-gamma-row-moments): its mean and `rstd`, as `LayerNorm`
    /// computes them ([`Shell::row_moments`]), written to elements `2 * index` and
    /// `2 * index + 1`.
    fn layer_norm_moments(&mut self, width: u32) -> Result<(), EmitError> {
        let index = self.index;
        let w = self.uint(width);
        let base = self.mul(index, w)?;
        let (mean, rstd) = self.row_moments(0, base, width)?;
        let (one, two) = (self.uint(1), self.uint(2));
        let at = self.mul(index, two)?;
        self.write(at, mean)?;
        let next = self.add(at, one)?;
        self.write(next, rstd)
    }

    /// Row `index` of the gradient of `LayerNorm` with respect to `x`: with `gx = g * gamma`
    /// and `xhat = (x - mean) * rstd`, `rstd * ((gx - mean(gx)) - xhat * mean(gx * xhat))`, the
    /// first mean a sum from the first term, the second a sum of products.
    fn layer_norm_grad_x(&mut self, width: u32) -> Result<(), EmitError> {
        let index = self.index;
        let w = self.uint(width);
        let base = self.mul(index, w)?;
        let (mean, rstd) = self.row_moments(0, base, width)?;
        let (zero, one, var, acc) = (self.uint(0), self.uint(1), self.u32_vars[0], self.f32_var);
        // `g * gamma` and `(x - mean) * rstd` of element `j`.
        let gx = |m: &mut Shell, j: Word| -> Result<Word, EmitError> {
            let at = m.add(base, j)?;
            let g = m.read(2, at)?;
            let gamma = m.read(1, j)?;
            m.f_mul(g, gamma)
        };
        let xhat = |m: &mut Shell, j: Word| -> Result<Word, EmitError> {
            let at = m.add(base, j)?;
            let x = m.read(0, at)?;
            let d = m.f_sub(x, mean)?;
            m.f_mul(d, rstd)
        };
        let first = gx(self, zero)?;
        self.b.store(acc, first, None, [])?;
        self.counted(var, one, w, one, |m, j| {
            let term = gx(m, j)?;
            m.accumulate(term)
        })?;
        let total = self.b.load(self.f32_ty, None, acc, None, [])?;
        let mean_gx = self.over_width(total, width)?;
        let first_xhat = xhat(self, zero)?;
        let product = self.f_mul(first, first_xhat)?;
        self.b.store(acc, product, None, [])?;
        self.counted(var, one, w, one, |m, j| {
            let a = gx(m, j)?;
            let b = xhat(m, j)?;
            let sum = m.b.load(m.f32_ty, None, acc, None, [])?;
            let fused = m.f_fma(a, b, sum)?;
            m.b.store(acc, fused, None, [])?;
            Ok(())
        })?;
        let products = self.b.load(self.f32_ty, None, acc, None, [])?;
        let mean_gx_xhat = self.over_width(products, width)?;
        self.counted(var, zero, w, one, |m, j| {
            let a = gx(m, j)?;
            let b = xhat(m, j)?;
            let centered = m.f_sub(a, mean_gx)?;
            let along = m.f_mul(b, mean_gx_xhat)?;
            let d = m.f_sub(centered, along)?;
            let y = m.f_mul(rstd, d)?;
            let at = m.add(base, j)?;
            m.write(at, y)
        })
    }

    /// Row `index`: the maximum from the first value, a value replacing it only when greater;
    /// `exp(x - max)` written per element; their sum from the first term; each divided by it.
    fn softmax(&mut self, width: u32) -> Result<(), EmitError> {
        let index = self.index;
        let w = self.uint(width);
        let base = self.mul(index, w)?;
        let (zero, one, var, acc) = (self.uint(0), self.uint(1), self.u32_vars[0], self.f32_var);
        let first = self.read(0, base)?;
        self.b.store(acc, first, None, [])?;
        self.counted(var, one, w, one, |m, j| {
            let at = m.add(base, j)?;
            let v = m.read(0, at)?;
            let max = m.b.load(m.f32_ty, None, acc, None, [])?;
            let greater = m.b.f_ord_greater_than(m.bool_ty, None, v, max)?;
            let kept = m.b.select(m.f32_ty, None, greater, v, max)?;
            m.b.store(acc, kept, None, [])?;
            Ok(())
        })?;
        let max = self.b.load(self.f32_ty, None, acc, None, [])?;
        self.counted(var, zero, w, one, |m, j| {
            let at = m.add(base, j)?;
            let v = m.read(0, at)?;
            let d = m.f_sub(v, max)?;
            let e = m.exp(d)?;
            m.write(at, e)
        })?;
        let first = self.read_result(base)?;
        self.b.store(acc, first, None, [])?;
        self.counted(var, one, w, one, |m, j| {
            let at = m.add(base, j)?;
            let e = m.read_result(at)?;
            m.accumulate(e)
        })?;
        let total = self.b.load(self.f32_ty, None, acc, None, [])?;
        self.counted(var, zero, w, one, |m, j| {
            let at = m.add(base, j)?;
            let e = m.read_result(at)?;
            let y = m.f_div(e, total)?;
            m.write(at, y)
        })
    }

    /// Row `index` of the gradient of `Softmax` (operands `y`, `g`): `d` the sum of products
    /// `g * y`, then `y * (g - d)` per element.
    fn softmax_grad(&mut self, width: u32) -> Result<(), EmitError> {
        let index = self.index;
        let w = self.uint(width);
        let base = self.mul(index, w)?;
        let (zero, one, var, acc) = (self.uint(0), self.uint(1), self.u32_vars[0], self.f32_var);
        let y0 = self.read(0, base)?;
        let g0 = self.read(1, base)?;
        let first = self.f_mul(g0, y0)?;
        self.b.store(acc, first, None, [])?;
        self.counted(var, one, w, one, |m, j| {
            let at = m.add(base, j)?;
            let y = m.read(0, at)?;
            let g = m.read(1, at)?;
            let sum = m.b.load(m.f32_ty, None, acc, None, [])?;
            let fused = m.f_fma(g, y, sum)?;
            m.b.store(acc, fused, None, [])?;
            Ok(())
        })?;
        let d = self.b.load(self.f32_ty, None, acc, None, [])?;
        self.counted(var, zero, w, one, |m, j| {
            let at = m.add(base, j)?;
            let y = m.read(0, at)?;
            let g = m.read(1, at)?;
            let diff = m.f_sub(g, d)?;
            let out = m.f_mul(y, diff)?;
            m.write(at, out)
        })
    }

    /// Where an invocation of a column sum of `tile` over rows of `width` stands: its column of
    /// the workgroup's (`local % columns`), its slot (`local / columns`), its column of the
    /// rows, that column clamped to the last for reads, and whether it adds: the invocations
    /// of slot 0 whose column is inside the rows.
    fn column_lanes(&mut self, tile: ColumnTile, width: u32) -> Result<ColumnLanes, EmitError> {
        let local = self.local.expect("a module with workgroup memory");
        let (group, columns) = (self.uint(GROUP), self.uint(tile.columns));
        let workgroup = self.div(self.index, group)?;
        let lane = self.rem(local, columns)?;
        let slot = self.div(local, columns)?;
        let origin = self.mul(workgroup, columns)?;
        let column = self.add(origin, lane)?;
        let last_column = self.uint(width - 1);
        let read_column = self.min(column, last_column)?;
        let w = self.uint(width);
        let inside = self.less(column, w)?;
        let adds = self.less(local, columns)?;
        let adder = self.and(inside, adds)?;
        Ok(ColumnLanes {
            lane,
            slot,
            column,
            read_column,
            adder,
        })
    }

    /// `SumRows` over `[rows, width]` (story:vulkan-column-sums-coalesced): workgroup `w` sums
    /// columns `w * C ..` of the [`ColumnTile::sum_rows`] tile's `C` columns, invocation
    /// `c < C` column `c`, from row 0 and adding every later row in ascending order, as the CPU
    /// does. Rows `1 ..` are staged a tile at a time in workgroup memory, invocation `i`
    /// loading rows `i / C + S * k` of column `i % C`, so each row of a tile is one segment of
    /// `C` consecutive values. The loads of the next tile are issued before the adds of the
    /// current one, between two barriers every invocation reaches.
    fn sum_rows(&mut self, rows: u32, width: u32) -> Result<(), EmitError> {
        let tile = ColumnTile::sum_rows(width);
        let (tile_rows, slots_n) = (tile.rows, tile.slots);
        let ColumnLanes {
            lane,
            slot,
            column,
            read_column,
            adder,
        } = self.column_lanes(tile, width)?;
        let (var, acc) = (self.u32_vars[0], self.f32_var);
        let regs: Vec<Word> = self.matmul_vars[..tile.per_invocation() as usize].to_vec();
        let w = self.uint(width);
        let first = self.read(0, read_column)?;
        self.b.store(acc, first, None, [])?;
        let after = rows - 1;
        let full = after / tile_rows;
        let left = after % tile_rows;
        let last_row = self.uint(rows - 1);
        let load = |m: &mut Self, base: Word| -> Result<(), EmitError> {
            for (k, &reg) in regs.iter().enumerate() {
                let step = m.uint(slots_n * k as u32);
                let at_tile = m.add(slot, step)?;
                let row = m.add(base, at_tile)?;
                let row = m.min(row, last_row)?;
                let start = m.mul(row, w)?;
                let at = m.add(start, read_column)?;
                let x = m.read(0, at)?;
                m.b.store(reg, x, None, [])?;
            }
            Ok(())
        };
        let pitch = self.uint(tile.pitch());
        let segment = self.mul(lane, pitch)?;
        let pitch4 = self.uint(tile.pitch() / 4);
        let segment4 = self.mul(lane, pitch4)?;
        let stage = |m: &mut Self| -> Result<(), EmitError> {
            for (k, &reg) in regs.iter().enumerate() {
                let step = m.uint(slots_n * k as u32);
                let at_tile = m.add(slot, step)?;
                let at = m.add(segment, at_tile)?;
                let x = m.b.load(m.f32_ty, None, reg, None, [])?;
                m.store_shared(at, x)?;
            }
            m.barrier()
        };
        let add = |m: &mut Self, count: u32| -> Result<(), EmitError> {
            m.when(adder, |m| {
                let mut sum = m.b.load(m.f32_ty, None, acc, None, [])?;
                for v in 0..count.div_ceil(4) {
                    let at = m.uint(v);
                    let at = m.add(segment4, at)?;
                    let values = m.load_shared_vector(at)?;
                    for (j, &x) in values.iter().enumerate() {
                        if 4 * v + (j as u32) < count {
                            sum = m.f_add(sum, x)?;
                        }
                    }
                }
                m.b.store(acc, sum, None, [])?;
                Ok(())
            })?;
            m.barrier()
        };
        let one = self.uint(1);
        load(self, one)?;
        if full > 0 {
            let end = self.uint(1 + full * tile_rows);
            let step = self.uint(tile_rows);
            self.counted(var, one, end, step, |m, base| {
                stage(m)?;
                let next = m.add(base, step)?;
                load(m, next)?;
                add(m, tile_rows)
            })?;
        }
        if left > 0 {
            stage(self)?;
            add(self, left)?;
        }
        self.when(adder, |m| {
            let total = m.b.load(m.f32_ty, None, acc, None, [])?;
            m.write(column, total)
        })
    }

    /// The column sum of `LayerNormGradGamma` over `[rows, width]` (operands the row moments
    /// `[rows, 2]`, `x` and `g`; story:vulkan-layer-norm-grad-gamma-row-moments): workgroup `w`
    /// adds columns `w * C ..` of the [`ColumnTile::layer_norm_grad_gamma`] tile's `C` columns,
    /// invocation `c < C` column `c`, over the rows in ascending order, each term `g * xhat` fused
    /// into the sum, as the CPU does. Each sum starts at `-0.0`, the one value `v` with
    /// `fma(a, b, v) = a * b` for every `a` and `b`, so the first term of a column is the CPU's
    /// rounded first product.
    ///
    /// Invocation `i` loads rows `i / C + S * k` of column `i % C` of a tile, `x`, `g` and the
    /// row's mean and `rstd` its moments kernel wrote, so each row of a tile is one segment of
    /// `C` consecutive values. It stages `xhat = (x - mean) * rstd` and `g` in workgroup memory,
    /// and after a barrier the adding invocations add the tile's terms. The loads of the next
    /// tile are issued before the adds of the current one, between two barriers every invocation
    /// reaches. Nothing in the module is unrolled per column of the row: its size does not depend
    /// on the width.
    fn layer_norm_grad_gamma(&mut self, rows: u32, width: u32) -> Result<(), EmitError> {
        let tile = ColumnTile::layer_norm_grad_gamma(width);
        let (tile_rows, slots_n) = (tile.rows, tile.slots);
        let per = tile.per_invocation() as usize;
        let ColumnLanes {
            lane,
            slot,
            column,
            read_column,
            adder,
        } = self.column_lanes(tile, width)?;
        let (var, acc) = (self.u32_vars[0], self.f32_var);
        let regs: Vec<Word> = self.matmul_vars[..4 * per].to_vec();
        let (xs, rest) = regs.split_at(per);
        let (gs, rest) = rest.split_at(per);
        let (means, rstds) = rest.split_at(per);
        let (zero, one, two, w) = (self.uint(0), self.uint(1), self.uint(2), self.uint(width));
        let last_row = self.uint(rows - 1);
        let negative_zero = self.float(-0.0);
        self.b.store(acc, negative_zero, None, [])?;
        let load = |m: &mut Self, base: Word| -> Result<(), EmitError> {
            for k in 0..per {
                let step = m.uint(slots_n * k as u32);
                let at_tile = m.add(slot, step)?;
                let row = m.add(base, at_tile)?;
                let row = m.min(row, last_row)?;
                let start = m.mul(row, w)?;
                let at = m.add(start, read_column)?;
                let x = m.read(1, at)?;
                m.b.store(xs[k], x, None, [])?;
                let g = m.read(2, at)?;
                m.b.store(gs[k], g, None, [])?;
                let moment = m.mul(row, two)?;
                let mean = m.read(0, moment)?;
                m.b.store(means[k], mean, None, [])?;
                let next = m.add(moment, one)?;
                let rstd = m.read(0, next)?;
                m.b.store(rstds[k], rstd, None, [])?;
            }
            Ok(())
        };
        let pitch = self.uint(tile.pitch());
        let xhat_segment = self.mul(lane, pitch)?;
        let g_start = self.uint(tile.values());
        let g_segment = self.add(g_start, xhat_segment)?;
        let pitch4 = self.uint(tile.pitch() / 4);
        let xhat_segment4 = self.mul(lane, pitch4)?;
        let g_start4 = self.uint(tile.values() / 4);
        let g_segment4 = self.add(g_start4, xhat_segment4)?;
        let stage = |m: &mut Self| -> Result<(), EmitError> {
            for k in 0..per {
                let step = m.uint(slots_n * k as u32);
                let at_tile = m.add(slot, step)?;
                let x = m.b.load(m.f32_ty, None, xs[k], None, [])?;
                let g = m.b.load(m.f32_ty, None, gs[k], None, [])?;
                let mean = m.b.load(m.f32_ty, None, means[k], None, [])?;
                let rstd = m.b.load(m.f32_ty, None, rstds[k], None, [])?;
                let d = m.f_sub(x, mean)?;
                let xhat = m.f_mul(d, rstd)?;
                let at = m.add(xhat_segment, at_tile)?;
                m.store_shared(at, xhat)?;
                let at = m.add(g_segment, at_tile)?;
                m.store_shared(at, g)?;
            }
            m.barrier()
        };
        let add = |m: &mut Self, count: u32| -> Result<(), EmitError> {
            m.when(adder, |m| {
                let mut sum = m.b.load(m.f32_ty, None, acc, None, [])?;
                for v in 0..count.div_ceil(4) {
                    let at = m.uint(v);
                    let xhat_at = m.add(xhat_segment4, at)?;
                    let xhats = m.load_shared_vector(xhat_at)?;
                    let g_at = m.add(g_segment4, at)?;
                    let gs = m.load_shared_vector(g_at)?;
                    let terms = (count - 4 * v).min(4) as usize;
                    for (&xhat, &g) in xhats.iter().zip(&gs).take(terms) {
                        sum = m.f_fma(g, xhat, sum)?;
                    }
                }
                m.b.store(acc, sum, None, [])?;
                Ok(())
            })?;
            m.barrier()
        };
        load(self, zero)?;
        let (full, left) = (rows / tile_rows, rows % tile_rows);
        if full > 0 {
            let end = self.uint(full * tile_rows);
            let step = self.uint(tile_rows);
            self.counted(var, zero, end, step, |m, base| {
                stage(m)?;
                let next = m.add(base, step)?;
                load(m, next)?;
                add(m, tile_rows)
            })?;
        }
        if left > 0 {
            stage(self)?;
            add(self, left)?;
        }
        self.when(adder, |m| {
            let total = m.b.load(m.f32_ty, None, acc, None, [])?;
            m.write(column, total)
        })
    }
}

/// [`Shell::column_lanes`].
#[derive(Clone, Copy)]
struct ColumnLanes {
    lane: Word,
    slot: Word,
    column: Word,
    read_column: Word,
    adder: Word,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::init::{Stream, block, dropout_keeps};

    fn uniform(w: u64) -> f64 {
        (w as f64 + 0.5) / 4_294_967_296.0
    }

    /// The threshold is the CPU's keep rule at and on both sides of it, for drop
    /// probabilities that are and are not the exact uniform of a word.
    #[test]
    fn the_keep_threshold_is_the_cpus_rule_at_its_edges() {
        let exact = (uniform(12_345) as f32, uniform(1 << 23) as f32);
        let ps = [
            0.0,
            f32::MIN_POSITIVE,
            0.1,
            0.25,
            0.3,
            0.5,
            0.9,
            exact.0,
            exact.1,
            1.0 - f32::EPSILON / 2.0,
        ];
        for p in ps {
            let t = u64::from(keep_threshold(p));
            let keeps = |w: u64| uniform(w) >= f64::from(p);
            assert!(keeps(t), "p {p}: word {t} is kept");
            if t > 0 {
                assert!(!keeps(t - 1), "p {p}: word {} is dropped", t - 1);
            }
        }
        assert_eq!(keep_threshold(0.0), 0);
    }

    /// Word 0 of the dropout block at or past the threshold is exactly the CPU's
    /// `dropout_keeps`, over the first 4096 elements of a draw at three drop probabilities.
    #[test]
    fn the_threshold_keeps_what_the_cpu_keeps() {
        let (seed, id) = (0x0123_4567_89ab_cdef_u64, 0xfeed_beef_dead_f00d_u64);
        for p in [0.1_f32, 0.25, 0.5] {
            let t = keep_threshold(p);
            for i in 0..4096 {
                let w = block(seed, Stream::Dropout, id, i)[0];
                assert_eq!(w >= t, dropout_keeps(seed, id, i, p), "p {p} element {i}");
            }
        }
    }
}
