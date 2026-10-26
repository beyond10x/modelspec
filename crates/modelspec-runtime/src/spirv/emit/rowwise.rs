//! The modules of the thirteen row-wise kernels (DESIGN.md § 5.2): `Embedding`, `AddRows`,
//! `Gelu`, `GeluGrad`, `Dropout` and `DropoutGrad` one invocation per output element;
//! `EmbeddingGrad`, `LayerNorm`, `LayerNormGradX`, `Softmax` and `SoftmaxGrad` one per output
//! row; `SumRows` and `LayerNormGradGamma` one per output column.
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
//! reads past the table; the CPU refuses such an id before it runs. `EmbeddingGrad` zeroes its
//! table row, then adds the gradient row of every id equal to it in ascending id position,
//! without atomics. `LayerNormGradGamma` runs workgroups of 64 columns; for each block of 64
//! rows, invocation `i` of the workgroup computes the mean and reciprocal standard deviation of
//! row `i` of the block into workgroup memory, and after a barrier each invocation adds its
//! column's terms of the block's rows in ascending order. `Dropout` and `DropoutGrad` compute
//! the Philox4x32-10 block of each element from the site's RNG token and the run key, the
//! specialization constants `crate::vulkan::pipeline::SEED_SPEC_IDS`, and keep the element when
//! word 0 is at least [`keep_threshold`] of `p`.

use super::{EmitError, Kernel, KernelKind, Operand, Shell, Word, operands_error, static_shape};
use modelspec_core::ir::tensor::DType;

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
/// The workgroup size of every row-wise kernel, and the rows `LayerNormGradGamma` stages.
const GROUP: u32 = 64;

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
        if kernel.dtype != DType::F32 || kernel.result.dtype != DType::F32 {
            return Err(wrong("a row-wise kernel writes f32"));
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
            ("EmbeddingGrad", [I32, F32], [[n], [gn, h]], [v, rh], []) if n == gn && h == rh => (
                PerRow,
                u(*v),
                Rowwise::EmbeddingGrad {
                    ids: u(*n),
                    width: u(*h),
                },
            ),
            ("AddRows", [F32, F32], [[n, h], [bh]], [rn, rh], [])
                if h == bh && (n, h) == (rn, rh) =>
            {
                (PerElement, out, Rowwise::AddRows { width: u(*h) })
            }
            ("SumRows", [F32], [[n, h]], [rh], []) if h == rh => (
                PerColumn,
                u(*h),
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
            ("LayerNormGradGamma", [F32, F32], [[n, h], g], [rh], [])
                if h == rh && *g == [*n, *h] =>
            {
                (
                    PerColumn,
                    u(*h),
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
            Rowwise::LayerNormGradGamma { .. } => Some(2 * GROUP),
            _ => None,
        }
    }

    /// The `f32` function variables the module keeps beside the shell's one.
    pub(super) fn f32_vars(self) -> usize {
        match self {
            Rowwise::LayerNormGradGamma { .. } => 1,
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

    /// `1 / sqrt(v)` within 1 ulp of the CPU's, which rounds the square root first: the device's
    /// `s = sqrt(v)` and `q = 1 / s` (neither correctly rounded on Vulkan), the root corrected
    /// to `s' = s + fma(0 - s, s, v) * (q * 0.5)`, then the reciprocal to
    /// `q + q * fma(0 - q, s', 1)`, one Newton step from `q`. `q` stands where that last
    /// residual is 0 or NaN: an exact reciprocal, an infinite `v` (`q = 0`) or a NaN.
    fn reciprocal_sqrt(&mut self, v: Word) -> Result<Word, EmitError> {
        let (zero, half, one) = (self.float(0.0), self.float(0.5), self.float(1.0));
        let s = self.sqrt(v)?;
        let q = self.f_div(one, s)?;
        let neg_s = self.f_sub(zero, s)?;
        let root_residual = self.f_fma(neg_s, s, v)?;
        let half_q = self.f_mul(q, half)?;
        let root = self.f_fma(root_residual, half_q, s)?;
        let neg_q = self.f_sub(zero, q)?;
        let residual = self.f_fma(neg_q, root, one)?;
        let corrected = self.f_fma(residual, q, q)?;
        self.where_residual(residual, corrected, q)
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
        let (index, count) = (self.index, self.count);
        let inside = self.less(index, count)?;
        self.when(inside, |m| match rowwise {
            Rowwise::Embedding { table, width } => m.embedding(table, width),
            Rowwise::EmbeddingGrad { ids, width } => m.embedding_grad(ids, width),
            Rowwise::AddRows { width } => {
                let w = m.uint(width);
                let j = m.rem(index, w)?;
                let x = m.read(0, index)?;
                let b = m.read(1, j)?;
                let sum = m.f_add(x, b)?;
                m.write(index, sum)
            }
            Rowwise::SumRows { rows, width } => {
                let (var, acc) = (m.u32_vars[0], m.f32_var);
                let first = m.read(0, index)?;
                m.b.store(acc, first, None, [])?;
                let (one, end, w) = (m.uint(1), m.uint(rows), m.uint(width));
                m.counted(var, one, end, one, |m, n| {
                    let row = m.mul(n, w)?;
                    let at = m.add(row, index)?;
                    let x = m.read(0, at)?;
                    m.accumulate(x)
                })?;
                let total = m.b.load(m.f32_ty, None, acc, None, [])?;
                m.write(index, total)
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
            Rowwise::LayerNormGradX { width } => m.layer_norm_grad_x(width),
            Rowwise::Softmax { width } => m.softmax(width),
            Rowwise::SoftmaxGrad { width } => m.softmax_grad(width),
            Rowwise::LayerNormGradGamma { .. } => unreachable!("handled above"),
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

    /// Table row `index` of the gradient: `0.0`, then plus gradient row `n` for every `n` in
    /// ascending order whose id is `index`.
    fn embedding_grad(&mut self, ids: u32, width: u32) -> Result<(), EmitError> {
        let index = self.index;
        let [n_var, j_var] = self.u32_vars;
        let (zero, one) = (self.uint(0), self.uint(1));
        let (w, end) = (self.uint(width), self.uint(ids));
        let base = self.mul(index, w)?;
        let f_zero = self.float(0.0);
        self.counted(j_var, zero, w, one, |m, j| {
            let at = m.add(base, j)?;
            m.write(at, f_zero)
        })?;
        self.counted(n_var, zero, end, one, |m, n| {
            let id = m.read_u32(0, n)?;
            let hit = m.equal(id, index)?;
            m.when(hit, |m| {
                let g_base = m.mul(n, w)?;
                m.counted(j_var, zero, w, one, |m, j| {
                    let at = m.add(base, j)?;
                    let current = m.read_result(at)?;
                    let g_at = m.add(g_base, j)?;
                    let g = m.read(1, g_at)?;
                    let sum = m.f_add(current, g)?;
                    m.write(at, sum)
                })
            })
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

    /// Column `index` of the gradient of `LayerNorm` with respect to `gamma` (operands `x`,
    /// `g`): over the rows in ascending order, `g * xhat` for row 0 and each later term fused
    /// into the sum. Each block of 64 rows stages its rows' mean and `rstd` in workgroup
    /// memory, row `i` of the block computed by invocation `i`, between two barriers that every
    /// invocation of the workgroup reaches.
    fn layer_norm_grad_gamma(&mut self, rows: u32, width: u32) -> Result<(), EmitError> {
        let (index, count) = (self.index, self.count);
        let local = self.local.expect("a module with workgroup memory");
        let [block_var, t_var] = self.u32_vars;
        let column_acc = self.matmul_vars[0];
        let (zero, one, two) = (self.uint(0), self.uint(1), self.uint(2));
        let (w, n_rows, group) = (self.uint(width), self.uint(rows), self.uint(GROUP));
        let f_zero = self.float(0.0);
        self.b.store(column_acc, f_zero, None, [])?;
        let inside = self.less(index, count)?;
        self.counted(block_var, zero, n_rows, group, |m, block| {
            let row = m.add(block, local)?;
            let staged = m.less(row, n_rows)?;
            m.when(staged, |m| {
                let base = m.mul(row, w)?;
                let (mean, rstd) = m.row_moments(0, base, width)?;
                let at = m.mul(local, two)?;
                m.store_shared(at, mean)?;
                let next = m.add(at, one)?;
                m.store_shared(next, rstd)
            })?;
            m.barrier()?;
            m.when(inside, |m| {
                let left = m.sub(n_rows, block)?;
                let end = m.min(left, group)?;
                m.counted(t_var, zero, end, one, |m, t| {
                    let n = m.add(block, t)?;
                    let at = m.mul(t, two)?;
                    let mean = m.load_shared(at)?;
                    let next = m.add(at, one)?;
                    let rstd = m.load_shared(next)?;
                    let row = m.mul(n, w)?;
                    let element = m.add(row, index)?;
                    let x = m.read(0, element)?;
                    let g = m.read(1, element)?;
                    let d = m.f_sub(x, mean)?;
                    let xhat = m.f_mul(d, rstd)?;
                    let product = m.f_mul(g, xhat)?;
                    let sum = m.b.load(m.f32_ty, None, column_acc, None, [])?;
                    let fused = m.f_fma(g, xhat, sum)?;
                    let first = m.equal(n, zero)?;
                    let next_sum = m.b.select(m.f32_ty, None, first, product, fused)?;
                    m.b.store(column_acc, next_sum, None, [])?;
                    Ok(())
                })
            })?;
            m.barrier()
        })?;
        self.when(inside, |m| {
            let total = m.b.load(m.f32_ty, None, column_acc, None, [])?;
            m.write(index, total)
        })
    }
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
