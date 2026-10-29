//! The modules of the two held-out evaluation kernels (story:vulkan-eval-metrics-on-device),
//! which reduce the logits of a `HeldOut` batch on the device to what the host tally of
//! `crate::train::eval` needs from them: the weighted sum of the row losses and the count of the
//! correct selected rows, so an evaluation downloads [`HELD_OUT_STATE`] floats instead of every
//! batch's logits.
//!
//! `HeldOutRows` runs one invocation per row `n` of the logits `[N, V]` and writes two floats:
//! at `2n` the row loss `max + ln(Σ exp(x - max)) - x[label]` with the instructions of the
//! `CrossEntropy` kernel (`emit/attention.rs`): the maximum starts at the row's first logit and
//! is replaced only by a greater one, the sum starts at `-0.0` and adds each `exp(x - max)` in
//! ascending class order, and a label outside the classes reads class 0, which the host refuses
//! before it reads the result; at `2n + 1` `1.0` when the row holds no NaN and its first largest
//! logit, the lowest class among ties, is its label, else `0.0`, the CPU's accuracy rule.
//!
//! `HeldOutSum` runs one workgroup of 64 over the `N` rows of one batch: its invocations stage
//! the row loss, the hit and the weight of 64 rows at a time in workgroup memory, and after a
//! barrier invocation 0 adds them in ascending row order into the state `[hi, lo, low, high]`.
//! The weighted loss `w · nll` is kept exactly as its rounded product and the product's error
//! (one `Fma`), and added to the unevaluated sum `hi + lo` by a two-sum, so the state carries
//! the numerator of the host's `f64` sum to about 48 bits; a sum that is not finite is the plain
//! `f32` sum `hi + w · nll`. The rows of positive weight that hit are counted in `low`, below
//! 65536, and `high`, the carries, both whole numbers in `f32`. The first module of an
//! evaluation starts the state at `[-0.0, +0.0, 0, 0]`, the next ones from the state the module
//! before wrote. Every float add, subtract, multiply and `Fma` is decorated `NoContraction`.

use super::{EmitError, Kernel, KernelKind, Operand, Shell, Word, operands_error, static_shape};
use modelspec_core::ir::tensor::{DType, TensorType};
use rspirv::spirv::GlslStd450Op;

/// The workgroup size of both kernels, and the rows `HeldOutSum` stages at a time.
const GROUP: u32 = 64;
/// The floats `HeldOutSum` stages per row: the row loss, the hit and the weight.
const STAGED: u32 = 3;
/// The floats of the state `HeldOutSum` writes: the numerator's high and low parts, then the
/// correct count below [`COUNT_LIMB`] and its carries.
pub const HELD_OUT_STATE: u64 = 4;
/// The base of the two limbs of the correct count.
pub const COUNT_LIMB: f32 = 65536.0;
/// The most workgroups of one dimension of a dispatch every Vulkan device runs.
const MAX_GROUPS: u64 = 65_535;

/// The kernels that reduce the logits `[rows, classes]` of each `HeldOut` batch on the device.
#[derive(Clone, Debug, PartialEq)]
pub struct HeldOutKernels {
    /// `HeldOutRows`: the logits `f32 [rows, classes]` and the labels `i32 [rows]` into
    /// `f32 [rows, 2]`, each row's loss and hit.
    pub rows: Kernel,
    /// `HeldOutSum` of the first batch of an evaluation: the rows' losses and hits and the
    /// weights `f32 [rows]` into the state `f32 [4]`, started at `[-0.0, +0.0, 0, 0]`.
    pub first: Kernel,
    /// `HeldOutSum` of every later batch: the state added to where the module before left it.
    pub next: Kernel,
}

/// The held-out kernels of batches of `rows` rows of `classes` logits, or why there are none:
/// either is 0, or `rows · classes` does not fit `u32`.
pub fn held_out_kernels(rows: u64, classes: u64) -> Result<HeldOutKernels, EmitError> {
    let fits = |n: Option<u64>| n.and_then(|n| u32::try_from(n).ok()).filter(|&n| n > 0);
    if fits(rows.checked_mul(classes)).is_none() || fits(rows.checked_mul(2)).is_none() {
        return Err(EmitError::Operands {
            op: "HeldOutRows",
            reason: format!(
                "held-out batches of {rows} rows of {classes} logits are not 1 to 2^32 - 1 \
                 elements"
            ),
        });
    }
    let groups = rows.div_ceil(u64::from(GROUP));
    let spread = [
        groups.min(MAX_GROUPS) as u32,
        groups.div_ceil(MAX_GROUPS) as u32,
        1,
    ];
    let rows_kernel = kernel(
        KernelKind::PerRow,
        "HeldOutRows",
        vec![
            Operand::Buffer(TensorType::f32(&[rows, classes])),
            Operand::Buffer(TensorType::i32(&[rows])),
        ],
        TensorType::f32(&[rows, 2]),
        spread,
    );
    let sum = |start: f32| {
        kernel(
            KernelKind::LossSum,
            "HeldOutSum",
            vec![
                Operand::Buffer(TensorType::f32(&[rows, 2])),
                Operand::Buffer(TensorType::f32(&[rows])),
                Operand::Immediate(start),
            ],
            TensorType::f32(&[HELD_OUT_STATE]),
            [1, 1, 1],
        )
    };
    Ok(HeldOutKernels {
        rows: rows_kernel,
        first: sum(1.0),
        next: sum(0.0),
    })
}

/// The numerator and the correct count of the state `[hi, lo, low, high]` the last
/// `HeldOutSum` of an evaluation wrote: `hi` alone when it is not finite or `lo` is zero, so a
/// sum of zeros keeps its sign, else `hi + lo` in `f64`; and `high · 65536 + low`.
pub fn held_out_sums(state: [f32; 4]) -> (f64, u64) {
    let [hi, lo, low, high] = state;
    let numerator = if !hi.is_finite() || lo == 0.0 {
        f64::from(hi)
    } else {
        f64::from(hi) + f64::from(lo)
    };
    let limb = COUNT_LIMB as u64;
    (numerator, high as u64 * limb + low as u64)
}

fn kernel(
    kind: KernelKind,
    op: &'static str,
    operands: Vec<Operand>,
    result: TensorType,
    workgroups: [u32; 3],
) -> Kernel {
    let mut kernel = Kernel {
        id: Kernel::module_id(""),
        kind,
        op,
        dtype: DType::F32,
        operands,
        result,
        reduces: None,
        epilogue: None,
        transposed: [false, false],
        workgroup_size: GROUP,
        workgroups,
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

/// What a held-out module computes, its constants taken from the kernel's shapes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum HeldOutBody {
    /// `HeldOutRows` over rows of `classes` logits.
    Rows { classes: u32 },
    /// `HeldOutSum` over `rows` rows, starting the state when `first`.
    Sum { rows: u32, first: bool },
}

impl HeldOutBody {
    /// The module of `kernel`, or why its kind, operands or workgroups are not those
    /// [`held_out_kernels`] gives it.
    pub(super) fn of(kernel: &Kernel) -> Result<Self, EmitError> {
        let wrong = |what: &str| operands_error(kernel, what.to_owned());
        if kernel.workgroup_size != GROUP || kernel.dtype != DType::F32 {
            return Err(wrong("a held-out kernel runs f32 workgroups of 64"));
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
        let fits = |n: u64, m: u64| {
            n.checked_mul(m)
                .and_then(|n| u32::try_from(n).ok())
                .is_some_and(|n| n > 0)
        };
        match (kernel.kind, kernel.op) {
            (KernelKind::PerRow, "HeldOutRows") => match (&buffers[..], &result[..]) {
                ([(DType::F32, logits), (DType::I32, labels)], [n, 2])
                    if logits.len() == 2
                        && labels[..] == [*n]
                        && logits[0] == *n
                        && fits(*n, logits[1])
                        && fits(*n, 2)
                        && immediates.is_empty() =>
                {
                    let covered = kernel
                        .workgroups
                        .iter()
                        .map(|&w| u64::from(w))
                        .product::<u64>()
                        * u64::from(GROUP);
                    if covered < *n {
                        return Err(wrong("the workgroups of 64 cover every row"));
                    }
                    Ok(HeldOutBody::Rows {
                        classes: logits[1] as u32,
                    })
                }
                _ => Err(wrong(
                    "HeldOutRows reads f32 logits [n, v] and i32 labels [n] into f32 [n, 2]",
                )),
            },
            (KernelKind::LossSum, "HeldOutSum") => {
                match (&buffers[..], &result[..], &immediates[..]) {
                    ([(DType::F32, rows), (DType::F32, weights)], [state], [start])
                        if rows.len() == 2
                            && rows[1] == 2
                            && weights[..] == [rows[0]]
                            && fits(rows[0], 2)
                            && *state == HELD_OUT_STATE
                            && (*start == 1.0 || *start == 0.0)
                            && kernel.workgroups == [1, 1, 1] =>
                    {
                        Ok(HeldOutBody::Sum {
                            rows: rows[0] as u32,
                            first: *start == 1.0,
                        })
                    }
                    _ => Err(wrong(
                        "HeldOutSum reads f32 [n, 2] and f32 weights [n] into f32 [4] in one \
                     workgroup, its start flag last",
                    )),
                }
            }
            _ => Err(EmitError::Unsupported {
                kind: kernel.kind,
                op: kernel.op,
            }),
        }
    }

    /// The `f32` values of workgroup memory: three per staged row of `HeldOutSum`.
    pub(super) fn shared(self) -> Option<u32> {
        match self {
            HeldOutBody::Rows { .. } => None,
            HeldOutBody::Sum { .. } => Some(STAGED * GROUP),
        }
    }

    /// The `f32` function variables the module keeps beside the shell's one: the row sum of
    /// `HeldOutRows`; the four parts of the state of `HeldOutSum`.
    pub(super) fn f32_vars(self) -> usize {
        match self {
            HeldOutBody::Rows { .. } => 1,
            HeldOutBody::Sum { .. } => HELD_OUT_STATE as usize,
        }
    }

    /// The `u32` function variables the module keeps beside the shell's two: the first largest
    /// class and the NaN flag of `HeldOutRows`; the term counter of `HeldOutSum`.
    pub(super) fn u32_vars(self) -> usize {
        match self {
            HeldOutBody::Rows { .. } => 2,
            HeldOutBody::Sum { .. } => 1,
        }
    }
}

impl Shell {
    /// The module body of `body`.
    pub(super) fn held_out(&mut self, body: HeldOutBody) -> Result<(), EmitError> {
        match body {
            HeldOutBody::Rows { classes } => {
                let inside = self.less(self.index, self.count)?;
                self.when(inside, |m| m.held_out_row(classes))
            }
            HeldOutBody::Sum { rows, first } => self.held_out_sum(rows, first),
        }
    }

    /// Row `index` of `HeldOutRows`: its loss and its hit.
    fn held_out_row(&mut self, classes: u32) -> Result<(), EmitError> {
        let index = self.index;
        let (col_var, max_var, total_var) = (self.u32_vars[1], self.f32_var, self.matmul_vars[0]);
        let (arg_var, nan_var) = (self.extra_u32_vars[0], self.extra_u32_vars[1]);
        let (zero, one, v) = (self.uint(0), self.uint(1), self.uint(classes));
        let base = self.mul(index, v)?;
        let first = self.read(0, base)?;
        self.b.store(max_var, first, None, [])?;
        self.b.store(arg_var, zero, None, [])?;
        let first_nan = self.b.is_nan(self.bool_ty, None, first)?;
        let flag = self.b.select(self.u32_ty, None, first_nan, one, zero)?;
        self.b.store(nan_var, flag, None, [])?;
        self.counted(col_var, one, v, one, |m, c| {
            let at = m.add(base, c)?;
            let x = m.read(0, at)?;
            let max = m.b.load(m.f32_ty, None, max_var, None, [])?;
            let greater = m.b.f_ord_greater_than(m.bool_ty, None, x, max)?;
            let kept = m.b.select(m.f32_ty, None, greater, x, max)?;
            m.b.store(max_var, kept, None, [])?;
            let arg = m.b.load(m.u32_ty, None, arg_var, None, [])?;
            let arg = m.b.select(m.u32_ty, None, greater, c, arg)?;
            m.b.store(arg_var, arg, None, [])?;
            let nan = m.b.load(m.u32_ty, None, nan_var, None, [])?;
            let is_nan = m.b.is_nan(m.bool_ty, None, x)?;
            let nan = m.b.select(m.u32_ty, None, is_nan, one, nan)?;
            m.b.store(nan_var, nan, None, [])?;
            Ok(())
        })?;
        let max = self.b.load(self.f32_ty, None, max_var, None, [])?;
        let start = self.float(-0.0);
        self.b.store(total_var, start, None, [])?;
        self.counted(col_var, zero, v, one, |m, c| {
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
        let label = self.read_u32(1, index)?;
        let valid = self.less(label, v)?;
        let class = self.b.select(self.u32_ty, None, valid, label, zero)?;
        let target_at = self.add(base, class)?;
        let target = self.read(0, target_at)?;
        let log = self.glsl_f32(GlslStd450Op::Log, &[total])?;
        let shifted = self.f_add(max, log)?;
        let loss = self.f_sub(shifted, target)?;
        let arg = self.b.load(self.u32_ty, None, arg_var, None, [])?;
        let nan = self.b.load(self.u32_ty, None, nan_var, None, [])?;
        let same = self.equal(arg, label)?;
        let clean = self.equal(nan, zero)?;
        let hit = self.and(same, clean)?;
        let (f_one, f_zero) = (self.float(1.0), self.float(0.0));
        let hit = self.b.select(self.f32_ty, None, hit, f_one, f_zero)?;
        let two = self.uint(2);
        let out = self.mul(index, two)?;
        self.write(out, loss)?;
        let out = self.add(out, one)?;
        self.write(out, hit)
    }

    /// The one workgroup of `HeldOutSum` over `rows` rows.
    fn held_out_sum(&mut self, rows: u32, first: bool) -> Result<(), EmitError> {
        let local = self.local.expect("a module with workgroup memory");
        let block_var = self.u32_vars[0];
        let term_var = self.extra_u32_vars[0];
        let state = [
            self.matmul_vars[0],
            self.matmul_vars[1],
            self.matmul_vars[2],
            self.matmul_vars[3],
        ];
        let (zero, one) = (self.uint(0), self.uint(1));
        let (n_rows, group, staged) = (self.uint(rows), self.uint(GROUP), self.uint(STAGED));
        let blocks = self.uint(rows.div_ceil(GROUP));
        let first_invocation = self.equal(local, zero)?;
        self.when(first_invocation, |m| {
            if first {
                let starts = [m.float(-0.0), m.float(0.0), m.float(0.0), m.float(0.0)];
                for (var, start) in state.into_iter().zip(starts) {
                    m.b.store(var, start, None, [])?;
                }
            } else {
                for (at, var) in state.into_iter().enumerate() {
                    let at = m.uint(at as u32);
                    let value = m.read_result(at)?;
                    m.b.store(var, value, None, [])?;
                }
            }
            Ok(())
        })?;
        self.counted(block_var, zero, blocks, one, |m, block| {
            let base = m.mul(block, group)?;
            let row = m.add(base, local)?;
            let inside = m.less(row, n_rows)?;
            m.when(inside, |m| {
                let two = m.uint(2);
                let at = m.mul(row, two)?;
                let loss = m.read(0, at)?;
                let at = m.add(at, one)?;
                let hit = m.read(0, at)?;
                let w = m.read(1, row)?;
                let slot = m.mul(local, staged)?;
                m.store_shared(slot, loss)?;
                let slot = m.add(slot, one)?;
                m.store_shared(slot, hit)?;
                let slot = m.add(slot, one)?;
                m.store_shared(slot, w)
            })?;
            m.barrier()?;
            m.when(first_invocation, |m| {
                let left = m.sub(n_rows, base)?;
                let len = m.min(left, group)?;
                m.counted(term_var, zero, len, one, |m, t| {
                    let slot = m.mul(t, staged)?;
                    let loss = m.load_shared(slot)?;
                    let slot = m.add(slot, one)?;
                    let hit = m.load_shared(slot)?;
                    let slot = m.add(slot, one)?;
                    let w = m.load_shared(slot)?;
                    m.held_out_term(state, loss, hit, w)
                })
            })?;
            m.barrier()
        })?;
        self.when(first_invocation, |m| {
            for (at, var) in state.into_iter().enumerate() {
                let value = m.b.load(m.f32_ty, None, var, None, [])?;
                let at = m.uint(at as u32);
                m.write(at, value)?;
            }
            Ok(())
        })
    }

    /// Adds one row's weighted loss `w · loss` to the numerator `hi + lo` of `state`, and counts
    /// the row when its weight is positive and it hits.
    fn held_out_term(
        &mut self,
        state: [Word; 4],
        loss: Word,
        hit: Word,
        w: Word,
    ) -> Result<(), EmitError> {
        let [hi_var, lo_var, low_var, high_var] = state;
        let p = self.f_mul(w, loss)?;
        let neg_p = self.b.f_negate(self.f32_ty, None, p)?;
        let e = self.f_fma(w, loss, neg_p)?;
        let hi = self.b.load(self.f32_ty, None, hi_var, None, [])?;
        let lo = self.b.load(self.f32_ty, None, lo_var, None, [])?;
        // The two-sum of `hi` and `p`: `s + err` is exactly `hi + p`.
        let s = self.f_add(hi, p)?;
        let bb = self.f_sub(s, hi)?;
        let s_bb = self.f_sub(s, bb)?;
        let hi_err = self.f_sub(hi, s_bb)?;
        let p_err = self.f_sub(p, bb)?;
        let err = self.f_add(hi_err, p_err)?;
        let errors = self.f_add(err, e)?;
        let t = self.f_add(lo, errors)?;
        // `s + t` renormalized; a zero `t` leaves `s`, so a sum of zeros keeps its sign.
        let h1 = self.f_add(s, t)?;
        let h1_s = self.f_sub(h1, s)?;
        let l1 = self.f_sub(t, h1_s)?;
        let f_zero = self.float(0.0);
        let t_zero = self.b.f_ord_equal(self.bool_ty, None, t, f_zero)?;
        let h2 = self.b.select(self.f32_ty, None, t_zero, s, h1)?;
        let l2 = self.b.select(self.f32_ty, None, t_zero, t, l1)?;
        let mut odd = Vec::new();
        for value in [h2, l2] {
            odd.push(self.b.is_nan(self.bool_ty, None, value)?);
            odd.push(self.b.is_inf(self.bool_ty, None, value)?);
        }
        let mut not_finite = odd[0];
        for &flag in &odd[1..] {
            not_finite = self.b.logical_or(self.bool_ty, None, not_finite, flag)?;
        }
        let hi = self.b.select(self.f32_ty, None, not_finite, s, h2)?;
        let lo = self.b.select(self.f32_ty, None, not_finite, lo, l2)?;
        self.b.store(hi_var, hi, None, [])?;
        self.b.store(lo_var, lo, None, [])?;

        let f_one = self.float(1.0);
        let selected = self.b.f_ord_greater_than(self.bool_ty, None, w, f_zero)?;
        let hits = self.b.f_ord_equal(self.bool_ty, None, hit, f_one)?;
        let counted = self.and(selected, hits)?;
        let low = self.b.load(self.f32_ty, None, low_var, None, [])?;
        let high = self.b.load(self.f32_ty, None, high_var, None, [])?;
        let low_up = self.f_add(low, f_one)?;
        let limb = self.float(COUNT_LIMB);
        let carry = self.b.f_ord_equal(self.bool_ty, None, low_up, limb)?;
        let low_next = self.b.select(self.f32_ty, None, carry, f_zero, low_up)?;
        let high_up = self.f_add(high, f_one)?;
        let high_next = self.b.select(self.f32_ty, None, carry, high_up, high)?;
        let low = self.b.select(self.f32_ty, None, counted, low_next, low)?;
        let high = self.b.select(self.f32_ty, None, counted, high_next, high)?;
        self.b.store(low_var, low, None, [])?;
        self.b.store(high_var, high, None, [])?;
        Ok(())
    }
}
