//! The CPU `matmul` kernels: `[M, K] x [K, N] -> [M, N]`, each output element the sum of its `K`
//! products in ascending `k`, starting from the `k = 0` product, every later product fused into
//! the sum with [`Element::mul_add`], rounded once to the element type (DESIGN.md § 2.2, § 12
//! ADR-009): the element the GPU's `Fma` chain computes.
//!
//! Two kernels compute the same bits. [`Kernel::Portable`] is the loop in order `i`, `k`, `j` and
//! runs everywhere. [`Kernel::Avx2`] runs on x86-64 CPUs with AVX2 and FMA, chosen at run time: it
//! holds a block of output rows by 256-bit vectors of output columns in registers across the whole
//! `k` loop, one output element per vector lane, with one fused multiply-add per later `k`. A lane
//! rounds as a scalar `mul_add` does, so each element sees the same sequence of roundings as in
//! the portable loop.
//!
//! The left operand is read row-major, or [`Lhs::Transposed`]: as the `[K, M]` value a
//! `transpose` would have turned into it, read in place.
//!
//! Both kernels write each output element once, into storage that is not filled first; only a
//! `K` of 0 writes zeros, which are then the result.

#[cfg(target_arch = "x86_64")]
use std::any::TypeId;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::tensor::Element;

/// See [`output_zero_fills`].
static OUTPUT_ZERO_FILLS: AtomicUsize = AtomicUsize::new(0);

/// The number of `matmul` calls in this process that zero-filled the output before handing it to
/// a `T::matmul_avx2` hook under `Kernel::Avx2`, whether or not the hook then wrote over the
/// zeros. The zeros of a `k` of 0 are the result, not such a fill, and are not counted.
///
/// A test-only instrument, not API: the tests `matmul_output_zero_fills` and
/// `adversary_matmul_output_zero_fills` read it; the kernels never touch it.
#[doc(hidden)]
pub fn output_zero_fills() -> usize {
    OUTPUT_ZERO_FILLS.load(Ordering::Relaxed)
}

/// A `matmul` kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// The loop in order `i`, `k`, `j`; every CPU and target.
    Portable,
    /// Register blocks of 256-bit vectors; x86-64 with AVX2 and FMA, for `f32` and `f64`.
    Avx2,
}

impl Kernel {
    /// Whether this CPU can run the kernel.
    pub fn is_available(self) -> bool {
        match self {
            Kernel::Portable => true,
            Kernel::Avx2 => avx2_detected(),
        }
    }

    /// The kernel the interpreter runs on this CPU: AVX2 where AVX2 and FMA are available, else
    /// the portable one.
    pub fn detected() -> Kernel {
        if avx2_detected() {
            Kernel::Avx2
        } else {
            Kernel::Portable
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_detected() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_detected() -> bool {
    false
}

/// How the left operand's elements are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lhs {
    /// `[M, K]` row-major: element `(i, p)` at `i * K + p`.
    RowMajor,
    /// `[K, M]` row-major, read as its transpose: element `(i, p)` at `p * M + i`.
    Transposed,
}

impl Lhs {
    /// The distance between rows and between columns of the `[M, K]` operand.
    fn strides(self, m: usize, k: usize) -> (usize, usize) {
        match self {
            Lhs::RowMajor => (k, 1),
            Lhs::Transposed => (1, m),
        }
    }
}

/// `a x b` by `kernel`, `a` holding `m * k` elements laid out as `lhs`, `b` the `[k, n]` operand
/// row-major; `None` when this CPU cannot run `kernel` or it has no code for `T`.
///
/// # Panics
///
/// When `m * k`, `k * n` or `m * n` overflows `usize`, or `a` or `b` does not hold exactly its
/// operand's element count.
pub fn matmul<T: Element>(
    kernel: Kernel,
    a: &[T],
    lhs: Lhs,
    b: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<T>> {
    let mut out = Vec::new();
    matmul_into(kernel, &mut out, a, lhs, b, m, k, n).then_some(out)
}

/// [`matmul`], appending the `m * n` elements to `out`; false, with `out` unchanged, where
/// [`matmul`] returns `None`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_into<T: Element>(
    kernel: Kernel,
    out: &mut Vec<T>,
    a: &[T],
    lhs: Lhs,
    b: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let (mk, kn, mn) = element_counts(m, k, n);
    assert_eq!(a.len(), mk, "left operand of {m} x {k}");
    assert_eq!(b.len(), kn, "right operand of {k} x {n}");
    if !kernel.is_available() {
        return false;
    }
    let start = out.len();
    let end = start
        .checked_add(mn)
        .unwrap_or_else(|| panic!("{start} + {mn} elements overflow `usize`"));
    if k == 0 {
        // Each element is the sum of no products.
        out.resize(end, T::ZERO);
        return true;
    }
    if mn == 0 {
        return true;
    }
    match kernel {
        Kernel::Portable => {
            portable(out, a, lhs, b, m, k, n);
            true
        }
        Kernel::Avx2 => {
            if let Some(done) = avx2_uninit(out, a, lhs, b, m, k, n) {
                return done;
            }
            // A type this module has no vector code for runs its own hook, over zeros.
            OUTPUT_ZERO_FILLS.fetch_add(1, Ordering::Relaxed);
            out.resize(end, T::ZERO);
            let done = T::matmul_avx2(&mut out[start..], a, lhs, b, m, k, n);
            if !done {
                out.truncate(start);
            }
            done
        }
    }
}

/// [`Kernel::Avx2`] for `f32` and `f64`, appending the `m * n` elements to `out` without filling
/// them first: `Some(true)` once written, `Some(false)` with `out` unchanged where the CPU has no
/// AVX2 or no FMA, `None` for any other type. `m`, `k` and `n` are nonzero and the slices hold
/// their counts.
#[allow(clippy::too_many_arguments)]
fn avx2_uninit<T: Element>(
    out: &mut Vec<T>,
    a: &[T],
    lhs: Lhs,
    b: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Option<bool> {
    let start = out.len();
    let mn = m * n;
    out.reserve(mn);
    let dst = &mut out.spare_capacity_mut()[..mn];
    let done = write_avx2(dst, a, lhs, b, m, k, n)?;
    if done {
        // SAFETY: the capacity was reserved above, and `write_avx2` returned true, so it wrote
        // every one of the `m * n` elements after `start`.
        unsafe { out.set_len(start + mn) };
    }
    Some(done)
}

/// The AVX2 kernel of `T` into `dst`, when `T` is `f32` or `f64`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn write_avx2<T: Element>(
    dst: &mut [MaybeUninit<T>],
    a: &[T],
    lhs: Lhs,
    b: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Option<bool> {
    if let (Some(dst), Some(a), Some(b)) = (same_mut::<T, f32>(dst), same(a), same(b)) {
        return Some(avx2::f32x8::matmul_uninit(dst, a, lhs, b, m, k, n));
    }
    if let (Some(dst), Some(a), Some(b)) = (same_mut::<T, f64>(dst), same(a), same(b)) {
        return Some(avx2::f64x4::matmul_uninit(dst, a, lhs, b, m, k, n));
    }
    None
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn write_avx2<T: Element>(
    _dst: &mut [MaybeUninit<T>],
    _a: &[T],
    _lhs: Lhs,
    _b: &[T],
    _m: usize,
    _k: usize,
    _n: usize,
) -> Option<bool> {
    None
}

/// `s` as a slice of `U`, where `T` is `U`.
#[cfg(target_arch = "x86_64")]
fn same<T: 'static, U: 'static>(s: &[T]) -> Option<&[U]> {
    // SAFETY: `T` and `U` are one type, so the slice is unchanged.
    (TypeId::of::<T>() == TypeId::of::<U>())
        .then(|| unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<U>(), s.len()) })
}

/// [`same`] for writable, possibly uninitialised storage.
#[cfg(target_arch = "x86_64")]
fn same_mut<T: 'static, U: 'static>(s: &mut [MaybeUninit<T>]) -> Option<&mut [MaybeUninit<U>]> {
    // SAFETY: `T` and `U` are one type, so the slice is unchanged.
    (TypeId::of::<T>() == TypeId::of::<U>()).then(|| unsafe {
        std::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<MaybeUninit<U>>(), s.len())
    })
}

/// `m * k`, `k * n` and `m * n`, the element counts of the left operand, the right operand and
/// the result.
///
/// # Panics
///
/// When one of them overflows `usize`: no slice can hold it, and a wrapped count would let an
/// empty slice pass a length check.
pub(crate) fn element_counts(m: usize, k: usize, n: usize) -> (usize, usize, usize) {
    let count = |x: usize, y: usize| {
        x.checked_mul(y)
            .unwrap_or_else(|| panic!("{x} x {y} elements overflow `usize`"))
    };
    (count(m, k), count(k, n), count(m, n))
}

/// Loop order `i`, `k`, `j`, so the inner loop reads a row of `b`: each output row is appended to
/// `out` as its `k = 0` products (not `+0`, so an all-`-0` sum stays `-0`) and every later `k`
/// fuses its products into the sums in turn ([`Element::mul_add`]), the same sequence of
/// roundings per element as summing over `k` directly. `k` and `n` are nonzero.
fn portable<T: Element>(
    out: &mut Vec<T>,
    a: &[T],
    lhs: Lhs,
    b: &[T],
    m: usize,
    k: usize,
    n: usize,
) {
    let (rs, cs) = lhs.strides(m, k);
    let (b_first, b_rest) = b.split_at(n);
    out.reserve(m * n);
    for i in 0..m {
        let x = a[i * rs];
        let start = out.len();
        out.extend(b_first.iter().map(|&y| x * y));
        let row = &mut out[start..];
        for (p, b_row) in (1..k).zip(b_rest.chunks_exact(n)) {
            let x = a[i * rs + p * cs];
            for (acc, &y) in row.iter_mut().zip(b_row) {
                *acc = T::mul_add(x, y, *acc);
            }
        }
    }
}

/// The AVX2 kernel, one module per element type.
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use std::arch::x86_64::{
        __m256i, _mm256_cmpgt_epi32, _mm256_cmpgt_epi64, _mm256_set1_epi32, _mm256_set1_epi64x,
        _mm256_setr_epi32, _mm256_setr_epi64x,
    };

    /// Output rows per register block. With two vectors of columns that is 12 accumulators, two
    /// loaded row segments of `b` and one broadcast element of `a`: 15 of the 16 vector registers.
    const ROWS: usize = 6;

    /// The lanes below `lanes` of eight 32-bit lanes set.
    #[target_feature(enable = "avx2")]
    fn mask_32(lanes: usize) -> __m256i {
        _mm256_cmpgt_epi32(
            _mm256_set1_epi32(lanes as i32),
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
        )
    }

    /// The lanes below `lanes` of four 64-bit lanes set.
    #[target_feature(enable = "avx2")]
    fn mask_64(lanes: usize) -> __m256i {
        _mm256_cmpgt_epi64(
            _mm256_set1_epi64x(lanes as i64),
            _mm256_setr_epi64x(0, 1, 2, 3),
        )
    }

    macro_rules! kernel {
        (
            $module:ident, $e:ty, $v:ty, $lanes:literal, $mask:ident,
            $setzero:ident, $set1:ident, $loadu:ident, $maskload:ident,
            $storeu:ident, $maskstore:ident, $mul:ident, $fmadd:ident $(,)?
        ) => {
            pub(crate) mod $module {
                use std::arch::x86_64::{
                    $fmadd, $loadu, $maskload, $maskstore, $mul, $set1, $setzero, $storeu,
                    __m256i,
                };
                use std::mem::MaybeUninit;

                use super::super::Lhs;
                use super::{$mask, ROWS};

                const LANES: usize = $lanes;

                /// Where a register block reads its operands: `a` at element `(i0, 0)` of the left
                /// operand, whose element `(i0 + r, p)` is at `a + r * rs + p * cs`; `b` at element
                /// `(0, j0)` of the right operand, row stride `n`; `k` rows of `b`, at least one.
                #[derive(Clone, Copy)]
                struct Panel {
                    a: *const $e,
                    rs: usize,
                    cs: usize,
                    b: *const $e,
                    n: usize,
                    k: usize,
                }

                /// `out = a x b` for the `[m, k]` left operand laid out as `lhs`; `m`, `k` and `n`
                /// are nonzero. False, writing nothing, where the CPU has no AVX2 or no FMA.
                pub(crate) fn matmul(
                    out: &mut [$e],
                    a: &[$e],
                    lhs: Lhs,
                    b: &[$e],
                    m: usize,
                    k: usize,
                    n: usize,
                ) -> bool {
                    let out = out as *mut [$e] as *mut [MaybeUninit<$e>];
                    // SAFETY: `MaybeUninit<$e>` has the layout of `$e`, and `matmul_uninit`
                    // stores only `$e` values, so every element of `out` stays initialised.
                    matmul_uninit(unsafe { &mut *out }, a, lhs, b, m, k, n)
                }

                /// [`matmul`] into storage that need not be initialised: true once it has
                /// written every one of the `m * n` elements of `out`, false, writing nothing,
                /// where the CPU has no AVX2 or no FMA.
                pub(crate) fn matmul_uninit(
                    out: &mut [MaybeUninit<$e>],
                    a: &[$e],
                    lhs: Lhs,
                    b: &[$e],
                    m: usize,
                    k: usize,
                    n: usize,
                ) -> bool {
                    assert!(m > 0 && k > 0 && n > 0);
                    let (mk, kn, mn) = super::super::element_counts(m, k, n);
                    assert!(out.len() == mn && a.len() == mk && b.len() == kn);
                    if !(std::arch::is_x86_feature_detected!("avx2")
                        && std::arch::is_x86_feature_detected!("fma"))
                    {
                        return false;
                    }
                    let (rs, cs) = lhs.strides(m, k);
                    // SAFETY: AVX2 and FMA are detected above, and the lengths asserted above are
                    // the ones `run` requires.
                    unsafe { run(out, a, rs, cs, b, m, k, n) };
                    true
                }

                /// Row blocks of `ROWS` rows and a last one of the rest, together storing every
                /// element of `out` once.
                ///
                /// # Safety
                ///
                /// AVX2 and FMA are available; `out` holds `m * n` elements, `b` `k * n`, and `a`
                /// an element at `i * rs + p * cs` for every `i < m`, `p < k`; `m`, `k`, `n`
                /// nonzero.
                #[target_feature(enable = "avx2,fma")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn run(
                    out: &mut [MaybeUninit<$e>],
                    a: &[$e],
                    rs: usize,
                    cs: usize,
                    b: &[$e],
                    m: usize,
                    k: usize,
                    n: usize,
                ) {
                    let mut i0 = 0;
                    while i0 < m {
                        let rows = (m - i0).min(ROWS);
                        let panel = Panel {
                            // SAFETY: `i0 < m`, so `i0 * rs` is the offset of element `(i0, 0)`
                            // of `a`, in bounds.
                            a: unsafe { a.as_ptr().add(i0 * rs) },
                            rs,
                            cs,
                            b: b.as_ptr(),
                            n,
                            k,
                        };
                        // SAFETY: element `(i0, 0)` of `out`, which has `rows` rows of `n`
                        // elements from there on.
                        let dst = unsafe { out.as_mut_ptr().cast::<$e>().add(i0 * n) };
                        // SAFETY: `panel` and `dst` cover `rows` rows from `i0` and all `n`
                        // columns, as `columns` requires; AVX2 and FMA are available by this
                        // function's contract.
                        unsafe {
                            match rows {
                                6 => columns::<6>(dst, panel),
                                5 => columns::<5>(dst, panel),
                                4 => columns::<4>(dst, panel),
                                3 => columns::<3>(dst, panel),
                                2 => columns::<2>(dst, panel),
                                _ => columns::<1>(dst, panel),
                            }
                        }
                        i0 += rows;
                    }
                }

                /// Column panels of two vectors, then one vector, then the last partial vector
                /// under a lane mask.
                ///
                /// # Safety
                ///
                /// AVX2 and FMA are available; `panel` reads `R` rows of `a` and all `n` columns
                /// of `b`, and `out` has `R` rows of `n` writable elements, row stride `n`.
                #[target_feature(enable = "avx2,fma")]
                #[inline]
                unsafe fn columns<const R: usize>(out: *mut $e, panel: Panel) {
                    let n = panel.n;
                    let all = $mask(LANES);
                    let mut j0 = 0;
                    while j0 + 2 * LANES <= n {
                        // SAFETY: columns `j0 .. j0 + 2 * LANES` are within `n`.
                        unsafe { block::<R, 2, false>(out.add(j0), panel.at(j0), all) };
                        j0 += 2 * LANES;
                    }
                    if j0 + LANES <= n {
                        // SAFETY: columns `j0 .. j0 + LANES` are within `n`.
                        unsafe { block::<R, 1, false>(out.add(j0), panel.at(j0), all) };
                        j0 += LANES;
                    }
                    if j0 < n {
                        // SAFETY: the mask enables only the `n - j0` columns left.
                        unsafe { block::<R, 1, true>(out.add(j0), panel.at(j0), $mask(n - j0)) };
                    }
                }

                impl Panel {
                    /// The same rows, `b` from column `j0`.
                    ///
                    /// # Safety
                    ///
                    /// `j0 < n`.
                    #[inline]
                    unsafe fn at(self, j0: usize) -> Panel {
                        Panel {
                            // SAFETY: `j0 < n`, a column of `b`'s first row.
                            b: unsafe { self.b.add(j0) },
                            ..self
                        }
                    }
                }

                /// `V` vectors of row `p` of `b`, the last one under `mask` when `MASKED`.
                ///
                /// # Safety
                ///
                /// AVX2 and FMA are available; `row` points at `V * LANES` readable elements, or,
                /// when `MASKED`, `(V - 1) * LANES` and the lanes of `mask` after them.
                #[target_feature(enable = "avx2,fma")]
                #[inline]
                unsafe fn load<const V: usize, const MASKED: bool>(
                    row: *const $e,
                    mask: __m256i,
                ) -> [$v; V] {
                    let mut lanes = [$setzero(); V];
                    for (v, lane) in lanes.iter_mut().enumerate() {
                        // SAFETY: vector `v` of `row` is readable, the last one only in the lanes
                        // of `mask` when `MASKED`, by this function's contract.
                        *lane = unsafe {
                            if MASKED && v == V - 1 {
                                $maskload(row.add(v * LANES), mask)
                            } else {
                                $loadu(row.add(v * LANES))
                            }
                        };
                    }
                    lanes
                }

                /// One register block: `R` output rows by `V` vectors of output columns, held in
                /// registers across all `k`. The `k = 0` products start every lane, and each later
                /// `k` fuses its product into the lane with one rounding, as the portable loop's
                /// `mul_add` does.
                ///
                /// # Safety
                ///
                /// AVX2 and FMA are available; `panel` reads `R` rows of `a` and `V * LANES`
                /// columns of `b` (when `MASKED`, of the last vector only the lanes of `mask`),
                /// and `out` has the same rows and columns writable, row stride `panel.n`.
                #[target_feature(enable = "avx2,fma")]
                #[inline]
                unsafe fn block<const R: usize, const V: usize, const MASKED: bool>(
                    out: *mut $e,
                    panel: Panel,
                    mask: __m256i,
                ) {
                    let Panel { a, rs, cs, b, n, k } = panel;
                    // SAFETY: row 0 of `b`, the block's columns.
                    let first = unsafe { load::<V, MASKED>(b, mask) };
                    let mut acc = [[$setzero(); V]; R];
                    for (r, acc_row) in acc.iter_mut().enumerate() {
                        // SAFETY: element `(r, 0)` of the block's rows of `a`.
                        let x = $set1(unsafe { *a.add(r * rs) });
                        for (acc, &y) in acc_row.iter_mut().zip(&first) {
                            *acc = $mul(x, y);
                        }
                    }
                    for p in 1..k {
                        // SAFETY: row `p < k` of `b`, the block's columns.
                        let row = unsafe { load::<V, MASKED>(b.add(p * n), mask) };
                        for (r, acc_row) in acc.iter_mut().enumerate() {
                            // SAFETY: element `(r, p)` of the block's rows of `a`.
                            let x = $set1(unsafe { *a.add(r * rs + p * cs) });
                            for (acc, &y) in acc_row.iter_mut().zip(&row) {
                                *acc = $fmadd(x, y, *acc);
                            }
                        }
                    }
                    for (r, acc_row) in acc.iter().enumerate() {
                        for (v, &lane) in acc_row.iter().enumerate() {
                            // SAFETY: vector `v` of output row `r` is writable, the last one
                            // only in the lanes of `mask` when `MASKED`.
                            unsafe {
                                let dst = out.add(r * n + v * LANES);
                                if MASKED && v == V - 1 {
                                    $maskstore(dst, mask, lane);
                                } else {
                                    $storeu(dst, lane);
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    kernel!(
        f32x8,
        f32,
        std::arch::x86_64::__m256,
        8,
        mask_32,
        _mm256_setzero_ps,
        _mm256_set1_ps,
        _mm256_loadu_ps,
        _mm256_maskload_ps,
        _mm256_storeu_ps,
        _mm256_maskstore_ps,
        _mm256_mul_ps,
        _mm256_fmadd_ps,
    );

    kernel!(
        f64x4,
        f64,
        std::arch::x86_64::__m256d,
        4,
        mask_64,
        _mm256_setzero_pd,
        _mm256_set1_pd,
        _mm256_loadu_pd,
        _mm256_maskload_pd,
        _mm256_storeu_pd,
        _mm256_maskstore_pd,
        _mm256_mul_pd,
        _mm256_fmadd_pd,
    );
}
