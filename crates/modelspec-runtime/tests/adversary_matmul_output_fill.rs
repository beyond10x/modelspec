//! Adversary pass for story `cpu-matmul-output-skips-zero-fill`.
//!
//! The output of the CPU `matmul` is no longer filled before the kernel writes it: the portable
//! kernel appends each row, and the AVX2 kernel for `f32` and `f64` writes into the spare
//! capacity of the output and then sets its length. An element the kernel skips is then whatever
//! the allocator left there, not `+0`.
//!
//! This binary installs a global allocator that fills every fresh allocation (and every grown
//! tail of a reallocation) with `0xFF` bytes, which is a NaN in `f32` and in `f64`. With finite
//! operands that cannot produce a NaN, every element of every result must equal, by bit pattern,
//! the ascending-`k` sum of DESIGN.md § 2.2, each later product fused into it (§ 12 ADR-009), for
//! every `m` and `n` in `1..=33` and `k` in
//! `0..=17` (every row block and column panel edge of the AVX2 kernel), in both left-operand
//! layouts, through each kernel this CPU runs, through `evaluate`, and for an element type the
//! AVX2 kernel has no code for.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::ops::{Add, Div, Mul, Sub};

use modelspec_core::ir::tensor::{TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::matmul::{self, Kernel, Lhs};
use modelspec_runtime::cpu::{Element, HostTensor, evaluate};

/// The system allocator, every byte it hands out set to `0xFF` first.
struct Poison;

// SAFETY: every call forwards to `System` with the same layout; the fill writes only bytes of
// the block just returned.
unsafe impl GlobalAlloc for Poison {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded as received.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            // SAFETY: `p` holds `layout.size()` writable bytes.
            unsafe { p.write_bytes(0xFF, layout.size()) };
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded as received.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded as received.
        let q = unsafe { System.realloc(ptr, layout, new_size) };
        if !q.is_null() && new_size > layout.size() {
            // SAFETY: the bytes after the old size are the new, writable tail.
            unsafe {
                q.add(layout.size())
                    .write_bytes(0xFF, new_size - layout.size())
            };
        }
        q
    }
}

#[global_allocator]
static POISON: Poison = Poison;

trait Real: Element {
    fn of(v: f64) -> Self;
    fn bits(self) -> u64;
    fn subnormal() -> Self;
}

impl Real for f32 {
    fn of(v: f64) -> Self {
        v as f32
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn subnormal() -> Self {
        f32::from_bits(3)
    }
}

impl Real for f64 {
    fn of(v: f64) -> Self {
        v
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn subnormal() -> Self {
        f64::from_bits(5)
    }
}

/// A fixed-seed xorshift.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A finite element: small values with fractional bits, signed zeros and subnormals, so
    /// rounding, `-0` sums and subnormal products all occur, and no NaN or infinity can.
    fn element<T: Real>(&mut self) -> T {
        match self.next() % 16 {
            0 => T::of(-0.0),
            1 => T::of(0.0),
            2 => T::subnormal(),
            _ => {
                let v = (self.next() % 20_001) as f64 / 1_000.0 - 10.0;
                T::of(v / 3.0)
            }
        }
    }
}

/// The ascending-`k` sum, from the `k = 0` product, each later product fused into it; `+0` where
/// `k` is 0.
fn reference<T: Real>(a: &[T], lhs: Lhs, b: &[T], m: usize, k: usize, n: usize) -> Vec<u64> {
    let at = |i: usize, p: usize| match lhs {
        Lhs::RowMajor => a[i * k + p],
        Lhs::Transposed => a[p * m + i],
    };
    let mut out = Vec::new();
    for i in 0..m {
        for j in 0..n {
            let mut acc = if k == 0 { T::ZERO } else { at(i, 0) * b[j] };
            for p in 1..k {
                acc = at(i, p).mul_add(b[p * n + j], acc);
            }
            out.push(acc.bits());
        }
    }
    out
}

fn kernels() -> Vec<Kernel> {
    [Kernel::Portable, Kernel::Avx2]
        .into_iter()
        .filter(|k| k.is_available())
        .collect()
}

/// The detector works: storage this allocator hands out reads as NaN until it is written.
#[test]
fn the_poisoning_allocator_makes_an_unwritten_element_a_nan() {
    let v: Vec<f32> = Vec::with_capacity(64);
    // SAFETY: the allocator wrote every byte of the block, so each element is a valid `f32`; it
    // is read through the raw pointer, not through the `Vec`.
    let seen: Vec<u32> = (0..64)
        .map(|i| unsafe { std::ptr::read_volatile(v.as_ptr().add(i)) }.to_bits())
        .collect();
    assert!(seen.iter().all(|&b| b == u32::MAX), "{seen:x?}");
}

fn every_edge_shape<T: Real>() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut bad = Vec::new();
    for kernel in kernels() {
        for lhs in [Lhs::RowMajor, Lhs::Transposed] {
            for m in 1..=33usize {
                for n in 1..=33usize {
                    for k in 0..=17usize {
                        let a: Vec<T> = (0..m * k).map(|_| rng.element()).collect();
                        let b: Vec<T> = (0..k * n).map(|_| rng.element()).collect();
                        let got = matmul::matmul(kernel, &a, lhs, &b, m, k, n)
                            .unwrap_or_else(|| panic!("{kernel:?} declined {m} x {k} x {n}"));
                        let got: Vec<u64> = got.iter().map(|x| x.bits()).collect();
                        let want = reference(&a, lhs, &b, m, k, n);
                        if got != want {
                            let first = got.iter().zip(&want).position(|(g, w)| g != w);
                            bad.push(format!(
                                "{kernel:?} {lhs:?} {m} x {k} x {n}: len {} want {}, first \
                                 differing element {first:?}",
                                got.len(),
                                want.len()
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} shape(s) differ, first: {:?}",
        bad.len(),
        &bad[..bad.len().min(5)]
    );
}

#[test]
fn every_output_element_is_written_for_every_edge_shape_in_f32() {
    every_edge_shape::<f32>();
}

#[test]
fn every_output_element_is_written_for_every_edge_shape_in_f64() {
    every_edge_shape::<f64>();
}

/// The interpreter's path: the output `Vec` comes from `try_reserve_exact`, so its capacity is
/// exactly `m * n`, and `matmul_into` appends to it.
#[test]
fn every_output_element_is_written_through_evaluate() {
    let mut rng = Rng(0x0123_4567_89AB_CDEF);
    for (m, k, n) in [(1, 1, 1), (7, 3, 17), (13, 17, 9), (33, 5, 31), (6, 16, 24)] {
        let a: Vec<f32> = (0..m * k).map(|_| rng.element()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.element()).collect();
        let (mu, ku, nu) = (m as u64, k as u64, n as u64);
        let mut bld = TensorProgramBuilder::new();
        let x = bld.input("a", TensorType::f32(&[mu, ku])).unwrap();
        let y = bld.input("b", TensorType::f32(&[ku, nu])).unwrap();
        let z = bld.matmul(x, y).unwrap();
        bld.output(z).unwrap();
        let program = bld.finish();
        let inputs = BTreeMap::from([
            (
                "a".to_string(),
                HostTensor::new(vec![mu, ku], a.clone()).unwrap(),
            ),
            (
                "b".to_string(),
                HostTensor::new(vec![ku, nu], b.clone()).unwrap(),
            ),
        ]);
        let out = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
        let got: Vec<u64> = out[0].data().iter().map(|x| x.bits()).collect();
        assert_eq!(
            got,
            reference(&a, Lhs::RowMajor, &b, m, k, n),
            "{m} x {k} x {n}"
        );
    }
}

/// An element type with no AVX2 code: `f64` in a wrapper.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct Wrapped(f64);

impl Add for Wrapped {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Wrapped(self.0 + rhs.0)
    }
}

impl Sub for Wrapped {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Wrapped(self.0 - rhs.0)
    }
}

impl Mul for Wrapped {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Wrapped(self.0 * rhs.0)
    }
}

impl Div for Wrapped {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Wrapped(self.0 / rhs.0)
    }
}

impl Element for Wrapped {
    const ZERO: Self = Wrapped(0.0);
    fn from_f32(value: f32) -> Self {
        Wrapped(f64::from(value))
    }
    fn from_count(count: u64) -> Self {
        Wrapped(count as f64)
    }
    /// The fused multiply-add of `f64`, so the reference of `f64` is this type's too.
    fn mul_add(self, b: Self, c: Self) -> Self {
        Wrapped(self.0.mul_add(b.0, c.0))
    }
    fn exp(self) -> Self {
        Wrapped(self.0.exp())
    }
}

/// The portable kernel writes every element for a type with no AVX2 code; under `Kernel::Avx2`
/// that type is declined where a product is needed, and gets `+0`s or nothing where none is.
#[test]
fn a_type_without_avx2_code_is_written_by_the_portable_kernel_and_zeros_where_k_is_0() {
    let mut rng = Rng(42);
    for m in 1..=13usize {
        for n in 1..=13usize {
            for k in 0..=7usize {
                let a: Vec<f64> = (0..m * k).map(|_| rng.element()).collect();
                let b: Vec<f64> = (0..k * n).map(|_| rng.element()).collect();
                let wa: Vec<Wrapped> = a.iter().map(|&x| Wrapped(x)).collect();
                let wb: Vec<Wrapped> = b.iter().map(|&x| Wrapped(x)).collect();
                let got =
                    matmul::matmul(Kernel::Portable, &wa, Lhs::Transposed, &wb, m, k, n).unwrap();
                let got: Vec<u64> = got.iter().map(|x| x.0.to_bits()).collect();
                assert_eq!(
                    got,
                    reference(&a, Lhs::Transposed, &b, m, k, n),
                    "{m} x {k} x {n}"
                );
                if Kernel::Avx2.is_available() {
                    let avx2 = matmul::matmul(Kernel::Avx2, &wa, Lhs::RowMajor, &wb, m, k, n);
                    if k == 0 {
                        let avx2 = avx2.expect("k = 0 needs no kernel");
                        assert_eq!(avx2.len(), m * n);
                        assert!(avx2.iter().all(|x| x.0.to_bits() == 0), "{avx2:?}");
                    } else {
                        assert!(avx2.is_none(), "{m} x {k} x {n}");
                    }
                }
            }
        }
    }
    for (m, k, n) in [(0, 3, 4), (4, 3, 0), (0, 0, 0)] {
        for kernel in kernels() {
            let wa = vec![Wrapped(1.5); m * k];
            let wb = vec![Wrapped(-2.0); k * n];
            let got = matmul::matmul(kernel, &wa, Lhs::RowMajor, &wb, m, k, n);
            assert_eq!(got, Some(Vec::new()), "{kernel:?} {m} x {k} x {n}");
        }
    }
}

/// A zero dimension: `m * n` elements, each `+0` (not the allocator's bytes) where `k` is 0.
#[test]
fn a_zero_dimension_gives_positive_zeros_from_fresh_storage() {
    for kernel in kernels() {
        for lhs in [Lhs::RowMajor, Lhs::Transposed] {
            for (m, k, n) in [
                (0, 0, 0),
                (0, 4, 9),
                (9, 4, 0),
                (33, 0, 33),
                (1, 0, 1),
                (6, 0, 17),
            ] {
                let a = vec![-1.5f64; m * k];
                let b = vec![-2.0f64; k * n];
                let got = matmul::matmul(kernel, &a, lhs, &b, m, k, n).unwrap();
                assert_eq!(got.len(), m * n, "{kernel:?} {lhs:?} {m} {k} {n}");
                assert!(
                    got.iter().all(|x| x.to_bits() == 0),
                    "{kernel:?} {lhs:?} {m} {k} {n}: {got:?}"
                );
            }
        }
    }
}

/// A panic inside the kernel (a short operand) unwinds without leaving a result anybody can read,
/// and the next call on the same thread is correct.
#[test]
fn a_panicking_call_leaves_nothing_behind_and_the_next_call_is_whole() {
    for kernel in kernels() {
        let caught = std::panic::catch_unwind(|| {
            matmul::matmul(kernel, &[1.0f32; 5], Lhs::RowMajor, &[1.0f32; 6], 2, 3, 2)
        });
        assert!(caught.is_err(), "{kernel:?}");
        let got =
            matmul::matmul(kernel, &[1.0f32; 6], Lhs::RowMajor, &[1.0f32; 6], 2, 3, 2).unwrap();
        assert_eq!(got, vec![3.0f32; 4], "{kernel:?}");
    }
}
