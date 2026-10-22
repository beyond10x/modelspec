//! Adversary cases for story `cpu-matmul-avx2-blocking`.
//!
//! 1. Soundness of the public kernel entry points: `matmul::matmul` and `Element::matmul_avx2` are
//!    safe functions whose length checks multiply `m * k`, `k * n` and `m * n` without overflow
//!    checks. In a release build (overflow checks off) dimensions whose products wrap to zero pass
//!    those checks with empty slices, and the AVX2 kernel then loads and stores through pointers
//!    to zero-length buffers. The documented contract is a panic ("When `a` or `b` does not hold
//!    exactly its operand's element count"); these cases assert that panic.
//! 2. Bit equality on directed shapes the seeded suite does not draw: products that overflow to
//!    infinity mid-sum over a long `k`, subnormal products, all-`-0.0` sums in every block lane
//!    position, `m`, `k` or `n` of 1, each kernel and each left-operand layout.

use modelspec_runtime::cpu::Element;
use modelspec_runtime::cpu::matmul::{self, Kernel, Lhs};

/// `2^32`: `m = k = n = 2^32` makes every product of two of them `2^64`, which wraps to zero.
#[cfg(target_pointer_width = "64")]
const WRAP: usize = 1 << 32;

#[cfg(target_pointer_width = "64")]
#[test]
#[should_panic]
fn the_public_avx2_matmul_panics_when_the_element_counts_overflow() {
    if !Kernel::Avx2.is_available() {
        panic!("no AVX2 on this host; the case is about the AVX2 kernel");
    }
    let _ = matmul::matmul::<f32>(Kernel::Avx2, &[], Lhs::RowMajor, &[], WRAP, WRAP, WRAP);
}

#[cfg(target_pointer_width = "64")]
#[test]
#[should_panic]
fn the_public_element_avx2_hook_panics_when_the_element_counts_overflow() {
    if !Kernel::Avx2.is_available() {
        panic!("no AVX2 on this host; the case is about the AVX2 kernel");
    }
    let mut out: [f64; 0] = [];
    let _ = f64::matmul_avx2(&mut out, &[], Lhs::Transposed, &[], WRAP, WRAP, WRAP);
}

/// The 0.1.1 loop: `i`, `j`, then the `K` products in ascending `k` from the first term.
fn reference<T: Element>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut terms = (0..k).map(|p| a[i * k + p] * b[p * n + j]);
            let acc = match terms.next() {
                Some(first) => terms.fold(first, |acc, x| acc + x),
                None => T::ZERO,
            };
            out.push(acc);
        }
    }
    out
}

fn transposed<T: Copy>(a: &[T], m: usize, k: usize) -> Vec<T> {
    (0..k)
        .flat_map(|p| (0..m).map(move |i| a[i * k + p]))
        .collect()
}

fn kernels() -> Vec<Kernel> {
    [Kernel::Portable, Kernel::Avx2]
        .into_iter()
        .filter(|k| k.is_available())
        .collect()
}

trait Bits: Element {
    fn bits(self) -> u64;
    fn is_nan_(self) -> bool;
    fn of(v: f64) -> Self;
}

impl Bits for f32 {
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn is_nan_(self) -> bool {
        self.is_nan()
    }
    fn of(v: f64) -> Self {
        v as f32
    }
}

impl Bits for f64 {
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn is_nan_(self) -> bool {
        self.is_nan()
    }
    fn of(v: f64) -> Self {
        v
    }
}

/// Every kernel and layout against the reference; returns the differing cases.
fn check<T: Bits>(label: &str, a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<String> {
    let want = reference(a, b, m, k, n);
    let mut bad = Vec::new();
    for kernel in kernels() {
        for lhs in [Lhs::RowMajor, Lhs::Transposed] {
            let a = match lhs {
                Lhs::RowMajor => a.to_vec(),
                Lhs::Transposed => transposed(a, m, k),
            };
            let got = matmul::matmul(kernel, &a, lhs, b, m, k, n).expect("available");
            if let Some(i) = (0..want.len()).find(|&i| {
                !((want[i].is_nan_() && got[i].is_nan_()) || want[i].bits() == got[i].bits())
            }) {
                bad.push(format!(
                    "{label} {kernel:?} {lhs:?} m={m} k={k} n={n} element {i}: want {:#x} got {:#x}",
                    want[i].bits(),
                    got[i].bits()
                ));
            }
        }
    }
    bad
}

/// A deterministic element source.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn sign(&mut self) -> f64 {
        if self.next() & 1 == 0 { 1.0 } else { -1.0 }
    }
}

const MS: [usize; 6] = [1, 5, 6, 7, 12, 13];
const NS: [usize; 10] = [1, 3, 4, 5, 7, 8, 9, 16, 17, 33];

fn directed<T: Bits>() -> Vec<String> {
    let mut bad = Vec::new();
    let mut rng = Rng(0xadd5_a1d0);
    let max = if std::mem::size_of::<T>() == 4 {
        f32::MAX as f64
    } else {
        f64::MAX
    };
    let tiny = if std::mem::size_of::<T>() == 4 {
        f32::MIN_POSITIVE as f64
    } else {
        f64::MIN_POSITIVE
    };
    for &m in &MS {
        for &n in &NS {
            // Overflow to infinity mid-sum over a long k, with sign changes so inf - inf appears.
            let k = 1500;
            let a: Vec<T> = (0..m * k)
                .map(|_| T::of(rng.sign() * max.sqrt() * (0.5 + rng.unit())))
                .collect();
            let b: Vec<T> = (0..k * n)
                .map(|_| T::of(rng.sign() * max.sqrt() * (0.5 + rng.unit()) / 64.0))
                .collect();
            bad.extend(check("overflow", &a, &b, m, k, n));
            // Subnormal products and sums.
            let k = 37;
            let a: Vec<T> = (0..m * k)
                .map(|_| T::of(rng.sign() * tiny.sqrt() * (0.5 + rng.unit())))
                .collect();
            let b: Vec<T> = (0..k * n)
                .map(|_| T::of(rng.sign() * tiny.sqrt() * (0.5 + rng.unit()) / 1024.0))
                .collect();
            bad.extend(check("subnormal", &a, &b, m, k, n));
            // Subnormal operands.
            let a: Vec<T> = (0..m * k)
                .map(|_| T::of(rng.sign() * tiny * rng.unit()))
                .collect();
            let b: Vec<T> = (0..k * n).map(|_| T::of(rng.sign() * rng.unit())).collect();
            bad.extend(check("subnormal-operand", &a, &b, m, k, n));
            // All-`-0.0` sums: every product -0 in every lane, for k = 1, 2, 3.
            for k in 1..=3 {
                let a: Vec<T> = (0..m * k).map(|_| T::of(-0.0)).collect();
                let b: Vec<T> = (0..k * n).map(|_| T::of(rng.unit() + 0.5)).collect();
                bad.extend(check("negzero", &a, &b, m, k, n));
                let a: Vec<T> = (0..m * k)
                    .map(|_| T::of(if rng.next() & 1 == 0 { -0.0 } else { 0.0 }))
                    .collect();
                let b: Vec<T> = (0..k * n)
                    .map(|_| T::of(rng.sign() * (rng.unit() + 0.5)))
                    .collect();
                bad.extend(check("mixed-zero", &a, &b, m, k, n));
            }
        }
    }
    bad
}

#[test]
fn each_kernel_equals_the_0_1_1_loop_on_directed_edge_values_in_f32() {
    let bad = directed::<f32>();
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn each_kernel_equals_the_0_1_1_loop_on_directed_edge_values_in_f64() {
    let bad = directed::<f64>();
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.join("\n"));
}
