//! The production `matmul` paths fill no output before the kernel writes it (story
//! `cpu-matmul-output-skips-zero-fill`, regression guard of adversary pass 1).
//!
//! A filled output and an unfilled one hold the same bits, so no comparison of results can tell
//! them apart. [`matmul::output_zero_fills`] counts the fills instead: every path that fills the
//! output with zeros before a kernel overwrites it adds one. The zeros of a `k` of 0 are the
//! result itself and are not counted.
//!
//! The counter is global to the process, so this binary holds one test only: no other test can
//! move it between the two reads.

use modelspec_runtime::cpu::Element;
use modelspec_runtime::cpu::matmul::{self, Kernel, Lhs};

fn operands<T: Element>(m: usize, k: usize, n: usize) -> (Vec<T>, Vec<T>) {
    let a = (0..m * k)
        .map(|i| T::from_f32(i as f32 * 0.5 - 3.0))
        .collect();
    let b = (0..k * n)
        .map(|i| T::from_f32(1.25 - i as f32 * 0.125))
        .collect();
    (a, b)
}

/// The fills of `matmul` by `kernel` over a set of shapes with `k > 0`, in `T`.
fn fills<T: Element>(kernel: Kernel) -> usize {
    let before = matmul::output_zero_fills();
    for lhs in [Lhs::RowMajor, Lhs::Transposed] {
        for (m, k, n) in [(1, 1, 1), (3, 4, 5), (7, 3, 17), (13, 9, 33), (100, 64, 10)] {
            let (a, b) = operands::<T>(m, k, n);
            let out = matmul::matmul(kernel, &a, lhs, &b, m, k, n).expect("kernel available");
            assert_eq!(out.len(), m * n);
        }
    }
    matmul::output_zero_fills() - before
}

#[test]
fn the_detected_and_the_portable_kernel_fill_no_output_where_k_is_positive() {
    let detected = Kernel::detected();
    let got = [
        ("f32", detected, fills::<f32>(detected)),
        ("f64", detected, fills::<f64>(detected)),
        ("f32", Kernel::Portable, fills::<f32>(Kernel::Portable)),
        ("f64", Kernel::Portable, fills::<f64>(Kernel::Portable)),
    ];
    let filled: Vec<_> = got.iter().filter(|(_, _, n)| *n != 0).collect();
    assert!(
        filled.is_empty(),
        "output zero-fills before the kernel wrote it: {filled:?}"
    );
}
