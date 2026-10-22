//! Adversary pass 2 for story `cpu-matmul-avx2-blocking`: the overflow-checked element counts.
//!
//! Every public path into the AVX2 kernel must panic, not read or write out of bounds, when the
//! dimensions overflow, when a slice is longer or shorter than its count, and must still return
//! `m * n` zeros (`+0`) when a dimension is zero.

#[cfg(target_arch = "x86_64")]
use modelspec_runtime::cpu::Element;
use modelspec_runtime::cpu::matmul::{self, Kernel, Lhs};

fn kernels() -> Vec<Kernel> {
    [Kernel::Portable, Kernel::Avx2]
        .into_iter()
        .filter(|k| k.is_available())
        .collect()
}

/// Whether `f` panics, for every available kernel.
fn panics_everywhere(f: impl Fn(Kernel) + std::panic::RefUnwindSafe) -> Vec<Kernel> {
    kernels()
        .into_iter()
        .filter(|&kernel| std::panic::catch_unwind(|| f(kernel)).is_ok())
        .collect()
}

#[cfg(target_pointer_width = "64")]
#[test]
fn only_the_result_count_overflows_and_every_kernel_panics() {
    let big = 1usize << 33;
    let silent = panics_everywhere(|kernel| {
        let _ = matmul::matmul::<f32>(kernel, &[], Lhs::RowMajor, &[], big, 0, big);
    });
    assert!(silent.is_empty(), "no panic for {silent:?}");
}

#[cfg(target_pointer_width = "64")]
#[test]
fn a_right_operand_count_that_overflows_with_m_zero_panics() {
    let big = 1usize << 40;
    let silent = panics_everywhere(|kernel| {
        let _ = matmul::matmul::<f64>(kernel, &[], Lhs::Transposed, &[], 0, big, big);
    });
    assert!(silent.is_empty(), "no panic for {silent:?}");
}

#[test]
fn slices_longer_or_shorter_than_their_counts_panic() {
    for (a_len, b_len) in [(7, 6), (6, 7), (5, 6), (6, 5), (0, 0)] {
        let silent = panics_everywhere(|kernel| {
            let a = vec![1.0f32; a_len];
            let b = vec![1.0f32; b_len];
            let _ = matmul::matmul(kernel, &a, Lhs::RowMajor, &b, 2, 3, 2);
        });
        assert!(
            silent.is_empty(),
            "a {a_len} b {b_len}: no panic for {silent:?}"
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn the_element_hook_checks_out_as_well() {
    if !Kernel::Avx2.is_available() {
        return;
    }
    for out_len in [3usize, 5] {
        let caught = std::panic::catch_unwind(|| {
            let mut out = vec![0.0f32; out_len];
            f32::matmul_avx2(&mut out, &[1.0; 2], Lhs::RowMajor, &[1.0; 2], 2, 1, 2)
        });
        assert!(
            caught.is_err(),
            "out of {out_len} for a 2 x 2 result accepted"
        );
    }
}

#[test]
fn a_zero_dimension_still_gives_m_times_n_positive_zeros() {
    for kernel in kernels() {
        for lhs in [Lhs::RowMajor, Lhs::Transposed] {
            for (m, k, n) in [
                (0, 5, 3),
                (4, 0, 1000),
                (13, 0, 1),
                (1, 0, 17),
                (3, 9, 0),
                (0, 0, 9),
            ] {
                let a = vec![-0.0f64; m * k];
                let b = vec![-1.0f64; k * n];
                let got = matmul::matmul(kernel, &a, lhs, &b, m, k, n).unwrap();
                assert_eq!(got.len(), m * n, "{kernel:?} {lhs:?} {m} {k} {n}");
                assert!(
                    got.iter().all(|x| x.to_bits() == 0),
                    "{kernel:?} {lhs:?} {m} {k} {n}"
                );
            }
        }
    }
}
