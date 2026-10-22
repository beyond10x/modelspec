//! Adversary pass 2 on `story:m1-f-forward-matmul-tiled`, no Vulkan instance.
//!
//! `spirv_val.rs` `a_matmul_module_is_no_larger_at_any_k_than_at_a_k_of_sixteen_tiles` bounds a
//! large-`k` module by the largest module at `k` up to 16 tiles plus its `k`-dependent constants
//! (`common::matmul_k_constants`). A small `k` reads its straight tiles after the loop at offsets
//! `11..=14` tiles deep, which can coincide with other constants of the module (the staging slot
//! offsets, `m`, the pre-loop and loop-body offsets of `b`); at a large `k` they do not, and the
//! trip count of the loop is a new constant too. Pass 2 found a fixed slack of 4 constants too
//! small at `[176|208, k] x [k, 2]` (5 constants, `review-result:m1-f-forward-adversary-2`). Same
//! measure as that test, every remainder of `k` modulo 4 tiles, at more shapes.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use modelspec_core::ir::kernel::MatMulTile;
use modelspec_runtime::spirv::emit::emit;

#[test]
fn a_large_k_matmul_module_stays_within_its_k_dependent_constants_of_the_sixteen_tile_bound_at_more_shapes()
 {
    let mut over = Vec::new();
    for (m, n) in [
        (100, 64),
        (5, 3),
        (704, 16),
        (704, 8),
        (352, 416),
        (208, 2),
        (176, 2),
    ] {
        let depth = u64::from(MatMulTile::of(m, 1 << 20, n).depth);
        let words = |k: u64| -> (usize, u32) {
            let module = emit(&common::matmul_kernel(m, k, n)).expect("the matmul emits");
            (module.len(), module[3])
        };
        let mut small = (0, 0);
        for at in 1..=16 * depth {
            let (len, bound) = words(at);
            small = (small.0.max(len), small.1.max(bound));
        }
        let limit = u64::from(u32::MAX) / n.max(m);
        let mut worst = (0usize, 0u32, 0u64, 0usize);
        for base in [10_000, 100_000, 1_000_000] {
            for at in base..base + 4 * depth {
                if at > limit {
                    continue;
                }
                let (len, bound) = words(at);
                let spare = common::matmul_k_constants(at, &MatMulTile::of(m, at, n));
                if bound > worst.1 {
                    worst = (len, bound, at, spare);
                }
                if len > small.0 + common::CONSTANT_WORDS * spare || bound > small.1 + spare as u32
                {
                    over.push(format!(
                        "[{m}, {at}] x [{at}, {n}] depth {depth}: {len} words, id bound {bound}, \
                         above {small:?} by more than its {spare} k-dependent constants"
                    ));
                }
            }
        }
        eprintln!(
            "[{m}, k] x [k, {n}] depth {depth}: up to 16 tiles {small:?}, worst large k {} gives {} words, id bound {} (+{} ids, +{} words, {} k-dependent constants)",
            worst.2,
            worst.0,
            worst.1,
            worst.1 as i64 - small.1 as i64,
            worst.0 as i64 - small.0 as i64,
            worst.3
        );
    }
    assert!(
        over.is_empty(),
        "beyond the 16-tile bound by more than the k-dependent constants: {:#?}",
        &over[..over.len().min(12)]
    );
}
