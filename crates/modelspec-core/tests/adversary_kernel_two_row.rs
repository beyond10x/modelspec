//! Adversary pass on `story:m1-f-weight-grad-matmul-tiled`: `MatMulTile::of` after two-row tiles
//! took tiles of up to 16 values read 2 ahead (`MATMUL_TWO_ROW_TILE`).
//!
//! 1. One-row tiles are unchanged: every one-row tile over a grid of shapes equals the tile the
//!    base commit `2449897` computed, its rule copied here as written there.
//! 2. Every two-row tile over the same grid reads 2 ahead, has a depth of `min(k, 16)` halved
//!    only while the staged values overflow 16 KiB, stages at most 16 KiB, and has two buffers
//!    exactly when it has more than one tile.
//! 3. Literal tiles of the two-row shapes whose depth the memory limit lowers below 16.

use modelspec_core::ir::kernel::{MIN_SHARED_MEMORY_BYTES, MatMulTile};

/// `MatMulTile::of` at `2449897`, verbatim but for the name.
fn base_of(m: u64, k: u64, n: u64) -> MatMulTile {
    let workgroup_size = 128;
    let cols = n.clamp(1, 16).next_power_of_two() as u32;
    let one_row = m.div_ceil(u64::from(workgroup_size / cols)) * n.div_ceil(u64::from(cols));
    let mut tile = MatMulTile {
        workgroup_size,
        cols,
        rows: if one_row >= 256 { 2 } else { 1 },
        depth: k.clamp(1, 64) as u32,
        ahead: 6,
        buffers: 1,
    };
    loop {
        tile.buffers = if u64::from(tile.depth) < k { 2 } else { 1 };
        if tile.depth == 1 || tile.shared_bytes() <= MIN_SHARED_MEMORY_BYTES {
            return tile;
        }
        tile.depth /= 2;
    }
}

fn grid() -> Vec<(u64, u64, u64)> {
    let ms = [
        1, 2, 7, 8, 9, 100, 784, 1016, 1017, 2040, 2041, 2048, 2049, 8160, 8161, 16_320, 16_321,
        32_640, 32_641, 70_000,
    ];
    let ks = [
        1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 100, 129, 1000, 4099,
    ];
    let ns = [1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 64];
    let mut out = Vec::new();
    for &m in &ms {
        for &k in &ks {
            for &n in &ns {
                out.push((m, k, n));
            }
        }
    }
    out
}

#[test]
fn every_one_row_tile_is_the_base_commits_tile() {
    let mut one_row = 0;
    for (m, k, n) in grid() {
        let base = base_of(m, k, n);
        if base.rows != 1 {
            continue;
        }
        one_row += 1;
        assert_eq!(
            MatMulTile::of(m, k, n),
            base,
            "[{m}, {k}] x [{k}, {n}]: a one-row tile changed"
        );
    }
    assert!(one_row > 1_000, "{one_row} one-row shapes");
}

#[test]
fn every_two_row_tile_reads_2_ahead_at_the_deepest_depth_up_to_16_that_fits() {
    let mut two_row = 0;
    for (m, k, n) in grid() {
        let tile = MatMulTile::of(m, k, n);
        let base = base_of(m, k, n);
        let what = format!("[{m}, {k}] x [{k}, {n}]: {tile:?}");
        assert_eq!(tile.rows, base.rows, "{what}: the row rule moved");
        if tile.rows != 2 {
            continue;
        }
        two_row += 1;
        assert_eq!(
            (tile.workgroup_size, tile.cols, tile.ahead),
            (128, base.cols, 2),
            "{what}"
        );
        assert!(tile.shared_bytes() <= MIN_SHARED_MEMORY_BYTES, "{what}");
        assert!(u64::from(tile.depth) <= k.min(16), "{what}");
        let tiles = k.div_ceil(u64::from(tile.depth));
        assert_eq!(tile.buffers, if tiles > 1 { 2 } else { 1 }, "{what}");
        // The depth is min(k, 16) or the memory limit forced every halving above it.
        let start = k.min(16) as u32;
        if tile.depth != start {
            let mut above = tile;
            above.depth = tile.depth * 2;
            above.buffers = if u64::from(above.depth) < k { 2 } else { 1 };
            assert!(
                above.depth <= start && above.shared_bytes() > MIN_SHARED_MEMORY_BYTES,
                "{what}: depth {} would fit ({} bytes)",
                above.depth,
                above.shared_bytes()
            );
        }
    }
    assert!(two_row > 300, "{two_row} two-row shapes");
}

#[test]
fn two_row_tiles_whose_depth_the_memory_limit_lowers() {
    // (m, k, n, depth, buffers): 256 rows of 1 column, 128 rows of 2 columns.
    let cases: [(u64, u64, u64, u32, u32); 8] = [
        (32_641, 100, 1, 4, 2),
        (32_641, 16, 1, 4, 2),
        (32_641, 15, 1, 15, 1),
        (32_641, 5, 1, 5, 1),
        (32_641, 17, 1, 4, 2),
        (16_321, 100, 2, 8, 2),
        (16_321, 16, 2, 16, 1),
        (16_321, 17, 2, 8, 2),
    ];
    for (m, k, n, depth, buffers) in cases {
        let tile = MatMulTile::of(m, k, n);
        let what = format!("[{m}, {k}] x [{k}, {n}]: {tile:?}");
        assert_eq!(tile.rows, 2, "{what}");
        assert_eq!(tile.ahead, 2, "{what}");
        assert_eq!((tile.depth, tile.buffers), (depth, buffers), "{what}");
        assert!(tile.shared_bytes() <= MIN_SHARED_MEMORY_BYTES, "{what}");
    }
}
