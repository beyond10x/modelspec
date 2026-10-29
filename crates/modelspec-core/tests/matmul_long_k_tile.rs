//! `story:vulkan-long-k-weight-grad-matmul`: a matmul whose `k` of at least 16,384 leaves its
//! output too small for the square tile takes a long-`k` tile.
//!
//! - The three weight-gradient matmuls of the decider model take [`MATMUL_LONG_K_NARROW_BLOCK`]
//!   (`[256, 65536]ᵀ x [65536, 256]`) and [`MATMUL_LONG_K_BLOCK`] (`[256, 65536]ᵀ x
//!   [65536, 1024]`, `[1024, 65536]ᵀ x [65536, 256]`), under every invocation limit.
//! - A `k` below 16,384, fewer than 32 rows, or fewer workgroups than a long-`k` block's bound,
//!   keeps the tile it had; the shapes of `1..=10_000` are pinned by `matmul_tile_from_profile`.
//! - Every long-`k` tile reads vectors, stages within the 16 KiB every Vulkan device provides
//!   with either operand transposed, every stride and offset a multiple of 4 values, and is the
//!   tile of its own workgroup size, so a kernel's signature names the tile it runs.

use modelspec_core::ir::kernel::{
    MATMUL_LARGE_BLOCK, MATMUL_LONG_K_BLOCK, MATMUL_LONG_K_NARROW_BLOCK, MIN_SHARED_MEMORY_BYTES,
    MatMulBlock, MatMulLimits, MatMulTile,
};

fn is_block(tile: &MatMulTile, block: &MatMulBlock) -> bool {
    (
        tile.workgroup_size,
        tile.cols,
        tile.rows,
        tile.cols_per_invocation,
        tile.depth,
        tile.ahead,
        tile.b_pad,
    ) == (
        block.workgroup_size,
        block.cols,
        block.rows,
        block.cols_per_invocation,
        block.depth,
        block.ahead,
        0,
    )
}

const FLAGS: [[bool; 2]; 4] = [[false, false], [true, false], [false, true], [true, true]];

#[test]
fn the_decider_weight_gradients_take_the_long_k_tiles() {
    for ((m, k, n), block, workgroups) in [
        ((256, 65536, 256), MATMUL_LONG_K_NARROW_BLOCK, 64),
        ((256, 65536, 1024), MATMUL_LONG_K_BLOCK, 128),
        ((1024, 65536, 256), MATMUL_LONG_K_BLOCK, 128),
    ] {
        for limit in [64, 128, 256, 1024] {
            let tile = MatMulTile::of_invocations(m, k, n, limit);
            assert!(
                is_block(&tile, &block),
                "[{m}, {k}] x [{k}, {n}] at {limit}: {tile}"
            );
            assert_eq!(tile.buffers, 2, "{tile}");
            assert_eq!(tile.workgroups(m, n), workgroups, "{tile}");
            assert!(tile.long_k() && tile.vector_reads(), "{tile}");
        }
        let limits = MatMulLimits {
            invocations: 1024,
            shared_bytes: MIN_SHARED_MEMORY_BYTES,
        };
        assert_eq!(
            MatMulTile::for_limits(m, k, n, [true, false], limits),
            MatMulTile::of(m, k, n)
        );
    }
    assert_eq!(
        MatMulTile::of(256, 65536, 256).to_string(),
        "tile workgroup_size 128 cols 32 rows 4 cols_per_invocation 2 depth 16 ahead 4 \
         buffers 2 b_pad 0"
    );
    assert_eq!(
        MatMulTile::of(256, 65536, 1024).to_string(),
        "tile workgroup_size 64 cols 32 rows 8 cols_per_invocation 4 depth 16 ahead 4 \
         buffers 2 b_pad 0"
    );
}

#[test]
fn a_shorter_k_or_fewer_workgroups_keep_the_tile_they_had() {
    for (m, k, n) in [
        (256, 16383, 256),
        (256, 16383, 1024),
        (784, 8192, 256),
        (128, 4096, 128),
        (512, 4096, 128),
        (128, 4096, 8192),
        // 63 workgroups of 32 by 32, 2 by 32, and `n` under 32.
        (224, 65536, 288),
        (32, 65536, 64),
        (4096, 65536, 31),
        // Fewer than 32 rows (finding LK-1).
        (31, 65536, 4096),
        (16, 16384, 2048),
        (1, 65536, 8192),
    ] {
        let tile = MatMulTile::of(m, k, n);
        assert!(!tile.long_k(), "[{m}, {k}] x [{k}, {n}]: {tile}");
        assert!(!tile.vector_reads(), "[{m}, {k}] x [{k}, {n}]: {tile}");
    }
    // The wide tile keeps the outputs it took.
    let wide = MatMulTile::of(1024, 65536, 1024);
    assert_eq!(
        (wide.cols, wide.rows, wide.cols_per_invocation),
        (128, 8, 8)
    );
}

#[test]
fn every_long_k_tile_reads_vectors_and_stages_within_the_vulkan_minimum() {
    let mut state = 0x6c6f_6e67_u64;
    let mut next = |bound: u64| {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        1 + (z ^ (z >> 31)) % bound
    };
    let (mut narrow, mut block) = (0, 0);
    for _ in 0..20_000 {
        let (m, k, n) = (next(4096), 16_383 + next(100_000), next(4096));
        for limit in [128, 256, 1024] {
            let tile = MatMulTile::of_invocations(m, k, n, limit);
            let case = format!("[{m}, {k}] x [{k}, {n}] at {limit}: {tile}");
            let large = tile.workgroup_size == MATMUL_LARGE_BLOCK.workgroup_size;
            assert_eq!(tile.vector_reads(), large || tile.long_k(), "{case}");
            if !tile.long_k() {
                continue;
            }
            if is_block(&tile, &MATMUL_LONG_K_BLOCK) {
                block += 1;
                assert!(
                    tile.workgroups(m, n) >= MATMUL_LONG_K_BLOCK.from_workgroups,
                    "{case}"
                );
            } else {
                assert!(is_block(&tile, &MATMUL_LONG_K_NARROW_BLOCK), "{case}");
                narrow += 1;
                assert!(
                    tile.workgroups(m, n) >= MATMUL_LONG_K_NARROW_BLOCK.from_workgroups,
                    "{case}"
                );
            }
            assert!(n >= u64::from(tile.cols), "{case}");
            assert!(m >= MATMUL_LONG_K_BLOCK.from_m, "{case}");
            assert_eq!(
                MatMulTile::of_invocations(m, k, n, tile.workgroup_size),
                tile,
                "{case}: the tile of its own workgroup size"
            );
            for flags in FLAGS {
                let layout = tile.layout(flags);
                assert_eq!(layout, tile.vector_layout(flags), "{case}");
                assert!(
                    layout.shared_bytes() <= MIN_SHARED_MEMORY_BYTES,
                    "{case} {flags:?}"
                );
                for value in [
                    layout.a_stride,
                    layout.b_stride,
                    layout.b_offset,
                    layout.buffer_values,
                ] {
                    assert!(value.is_multiple_of(4), "{case} {flags:?} {layout:?}");
                }
            }
        }
    }
    assert!(
        narrow > 100 && block > 100,
        "{narrow} narrow, {block} block"
    );
}
