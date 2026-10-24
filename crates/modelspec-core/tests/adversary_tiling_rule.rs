//! Adversary cases for `story:matmul-shape-aware-tiling`: `MatMulTile::of` read against the rule
//! DESIGN.md § 5.2 states in words, written out here independently of the code, and against the
//! rule of the base commit for every shape the block rule does not take.
//!
//! - The rule: the wide tile for an output of at least 128 columns, `k` of at least 64 and at
//!   least 128 blocks of 64 by 128; else the square tile for an output of at least 64 columns
//!   and at least 128 blocks of 64 by 64; else one column per invocation.
//! - Every tile the rule gives fits [`MAX_WORKGROUP_INVOCATIONS`] and
//!   [`MIN_SHARED_MEMORY_BYTES`], covers its output, has whole column lanes and is a function of
//!   the shape alone; `cols_per_invocation` is 1 whenever no block applies.
//! - Acceptance `register-blocked`: every output of at least 1024 by 1024 is blocked at least
//!   4 by 4.
//! - A shape the block rule does not take keeps the base commit's tile field for field, its
//!   `b` stride and its workgroup memory, so only the printed tile changes.
//!
//! The shapes: every triple of 46 boundary values in `1..=10_000`, then 200,000 triples drawn
//! from a fixed-seed generator over the same range.

use modelspec_core::ir::kernel::{MIN_SHARED_MEMORY_BYTES, MatMulTile};
use modelspec_core::ir::verify::MAX_WORKGROUP_INVOCATIONS;

const EDGES: [u64; 46] = [
    1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 511,
    512, 513, 783, 784, 785, 1000, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8127,
    8128, 8129, 8191, 8192, 8193, 9999, 10_000,
];

/// Which tile DESIGN.md § 5.2 says the shape takes.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Rule {
    Wide,
    Square,
    OneColumn,
}

fn rule(m: u64, k: u64, n: u64) -> Rule {
    let wide_blocks = m.div_ceil(64) * n.div_ceil(128);
    let square_blocks = m.div_ceil(64) * n.div_ceil(64);
    if n >= 128 && k >= 64 && wide_blocks >= 128 {
        Rule::Wide
    } else if n >= 64 && square_blocks >= 128 {
        Rule::Square
    } else {
        Rule::OneColumn
    }
}

/// The base commit's `MatMulTile::of` (269b557, `crates/modelspec-core/src/ir/kernel.rs`), as
/// `(workgroup_size, cols, rows, depth, ahead, buffers, b_stride, shared_bytes)`.
fn base_tile(m: u64, k: u64, n: u64) -> (u32, u32, u32, u32, u32, u32, u32, u32) {
    let workgroup_size = 128u32;
    let cols = n.clamp(1, 16).next_power_of_two() as u32;
    let one_row = m.div_ceil(u64::from(workgroup_size / cols)) * n.div_ceil(u64::from(cols));
    let (rows, depth, ahead) = if one_row >= 256 {
        (2u32, 16u32, 2u32)
    } else {
        (1, 64, 6)
    };
    let mut depth = k.clamp(1, u64::from(depth)) as u32;
    loop {
        let buffers = if u64::from(depth) < k { 2 } else { 1 };
        let b_stride = depth.next_multiple_of(4) + 4;
        let rows_per_workgroup = (workgroup_size / cols) * rows;
        let shared_bytes = buffers * (rows_per_workgroup * depth + cols * b_stride) * 4;
        if depth == 1 || shared_bytes <= MIN_SHARED_MEMORY_BYTES {
            return (
                workgroup_size,
                cols,
                rows,
                depth,
                ahead,
                buffers,
                b_stride,
                shared_bytes,
            );
        }
        depth /= 2;
    }
}

/// Every way `tile` can break the rule for `[m, k] x [k, n]`, empty when it keeps it.
fn breaks(m: u64, k: u64, n: u64) -> Vec<String> {
    let tile = MatMulTile::of(m, k, n);
    let mut wrong = Vec::new();
    let mut check = |ok: bool, what: &str| {
        if !ok {
            wrong.push(format!("[{m}, {k}] x [{k}, {n}] {tile}: {what}"));
        }
    };
    check(tile == MatMulTile::of(m, k, n), "a function of the shape");
    check(
        (1..=MAX_WORKGROUP_INVOCATIONS).contains(&tile.workgroup_size),
        "the plan verifier's workgroup bound",
    );
    check(
        tile.shared_bytes() <= MIN_SHARED_MEMORY_BYTES,
        "16 KiB of workgroup memory",
    );
    check(
        tile.depth >= 1 && u64::from(tile.depth) <= k.max(1),
        "a depth in 1..=k",
    );
    check(
        tile.buffers == if u64::from(tile.depth) < k { 2 } else { 1 },
        "two buffers exactly when there is more than one tile",
    );
    check(
        tile.cols_per_invocation.is_power_of_two()
            && tile.cols.is_multiple_of(tile.cols_per_invocation),
        "whole column lanes",
    );
    check(
        tile.workgroup_size.is_multiple_of(tile.lanes()),
        "whole rows of lanes",
    );
    check(
        u64::from(tile.workgroup_size) * u64::from(tile.rows * tile.cols_per_invocation)
            == u64::from(tile.rows_per_workgroup()) * u64::from(tile.cols),
        "each element of a block is one invocation's",
    );
    let row_blocks = m.div_ceil(u64::from(tile.rows_per_workgroup()));
    let col_blocks = n.div_ceil(u64::from(tile.cols));
    check(
        row_blocks * u64::from(tile.rows_per_workgroup()) >= m
            && col_blocks * u64::from(tile.cols) >= n
            && tile.workgroups(m, n) == row_blocks * col_blocks,
        "the workgroups cover the output",
    );
    check(
        u32::try_from(tile.workgroups(m, n)).is_ok(),
        "the workgroups fit one u32 dimension",
    );
    match rule(m, k, n) {
        Rule::Wide => check(
            (tile.cols, tile.rows, tile.cols_per_invocation) == (128, 8, 8)
                && tile.rows_per_workgroup() == 64
                && tile.depth == k.min(8) as u32
                && tile.ahead == 1
                && tile.b_pad == 0,
            "DESIGN.md § 5.2: the wide tile, 8 by 8 of a block of 64 by 128, depth 8",
        ),
        Rule::Square => check(
            (tile.cols, tile.rows, tile.cols_per_invocation) == (64, 8, 4)
                && tile.rows_per_workgroup() == 64
                && tile.depth == k.min(16) as u32
                && tile.ahead == 1
                && tile.b_pad == 0,
            "DESIGN.md § 5.2: the square tile, 8 by 4 of a block of 64 by 64, depth 16",
        ),
        Rule::OneColumn => {
            check(
                tile.cols_per_invocation == 1,
                "one column per invocation when no block applies",
            );
            let (workgroup_size, cols, rows, depth, ahead, buffers, b_stride, shared_bytes) =
                base_tile(m, k, n);
            check(
                (
                    tile.workgroup_size,
                    tile.cols,
                    tile.rows,
                    tile.depth,
                    tile.ahead,
                    tile.buffers,
                    tile.b_stride(),
                    tile.shared_bytes(),
                ) == (
                    workgroup_size,
                    cols,
                    rows,
                    depth,
                    ahead,
                    buffers,
                    b_stride,
                    shared_bytes,
                ),
                "the base commit's tile, b stride and workgroup memory",
            );
        }
    }
    if m >= 1024 && n >= 1024 {
        check(
            tile.rows >= 4 && tile.cols_per_invocation >= 4,
            "acceptance register-blocked: at least 4 by 4 from 1024 by 1024",
        );
    }
    wrong
}

#[test]
fn every_tile_on_the_boundary_grid_keeps_the_design_rule() {
    let mut wrong = Vec::new();
    let mut taken = [0usize; 3];
    for m in EDGES {
        for k in EDGES {
            for n in EDGES {
                taken[rule(m, k, n) as usize] += 1;
                wrong.extend(breaks(m, k, n));
            }
        }
    }
    eprintln!(
        "boundary grid: {} shapes, wide {} square {} one column {}",
        EDGES.len().pow(3),
        taken[0],
        taken[1],
        taken[2]
    );
    assert!(
        taken.iter().all(|&t| t > 0),
        "the grid reaches every tile: {taken:?}"
    );
    assert!(
        wrong.is_empty(),
        "{} breaks, first {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(20)]
    );
}

#[test]
fn every_tile_of_200_000_drawn_shapes_keeps_the_design_rule() {
    let mut state = 0x7469_6c65_7275_6c65u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % 10_000 + 1
    };
    let mut wrong = Vec::new();
    for _ in 0..200_000 {
        let (m, k, n) = (next(), next(), next());
        wrong.extend(breaks(m, k, n));
    }
    assert!(
        wrong.is_empty(),
        "{} breaks, first {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(20)]
    );
}

#[test]
fn the_thresholds_fall_where_the_design_puts_them() {
    // `[m, k, n]` and the tile DESIGN.md § 5.2 names, one step either side of each bound.
    let cases: [((u64, u64, u64), Rule); 12] = [
        ((8192, 64, 64), Rule::Square),
        ((8128, 64, 64), Rule::OneColumn),
        ((8192, 64, 63), Rule::OneColumn),
        ((512, 64, 2048), Rule::Wide),
        ((448, 64, 2048), Rule::Square),
        ((512, 63, 2048), Rule::Square),
        ((512, 64, 127), Rule::OneColumn),
        ((8192, 64, 127), Rule::Square),
        ((1, 64, 16_385), Rule::Wide),
        ((1, 1, 8192), Rule::Square),
        ((1, 1, 8128), Rule::OneColumn),
        ((784, 8192, 1024), Rule::Square),
    ];
    for ((m, k, n), expected) in cases {
        assert_eq!(rule(m, k, n), expected, "[{m}, {k}] x [{k}, {n}]");
        let tile = MatMulTile::of(m, k, n);
        let (cols, per) = match expected {
            Rule::Wide => (128, 8),
            Rule::Square => (64, 4),
            Rule::OneColumn => (tile.cols, 1),
        };
        assert_eq!(
            (tile.cols, tile.cols_per_invocation),
            (cols, per),
            "[{m}, {k}] x [{k}, {n}]: {tile}"
        );
    }
}
