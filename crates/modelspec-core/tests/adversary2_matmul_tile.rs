//! Adversary pass 2 on `story:m1-f-forward-matmul-tiled`: the two-row threshold of
//! `MatMulTile::of` at its boundary, as literals. DESIGN.md § 5.2: two rows per invocation "when
//! the block of 8 rows needs at least 256 workgroups". No shape of `kernel_plan.rs` has exactly
//! 256 workgroups of the one-row tile, and it derives its expected rows from the same constant.

use modelspec_core::ir::kernel::MatMulTile;

#[test]
fn a_matmul_tile_takes_two_rows_from_exactly_256_one_row_workgroups() {
    // (m, k, n, one-row workgroups, rows)
    let cases: [(u64, u64, u64, u64, u32); 10] = [
        (2040, 773, 16, 255, 1),
        (2041, 773, 16, 256, 2),
        (2048, 64, 16, 256, 2),
        (2049, 773, 16, 257, 2),
        (8160, 300, 3, 255, 1),
        (8161, 300, 3, 256, 2),
        (32640, 100, 1, 255, 1),
        (32641, 100, 1, 256, 2),
        (1016, 500, 17, 254, 1),
        (1017, 500, 17, 256, 2),
    ];
    for (m, k, n, one_row, rows) in cases {
        let tile = MatMulTile::of(m, k, n);
        assert_eq!(
            tile.rows, rows,
            "[{m}, {k}] x [{k}, {n}], {one_row} one-row workgroups: {tile:?}"
        );
        assert_eq!(
            tile.workgroups(m, n),
            one_row.div_ceil(u64::from(rows)),
            "[{m}, {k}] x [{k}, {n}]: {tile:?}"
        );
    }
}
