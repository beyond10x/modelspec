//! `story:matmul-shape-aware-tiling`, acceptance `register-blocked`: for a matmul whose output
//! has at least 1024 rows and 1024 columns, `MatMulTile::of` gives each invocation a block of
//! `rows` by `cols_per_invocation` output elements, at least 4 by 4, the tile prints the block,
//! and the kernel signature changes with it. Small shapes, the MNIST step among them, keep one
//! column per invocation.

use modelspec_core::ir::kernel::{MIN_SHARED_MEMORY_BYTES, MatMulTile};

/// Outputs of at least 1024 by 1024: the `mlp_sweep` forward matmuls of that size at b8192,
/// whole and partial blocks in each of `m` and `n`, and `k` from 1 to 8192.
const LARGE: [(u64, u64, u64); 11] = [
    (8192, 784, 4096),
    (8192, 784, 1024),
    (1024, 1000, 1024),
    (1024, 784, 1024),
    (1025, 1000, 1031),
    (1024, 1, 1024),
    (1031, 7, 1029),
    (2049, 129, 1025),
    (4096, 8192, 4096),
    (1100, 33, 1500),
    (65_536, 64, 1024),
];

#[test]
fn a_matmul_of_at_least_1024_by_1024_outputs_is_register_blocked_at_least_4_by_4() {
    for (m, k, n) in LARGE {
        assert!(
            m >= 1024 && n >= 1024,
            "[{m}, {k}] x [{k}, {n}] is a large shape"
        );
        let tile = MatMulTile::of(m, k, n);
        let what = format!("[{m}, {k}] x [{k}, {n}]: {tile}");
        assert_eq!(
            tile,
            MatMulTile::of(m, k, n),
            "{what}: a function of the shape"
        );
        assert!(tile.rows >= 4, "{what}: rows per invocation");
        assert!(
            tile.cols_per_invocation >= 4,
            "{what}: columns per invocation"
        );
        assert!(
            tile.cols.is_multiple_of(tile.cols_per_invocation),
            "{what}: whole column lanes"
        );
        let lanes = tile.cols / tile.cols_per_invocation;
        assert!(tile.workgroup_size.is_multiple_of(lanes), "{what}");
        assert_eq!(
            u64::from(tile.workgroup_size) * u64::from(tile.rows * tile.cols_per_invocation),
            u64::from(tile.rows_per_workgroup()) * u64::from(tile.cols),
            "{what}: each output element of a block is one invocation's"
        );
        assert!(tile.shared_bytes() <= MIN_SHARED_MEMORY_BYTES, "{what}");
        assert!(
            tile.workgroups(m, n) * u64::from(tile.rows_per_workgroup()) * u64::from(tile.cols)
                >= m * n,
            "{what}: the workgroups cover the output"
        );
        let printed = tile.to_string();
        for field in [
            format!(" rows {} ", tile.rows),
            format!(" cols_per_invocation {} ", tile.cols_per_invocation),
        ] {
            assert!(printed.contains(&field), "{what}: prints `{field}`");
        }
    }
}

#[test]
fn the_mnist_matmuls_keep_one_column_per_invocation() {
    for (m, k, n) in [
        (100, 784, 64),
        (784, 100, 64),
        (100, 64, 10),
        (64, 100, 10),
        (100, 10, 64),
    ] {
        let tile = MatMulTile::of(m, k, n);
        assert_eq!(
            tile.cols_per_invocation, 1,
            "[{m}, {k}] x [{k}, {n}]: {tile}"
        );
    }
}

#[test]
fn a_tile_that_differs_only_in_columns_per_invocation_prints_differently() {
    let tile = MatMulTile::of(1024, 1024, 1024);
    let other = MatMulTile {
        cols_per_invocation: tile.cols_per_invocation / 2,
        ..tile
    };
    assert_ne!(tile.to_string(), other.to_string());
}
