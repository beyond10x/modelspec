//! `story:matmul-shape-aware-tiling`, acceptance `register-blocked` and `same-bits`, without a
//! device: the module of a register-blocked matmul passes `spirv-val --target-env vulkan1.2`,
//! runs its tile's workgroup size, stages exactly its tile's workgroup memory, keeps one
//! accumulator chain per output element of its block (`rows · cols_per_invocation` chains of
//! `2k - 1` operations over the tiles it emits straight), every `OpFMul` and `OpFAdd` decorated
//! `NoContraction`, and is no larger at a long `k` than at a short one but for the constants
//! that depend on `k`. Needs `spirv-val` on the path and fails, not skips, without it.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use modelspec_core::ir::kernel::{Epilogue, MIN_SHARED_MEMORY_BYTES, MatMulTile};
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::spirv::validate::validate;

const OP_EXECUTION_MODE: u32 = 16;
const OP_TYPE_ARRAY: u32 = 28;
const OP_CONSTANT: u32 = 43;
const OP_LOOP_MERGE: u32 = 246;
const MODE_LOCAL_SIZE: u32 = 17;

/// Shapes whose tile is register-blocked: partial blocks, `k = 1`, a `k` shorter than a tile
/// and `k` long enough for a tile loop of several passes.
const SHAPES: [(u64, u64, u64); 6] = [
    (1024, 1024, 1024),
    (1025, 1000, 1031),
    (1024, 1, 1024),
    (1031, 7, 1029),
    (8192, 784, 4096),
    (4096, 8192, 4096),
];

const EPILOGUES: [Option<Epilogue>; 3] = [
    None,
    Some(Epilogue::Relu {
        writes_pre_activation: false,
    }),
    Some(Epilogue::Relu {
        writes_pre_activation: true,
    }),
];

fn local_size(module: &[u32]) -> Vec<u32> {
    common::words::instructions(module)
        .into_iter()
        .find(|(op, w)| *op == OP_EXECUTION_MODE && w[2] == MODE_LOCAL_SIZE)
        .map(|(_, w)| w[3..].to_vec())
        .expect("a LocalSize")
}

/// The length of each `OpTypeArray` of `module`, read from its `u32` constant.
fn array_lengths(module: &[u32]) -> Vec<u32> {
    let all = common::words::instructions(module);
    let constant = |id: u32| {
        all.iter()
            .find(|(op, w)| *op == OP_CONSTANT && w[2] == id)
            .map(|(_, w)| w[3])
            .expect("an array length is a constant")
    };
    all.iter()
        .filter(|(op, _)| *op == OP_TYPE_ARRAY)
        .map(|(_, w)| constant(w[3]))
        .collect()
}

fn loops(module: &[u32]) -> usize {
    common::words::instructions(module)
        .iter()
        .filter(|(op, _)| *op == OP_LOOP_MERGE)
        .count()
}

/// The tiles of the tile loop body (`emit.rs` `MATMUL_UNROLL`).
const LOOP_BODY: u64 = 4;

/// The tiles a `k`-deep matmul with `tile` runs in its loop, in whole loop bodies.
fn looped(k: u64, tile: &MatMulTile) -> u64 {
    let candidates = (k / u64::from(tile.depth)).saturating_sub(u64::from(tile.ahead) + 1);
    candidates - candidates % LOOP_BODY
}

#[test]
fn a_register_blocked_matmul_module_validates_with_one_uncontracted_chain_per_element() {
    let mut total = 0;
    for (m, k, n) in SHAPES {
        let tile = MatMulTile::of(m, k, n);
        let what = format!("[{m}, {k}] x [{k}, {n}], {tile}");
        assert!(
            tile.rows >= 4 && tile.cols_per_invocation >= 4,
            "{what}: register-blocked"
        );
        assert!(tile.shared_bytes() <= MIN_SHARED_MEMORY_BYTES, "{what}");
        for epilogue in EPILOGUES {
            let kernel = common::matmul_kernel_with(m, k, n, epilogue);
            let module = emit(&kernel).unwrap_or_else(|error| panic!("{what}: {error}"));
            validate(&module).unwrap_or_else(|error| panic!("{what}: {error}"));
            assert_eq!(local_size(&module), [tile.workgroup_size, 1, 1], "{what}");
            assert_eq!(
                array_lengths(&module),
                [tile.shared_values()],
                "{what}: one f32 array of every buffer"
            );
            let looped = looped(k, &tile);
            assert_eq!(loops(&module), usize::from(looped > 0), "{what}");
            let emitted_k = k - looped.saturating_sub(LOOP_BODY) * u64::from(tile.depth);
            let chains = u64::from(tile.rows * tile.cols_per_invocation);
            let (missing, all) = common::words::uncontracted_arithmetic(&module);
            assert_eq!(
                all as u64,
                chains * (2 * emitted_k - 1),
                "{what}: one chain of products and sums per output element of the block"
            );
            assert!(
                missing.is_empty(),
                "{what}: no NoContraction on {missing:?}"
            );
            total += 1;
        }
    }
    eprintln!("register blocked: {total} modules pass spirv-val");
}

#[test]
fn a_register_blocked_matmul_module_does_not_grow_with_k() {
    let (m, n) = (1024, 1024);
    let words = |k: u64| emit(&common::matmul_kernel(m, k, n)).expect("emits").len();
    let depth = u64::from(MatMulTile::of(m, 1 << 16, n).depth);
    let small = (1..=16 * depth)
        .step_by(1)
        .map(words)
        .max()
        .expect("some k");
    for k in [10_000, 65_536, 100_003] {
        let tile = MatMulTile::of(m, k, n);
        assert!(tile.cols_per_invocation >= 4, "{tile}");
        let spare = common::matmul_k_constants(k, &tile);
        let len = words(k);
        assert!(
            len <= small + common::CONSTANT_WORDS * spare,
            "[{m}, {k}] x [{k}, {n}]: {len} words above {small} by more than {spare} constants"
        );
    }
}
