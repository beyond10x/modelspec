//! Adversary pass on `story:vulkan-long-k-weight-grad-matmul`: the long-`k` tiles of
//! `MatMulTile::of_invocations` (64 invocations, a 64 by 32 block, and 128 invocations, a 32 by
//! 32 block, both depth 16 read 4 tiles ahead) attacked at every selection boundary.
//!
//! Host cases (no device):
//! - the tile of every boundary shape (`k` 16383/16384/16385, `n` 31/32/33, 127/128/129 blocks of
//!   64 by 32 and 63/64/65 of 32 by 32) is the one the claimed rule names, under every invocation
//!   limit, workgroup memory and transposed flag, and over sampled shapes;
//! - every shape that selects a long-`k` tile stages at most 13,312 bytes, and its emitted module
//!   declares at most that much workgroup memory;
//! - the staging stores of a long-`k` tile into workgroup memory, modelled as the emitter
//!   computes them, take one address per bank per subgroup store: the fixed-line rule
//!   (`a` transposed, `b` as is), and the other two operands (`a` as is, `b` transposed);
//! - no matmul module below `k` = 16384 changed: the words `emit` and `emit_module` give for the
//!   matmul kernels of BERT-tiny, transformer_block, the three `mlp_sweep` sources and MNIST, and
//!   for sampled synthetic shapes, hash as at the base commit 2fe753ab
//!   (`tests/fixtures/adversary_long_k_base_modules.txt`, recorded there with
//!   `MODELSPEC_ADV_LONG_K_RECORD=1`).
//!
//! Device cases (ignored, they need a Vulkan device):
//! - every boundary shape and a `k` sweep over tails and tile counts modulo 4, all four
//!   transposed combinations and the three epilogues, bit for bit against the CPU AVX2 kernel;
//! - NaN, infinities, signed zeros, overflow mid-sum and subnormals;
//! - timing: each long-`k` tile against the tile it replaced, on a shape just past each
//!   boundary. The replaced tile is that of the same `[m, n]` at `k` = 16383, which is the tile
//!   the base commit gave every `k` of that `[m, n]` (`adv_lk_the_replaced_tile_is_that_of_k_16383`).
//!   A shape of fewer than 32 rows keeps that tile (finding LK-1) and is not timed.
//!
//! Finding LK-2 (`adv_lk_a_as_is_and_b_transposed_stage_without_bank_conflicts`) is ignored as
//! INFEASIBLE, with its reason.
//!
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test adversary_matmul_long_k -- --ignored`
#![cfg(feature = "vulkan")]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact, compile_artifact_for};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    Epilogue, Kernel, KernelKind, KernelModuleId, MatMulLimits, MatMulTile, Operand,
};
use modelspec_core::ir::tensor::{DType, TensorType};
use modelspec_runtime::cpu::matmul::{self as cpu, Lhs};
use modelspec_runtime::spirv::emit::{emit, emit_module, emit_module_with};
use modelspec_runtime::spirv::validate::validate;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, Vulkan};

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

const RECORD_ENV: &str = "MODELSPEC_ADV_LONG_K_RECORD";

const GUARD_WORDS: usize = 16;
const UNWRITTEN: u32 = 0x7fc0_dead;

const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_VARIABLE: u32 = 59;
const STORAGE_WORKGROUP: u32 = 4;

const FLAGS: [[bool; 2]; 4] = [[false, false], [true, false], [false, true], [true, true]];

fn profile() -> DeviceProfile {
    DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile")
}

/// The profile with only the Vulkan minimum of workgroup memory.
fn min_profile() -> DeviceProfile {
    let mut p = profile();
    p.max_shared_memory_bytes = 16_384;
    p
}

/// The tiles a matmul can take, read from the tile's fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tile {
    Large,
    Wide,
    Square,
    LongK,
    Narrow,
    Fallback,
}

fn kind(tile: &MatMulTile) -> Tile {
    match (
        tile.workgroup_size,
        tile.cols,
        tile.rows,
        tile.cols_per_invocation,
        tile.ahead,
    ) {
        (256, 128, 8, 8, 1) => Tile::Large,
        (128, 128, 8, 8, 1) => Tile::Wide,
        (128, 64, 8, 4, 1) => Tile::Square,
        (64, 32, 8, 4, 4) => Tile::LongK,
        (128, 32, 4, 2, 4) => Tile::Narrow,
        (128, _, _, 1, _) => Tile::Fallback,
        _ => panic!("a tile of no known kind: {tile}"),
    }
}

/// The tile the commit message and DESIGN.md § 5.2 name for `[m, k] x [k, n]` at an invocation
/// limit: large, wide and square first, then the 64 by 32 long-`k` block at `k >= 16384`,
/// `n >= 32`, `m >= 32` and at least 128 blocks, then the 32 by 32 one at least 64 blocks.
/// The `m` floor is finding LK-1's fix: below 32 rows the long-`k` tiles were slower than the
/// tile they replaced.
fn claimed(m: u64, k: u64, n: u64, limit: u32) -> Tile {
    let blocks = |rows: u64, cols: u64| m.div_ceil(rows) * n.div_ceil(cols);
    if limit >= 256 && n >= 128 && k >= 64 && blocks(128, 128) >= 128 {
        Tile::Large
    } else if n >= 128 && k >= 64 && blocks(64, 128) >= 128 {
        Tile::Wide
    } else if n >= 64 && blocks(64, 64) >= 128 {
        Tile::Square
    } else if k >= 16_384 && n >= 32 && m >= 32 && blocks(64, 32) >= 128 {
        Tile::LongK
    } else if k >= 16_384 && n >= 32 && m >= 32 && blocks(32, 32) >= 64 {
        Tile::Narrow
    } else {
        Tile::Fallback
    }
}

/// Every selection boundary, `(m, k, n)` and the tile the rule names.
const BOUNDARIES: [((u64, u64, u64), Tile); 27] = [
    // k 16383 / 16384 / 16385 at the decider's outputs.
    ((256, 16_383, 1024), Tile::Fallback),
    ((256, 16_384, 1024), Tile::LongK),
    ((256, 16_385, 1024), Tile::LongK),
    ((256, 16_383, 256), Tile::Fallback),
    ((256, 16_384, 256), Tile::Narrow),
    ((256, 16_385, 256), Tile::Narrow),
    // n 31 / 32 / 33.
    ((4096, 16_384, 31), Tile::Fallback),
    ((4096, 16_384, 32), Tile::Narrow),
    ((4096, 16_384, 33), Tile::LongK),
    // 127 / 128 / 129 blocks of 64 by 32, by columns and by rows.
    ((32, 16_384, 4064), Tile::Narrow),
    ((32, 16_384, 4096), Tile::LongK),
    ((32, 16_384, 4128), Tile::LongK),
    ((8128, 16_384, 32), Tile::Narrow),
    ((8129, 16_384, 32), Tile::LongK),
    ((8193, 16_384, 32), Tile::LongK),
    // m 31 / 33 beside the 32 above (finding LK-1), and one row, which took a long-`k` tile
    // before it.
    ((31, 16_384, 4096), Tile::Fallback),
    ((33, 16_384, 4096), Tile::LongK),
    ((31, 16_384, 2048), Tile::Fallback),
    ((33, 16_384, 2048), Tile::Narrow),
    ((1, 16_384, 4096), Tile::Fallback),
    ((1, 16_384, 2048), Tile::Fallback),
    // 63 / 64 / 65 blocks of 32 by 32, by columns and by rows.
    ((32, 16_384, 2016), Tile::Fallback),
    ((32, 16_384, 2048), Tile::Narrow),
    ((32, 16_384, 2080), Tile::Narrow),
    ((2016, 16_384, 32), Tile::Fallback),
    ((2048, 16_384, 32), Tile::Narrow),
    ((2080, 16_384, 32), Tile::Narrow),
];

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A dimension between 1 and 2^bits, log-uniform.
fn dim(state: &mut u64, bits: u32) -> u64 {
    let width = 1 + splitmix(state) % u64::from(bits);
    let low = 1u64 << (width - 1);
    low + splitmix(state) % low.max(1)
}

#[test]
fn adv_lk_every_boundary_shape_takes_the_claimed_tile_under_every_limit() {
    let mut wrong = Vec::new();
    for ((m, k, n), want) in BOUNDARIES {
        let case = format!("[{m}, {k}] x [{k}, {n}]");
        let tile = MatMulTile::of(m, k, n);
        if kind(&tile) != want || claimed(m, k, n, 128) != want {
            wrong.push(format!(
                "{case}: of() {:?}, rule {:?}, table {want:?}",
                kind(&tile),
                claimed(m, k, n, 128)
            ));
        }
        for limit in [128, 255, 256, 1024] {
            for shared_bytes in [16_384, 49_152] {
                for flags in FLAGS {
                    let limits = MatMulLimits {
                        invocations: limit,
                        shared_bytes,
                    };
                    let got = MatMulTile::for_limits(m, k, n, flags, limits);
                    if kind(&got) != claimed(m, k, n, limit) {
                        wrong.push(format!(
                            "{case} at {limit} invocations, {shared_bytes} bytes, {flags:?}: {got}"
                        ));
                    }
                }
            }
        }
        if matches!(want, Tile::LongK | Tile::Narrow) {
            assert_eq!((tile.depth, tile.buffers, tile.b_pad), (16, 2, 0), "{case}");
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn adv_lk_selection_is_the_claimed_pure_function_of_the_shape_over_sampled_shapes() {
    let mut state = 0x00ad_0f1c_u64;
    let mut seen: BTreeMap<String, u32> = BTreeMap::new();
    let mut wrong = Vec::new();
    for i in 0..60_000 {
        let m = dim(&mut state, 14);
        let n = dim(&mut state, 14);
        let k = match i % 3 {
            0 => 16_384 - 64 + splitmix(&mut state) % 128,
            1 => dim(&mut state, 22),
            _ => 16_384 + splitmix(&mut state) % 300_000,
        };
        for limit in [128, 256, 1024] {
            let tile = MatMulTile::of_invocations(m, k, n, limit);
            let again = MatMulTile::of_invocations(m, k, n, limit);
            assert_eq!(tile, again);
            let want = claimed(m, k, n, limit);
            *seen.entry(format!("{want:?}")).or_default() += 1;
            if kind(&tile) != want && wrong.len() < 10 {
                wrong.push(format!(
                    "[{m}, {k}] x [{k}, {n}] at {limit}: {tile}, rule {want:?}"
                ));
            }
        }
    }
    eprintln!("sampled tiles: {seen:?}");
    assert!(seen.get("LongK").copied().unwrap_or(0) > 1000, "{seen:?}");
    assert!(seen.get("Narrow").copied().unwrap_or(0) > 1000, "{seen:?}");
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// The tile a long-`k` shape replaced: the base commit's `of_invocations` had no long-`k`
/// block, so every `k >= 64` of an `[m, n]` the large, wide and square tiles decline took the
/// one-column tile, whose rows and depth depend on `m` and `n` alone. The same `[m, n]` at
/// `k` = 16383 takes it in this tree: that is the A side of the timing case.
#[test]
fn adv_lk_the_replaced_tile_is_that_of_k_16383() {
    for ((m, k, n), want) in BOUNDARIES {
        if !matches!(want, Tile::LongK | Tile::Narrow) {
            continue;
        }
        let before = MatMulTile::of(m, 16_383, n);
        assert_eq!(
            kind(&before),
            Tile::Fallback,
            "[{m}, {k}] x [{k}, {n}]: {before}"
        );
        assert_eq!(claimed(m, 16_383, n, 128), Tile::Fallback);
        // The one-column tile's fields below the depth do not depend on `k` from 64 on.
        let at_64 = MatMulTile::of(m, 64, n);
        assert_eq!(
            (
                before.workgroup_size,
                before.cols,
                before.rows,
                before.ahead,
                before.b_pad
            ),
            (
                at_64.workgroup_size,
                at_64.cols,
                at_64.rows,
                at_64.ahead,
                at_64.b_pad
            ),
            "[{m}, {k}] x [{k}, {n}]"
        );
    }
}

/// `(f32 values of workgroup memory, as declared by the module's `Workgroup` variables) * 4`.
fn workgroup_bytes(words: &[u32]) -> u64 {
    let mut sizes: BTreeMap<u32, u64> = BTreeMap::new();
    let mut constants: BTreeMap<u32, u64> = BTreeMap::new();
    let mut pointees: BTreeMap<u32, u32> = BTreeMap::new();
    let mut total = 0;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        assert!(count > 0, "a zero-length instruction at {at}");
        let op = words[at] & 0xffff;
        let w = &words[at + 1..at + count];
        match op {
            OP_TYPE_FLOAT => {
                sizes.insert(w[0], u64::from(w[1] / 8));
            }
            OP_TYPE_VECTOR => {
                let size = sizes.get(&w[1]).copied().unwrap_or(0) * u64::from(w[2]);
                sizes.insert(w[0], size);
            }
            OP_CONSTANT => {
                constants.insert(w[1], u64::from(w[2]));
            }
            OP_TYPE_ARRAY => {
                let size = sizes.get(&w[1]).copied().unwrap_or(0) * constants[&w[2]];
                sizes.insert(w[0], size);
            }
            OP_TYPE_POINTER if w[1] == STORAGE_WORKGROUP => {
                pointees.insert(w[0], w[2]);
            }
            OP_VARIABLE if w[2] == STORAGE_WORKGROUP => {
                total += sizes[&pointees[&w[0]]];
            }
            _ => {}
        }
        at += count;
    }
    total
}

/// The matmul kernel of `[m, k] x [k, n]` with the tile `MatMulTile::of` gives it.
fn matmul(m: u64, k: u64, n: u64, epilogue: Option<Epilogue>, transposed: [bool; 2]) -> Kernel {
    matmul_at(m, k, n, epilogue, transposed, MatMulTile::of(m, k, n))
}

fn matmul_at(
    m: u64,
    k: u64,
    n: u64,
    epilogue: Option<Epilogue>,
    transposed: [bool; 2],
    tile: MatMulTile,
) -> Kernel {
    let mut operands = vec![
        Operand::Buffer(TensorType::f32(&[m, k])),
        Operand::Buffer(TensorType::f32(&[k, n])),
    ];
    if epilogue == Some(Epilogue::ReluGrad) {
        operands.push(Operand::Buffer(TensorType::f32(&[m, n])));
    }
    let mut kernel = Kernel {
        id: KernelModuleId([0; 16]),
        kind: KernelKind::MatMul,
        op: "MatMul",
        dtype: DType::F32,
        operands,
        result: TensorType::f32(&[m, n]),
        reduces: None,
        epilogue,
        transposed,
        softmax_stats: None,
        workgroup_size: tile.workgroup_size,
        workgroups: [
            u32::try_from(tile.workgroups(m, n)).expect("one dimension"),
            1,
            1,
        ],
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

#[test]
fn adv_lk_every_selected_shape_stages_within_13312_bytes() {
    let mut state = 0x5_7a6e_u64;
    let mut shapes: Vec<(u64, u64, u64)> = BOUNDARIES.iter().map(|&(s, _)| s).collect();
    shapes.extend([
        (256, 65_536, 256),
        (256, 65_536, 1024),
        (1024, 65_536, 256),
        (127, 16_391, 2047),
        (33, 16_391, 1025),
        (1, 1 << 18, 4096),
        (8, 1 << 18, 8192),
        (64, 1_000_000, 2048),
    ]);
    for _ in 0..40 {
        shapes.push((
            dim(&mut state, 12),
            16_384 + splitmix(&mut state) % 1_000_000,
            dim(&mut state, 13),
        ));
    }
    let (full, min) = (profile(), min_profile());
    let mut emitted = 0;
    for (m, k, n) in shapes {
        let tile = MatMulTile::of(m, k, n);
        if !matches!(kind(&tile), Tile::LongK | Tile::Narrow) {
            continue;
        }
        // The emitter binds operands of at most u32::MAX elements.
        if [m * k, k * n, m * n]
            .iter()
            .any(|&e| e > u64::from(u32::MAX))
        {
            continue;
        }
        for flags in FLAGS {
            let case = format!("[{m}, {k}] x [{k}, {n}] {flags:?} {tile}");
            let layout = tile.layout(flags);
            assert!(layout.shared_bytes() <= 13_312, "{case}: {layout:?}");
            assert_eq!(tile.layout_within(flags, 16_384), layout, "{case}");
            for p in [&full, &min] {
                let module = emit_module(&matmul(m, k, n, None, flags), p)
                    .unwrap_or_else(|e| panic!("{case}: {e}"));
                let bytes = workgroup_bytes(&module.spirv);
                assert!(bytes <= 13_312, "{case}: the module declares {bytes} bytes");
                assert_eq!(bytes, u64::from(layout.shared_bytes()), "{case}");
            }
            if emitted < 24 {
                let module = emit_module(&matmul(m, k, n, None, flags), &min).expect("emits");
                validate(&module.spirv).unwrap_or_else(|e| panic!("{case}: {e}"));
            }
            emitted += 1;
        }
    }
    assert!(emitted >= 40, "{emitted} modules");
}

/// The workgroup memory addresses one staging step of operand `a` (`is_a`) or `b` of a
/// long-`k` tile stores to, per invocation, as `Shell::matmul` computes them: an operand
/// contiguous across its lines (`a` transposed, `b` as is) by the fixed-line rule, invocation
/// `t` taking line `t % across` at `k` index `t / across + step * size / across`; any other by
/// `Staging` element `e = step * size + t`, `along = e % depth`, `across = e / depth`
/// (`run` = depth, along first). Both blocks are `k`-major (`vector_layout`).
fn long_k_stores(tile: &MatMulTile, flags: [bool; 2], is_a: bool) -> Vec<Vec<u32>> {
    let layout = tile.layout(flags);
    let size = tile.workgroup_size;
    let depth = tile.depth;
    let across = if is_a {
        tile.rows_per_workgroup()
    } else {
        tile.cols
    };
    let total = across * depth;
    let transposed = flags[usize::from(!is_a)];
    let contiguous_across = if is_a { transposed } else { !transposed };
    let place = |along: u32, line: u32| {
        if is_a {
            along * layout.a_stride + line
        } else {
            layout.b_offset + along * layout.b_stride + line
        }
    };
    let fixed = contiguous_across && size.is_multiple_of(across) && total.is_multiple_of(size);
    let mut steps = Vec::new();
    for step in 0..total.div_ceil(size) {
        let mut stores = Vec::new();
        for t in 0..size {
            if fixed {
                let along = t / across + step * size / across;
                stores.push(place(along, t % across));
            } else {
                let e = step * size + t;
                if e >= total {
                    continue;
                }
                let (along, line) = if contiguous_across {
                    // `run` = across: along = e / across, line = e % across.
                    (e / across, e % across)
                } else {
                    (e % depth, e / depth)
                };
                stores.push(place(along, line));
            }
        }
        steps.push(stores);
    }
    steps
}

/// The most distinct addresses one of 32 banks takes over the stores of one subgroup of 32
/// consecutive invocations, over every step.
fn worst_bank(steps: &[Vec<u32>]) -> u32 {
    steps
        .iter()
        .flat_map(|stores| stores.chunks(32))
        .map(|subgroup| {
            let mut banks: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
            for &at in subgroup {
                banks.entry(at % 32).or_default().insert(at);
            }
            banks.values().map(|s| s.len() as u32).max().unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

/// Every staged cell of the operand is stored exactly once per tile, inside the block.
fn covers_the_block(tile: &MatMulTile, flags: [bool; 2], is_a: bool, steps: &[Vec<u32>]) {
    let layout = tile.layout(flags);
    let across = if is_a {
        tile.rows_per_workgroup()
    } else {
        tile.cols
    };
    let mut want = BTreeSet::new();
    for along in 0..tile.depth {
        for line in 0..across {
            want.insert(if is_a {
                along * layout.a_stride + line
            } else {
                layout.b_offset + along * layout.b_stride + line
            });
        }
    }
    let got: Vec<u32> = steps.iter().flatten().copied().collect();
    let unique: BTreeSet<u32> = got.iter().copied().collect();
    assert_eq!(
        got.len(),
        unique.len(),
        "{tile} a {is_a}: a cell stored twice"
    );
    assert_eq!(unique, want, "{tile} a {is_a}: the stores are the block");
}

fn long_k_tiles() -> Vec<(u64, u64, u64)> {
    vec![
        (256, 65_536, 1024),
        (256, 65_536, 256),
        (1024, 65_536, 256),
        (127, 16_391, 2047),
        (33, 16_391, 1025),
    ]
}

/// The brief's case: the fixed-line staging of `a` transposed and `b` as is (the decider's
/// weight gradients) stores one address per bank per subgroup at long `k`.
#[test]
fn adv_lk_fixed_line_staging_stores_without_bank_conflicts() {
    for (m, k, n) in long_k_tiles() {
        let tile = MatMulTile::of(m, k, n);
        assert!(matches!(kind(&tile), Tile::LongK | Tile::Narrow), "{tile}");
        for flags in FLAGS {
            for is_a in [true, false] {
                let transposed = flags[usize::from(!is_a)];
                if is_a != transposed {
                    continue;
                }
                let steps = long_k_stores(&tile, flags, is_a);
                covers_the_block(&tile, flags, is_a, &steps);
                let worst = worst_bank(&steps);
                assert_eq!(
                    worst, 1,
                    "[{m}, {k}] x [{k}, {n}] {tile} {flags:?} a {is_a}"
                );
            }
        }
    }
}

/// The other two operands of a long-`k` tile, `a` read as is and `b` read transposed, are
/// staged by the general rule, 2 lines by 16 values of `k` per subgroup store, into the
/// `k`-major layout whose `b` stride is padded only for depths of 2 to 8: every subgroup store
/// should take one address per bank, as the square tile's padded layout and the block of 256
/// do for these operands. Finding LK-2: a transposed `b` at depth 16 stages 16 addresses in
/// one bank (ir/kernel.rs, `MatMulTile::vector_layout`), reached only by an edited BERT-tiny at
/// vocabulary 30522, whose long-`k` matmul with `b` transposed still ran in 0.417 of the time
/// of the tile it replaced in the adversary's measurement.
#[test]
#[ignore = "finding LK-2, INFEASIBLE: a transposed b at depth 16 stages 16 addresses in one bank; reached only by an edited BERT-tiny at vocabulary 30522, still faster than the replaced tile (ratio 0.417), so no measured shape gains from a fix"]
fn adv_lk_a_as_is_and_b_transposed_stage_without_bank_conflicts() {
    let mut found = Vec::new();
    for (m, k, n) in long_k_tiles() {
        let tile = MatMulTile::of(m, k, n);
        for flags in FLAGS {
            for is_a in [true, false] {
                let transposed = flags[usize::from(!is_a)];
                if is_a == transposed {
                    continue;
                }
                let steps = long_k_stores(&tile, flags, is_a);
                covers_the_block(&tile, flags, is_a, &steps);
                let worst = worst_bank(&steps);
                if worst != 1 {
                    found.push(format!(
                        "[{m}, {k}] x [{k}, {n}] {:?} {flags:?} {}: {worst} addresses in one bank",
                        kind(&tile),
                        if is_a { "a as is" } else { "b transposed" }
                    ));
                }
            }
        }
    }
    assert!(found.is_empty(), "{found:#?}");
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn base_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/adversary_long_k_base_modules.txt")
}

fn sweep(batch: u64, hidden: u64) -> String {
    format!(
        "model SweepMlp {{
    param w1: Tensor<f32, [784, {hidden}]> = xavier_uniform();
    param w2: Tensor<f32, [{hidden}, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [{batch}, 784]>)
        -> Tensor<f32, [{batch}, 10]> {{
        return matmul(relu(matmul(x, w1)), w2);
    }}
}}

train Fit {{
    model = SweepMlp;
    input x: Tensor<f32, [{batch}, 784]>;
    target y: Tensor<f32, [{batch}, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 220;
    seed = 42;
    device = vulkan(0);
}}
"
    )
}

fn words_hash(words: &[u32]) -> String {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    blake3::hash(&bytes).to_hex().to_string()
}

/// `label -> blake3 of the words` of every matmul module of the example plans (with and
/// without the RTX 3090 profile, and under fast math) and of sampled synthetic matmuls.
fn module_hashes() -> BTreeMap<String, String> {
    let mut sources = Vec::new();
    for example in ["mnist_mlp", "bert_tiny", "transformer_block"] {
        let source = std::fs::read_to_string(repo_root().join(format!("examples/{example}.mspec")))
            .expect("the example reads");
        sources.push((example.to_owned(), source));
    }
    for (batch, hidden) in [(1000, 1024), (8192, 1024), (8192, 4096)] {
        sources.push((format!("sweep_b{batch}_h{hidden}"), sweep(batch, hidden)));
    }
    let full = profile();
    let mut out = BTreeMap::new();
    for (name, source) in &sources {
        for with_profile in [false, true] {
            let artifact = if with_profile {
                compile_artifact_for(name, source, Backend::Vulkan, Some(&full))
            } else {
                compile_artifact(name, source, Backend::Vulkan)
            }
            .unwrap_or_else(|d| panic!("{name} compiles: {d:?}"));
            for kernel in &artifact.execution.kernels {
                if kernel.kind != KernelKind::MatMul {
                    continue;
                }
                let signature = kernel.signature();
                if with_profile {
                    for fast in [false, true] {
                        let module = emit_module_with(kernel, &full, fast)
                            .unwrap_or_else(|e| panic!("{signature}: {e}"));
                        out.insert(
                            format!("profile fast {fast} {signature}"),
                            words_hash(&module.spirv),
                        );
                    }
                } else {
                    let words = emit(kernel).unwrap_or_else(|e| panic!("{signature}: {e}"));
                    out.insert(format!("plan {signature}"), words_hash(&words));
                }
            }
        }
    }
    // Synthetic shapes: every `k` below 16384 at sampled `[m, n]`, and `k` at and above 16384
    // where the rule keeps the old tile (`n` under 32, too few blocks, the square and wide
    // tiles).
    let mut state = 0x0bad_5eed_u64;
    let mut shapes = vec![
        (256, 16_383, 1024),
        (256, 16_383, 256),
        (4096, 16_383, 33),
        (1, 16_383, 4096),
        (8129, 16_383, 32),
        (4096, 65_536, 31),
        (224, 65_536, 288),
        (32, 65_536, 64),
        (1, 16_384, 2016),
        (2016, 16_384, 32),
        (1024, 65_536, 1024),
        (784, 65_536, 4096),
        (2048, 20_000, 2048),
    ];
    for _ in 0..120 {
        let k = 1 + splitmix(&mut state) % 16_383;
        shapes.push((dim(&mut state, 13), k, dim(&mut state, 13)));
    }
    for (i, &(m, k, n)) in shapes.iter().enumerate() {
        let flags = FLAGS[i % 4];
        let epilogue = match i % 5 {
            0 => Some(Epilogue::Relu {
                writes_pre_activation: false,
            }),
            1 => Some(Epilogue::Relu {
                writes_pre_activation: true,
            }),
            2 => Some(Epilogue::ReluGrad),
            _ => None,
        };
        let kernel = matmul(m, k, n, epilogue, flags);
        let signature = kernel.signature();
        let words = emit(&kernel).unwrap_or_else(|e| panic!("{signature}: {e}"));
        out.insert(format!("synthetic {signature}"), words_hash(&words));
        let tile = MatMulTile::for_profile(m, k, n, flags, &full);
        let kernel = matmul_at(m, k, n, epilogue, flags, tile);
        let signature = kernel.signature();
        let module = emit_module(&kernel, &full).unwrap_or_else(|e| panic!("{signature}: {e}"));
        out.insert(
            format!("synthetic profile {signature}"),
            words_hash(&module.spirv),
        );
    }
    out
}

/// Acceptance: every shape with `k` < 16384 keeps its tile and its module. The fixture holds
/// the hashes the base commit 2fe753ab gives for the same labels.
#[test]
fn adv_lk_matmul_modules_below_k_16384_are_the_words_of_the_base_commit() {
    let now = module_hashes();
    if std::env::var(RECORD_ENV).as_deref() == Ok("1") {
        let mut lines = String::new();
        for (label, hash) in &now {
            writeln!(lines, "{hash} {label}").expect("a string");
        }
        std::fs::write(base_fixture(), lines).expect("the fixture is written");
        eprintln!("recorded {} modules", now.len());
        return;
    }
    let recorded = std::fs::read_to_string(base_fixture()).expect("the base fixture reads");
    let before: BTreeMap<&str, &str> = recorded
        .lines()
        .map(|line| {
            let (hash, label) = line.split_once(' ').expect("hash and label");
            (label, hash)
        })
        .collect();
    let mut wrong = Vec::new();
    let mut long = Vec::new();
    let mut same = 0;
    for (label, hash) in &now {
        match before.get(label.as_str()) {
            Some(was) if *was == hash => same += 1,
            Some(was) => wrong.push(format!("{label}: {hash}, base {was}")),
            None => long.push(label.clone()),
        }
    }
    eprintln!(
        "{same} modules as at the base commit, {} differ, {} labels new",
        wrong.len(),
        long.len()
    );
    assert!(same >= 300, "{same} compared");
    assert!(long.is_empty(), "labels the base did not have: {long:#?}");
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Reach probe, prints only: the matmul kernels of BERT-tiny at the 30,522-token vocabulary of
/// Turc et al. and 1,024 tokens a step that take a long-`k` tile, with their transposed flags.
#[test]
#[ignore = "probe: prints the long-k matmuls of a BERT-tiny at vocabulary 30522"]
fn adv_lk_probe_long_k_matmuls_of_a_bert_at_vocab_30522() {
    let source = std::fs::read_to_string(repo_root().join("examples/bert_tiny.mspec"))
        .expect("the example reads");
    for (tokens, rows) in [("4096", "32"), ("1024", "8")] {
        let source = source
            .replace("8192", "30522")
            .replace("4096", tokens)
            .replace("[32, 128]", &format!("[{rows}, 128]"));
        let artifact = match compile_artifact("bert30522", &source, Backend::Vulkan) {
            Ok(a) => a,
            Err(d) => {
                eprintln!("tokens {tokens}: does not compile: {d:?}");
                continue;
            }
        };
        for kernel in &artifact.execution.kernels {
            if let Some(tile) = kernel.matmul_tile() {
                let (m, k, n) = Kernel::matmul_shape(&kernel.operands).expect("a matmul");
                if k >= 16_384 {
                    eprintln!(
                        "tokens {tokens}: [{m}, {k}] x [{k}, {n}] {:?} transposed {:?}",
                        kind(&tile),
                        kernel.transposed
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Device cases.

fn guard(i: usize) -> u32 {
    0x5afe_0000 | i as u32
}

fn guarded(values: &[f32]) -> Vec<f32> {
    let mut words = values.to_vec();
    words.extend((0..GUARD_WORDS).map(|i| f32::from_bits(guard(i))));
    words
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// Values in [-1, 1), every seventh a zero of alternating sign.
fn values(seed: u64, count: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|i| {
            let z = splitmix(&mut state);
            match i % 14 {
                3 => 0.0,
                10 => -0.0,
                _ => ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0,
            }
        })
        .collect()
}

fn transpose(values: &[f32], rows: u64, cols: u64) -> Vec<f32> {
    let (rows, cols) = (rows as usize, cols as usize);
    let mut out = vec![0.0; values.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    out
}

/// Runs `kernel` once over `inputs`; returns each written view whole, guard words included.
fn run(
    device: &Device<'_>,
    kernel: &Kernel,
    spirv: &[u32],
    inputs: &[&[f32]],
    count: usize,
) -> Vec<Vec<f32>> {
    let uploads: Vec<Vec<f32>> = inputs
        .iter()
        .map(|values| guarded(values))
        .chain((0..kernel.written()).map(|_| guarded(&vec![f32::from_bits(UNWRITTEN); count])))
        .collect();
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            device
                .device_local_buffer(bytes_of(v.len()))
                .expect("a buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            let mut b = device.staging_buffer(bytes_of(v.len())).expect("staging");
            b.write_f32(v).expect("written");
            b
        })
        .collect();
    let pipeline = device.kernel_pipeline(kernel, spirv).expect("a pipeline");
    let mut recording = device.record().expect("a recording");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("upload");
    }
    recording.barrier(Barrier::TransferToCompute);
    let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
    recording
        .dispatch_kernel(&pipeline, kernel, &bound)
        .expect("dispatch");
    recording.barrier(Barrier::ComputeToTransfer);
    for at in inputs.len()..buffers.len() {
        recording
            .copy(&buffers[at], &staging[at])
            .expect("download");
    }
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("finishes");
    staging[inputs.len()..]
        .iter_mut()
        .map(|b| b.read_f32().expect("reads"))
        .collect()
}

fn open() -> (Vulkan, DeviceProfile) {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("probed");
    let profile = candidates[0].profile.clone();
    (vulkan, profile)
}

/// How a device element is compared with the CPU's.
#[derive(Clone, Copy)]
enum Compare {
    /// Bit for bit; a NaN matches any NaN (the payload is not specified by either side).
    Bits,
    /// Within `abs` of the CPU's value, for subnormal operands on a device that may flush them.
    Within(f32),
}

struct Case {
    label: String,
    m: u64,
    k: u64,
    n: u64,
    flags: [bool; 2],
    epilogue: Option<Epilogue>,
    a: Vec<f32>,
    b: Vec<f32>,
    x: Option<Vec<f32>>,
    compare: Compare,
}

fn epilogue_cycle(i: usize) -> Option<Epilogue> {
    match i % 4 {
        1 => Some(Epilogue::Relu {
            writes_pre_activation: false,
        }),
        2 => Some(Epilogue::Relu {
            writes_pre_activation: true,
        }),
        3 => Some(Epilogue::ReluGrad),
        _ => None,
    }
}

fn plain_case(m: u64, k: u64, n: u64, flags: [bool; 2], epilogue: Option<Epilogue>) -> Case {
    let a = values(0x6164_7631 ^ (m << 40) ^ (k << 20) ^ n, m * k);
    let b = values(0x6164_7632 ^ (n << 40) ^ (k << 20) ^ m, k * n);
    let x = (epilogue == Some(Epilogue::ReluGrad)).then(|| values(0x6164_7633 ^ m ^ n, m * n));
    Case {
        label: format!("[{m}, {k}] x [{k}, {n}] {flags:?} {epilogue:?}"),
        m,
        k,
        n,
        flags,
        epilogue,
        a,
        b,
        x,
        compare: Compare::Bits,
    }
}

/// The CPU AVX2 kernel's outputs of `case`: the sum, then the epilogue's views.
fn want(case: &Case) -> Vec<Vec<f32>> {
    assert!(cpu::Kernel::Avx2.is_available(), "the CPU has AVX2 and FMA");
    let sum = cpu::matmul(
        cpu::Kernel::Avx2,
        &case.a,
        Lhs::RowMajor,
        &case.b,
        case.m as usize,
        case.k as usize,
        case.n as usize,
    )
    .expect("the AVX2 kernel runs");
    let relu = |v: &[f32]| v.iter().map(|&s| if s > 0.0 { s } else { 0.0 }).collect();
    match case.epilogue {
        None => vec![sum],
        Some(Epilogue::Relu {
            writes_pre_activation: false,
        }) => vec![relu(&sum)],
        Some(Epilogue::Relu {
            writes_pre_activation: true,
        }) => {
            let r = relu(&sum);
            vec![sum, r]
        }
        Some(Epilogue::ReluGrad) => {
            let x = case.x.as_ref().expect("x");
            vec![
                sum.iter()
                    .zip(x)
                    .map(|(&s, &x)| if x > 0.0 { s } else { 0.0 })
                    .collect(),
            ]
        }
    }
}

fn agrees(got: f32, want: f32, how: Compare) -> bool {
    if want.is_nan() {
        return got.is_nan();
    }
    match how {
        Compare::Bits => got.to_bits() == want.to_bits(),
        Compare::Within(abs) => got == want || (got - want).abs() <= abs,
    }
}

/// Runs every case on device 0; returns one line per wrong case.
fn check(cases: &[Case]) -> Vec<String> {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("negotiates");
    let device = vulkan.create_device(0, &negotiated).expect("opens");
    let mut wrong = Vec::new();
    let mut compared = 0u64;
    let mut tiles: BTreeMap<String, u32> = BTreeMap::new();
    for case in cases {
        let (m, k, n) = (case.m, case.k, case.n);
        let kernel = matmul(m, k, n, case.epilogue, case.flags);
        let tile = kernel.matmul_tile().expect("a tile");
        *tiles.entry(format!("{:?}", kind(&tile))).or_default() += 1;
        let module = emit_module(&kernel, &profile).expect("emits");
        let a = if case.flags[0] {
            transpose(&case.a, m, k)
        } else {
            case.a.clone()
        };
        let b = if case.flags[1] {
            transpose(&case.b, k, n)
        } else {
            case.b.clone()
        };
        let mut inputs: Vec<&[f32]> = vec![&a, &b];
        if let Some(x) = &case.x {
            inputs.push(x);
        }
        let count = (m * n) as usize;
        let got = run(&device, &kernel, &module.spirv, &inputs, count);
        let want = want(case);
        let label = format!("{} {:?}", case.label, kind(&tile));
        assert_eq!(got.len(), want.len(), "{label}");
        for (view, (got, want)) in got.iter().zip(&want).enumerate() {
            compared += count as u64;
            let differ: Vec<usize> = (0..count)
                .filter(|&i| !agrees(got[i], want[i], case.compare))
                .collect();
            if let Some(&i) = differ.first() {
                wrong.push(format!(
                    "{label} view {view}: {} of {count} differ, first [{}, {}] {:?} ({:#010x}) against the CPU's {:?} ({:#010x})",
                    differ.len(),
                    i as u64 / n,
                    i as u64 % n,
                    got[i],
                    got[i].to_bits(),
                    want[i],
                    want[i].to_bits()
                ));
            }
            let unwritten = (0..count)
                .filter(|&i| got[i].to_bits() == UNWRITTEN)
                .count();
            if unwritten > 0 {
                wrong.push(format!("{label} view {view}: {unwritten} unwritten"));
            }
            let changed = (0..GUARD_WORDS)
                .filter(|&i| got[count + i].to_bits() != guard(i))
                .count();
            if changed > 0 {
                wrong.push(format!(
                    "{label} view {view}: {changed} guard words changed"
                ));
            }
        }
        for message in vulkan.take_validation_messages() {
            wrong.push(format!("{label}: {message}"));
        }
    }
    eprintln!(
        "{} cases, {compared} elements compared with the CPU Avx2 kernel, tiles {tiles:?}, validation {}",
        cases.len(),
        if vulkan.validation_enabled() {
            "on"
        } else {
            "off"
        }
    );
    wrong
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adv_lk_boundary_shapes_and_k_tails_are_the_cpu_bits() {
    let mut cases = Vec::new();
    for (i, ((m, k, n), _)) in BOUNDARIES.iter().enumerate() {
        let flags = if m * k > 64_000_000 {
            [true, false]
        } else {
            FLAGS[i % 4]
        };
        cases.push(plain_case(*m, *k, *n, flags, epilogue_cycle(i)));
    }
    // Tails of 1 to 15 values of `k` after 1024 to 2047 full tiles, every count modulo 4.
    let ks = [
        16_384, 16_385, 16_399, 16_400, 16_415, 16_431, 16_447, 16_448, 16_463, 32_767,
    ];
    let mut i = 0;
    for (m, n) in [(128, 2048), (32, 2048)] {
        for k in ks {
            for flags in FLAGS {
                cases.push(plain_case(m, k, n, flags, epilogue_cycle(i)));
                i += 1;
            }
        }
    }
    // BERT-tiny at vocabulary 30522 and 1024 tokens a step: `[1024, 30522] x [30522, 128]`.
    cases.push(plain_case(1024, 30_522, 128, [false, true], None));
    // Partial row and column blocks with every epilogue.
    for (m, k, n) in [(127, 16_391, 2047), (33, 16_391, 1025)] {
        for flags in FLAGS {
            for e in 0..4 {
                cases.push(plain_case(m, k, n, flags, epilogue_cycle(e)));
            }
        }
    }
    let wrong = check(&cases);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// NaN, infinities, an overflow mid-sum, and sums of signed zeros only.
fn special_case(m: u64, k: u64, n: u64, flags: [bool; 2]) -> Case {
    let mut case = plain_case(m, k, n, flags, None);
    let (mu, ku, nu) = (m as usize, k as usize, n as usize);
    let a = &mut case.a;
    let b = &mut case.b;
    a[3 * ku] = f32::NAN;
    a[(mu - 1) * ku + ku - 1] = f32::NAN;
    b[16 * nu + 5] = f32::INFINITY;
    b[(ku - 1) * nu + nu - 1] = f32::NEG_INFINITY;
    b[(ku / 2) * nu + 9] = f32::INFINITY;
    b[(ku / 2 + 1) * nu + 9] = f32::NEG_INFINITY;
    // Row 10 by column 7: 3e38 + 3e38 overflows before -3e38 comes; any other order of the
    // first three products stays finite.
    for (p, v) in [(0, 3e38f32), (1, 3e38), (2, -3e38)] {
        a[10 * ku + p] = v;
        b[p * nu + 7] = 1.0;
    }
    // Row 20 positive by column 30 of -0: every product -0, the sum -0.
    for p in 0..ku {
        let v = a[20 * ku + p].abs();
        a[20 * ku + p] = if v == 0.0 || v.is_nan() { 0.5 } else { v };
        b[p * nu + 30] = -0.0;
    }
    // Row 21 of -0 by any column: products -0 or +0 by the sign of `b`.
    for p in 0..ku {
        a[21 * ku + p] = -0.0;
    }
    // Row 22 at column 31: 1e8, then 1, then -1e8 at the first, a middle and the last tile.
    for (p, v) in [(0, 1e8f32), (ku / 2, 1.0), (ku - 1, -1e8)] {
        a[22 * ku + p] = v;
        b[p * nu + 31] = 1.0;
    }
    case.label = format!("specials {}", case.label);
    case
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adv_lk_nan_infinity_signed_zero_and_overflow_are_the_cpu_bits() {
    let mut cases = Vec::new();
    for (m, k, n) in [(127, 16_400, 2047), (33, 16_400, 1025), (33, 16_391, 1025)] {
        for flags in FLAGS {
            cases.push(special_case(m, k, n, flags));
        }
    }
    let wrong = check(&cases);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adv_lk_subnormal_operands_are_the_cpu_bits_where_the_device_preserves_them() {
    let (_, profile) = open();
    let preserves = profile.shader_denorm_preserve_f32 == Some(true);
    let mut cases = Vec::new();
    for (m, k, n) in [(127, 16_400, 2047), (33, 16_391, 1025)] {
        for flags in [[true, false], [false, false]] {
            let mut case = plain_case(m, k, n, flags, None);
            for (i, v) in case.a.iter_mut().enumerate() {
                if i % 5 == 1 {
                    *v *= 1e-39;
                }
            }
            for (i, v) in case.b.iter_mut().enumerate() {
                if i % 7 == 2 {
                    *v *= 1e-38;
                }
            }
            case.compare = if preserves {
                Compare::Bits
            } else {
                Compare::Within(k as f32 * f32::MIN_POSITIVE * 2.0)
            };
            case.label = format!("subnormals {}", case.label);
            cases.push(case);
        }
    }
    eprintln!(
        "shader_denorm_preserve_f32 {:?}",
        profile.shader_denorm_preserve_f32
    );
    let wrong = check(&cases);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

// ---------------------------------------------------------------------------------------
// Timing.

/// Pairs of a shape just past a boundary (a long-`k` tile) and the same `[m, n]` at
/// `k` = 16383, the tile it replaced.
const TIMED: [((u64, u64), [bool; 2]); 22] = [
    ((256, 1024), [true, false]),
    ((256, 256), [true, false]),
    ((4096, 32), [true, false]),
    ((4096, 33), [true, false]),
    ((1, 4064), [true, false]),
    ((1, 4096), [true, false]),
    ((1, 2048), [true, false]),
    ((8129, 32), [true, false]),
    ((2048, 32), [true, false]),
    ((8, 4096), [true, false]),
    ((1, 4096), [false, false]),
    ((16, 4096), [true, false]),
    ((32, 4096), [true, false]),
    ((64, 4096), [true, false]),
    ((16, 2048), [true, false]),
    ((32, 2048), [true, false]),
    ((33, 4096), [true, false]),
    ((33, 2048), [true, false]),
    ((256, 1024), [false, false]),
    ((256, 1024), [false, true]),
    ((256, 256), [false, true]),
    // BERT-tiny at vocabulary 30522 and 1024 tokens: the gradient of its hidden rows.
    ((1024, 128), [false, true]),
];

const PAIRS: usize = 9;
const REPS: u32 = 10;

struct Timed<'d> {
    kernel: Kernel,
    pipeline: modelspec_runtime::vulkan::pipeline::Pipeline<'d>,
    buffers: Vec<Buffer<'d>>,
}

fn prepare<'d>(
    device: &'d Device<'_>,
    profile: &DeviceProfile,
    m: u64,
    k: u64,
    n: u64,
    flags: [bool; 2],
) -> Timed<'d> {
    let kernel = matmul(m, k, n, None, flags);
    let module = emit_module(&kernel, profile).expect("emits");
    let pipeline = device
        .kernel_pipeline(&kernel, &module.spirv)
        .expect("pipeline");
    let sizes = [m * k, k * n, m * n];
    let mut buffers = Vec::new();
    let mut staging = Vec::new();
    for (i, &count) in sizes.iter().enumerate() {
        let data = values(0x7469_6d65 + i as u64, count);
        let mut s = device
            .staging_buffer(bytes_of(data.len()))
            .expect("staging");
        s.write_f32(&data).expect("written");
        buffers.push(
            device
                .device_local_buffer(bytes_of(data.len()))
                .expect("buffer"),
        );
        staging.push(s);
    }
    let mut recording = device.record().expect("recording");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("upload");
    }
    recording.barrier(Barrier::TransferToCompute);
    let recorded = recording.finish().expect("ends");
    device
        .submit(recorded)
        .expect("submit")
        .wait()
        .expect("wait");
    Timed {
        kernel,
        pipeline,
        buffers,
    }
}

/// Microseconds per dispatch over `REPS` dispatches separated by compute barriers.
fn time(device: &Device<'_>, timed: &Timed<'_>, period_ns: f64) -> f64 {
    let pool = device.timestamp_pool(2).expect("a pool");
    let mut recording = device.record().expect("recording");
    recording.reset_timestamps(&pool);
    recording.timestamp(&pool, 0, true).expect("stamp");
    let bound: Vec<&Buffer<'_>> = timed.buffers.iter().collect();
    for _ in 0..REPS {
        recording
            .dispatch_kernel(&timed.pipeline, &timed.kernel, &bound)
            .expect("dispatch");
        recording.barrier(Barrier::ComputeToCompute);
    }
    recording.timestamp(&pool, 1, true).expect("stamp");
    let recorded = recording.finish().expect("ends");
    device
        .submit(recorded)
        .expect("submit")
        .wait()
        .expect("wait");
    let stamps = pool.read().expect("stamps");
    (stamps[1] - stamps[0]) as f64 * period_ns / 1000.0 / f64::from(REPS)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v[v.len() / 2]
}

/// Each long-`k` tile is not slower than the tile it replaced on a shape just past each
/// boundary: median of `PAIRS` alternating pairs, the order flipped each pair, with 3 %
/// allowed for noise.
#[test]
#[ignore = "needs a Vulkan device; run alone under the GPU lock"]
fn adv_lk_each_long_k_tile_is_not_slower_than_the_tile_it_replaced() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("negotiates");
    let device = vulkan.create_device(0, &negotiated).expect("opens");
    let period_ns = f64::from(device.clock().expect("timestamps").period_ns);
    let mut slower = Vec::new();
    for ((m, n), flags) in TIMED {
        let new = prepare(&device, &profile, m, 16_384, n, flags);
        let old = prepare(&device, &profile, m, 16_383, n, flags);
        let new_tile = new.kernel.matmul_tile().expect("tile");
        let old_tile = old.kernel.matmul_tile().expect("tile");
        if m < 32 {
            // Finding LK-1: fewer than 32 rows keep the tile they had, so nothing was replaced.
            assert_eq!(new_tile, old_tile, "[{m}, 16384] x [16384, {n}]");
            eprintln!("[{m}, 16384] x [16384, {n}] {flags:?}: keeps {new_tile}");
            continue;
        }
        assert!(
            matches!(kind(&new_tile), Tile::LongK | Tile::Narrow),
            "{new_tile}"
        );
        assert_eq!(kind(&old_tile), Tile::Fallback, "{old_tile}");
        // The pair's own warm-up: 1 s of both kernels, so the clocks are up after the host
        // prepared the operands (the GPU drops to P3 to P8 while it waits).
        let warm = std::time::Instant::now();
        while warm.elapsed() < std::time::Duration::from_secs(1) {
            time(&device, &new, period_ns);
            time(&device, &old, period_ns);
        }
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for pair in 0..PAIRS {
            if pair % 2 == 0 {
                a.push(time(&device, &new, period_ns));
                b.push(time(&device, &old, period_ns));
            } else {
                b.push(time(&device, &old, period_ns));
                a.push(time(&device, &new, period_ns));
            }
        }
        let (a, b) = (median(a), median(b));
        let line = format!(
            "[{m}, 16384] x [16384, {n}] {flags:?}: {:?} {a:.1} us, replaced tile (k 16383, rows {} depth {}) {b:.1} us, ratio {:.3}",
            kind(&new_tile),
            old_tile.rows,
            old_tile.depth,
            a / b
        );
        eprintln!("{line}");
        if a > b * 1.03 {
            slower.push(line);
        }
    }
    assert!(slower.is_empty(), "{slower:#?}");
}
