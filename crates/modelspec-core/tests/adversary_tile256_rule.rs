//! Adversary cases for story:matmul-tile-from-device-invocation-limit, without a device: the
//! tile rule and its workgroup-memory accounting, read against DESIGN.md § 5.2.
//!
//! - Workgroup memory, literal values: the staged bytes of the block, wide and square tiles for
//!   each flag combination, computed by hand from the layout DESIGN.md § 5.2 states.
//! - Workgroup memory, property: every tile `for_limits` gives for a profile, staged in the
//!   layout the emitter takes (`layout_within`), fits the profile's memory; a block of 256 is
//!   never staged without its padding; a one-column tile never pads `b` twice.

use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{MIN_SHARED_MEMORY_BYTES, MatMulLimits, MatMulTile};

const RTX3090: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../modelspec-runtime/tests/fixtures/rtx3090.device.json"
));

fn limited(invocations: u32, shared_bytes: u32) -> DeviceProfile {
    let mut profile = DeviceProfile::from_json(RTX3090).expect("the fixture is a valid profile");
    profile.max_workgroup_invocations = invocations;
    profile.max_shared_memory_bytes = shared_bytes;
    profile
}

const FLAGS: [[bool; 2]; 4] = [[false, false], [true, false], [false, true], [true, true]];

/// SplitMix64 shapes in `1..=bound`, then the edges of every rule.
fn shapes(count: usize, bound: u64) -> Vec<(u64, u64, u64)> {
    let mut state = 0x7469_6c65_3235_3621u64;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        1 + z % bound
    };
    let mut out: Vec<(u64, u64, u64)> = (0..count).map(|_| (next(), next(), next())).collect();
    let edges = [
        1, 2, 3, 4, 7, 8, 9, 12, 16, 24, 32, 63, 64, 65, 127, 128, 129, 255, 256, 783, 1023, 1025,
        2049, 8191, 8193,
    ];
    for &m in &edges {
        for &k in &edges {
            for &n in &edges {
                out.push((m, k, n));
            }
        }
    }
    out
}

/// `(m, k, n)` of one tile family, and the staged bytes, by hand, for each entry of [`FLAGS`].
///
/// Block (256 invocations, 16 lanes, 128 rows, depth 8, 2 buffers): `a` 128 lines, `b` 8 lines
/// of 128; a padded `a` line is 12, a padded `b` line 128 + 32 / 8 = 132.
/// Wide (128 invocations, 64 rows, depth 8, 2 buffers): `a` 64 lines.
/// Square (128 invocations, 64 rows, 64 columns, depth 16, 2 buffers): a padded `a` line is 20,
/// a padded `b` line 64 + 32 / 16 = 66.
type Literal = ((u64, u64, u64), u32, [u32; 4]);

const LITERAL: [Literal; 3] = [
    (
        (8192, 784, 4096),
        1024,
        [
            2 * (128 * 8 + 8 * 128) * 4,
            2 * (128 * 12 + 8 * 128) * 4,
            2 * (128 * 8 + 8 * 132) * 4,
            2 * (128 * 12 + 8 * 132) * 4,
        ],
    ),
    (
        (8192, 784, 4096),
        128,
        [
            2 * (64 * 8 + 8 * 128) * 4,
            2 * (64 * 12 + 8 * 128) * 4,
            2 * (64 * 8 + 8 * 132) * 4,
            2 * (64 * 12 + 8 * 132) * 4,
        ],
    ),
    (
        (8193, 300, 100),
        128,
        [
            2 * (64 * 16 + 16 * 64) * 4,
            2 * (64 * 20 + 16 * 64) * 4,
            2 * (64 * 16 + 16 * 66) * 4,
            2 * (64 * 20 + 16 * 66) * 4,
        ],
    ),
];

#[test]
fn the_staged_bytes_of_the_block_wide_and_square_tiles_are_those_of_the_layout_by_hand() {
    let mut wrong = Vec::new();
    for ((m, k, n), limit, want) in LITERAL {
        let tile = MatMulTile::of_invocations(m, k, n, limit);
        for (flags, want) in FLAGS.iter().zip(want) {
            let got = tile.layout(*flags).shared_bytes();
            if got != want {
                wrong.push(format!("{tile} {flags:?}: {got} bytes, by hand {want}"));
            }
        }
        if tile.layout([false, false]).shared_bytes() > MIN_SHARED_MEMORY_BYTES {
            wrong.push(format!("{tile}: unpadded over 16 KiB"));
        }
    }
    // The block reading both operands transposed needs 20,736 bytes: a profile of exactly that
    // takes it, one byte fewer takes the wide tile, with no refusal.
    let (m, k, n) = (8192, 784, 4096);
    let both = [true, true];
    let at = MatMulTile::for_profile(m, k, n, both, &limited(1024, 20_736));
    let short = MatMulTile::for_profile(m, k, n, both, &limited(1024, 20_735));
    if at.workgroup_size != 256 {
        wrong.push(format!("20,736 bytes: {at}"));
    }
    if short != MatMulTile::of(m, k, n) {
        wrong.push(format!("20,735 bytes: {short}"));
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn every_tile_a_profile_gets_is_staged_within_its_memory_and_a_block_keeps_its_padding() {
    let mut wrong = Vec::new();
    let mut blocks = 0u64;
    let mut dropped = 0u64;
    for (m, k, n) in shapes(20_000, 9000) {
        for invocations in [128, 255, 256, 1024] {
            for shared in [16_384, 16_640, 18_432, 20_479, 20_480, 20_736, 49_152] {
                let limits = MatMulLimits {
                    invocations,
                    shared_bytes: shared,
                };
                for flags in FLAGS {
                    let tile = MatMulTile::for_limits(m, k, n, flags, limits);
                    let case = format!("[{m}, {k}] x [{k}, {n}] {invocations} {shared} {flags:?}");
                    let staged = tile.layout_within(flags, shared);
                    if staged.shared_bytes() > shared {
                        wrong.push(format!("{case}: {tile} stages {}", staged.shared_bytes()));
                    }
                    if tile.workgroup_size > 128 {
                        blocks += 1;
                        if staged != tile.layout(flags) {
                            wrong.push(format!("{case}: the block drops its padding"));
                        }
                    }
                    if staged != tile.layout(flags) {
                        dropped += 1;
                    }
                    if tile.cols_per_invocation == 1
                        && tile.layout(flags).b_stride != tile.b_stride()
                    {
                        wrong.push(format!("{case}: a one-column b padded twice"));
                    }
                    if tile.cols_per_invocation > 1 && tile.b_pad != 0 {
                        wrong.push(format!("{case}: a register-blocked tile with b_pad"));
                    }
                    if tile.workgroup_size != 128 && tile.workgroup_size != 256 {
                        wrong.push(format!("{case}: {tile}"));
                    }
                    if invocations < 256 && tile != MatMulTile::of(m, k, n) {
                        wrong.push(format!("{case}: {tile} under {invocations}"));
                    }
                }
            }
        }
    }
    eprintln!("{blocks} blocks, {dropped} paddings dropped");
    assert!(
        blocks > 1000 && dropped > 0,
        "{blocks} blocks, {dropped} dropped"
    );
    assert!(
        wrong.is_empty(),
        "{} wrong, first {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(8)]
    );
}
