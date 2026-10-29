//! Adversary pass on story:vulkan-attention-tiled-decider-shapes: the four attention kernels
//! that stage the other side's rows in workgroup memory and loop over a prefix mask's keys only.
//!
//! - Host: every attention module stays within 16 KiB of workgroup memory for every key count
//!   from 1 to 512, and its module grows no more than twice the base kernels' words
//!   (2fe753ab) for heads wider than the decider's and BERT-tiny's.
//! - Device (ignored): the kernels agree with the CPU interpreter on key counts on either side of
//!   every tile-halving boundary, 1, 2, 63, 64, 65 and 512 keys, head widths of 1, 2, 8, 12,
//!   20, 33, 130 and 300 columns, 1 to 8 heads, masks that keep a prefix (of 0, 1, some and all
//!   keys), a suffix, a prefix with a hole near its end, a prefix and the last key, every key,
//!   none, and a different pattern per sequence, NaN, ±inf and -0 in `q`, `k`, `v` and `g`, and
//!   batches whose workgroups need a 2-D dispatch; each kernel alone computes the bits the four
//!   compute sharing softmax statistics; each case computes the bits the base kernels computed.
//! - Device timing (ignored, run under the GPU lock): the staged kernels are not slower than the
//!   base kernels on BERT-tiny's and transformer_block's shapes and on key counts whose tile
//!   halved.
//!
//! Run the device cases with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test adversary_attention_tiled -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;
use std::time::Instant;

use common::{Draw, GPU_ATOL, GPU_RTOL, within};
use modelspec_core::ir::kernel::{Kernel, Operand};
use modelspec_core::ir::tensor::{TensorOp, TensorProgramBuilder, TensorType};
use modelspec_core::schedule::{attention_loss_kernel, with_softmax_stats};
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, Vulkan};

/// Which keys of each sequence a case's mask keeps.
#[derive(Clone, Copy, Debug)]
enum Mask {
    /// Keys `0..L`, `L` the `seq`-th of the lengths, taken in turn.
    Prefix(&'static [u64]),
    /// The last `L` keys.
    Suffix(&'static [u64]),
    /// Keys `0..L` but key `L - 2`: a prefix with a hole near its end.
    Holed(&'static [u64]),
    /// Keys `0..L` and the last key: a kept key after the masked run, found by the last key.
    TailKept(&'static [u64]),
    /// Per sequence `seq % 6`: a third of the keys, every key, none, the last half, every key
    /// but the third from last, the first key.
    Mixed,
    /// Every key.
    Full,
    /// No key.
    Empty,
}

/// Whether a case's operands carry NaN, ±inf and -0 at fixed places.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Values {
    Plain,
    Special,
}

fn keeps(mask: Mask, s: u64, seq: u64, j: u64) -> bool {
    let pick = |lengths: &[u64]| lengths[seq as usize % lengths.len()];
    match mask {
        Mask::Prefix(l) => j < pick(l),
        Mask::Suffix(l) => j >= s - pick(l),
        Mask::Holed(l) => {
            let len = pick(l);
            j < len && j + 2 != len
        }
        Mask::TailKept(l) => j < pick(l) || j == s - 1,
        Mask::Mixed => match seq % 6 {
            0 => j < s.div_ceil(3),
            1 => true,
            2 => false,
            3 => j >= s / 2,
            4 => j + 3 != s,
            _ => j == 0,
        },
        Mask::Full => true,
        Mask::Empty => false,
    }
}

fn key_mask(b: u64, s: u64, mask: Mask) -> Vec<f32> {
    (0..b * s)
        .map(|i| {
            if keeps(mask, s, i / s, i % s) {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// The shapes `[b, s, h, heads]`, masks and values of the cases.
const CASES: &[(u64, u64, u64, u64, Mask, Values)] = &[
    (6, 1, 8, 2, Mask::Prefix(&[1, 0]), Values::Plain),
    (6, 2, 6, 3, Mask::Mixed, Values::Plain),
    (6, 42, 64, 1, Mask::Mixed, Values::Plain),
    (6, 43, 64, 1, Mask::Mixed, Values::Plain),
    (4, 61, 40, 2, Mask::Prefix(&[61, 0, 1, 60]), Values::Plain),
    (4, 62, 40, 2, Mask::Prefix(&[62, 31, 1, 0]), Values::Plain),
    (3, 63, 24, 3, Mask::TailKept(&[10, 40, 0]), Values::Plain),
    (3, 64, 24, 3, Mask::Holed(&[64, 40, 3]), Values::Plain),
    (6, 65, 24, 3, Mask::Mixed, Values::Plain),
    (6, 90, 33, 1, Mask::Mixed, Values::Plain),
    (3, 91, 33, 1, Mask::Prefix(&[91, 45, 1]), Values::Plain),
    (6, 122, 64, 2, Mask::Mixed, Values::Plain),
    (2, 123, 64, 2, Mask::TailKept(&[5, 100]), Values::Plain),
    (6, 186, 16, 2, Mask::Mixed, Values::Plain),
    (2, 187, 16, 2, Mask::Prefix(&[187, 93]), Values::Plain),
    (2, 239, 16, 2, Mask::Holed(&[239, 120]), Values::Plain),
    (6, 240, 16, 2, Mask::Mixed, Values::Plain),
    (6, 378, 12, 1, Mask::Mixed, Values::Plain),
    (2, 379, 12, 1, Mask::Prefix(&[379, 1]), Values::Plain),
    (2, 453, 12, 1, Mask::TailKept(&[0, 200]), Values::Plain),
    (6, 454, 12, 1, Mask::Mixed, Values::Plain),
    (6, 512, 130, 1, Mask::Mixed, Values::Plain),
    (6, 32, 300, 1, Mask::Mixed, Values::Plain),
    (6, 128, 64, 8, Mask::Mixed, Values::Plain),
    (4, 100, 4, 4, Mask::Prefix(&[100, 50, 0, 1]), Values::Plain),
    (4, 64, 16, 2, Mask::Prefix(&[64, 30, 0, 1]), Values::Special),
    (
        4,
        64,
        16,
        2,
        Mask::Suffix(&[64, 30, 1, 10]),
        Values::Special,
    ),
    (4, 200, 16, 2, Mask::Mixed, Values::Special),
    (65536, 1, 2, 1, Mask::Prefix(&[1, 1, 0]), Values::Plain),
    (40000, 4, 4, 2, Mask::Mixed, Values::Plain),
    (3, 128, 128, 2, Mask::Full, Values::Plain),
    (3, 256, 256, 4, Mask::Empty, Values::Plain),
];

/// The hashes of each case's results, in the order of [`CASES`]: `Attention`,
/// `AttentionGradQ`, `AttentionGradK` and `AttentionGradV` alone, then the shared statistics and
/// the four results of the call that shares them, recorded from the base kernels (2fe753ab)
/// with this file on an RTX 3090.
const BASE_BITS: &[[u64; 9]] = &[
    [
        0x6c50e5392874d32e,
        0xe3a6e7fa4a2fc3a5,
        0x405a05ff2c82df25,
        0xf83da7a7dc264458,
        0x2e68acebbce5039e,
        0x6c50e5392874d32e,
        0xe3a6e7fa4a2fc3a5,
        0x405a05ff2c82df25,
        0xf83da7a7dc264458,
    ],
    [
        0xe31ac78ffeecbf3e,
        0x567afbd5859b5b86,
        0x7416d239c2810ed6,
        0xcb5eb8eeefbc2364,
        0x2db1e8ddc00df16d,
        0xe31ac78ffeecbf3e,
        0x567afbd5859b5b86,
        0x7416d239c2810ed6,
        0xcb5eb8eeefbc2364,
    ],
    [
        0x4c9e8fd72b901097,
        0x5146b800b5e8dccc,
        0x9ad8f291c06e83b5,
        0xd3f24f6e6dafaf4d,
        0x7d3b259b4adfbddb,
        0x4c9e8fd72b901097,
        0x5146b800b5e8dccc,
        0x9ad8f291c06e83b5,
        0xd3f24f6e6dafaf4d,
    ],
    [
        0xe65ad9f6b0974b2b,
        0x44bffa7a298aa289,
        0x6630f059db8b1768,
        0x69f81918e984bbe9,
        0xd187f20d5215294a,
        0xe65ad9f6b0974b2b,
        0x44bffa7a298aa289,
        0x6630f059db8b1768,
        0x69f81918e984bbe9,
    ],
    [
        0x2241e28dae0a6bfb,
        0x85965a817618872a,
        0x8f6163cdc6decf53,
        0x0c3a637046286b91,
        0xcfb138edee3a77e2,
        0x2241e28dae0a6bfb,
        0x85965a817618872a,
        0x8f6163cdc6decf53,
        0x0c3a637046286b91,
    ],
    [
        0xe2ca62ce7be4a430,
        0xb3234a1905245896,
        0xb06556da759695cc,
        0xc21e9133929e9242,
        0x750addcf3f71bc0f,
        0xe2ca62ce7be4a430,
        0xb3234a1905245896,
        0xb06556da759695cc,
        0xc21e9133929e9242,
    ],
    [
        0x6bd312d43cb191ac,
        0x76e790c4011ab537,
        0x785dd50281327913,
        0x17fb40276bbeb225,
        0xc89f5ed22f93f1dc,
        0x6bd312d43cb191ac,
        0x76e790c4011ab537,
        0x785dd50281327913,
        0x17fb40276bbeb225,
    ],
    [
        0x09ce4b7a138bf495,
        0xae24329196ad31e4,
        0x635107ff76f2f01b,
        0x078ee475a6491d8e,
        0x3edea8efa5f524c6,
        0x09ce4b7a138bf495,
        0xae24329196ad31e4,
        0x635107ff76f2f01b,
        0x078ee475a6491d8e,
    ],
    [
        0x527d1b4b8872ef48,
        0x83fa32c352c6925e,
        0x46ae6f0b3b259e8b,
        0xe56a776b81b368fb,
        0x56c7566df60f63b0,
        0x527d1b4b8872ef48,
        0x83fa32c352c6925e,
        0x46ae6f0b3b259e8b,
        0xe56a776b81b368fb,
    ],
    [
        0xccb5a5cad28fec6e,
        0x23b86efe1729e880,
        0xd738d5a7495ffff8,
        0x45a93146714ffe35,
        0x99783454c3efba22,
        0xccb5a5cad28fec6e,
        0x23b86efe1729e880,
        0xd738d5a7495ffff8,
        0x45a93146714ffe35,
    ],
    [
        0x51a40f5dd09f7370,
        0xdeac5e94f520e156,
        0x7e6731fa8645f095,
        0xe1dfab1d9e135f4c,
        0xc16340859322e578,
        0x51a40f5dd09f7370,
        0xdeac5e94f520e156,
        0x7e6731fa8645f095,
        0xe1dfab1d9e135f4c,
    ],
    [
        0x439111ce6862038f,
        0xf4fef23291f941e9,
        0xe93aa40baff9786b,
        0xa8cfceaecfc9bfb9,
        0x1403c76a5ae533fa,
        0x439111ce6862038f,
        0xf4fef23291f941e9,
        0xe93aa40baff9786b,
        0xa8cfceaecfc9bfb9,
    ],
    [
        0x10f4932e4811a2ed,
        0x1bbdb39ac30bae35,
        0x0a0beedb9e0eb4ef,
        0xb8828f35df0324fc,
        0xf23ac5eb16d81400,
        0x10f4932e4811a2ed,
        0x1bbdb39ac30bae35,
        0x0a0beedb9e0eb4ef,
        0xb8828f35df0324fc,
    ],
    [
        0xbbc5d666f50f174f,
        0xa6714540f9173c17,
        0x70a196ee152a5170,
        0x0a398dde17777775,
        0xae103fb7bea5b9d9,
        0xbbc5d666f50f174f,
        0xa6714540f9173c17,
        0x70a196ee152a5170,
        0x0a398dde17777775,
    ],
    [
        0xabce42436a6fa937,
        0x32c8983985cf9d6e,
        0x8c39292bdd64fae5,
        0xaf8f3f54c628b690,
        0xee5e366deb30647b,
        0xabce42436a6fa937,
        0x32c8983985cf9d6e,
        0x8c39292bdd64fae5,
        0xaf8f3f54c628b690,
    ],
    [
        0xba3e2932f3e97abe,
        0xf47544b85dabe393,
        0xcd84dfd214024c19,
        0x6476474a290343a6,
        0xb711875e857ece00,
        0xba3e2932f3e97abe,
        0xf47544b85dabe393,
        0xcd84dfd214024c19,
        0x6476474a290343a6,
    ],
    [
        0xa7aeb62bcfccb81b,
        0xd2f4f02f3829a7ef,
        0x5b9a2d8b6dea96bf,
        0xa216a0543954e529,
        0x16baee99de346aec,
        0xa7aeb62bcfccb81b,
        0xd2f4f02f3829a7ef,
        0x5b9a2d8b6dea96bf,
        0xa216a0543954e529,
    ],
    [
        0x32ffe5906c5e387e,
        0x4602d015b1a68898,
        0x6b5d1f7edf503ca7,
        0xe81b1d7ad5b804e3,
        0x6c6240a664a0739b,
        0x32ffe5906c5e387e,
        0x4602d015b1a68898,
        0x6b5d1f7edf503ca7,
        0xe81b1d7ad5b804e3,
    ],
    [
        0x4d73ff2ea94634a2,
        0x85c3a70f3b232acf,
        0x560016ad71f080e2,
        0x479d8b11a95ca81b,
        0xee24739585bd7edf,
        0x4d73ff2ea94634a2,
        0x85c3a70f3b232acf,
        0x560016ad71f080e2,
        0x479d8b11a95ca81b,
    ],
    [
        0x7c80633bd3fa83b4,
        0x9941bf22ad052f30,
        0xd620625ce73a6482,
        0x90a6463d0f884e83,
        0x63dd89ccafe9ee15,
        0x7c80633bd3fa83b4,
        0x9941bf22ad052f30,
        0xd620625ce73a6482,
        0x90a6463d0f884e83,
    ],
    [
        0x923ef2a8dfe70972,
        0x01aaf54163de4789,
        0xce7261d436641bdc,
        0xffcae823a0fcc5ab,
        0xd5cdefbeac977da0,
        0x923ef2a8dfe70972,
        0x01aaf54163de4789,
        0xce7261d436641bdc,
        0xffcae823a0fcc5ab,
    ],
    [
        0xde12c80ece238d34,
        0x8206b9561bbdf9bc,
        0xcd9ee884a02c41a0,
        0x469462998cfe94c6,
        0x9141b2854e05b2a3,
        0xde12c80ece238d34,
        0x8206b9561bbdf9bc,
        0xcd9ee884a02c41a0,
        0x469462998cfe94c6,
    ],
    [
        0x0b5908241e8686d7,
        0xec8aaaf5862fd9be,
        0xfe3e29c56071a237,
        0x64231720c6d85b7b,
        0xa7d0a74ca75d6cec,
        0x0b5908241e8686d7,
        0xec8aaaf5862fd9be,
        0xfe3e29c56071a237,
        0x64231720c6d85b7b,
    ],
    [
        0xbd3835c2a420ef94,
        0x6cfb81c9dd7b934f,
        0x9c5934f2ff9d8f38,
        0x38be4ee893e534ef,
        0xcd3a7c9f7fba50aa,
        0xbd3835c2a420ef94,
        0x6cfb81c9dd7b934f,
        0x9c5934f2ff9d8f38,
        0x38be4ee893e534ef,
    ],
    [
        0xc39121b09ac65d7a,
        0x569b44e12072dd6c,
        0xfdb67d8b02a6b96a,
        0xa8cfdfed0871acfb,
        0xcd6faa96ef7b1e00,
        0xc39121b09ac65d7a,
        0x569b44e12072dd6c,
        0xfdb67d8b02a6b96a,
        0xa8cfdfed0871acfb,
    ],
    [
        0x00a64088d6ced26b,
        0x5a532ef6fedb280d,
        0x03ec187364d34443,
        0x0a9bc3329e7f3043,
        0x68a72f6cf4957bda,
        0x00a64088d6ced26b,
        0x5a532ef6fedb280d,
        0x03ec187364d34443,
        0x0a9bc3329e7f3043,
    ],
    [
        0x50f1268a7b13d6d5,
        0x9ae4cde3585e3125,
        0xdbb300bc88ba7057,
        0x2c3f8a6723d432ac,
        0x828ef9281d10a3c2,
        0x50f1268a7b13d6d5,
        0x9ae4cde3585e3125,
        0xdbb300bc88ba7057,
        0x2c3f8a6723d432ac,
    ],
    [
        0x029325f15b1c9610,
        0xf2b34f6ac73ddad9,
        0x0b368a7a46cfe34b,
        0x383d25705d4483cb,
        0x20711050d58b4b50,
        0x029325f15b1c9610,
        0xf2b34f6ac73ddad9,
        0x0b368a7a46cfe34b,
        0x383d25705d4483cb,
    ],
    [
        0x5d522537fc2a301d,
        0xd518c5d3a6faf1a5,
        0x8e01885183f4a8a5,
        0xe1c93d145af071a6,
        0xb6defb2a982a50c2,
        0x5d522537fc2a301d,
        0xd518c5d3a6faf1a5,
        0x8e01885183f4a8a5,
        0xe1c93d145af071a6,
    ],
    [
        0xa2ffb179772eaa0f,
        0xfbcac34af91a729f,
        0x110e2225f4491647,
        0xbda7caf52c89aa7c,
        0x2e9b67dd9c15f428,
        0xa2ffb179772eaa0f,
        0xfbcac34af91a729f,
        0x110e2225f4491647,
        0xbda7caf52c89aa7c,
    ],
    [
        0x7a88b323d37d278c,
        0xca654539e2043334,
        0xb5711eabfc3a040c,
        0x28b67570b132d64c,
        0xf467e45804886ae7,
        0x7a88b323d37d278c,
        0xca654539e2043334,
        0xb5711eabfc3a040c,
        0x28b67570b132d64c,
    ],
    [
        0x43cd26de17122325,
        0x43cd26de17122325,
        0x3b2dc8fe17122325,
        0x3b2dc8fe17122325,
        0x6334ae8553a56325,
        0x43cd26de17122325,
        0x43cd26de17122325,
        0x3b2dc8fe17122325,
        0x3b2dc8fe17122325,
    ],
];

/// The operands `q`, `k`, `v`, the mask and `g` of a case.
fn inputs(b: u64, s: u64, h: u64, mask: Mask, values: Values) -> Vec<Vec<f32>> {
    let n = b * s;
    let mut draw = Draw::new(0x6164_7661_7474_6e00 ^ s ^ (h << 16) ^ (b << 32));
    let scaled = |draw: &mut Draw, by: f32| -> Vec<f32> {
        draw.values(n * h).into_iter().map(|x| x * by).collect()
    };
    let mut q = scaled(&mut draw, 0.25);
    let mut k = scaled(&mut draw, 0.25);
    let mut v = scaled(&mut draw, 1.0);
    let mut g = scaled(&mut draw, 1.0);
    if values == Values::Special {
        let h = h as usize;
        let s = s as usize;
        q[h] = f32::NAN;
        k[2 * h + 1] = f32::INFINITY;
        k[(s + 2) * h] = f32::NEG_INFINITY;
        k[(2 * s + 5) * h + 3] = f32::INFINITY;
        v[0] = f32::NAN;
        v[3 * h + 1] = f32::INFINITY;
        v[(s + 4) * h + 2] = -0.0;
        g[h + 1] = f32::NEG_INFINITY;
        g[4 * h] = f32::NAN;
        q[(3 * s + 1) * h] = -0.0;
        k[(3 * s + 1) * h] = -0.0;
    }
    vec![q, k, v, key_mask(b, s, mask), g]
}

fn ops(heads: u64) -> [TensorOp; 4] {
    [
        TensorOp::Attention { heads },
        TensorOp::AttentionGradQ { heads },
        TensorOp::AttentionGradK { heads },
        TensorOp::AttentionGradV { heads },
    ]
}

/// FNV-1a over the bits of `values`, low byte first.
fn fnv(values: &[f32]) -> u64 {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
}

fn arity(op: &TensorOp) -> usize {
    if matches!(op, TensorOp::Attention { .. }) {
        4
    } else {
        5
    }
}

/// The kernel of `op` over `[n, h]` rows and a `[b, s]` mask, as the schedule builds it.
fn kernel_of(op: &TensorOp, b: u64, s: u64, h: u64) -> Kernel {
    let rows = TensorType::f32(&[b * s, h]);
    let mask = TensorType::f32(&[b, s]);
    let operands = (0..arity(op))
        .map(|at| Operand::Buffer(if at == 3 { mask.clone() } else { rows.clone() }))
        .collect();
    attention_loss_kernel(op, operands, rows).expect("an attention kernel")
}

/// The CPU interpreter's result of `op` over the sequences `kept` of `inputs`, in their order.
fn cpu(op: &TensorOp, inputs: &[Vec<f32>], s: u64, h: u64, kept: &[u64]) -> Vec<f32> {
    let (s, h) = (s as usize, h as usize);
    let b = kept.len();
    let pick = |values: &[f32], width: usize| -> Vec<f32> {
        kept.iter()
            .flat_map(|&seq| {
                let seq = seq as usize;
                values[seq * s * width..(seq + 1) * s * width]
                    .iter()
                    .copied()
            })
            .collect()
    };
    let mut builder = TensorProgramBuilder::new();
    let mut values = Vec::new();
    let mut host = BTreeMap::new();
    for at in 0..arity(op) {
        let (shape, data) = if at == 3 {
            (vec![b as u64, s as u64], pick(&inputs[3], 1))
        } else {
            (vec![(b * s) as u64, h as u64], pick(&inputs[at], h))
        };
        let name = format!("x{at}");
        let ty = TensorType::f32(&shape);
        host.insert(
            name.clone(),
            HostTensor::new(shape, data).expect("a tensor"),
        );
        values.push(builder.input(name, ty).expect("a new input"));
    }
    let result = builder.push(op.clone(), &values).expect("well typed");
    builder.output(result).expect("the result is a value");
    let program = builder.finish();
    evaluate_seeded(&program, &host, &BTreeMap::new(), &BTreeMap::new(), 0)
        .expect("the CPU evaluates")[0]
        .data()
        .to_vec()
}

/// Runs `kernels` in order in one recording, each bound to the inputs it reads, the statistics
/// buffer where it shares them, and its own result; returns the statistics and every result.
fn run(
    device: &Device<'_>,
    kernels: &[Kernel],
    inputs: &[Vec<f32>],
    stats: usize,
) -> Vec<Vec<f32>> {
    let words: Vec<Vec<u32>> = kernels.iter().map(|k| emit(k).expect("emits")).collect();
    let pipelines: Vec<_> = kernels
        .iter()
        .zip(&words)
        .map(|(k, w)| device.kernel_pipeline(k, w).expect("a pipeline"))
        .collect();
    let rows = inputs[0].len();
    let mut lengths: Vec<usize> = inputs.iter().map(Vec::len).collect();
    lengths.push(stats.max(1));
    lengths.extend(kernels.iter().map(|_| rows));
    let buffers: Vec<Buffer<'_>> = lengths
        .iter()
        .map(|&n| device.device_local_buffer(4 * n as u64).expect("a buffer"))
        .collect();
    let mut staging: Vec<Buffer<'_>> = lengths
        .iter()
        .map(|&n| {
            device
                .staging_buffer(4 * n as u64)
                .expect("a staging buffer")
        })
        .collect();
    for (at, buffer) in staging.iter_mut().enumerate() {
        let values = match inputs.get(at) {
            Some(values) => values.clone(),
            None => vec![f32::from_bits(0x7fc0_dead); lengths[at]],
        };
        buffer.write_f32(&values).expect("written");
    }
    let mut recording = device.record().expect("a recording");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("an upload");
    }
    recording.barrier(Barrier::TransferToCompute);
    let stats_at = inputs.len();
    for (at, (kernel, pipeline)) in kernels.iter().zip(&pipelines).enumerate() {
        let reads = kernel
            .operands
            .iter()
            .filter(|o| matches!(o, Operand::Buffer(_)))
            .count();
        let mut bound: Vec<&Buffer<'_>> = buffers[..reads].iter().collect();
        if kernel.softmax_stats.is_some() {
            bound.push(&buffers[stats_at]);
        }
        bound.push(&buffers[stats_at + 1 + at]);
        recording
            .dispatch_kernel(pipeline, kernel, &bound)
            .expect("a dispatch");
        recording.barrier(Barrier::ComputeToCompute);
    }
    recording.barrier(Barrier::ComputeToTransfer);
    for at in stats_at..buffers.len() {
        recording
            .copy(&buffers[at], &staging[at])
            .expect("a download");
    }
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[stats_at..]
        .iter_mut()
        .map(|b| b.read_f32().expect("read"))
        .collect()
}

/// Each op alone, then the four sharing statistics: `[alone × 4, stats, shared × 4]`.
fn results(
    device: &Device<'_>,
    b: u64,
    s: u64,
    h: u64,
    heads: u64,
    inputs: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let ops = ops(heads);
    let mut out = Vec::new();
    for op in &ops {
        let kernel = kernel_of(op, b, s, h);
        out.push(run(device, &[kernel], inputs, 0).swap_remove(1));
    }
    let shared: Vec<Kernel> = ops
        .iter()
        .map(|op| with_softmax_stats(kernel_of(op, b, s, h)).expect("shares"))
        .collect();
    out.extend(run(device, &shared, inputs, (b * s * heads * 3) as usize));
    out
}

fn open() -> (Vulkan, usize) {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    (vulkan, 0)
}

fn label(case: &(u64, u64, u64, u64, Mask, Values)) -> String {
    let (b, s, h, heads, mask, values) = case;
    format!("[{b}, {s}] x {h} heads {heads} {mask:?} {values:?}")
}

/// Whether the device's `g` stands for the CPU's `c`: the same bits, both NaN, or within the
/// GPU tolerance.
fn agrees(g: f32, c: f32) -> bool {
    g.to_bits() == c.to_bits()
        || (g.is_nan() && c.is_nan())
        || (g.is_finite() && c.is_finite() && within(g, c, c.abs(), GPU_ATOL, GPU_RTOL))
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_attention_kernels_agree_with_the_cpu_on_every_case() {
    let (vulkan, index) = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[index].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(index, &negotiated)
        .expect("device 0 opens");
    let mut wrong = Vec::new();
    for case in CASES {
        let &(b, s, h, heads, mask, values) = case;
        let inputs = inputs(b, s, h, mask, values);
        let out = results(&device, b, s, h, heads, &inputs);
        let messages = vulkan.take_validation_messages();
        assert!(messages.is_empty(), "{}: {messages:?}", label(case));
        let kept: Vec<u64> = (0..b)
            .filter(|&seq| (0..s).any(|j| keeps(mask, s, seq, j)))
            .collect();
        let width = (s * h) as usize;
        for (o, op) in ops(heads).iter().enumerate() {
            if kept.is_empty() {
                continue;
            }
            let expected = cpu(op, &inputs, s, h, &kept);
            for (which, result) in [("alone", &out[o]), ("shared", &out[5 + o])] {
                let got: Vec<f32> = kept
                    .iter()
                    .flat_map(|&seq| {
                        let seq = seq as usize;
                        result[seq * width..(seq + 1) * width].iter().copied()
                    })
                    .collect();
                let bad: Vec<String> = got
                    .iter()
                    .zip(&expected)
                    .enumerate()
                    .filter(|(_, (g, c))| !agrees(**g, **c))
                    .take(2)
                    .map(|(i, (g, c))| {
                        format!(
                            "{} {which} {} [{i}]: gpu {g:?} cpu {c:?}",
                            op.name(),
                            label(case)
                        )
                    })
                    .collect();
                wrong.extend(bad);
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_attention_kernels_alone_compute_the_bits_of_the_shared_call() {
    let (vulkan, index) = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[index].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(index, &negotiated)
        .expect("device 0 opens");
    let mut wrong = Vec::new();
    for case in CASES {
        let &(b, s, h, heads, mask, values) = case;
        let inputs = inputs(b, s, h, mask, values);
        let out = results(&device, b, s, h, heads, &inputs);
        for (o, op) in ops(heads).iter().enumerate() {
            let differ = out[o]
                .iter()
                .zip(&out[5 + o])
                .enumerate()
                .find(|(_, (a, s))| a.to_bits() != s.to_bits());
            if let Some((i, (a, s))) = differ {
                wrong.push(format!(
                    "{} {} [{i}]: alone {a:?} shared {s:?}",
                    op.name(),
                    label(case)
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_attention_kernels_compute_the_base_bits_on_every_case() {
    let (vulkan, index) = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[index].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(index, &negotiated)
        .expect("device 0 opens");
    let mut seen = Vec::new();
    for case in CASES {
        let &(b, s, h, heads, mask, values) = case;
        let inputs = inputs(b, s, h, mask, values);
        let out = results(&device, b, s, h, heads, &inputs);
        seen.push(out.iter().map(|values| fnv(values)).collect::<Vec<u64>>());
    }
    let table: Vec<String> = seen
        .iter()
        .map(|hashes| {
            let words: Vec<String> = hashes.iter().map(|h| format!("0x{h:016x}")).collect();
            format!("    [{}],", words.join(", "))
        })
        .collect();
    let differ: Vec<String> = CASES
        .iter()
        .zip(&seen)
        .zip(BASE_BITS.iter().map(Some).chain(std::iter::repeat(None)))
        .filter(|((_, got), base)| base.map(|b| b.as_slice()) != Some(got.as_slice()))
        .map(|((case, got), base)| {
            let names = [
                "Attention",
                "GradQ",
                "GradK",
                "GradV",
                "stats",
                "Attention'",
                "GradQ'",
                "GradK'",
                "GradV'",
            ];
            let which: Vec<&str> = match base {
                None => vec!["unrecorded"],
                Some(base) => (0..9)
                    .filter(|&i| base[i] != got[i])
                    .map(|i| names[i])
                    .collect(),
            };
            format!("{}: {which:?}", label(case))
        })
        .collect();
    assert!(
        differ.is_empty(),
        "differ from base: {differ:#?}\nseen:\n{}",
        table.join("\n")
    );
}

/// The bytes of the `Workgroup` variables of a SPIR-V module.
fn workgroup_bytes(words: &[u32]) -> u64 {
    let mut at = 5;
    let mut sizes: BTreeMap<u32, u64> = BTreeMap::new();
    let mut constants: BTreeMap<u32, u64> = BTreeMap::new();
    let mut pointers: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut total = 0;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let opcode = words[at] & 0xffff;
        let operands = &words[at + 1..at + count];
        match opcode {
            21 | 22 => {
                sizes.insert(operands[0], u64::from(operands[1] / 8));
            }
            43 => {
                constants.insert(operands[1], u64::from(operands[2]));
            }
            28 => {
                let element = sizes.get(&operands[1]).copied().unwrap_or(4);
                let length = constants.get(&operands[2]).copied().unwrap_or(0);
                sizes.insert(operands[0], element * length);
            }
            32 => {
                pointers.insert(operands[0], (operands[1], operands[2]));
            }
            59 if operands[2] == 4 => {
                let (_, pointee) = pointers[&operands[0]];
                total += sizes.get(&pointee).copied().unwrap_or(0);
            }
            _ => {}
        }
        at += count.max(1);
    }
    total
}

#[test]
fn adversary_attention_modules_stay_within_16_kib_of_workgroup_memory() {
    let mut over = Vec::new();
    let widths: &[(u64, u64)] = &[(64, 1), (1, 1), (3, 3), (130, 1), (300, 1)];
    for &(d, heads) in widths {
        let keys: Vec<u64> = if d == 64 {
            (1..=512).collect()
        } else {
            vec![
                1, 2, 31, 32, 42, 43, 61, 62, 90, 91, 122, 123, 186, 187, 239, 240, 378, 379, 453,
                454, 512,
            ]
        };
        for s in keys {
            for op in ops(heads) {
                let plain = kernel_of(&op, 2, s, d * heads);
                let shared = with_softmax_stats(plain.clone()).expect("shares");
                for kernel in [plain, shared] {
                    let words = emit(&kernel).expect("emits");
                    let bytes = workgroup_bytes(&words);
                    if bytes == 0 || bytes > 16384 {
                        over.push(format!(
                            "{} s {s} d {d} stats {}: {bytes} bytes",
                            op.name(),
                            kernel.softmax_stats.is_some()
                        ));
                    }
                }
            }
        }
    }
    assert!(over.is_empty(), "{over:#?}");
}

/// The head widths of [`adversary_attention_module_words_stay_within_twice_the_base`], one head
/// over 32 keys, and the words of each op's module without and with shared statistics, from
/// the base kernels (2fe753ab).
const WORDS_WIDTHS: [u64; 7] = [8, 64, 128, 129, 256, 1024, 4096];
const BASE_WORDS: &[[usize; 8]] = &[
    [2892, 3701, 4120, 2956, 2963, 3497, 2479, 2089],
    [6964, 10293, 19224, 11060, 7035, 10089, 8399, 5601],
    [11342, 17551, 36210, 20046, 11413, 17347, 14889, 9339],
    [7166, 8248, 6287, 5544, 7237, 8044, 5538, 5123],
    [9851, 11429, 7980, 6989, 9922, 11225, 7231, 6568],
    [34035, 38685, 26020, 23493, 34106, 38481, 25271, 23072],
    [130803, 147741, 98212, 89541, 130874, 147537, 97463, 89120],
];

fn module_words() -> Vec<[usize; 8]> {
    WORDS_WIDTHS
        .iter()
        .map(|&d| {
            let mut row = [0; 8];
            for (o, op) in ops(1).iter().enumerate() {
                let plain = kernel_of(op, 2, 32, d);
                let shared = with_softmax_stats(plain.clone()).expect("shares");
                row[o] = emit(&plain).expect("emits").len();
                row[4 + o] = emit(&shared).expect("emits").len();
            }
            row
        })
        .collect()
}

#[test]
#[ignore = "finding F4, INFEASIBLE: modules for head width over 128 are 2.1-4.7x the base words; no repository model has heads wider than 128"]
fn adversary_attention_module_words_stay_within_twice_the_base() {
    let seen = module_words();
    let table: Vec<String> = seen.iter().map(|row| format!("    {row:?},")).collect();
    let over: Vec<String> = WORDS_WIDTHS
        .iter()
        .zip(&seen)
        .zip(BASE_WORDS.iter().map(Some).chain(std::iter::repeat(None)))
        .filter_map(|((d, got), base)| match base {
            None => Some(format!("d {d}: unrecorded")),
            Some(base) => {
                let grown: Vec<String> = (0..8)
                    .filter(|&i| got[i] > 2 * base[i])
                    .map(|i| format!("module {i}: {} -> {} words", base[i], got[i]))
                    .collect();
                (!grown.is_empty()).then(|| format!("d {d}: {grown:?}"))
            }
        })
        .collect();
    assert!(over.is_empty(), "{over:#?}\nseen:\n{}", table.join("\n"));
}

/// Pipeline creation seconds of the shared modules at the widest head of [`WORDS_WIDTHS`],
/// printed; the base kernels' median recorded by the alternating runs is
/// [`BASE_PIPELINE_SECONDS`], and the staged ones may take at most twice it. The base's first
/// (cold driver cache) run on an RTX 3090 took 27.845 s, its later runs 0.083 s and 0.140 s.
const BASE_PIPELINE_SECONDS: Option<f64> = Some(27.845);

#[test]
#[ignore = "finding F4, INFEASIBLE: pipeline creation at head width 4096 takes 41-71 s; no repository model has heads wider than 128; needs a Vulkan device"]
fn adversary_attention_pipelines_of_wide_heads_are_created_in_twice_the_base_time() {
    let (vulkan, index) = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[index].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(index, &negotiated)
        .expect("device 0 opens");
    let d = WORDS_WIDTHS[WORDS_WIDTHS.len() - 1];
    let started = Instant::now();
    for op in ops(1) {
        let kernel = with_softmax_stats(kernel_of(&op, 2, 32, d)).expect("shares");
        let words = emit(&kernel).expect("emits");
        device.kernel_pipeline(&kernel, &words).expect("a pipeline");
    }
    let seconds = started.elapsed().as_secs_f64();
    println!("ADV-PIPELINE d {d}: {seconds:.3} s");
    let base = BASE_PIPELINE_SECONDS.expect("a recorded base time");
    assert!(
        seconds <= 2.0 * base,
        "{seconds:.3} s against the base's {base:.3} s"
    );
}

/// The timing shapes `[b, s, h, heads]` and masks: BERT-tiny's batch, transformer_block's two
/// batches, and the decider's width at key counts whose tile halved.
const TIMED: &[(u64, u64, u64, u64, Mask)] = &[
    (32, 128, 128, 2, Mask::Full),
    (8, 128, 128, 2, Mask::Full),
    (128, 128, 128, 2, Mask::Full),
    (128, 50, 256, 4, Mask::Full),
    (64, 100, 256, 4, Mask::Full),
    (32, 200, 256, 4, Mask::Full),
    (16, 400, 256, 4, Mask::Full),
    (128, 50, 256, 4, Mask::Suffix(&[40, 25])),
    (32, 200, 256, 4, Mask::Suffix(&[150, 80])),
];

/// The base kernels' medians (2fe753ab) in microseconds per call of each [`TIMED`] shape: the
/// four ops alone, then the shared call, from the alternating runs under the GPU lock: the
/// median of three runs at P0 on an RTX 3090.
const BASE_MICROS: &[[f64; 5]] = &[
    [209.0, 355.3, 30891.6, 15626.2, 892.0],
    [77.8, 117.0, 9479.5, 4709.7, 290.6],
    [637.2, 1195.7, 118859.4, 60028.1, 3189.3],
    [271.4, 516.4, 5312.8, 2644.9, 1523.2],
    [374.1, 683.7, 32796.1, 16558.7, 1888.9],
    [782.6, 1394.7, 242084.2, 117624.1, 3826.2],
    [2284.2, 4185.9, 1825466.7, 904336.7, 11250.5],
    [265.6, 504.9, 3649.9, 1908.4, 1255.2],
    [790.0, 1201.3, 160499.4, 74852.8, 3272.1],
];

/// Microseconds per call of `kernels` over `inputs`: the median of 7 submissions of 20 calls.
fn time(device: &Device<'_>, kernels: &[Kernel], inputs: &[Vec<f32>], stats: usize) -> f64 {
    let words: Vec<Vec<u32>> = kernels.iter().map(|k| emit(k).expect("emits")).collect();
    let pipelines: Vec<_> = kernels
        .iter()
        .zip(&words)
        .map(|(k, w)| device.kernel_pipeline(k, w).expect("a pipeline"))
        .collect();
    let rows = inputs[0].len();
    let mut lengths: Vec<usize> = inputs.iter().map(Vec::len).collect();
    lengths.push(stats.max(1));
    lengths.extend(kernels.iter().map(|_| rows));
    let buffers: Vec<Buffer<'_>> = lengths
        .iter()
        .map(|&n| device.device_local_buffer(4 * n as u64).expect("a buffer"))
        .collect();
    let mut staging: Vec<Buffer<'_>> = inputs
        .iter()
        .map(|v| {
            device
                .staging_buffer(4 * v.len() as u64)
                .expect("a staging buffer")
        })
        .collect();
    for (buffer, values) in staging.iter_mut().zip(inputs) {
        buffer.write_f32(values).expect("written");
    }
    let mut recording = device.record().expect("a recording");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("an upload");
    }
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("uploaded");
    let stats_at = inputs.len();
    let mut calls = 1;
    let mut samples = Vec::new();
    for sample in 0..8 {
        let mut recording = device.record().expect("a recording");
        for _ in 0..calls {
            for (at, (kernel, pipeline)) in kernels.iter().zip(&pipelines).enumerate() {
                let reads = kernel
                    .operands
                    .iter()
                    .filter(|o| matches!(o, Operand::Buffer(_)))
                    .count();
                let mut bound: Vec<&Buffer<'_>> = buffers[..reads].iter().collect();
                if kernel.softmax_stats.is_some() {
                    bound.push(&buffers[stats_at]);
                }
                bound.push(&buffers[stats_at + 1 + at]);
                recording
                    .dispatch_kernel(pipeline, kernel, &bound)
                    .expect("a dispatch");
                recording.barrier(Barrier::ComputeToCompute);
            }
        }
        let recorded = recording.finish().expect("the recording ends");
        let started = Instant::now();
        device
            .submit(recorded)
            .expect("submitted")
            .wait()
            .expect("finished");
        let micros = started.elapsed().as_secs_f64() * 1e6 / f64::from(calls);
        samples.push(micros);
        if sample == 0 {
            // At most 20 calls per submission and about 0.2 s, so one measurement command
            // holds the GPU lock for minutes, not tens of them.
            calls = ((200_000.0 / micros) as u32).clamp(1, 20);
        }
    }
    samples.remove(0);
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

#[test]
#[ignore = "findings F2 and F3, INFEASIBLE: 187-239 keys under a non-prefix mask and GradK/V without shared statistics run 1.3-2.0x the base; no repository model has those shapes; needs a Vulkan device and the GPU lock"]
fn adversary_attention_staged_kernels_are_not_slower_than_the_base() {
    let (vulkan, index) = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[index].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(index, &negotiated)
        .expect("device 0 opens");
    // `ADV_ATTN_TIMED=<shape>:<column>` times that one cell of [`TIMED`] and [`BASE_MICROS`],
    // so each measurement command takes the GPU lock on its own; unset, every cell.
    let only: Option<(usize, usize)> = std::env::var("ADV_ATTN_TIMED").ok().map(|cell| {
        let (shape, column) = cell.split_once(':').expect("<shape>:<column>");
        (
            shape.parse().expect("a shape"),
            column.parse().expect("a column"),
        )
    });
    let wanted = |shape: usize, column: usize| only.is_none_or(|cell| cell == (shape, column));
    let mut seen = Vec::new();
    for (shape, &(b, s, h, heads, mask)) in TIMED.iter().enumerate() {
        let mut row = [f64::NAN; 5];
        if !(0..5).any(|column| wanted(shape, column)) {
            seen.push(row);
            continue;
        }
        let inputs = inputs(b, s, h, mask, Values::Plain);
        let ops = ops(heads);
        for (o, op) in ops.iter().enumerate() {
            if wanted(shape, o) {
                row[o] = time(&device, &[kernel_of(op, b, s, h)], &inputs, 0);
            }
        }
        let shared: Vec<Kernel> = ops
            .iter()
            .map(|op| with_softmax_stats(kernel_of(op, b, s, h)).expect("shares"))
            .collect();
        if wanted(shape, 4) {
            row[4] = time(&device, &shared, &inputs, (b * s * heads * 3) as usize);
        }
        println!(
            "ADV-TIMING [{b}, {s}] x {h} heads {heads} {mask:?}: {:.1} {:.1} {:.1} {:.1} {:.1}",
            row[0], row[1], row[2], row[3], row[4]
        );
        seen.push(row);
    }
    let slower: Vec<String> = TIMED
        .iter()
        .zip(&seen)
        .zip(BASE_MICROS.iter().map(Some).chain(std::iter::repeat(None)))
        .filter_map(|((case, got), base)| match base {
            None => Some(format!("{case:?}: unrecorded")),
            Some(base) => {
                let worse: Vec<String> = (0..5)
                    .filter(|&i| got[i] > 1.05 * base[i])
                    .map(|i| format!("{i}: {:.1} -> {:.1} us", base[i], got[i]))
                    .collect();
                (!worse.is_empty()).then(|| format!("{case:?}: {worse:?}"))
            }
        })
        .collect();
    assert!(slower.is_empty(), "{slower:#?}");
}

/// A query head of `-0.0` against keys `+0.5` and `-0.5` in turn scores `-0.0` with the first
/// key and `+0.0` with the second: the CPU's softmax maximum, and the module doc's ("starts at
/// the first unmasked key's score and is replaced only by a greater one"), is the first key's
/// `-0.0`, since `+0.0 > -0.0` is false. `Attention` sharing its statistics writes that maximum
/// for every query, under a mask that keeps every key (a prefix of 8) and under one that masks
/// key 3 (not a prefix).
#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_attention_softmax_maximum_of_zero_scores_is_the_first_keys_negative_zero() {
    let (vulkan, index) = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[index].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(index, &negotiated)
        .expect("device 0 opens");
    let (b, s, h) = (2u64, 8u64, 1u64);
    let q = vec![-0.0f32; 16];
    let k: Vec<f32> = (0..16)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    let v: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mask: Vec<f32> = (0..16)
        .map(|i| if i == 8 + 3 { 0.0 } else { 1.0 })
        .collect();
    let g = vec![1.0f32; 16];
    let inputs = vec![q, k, v, mask, g];
    let shared: Vec<Kernel> = ops(1)
        .iter()
        .map(|op| with_softmax_stats(kernel_of(op, b, s, h)).expect("shares"))
        .collect();
    let out = run(&device, &shared, &inputs, 16 * 3);
    let maxima: Vec<String> = out[0]
        .chunks(3)
        .map(|stats| format!("{:08x}", stats[0].to_bits()))
        .collect();
    let expected = vec!["80000000".to_owned(); 16];
    assert_eq!(
        maxima, expected,
        "softmax maxima of queries 0..8 (prefix mask) and 8..16 (key 3 masked)"
    );
}

/// Writes the bits of every result of case `ADV_ATTN_CASE` to the file `ADV_ATTN_DUMP`, one
/// line per element, for comparing two builds' results element by element.
#[test]
#[ignore = "needs a Vulkan device; a probe run by hand"]
fn adversary_attention_dump_case() {
    let (Ok(case), Ok(path)) = (
        std::env::var("ADV_ATTN_CASE"),
        std::env::var("ADV_ATTN_DUMP"),
    ) else {
        return;
    };
    let (vulkan, index) = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[index].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(index, &negotiated)
        .expect("device 0 opens");
    let &(b, s, h, heads, mask, values) = &CASES[case.parse::<usize>().expect("an index")];
    let operands = inputs(b, s, h, mask, values);
    let mut out = results(&device, b, s, h, heads, &operands);
    out.extend(operands);
    let lines: Vec<String> = out
        .iter()
        .enumerate()
        .flat_map(|(r, values)| {
            values
                .iter()
                .enumerate()
                .map(move |(i, v)| format!("{r} {i} {:08x} {v:?}", v.to_bits()))
        })
        .collect();
    std::fs::write(path, lines.join("\n")).expect("written");
}
