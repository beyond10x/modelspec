//! Adversary cases for story:matmul-tile-from-device-invocation-limit, without a device: the
//! SPIR-V of the 256-invocation block and of the padded staging.
//!
//! - The block module of a shape has the arithmetic of the wide module of the same shape, op for
//!   op (opcode and the kind of each operand), every `OpFMul` and `OpFAdd` `NoContraction`,
//!   `LocalSize` 256, and a workgroup array of the length worked out by hand for its flags (in
//!   `f32` values; the block declares it as `vec4`, and a component of a `vec4` read from it
//!   counts as a read from workgroup memory); it passes `spirv-val`.
//! - The square tile reading `a` transposed, emitted for a device with 49,152 bytes of workgroup
//!   memory (padded) and with 16,384 (the padding dropped): the same arithmetic, and the array
//!   lengths by hand, 4,608 and 4,096 values.
//! - ADV-T256-3: the module bytes of that kernel differ between the two profiles while its cache
//!   key's `ir_hash` does not, though the loss sum, whose module also takes a value from the
//!   profile's workgroup memory, records it (` chunk <n>`) in its `ir_hash`; ignored with the id.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Kernel, KernelKind, KernelModuleId, MatMulTile, Operand};
use modelspec_core::ir::tensor::{DType, TensorType};
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::spirv::validate::validate;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

const OP_EXECUTION_MODE: u32 = 16;
const MODE_LOCAL_SIZE: u32 = 17;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_DECORATE: u32 = 71;
const OP_COMPOSITE_EXTRACT: u32 = 81;
const OP_F_ADD: u32 = 129;
const OP_F_MUL: u32 = 133;
const NO_CONTRACTION: u32 = 42;
const WORKGROUP: u32 = 4;
const FUNCTION: u32 = 7;

const FLAGS: [[bool; 2]; 4] = [[false, false], [true, false], [false, true], [true, true]];

fn profile(shared: u32) -> DeviceProfile {
    let mut profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile");
    profile.max_shared_memory_bytes = shared;
    profile
}

/// A matmul kernel of the tile `of_invocations` gives at `limit`.
fn matmul(m: u64, k: u64, n: u64, limit: u32, transposed: [bool; 2]) -> Kernel {
    let tile = MatMulTile::of_invocations(m, k, n, limit);
    let mut kernel = Kernel {
        id: KernelModuleId([0; 16]),
        kind: KernelKind::MatMul,
        op: "MatMul",
        dtype: DType::F32,
        operands: vec![
            Operand::Buffer(TensorType::f32(&[m, k])),
            Operand::Buffer(TensorType::f32(&[k, n])),
        ],
        result: TensorType::f32(&[m, n]),
        reduces: None,
        epilogue: None,
        transposed,
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

fn instructions(module: &[u32]) -> Vec<(u32, &[u32])> {
    let mut out = Vec::new();
    let mut at = 5;
    while at < module.len() {
        let count = (module[at] >> 16) as usize;
        assert!(count > 0, "a zero-length instruction at word {at}");
        out.push((module[at] & 0xffff, &module[at + 1..at + count]));
        at += count;
    }
    out
}

#[derive(Debug, PartialEq, Eq)]
struct Shape {
    /// Per `OpFMul` / `OpFAdd` in module order: the opcode and the kind of each operand; a
    /// component of a vector loaded from workgroup memory is a load from workgroup memory.
    arithmetic: Vec<(u32, String, String)>,
    /// The lengths of the workgroup memory arrays in `f32` values: a `vec4` element is 4.
    shared: Vec<u32>,
    /// The arithmetic results not decorated `NoContraction`.
    contracted: usize,
    /// `LocalSize` x.
    local_x: u32,
}

fn shape_of(module: &[u32]) -> Shape {
    let instructions = instructions(module);
    let mut defined: BTreeMap<u32, (u32, &[u32])> = BTreeMap::new();
    let mut variables: BTreeMap<u32, u32> = BTreeMap::new();
    let mut constants: BTreeMap<u32, u32> = BTreeMap::new();
    let mut pointers: BTreeMap<u32, u32> = BTreeMap::new();
    let mut arrays: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut vectors: BTreeMap<u32, u32> = BTreeMap::new();
    let mut no_contraction = BTreeSet::new();
    let mut local_x = 0;
    for &(op, words) in &instructions {
        match op {
            OP_EXECUTION_MODE if words[1] == MODE_LOCAL_SIZE => local_x = words[2],
            OP_VARIABLE => {
                variables.insert(words[1], words[2]);
                defined.insert(words[1], (op, words));
            }
            OP_CONSTANT => {
                constants.insert(words[1], words[2]);
            }
            OP_TYPE_POINTER => {
                pointers.insert(words[0], words[2]);
            }
            OP_TYPE_VECTOR => {
                vectors.insert(words[0], words[2]);
            }
            OP_TYPE_ARRAY => {
                arrays.insert(words[0], (words[1], words[2]));
            }
            OP_DECORATE if words[1] == NO_CONTRACTION => {
                no_contraction.insert(words[0]);
            }
            OP_LOAD | OP_ACCESS_CHAIN | OP_COMPOSITE_EXTRACT | OP_F_ADD | OP_F_MUL => {
                defined.insert(words[1], (op, words));
            }
            _ => {}
        }
    }
    let kind = |id: u32| -> String {
        let id = match defined.get(&id) {
            Some((OP_COMPOSITE_EXTRACT, words)) => words[2],
            _ => id,
        };
        match defined.get(&id) {
            Some((OP_F_MUL, _)) => "fmul".to_owned(),
            Some((OP_F_ADD, _)) => "fadd".to_owned(),
            Some((OP_LOAD, words)) => match defined.get(&words[2]) {
                Some((OP_VARIABLE, v)) if v[2] == FUNCTION => "load function".to_owned(),
                Some((OP_ACCESS_CHAIN, chain)) => match variables.get(&chain[2]) {
                    Some(&WORKGROUP) => "load workgroup".to_owned(),
                    _ => "load other chain".to_owned(),
                },
                _ => "load other".to_owned(),
            },
            _ => "other".to_owned(),
        }
    };
    let mut arithmetic = Vec::new();
    let mut contracted = 0;
    for &(op, words) in &instructions {
        if op == OP_F_ADD || op == OP_F_MUL {
            if !no_contraction.contains(&words[1]) {
                contracted += 1;
            }
            arithmetic.push((op, kind(words[2]), kind(words[3])));
        }
    }
    let mut shared = Vec::new();
    for &(op, words) in &instructions {
        if op == OP_VARIABLE && words[2] == WORKGROUP {
            let (element, length) = arrays[&pointers[&words[0]]];
            shared.push(constants[&length] * vectors.get(&element).copied().unwrap_or(1));
        }
    }
    Shape {
        arithmetic,
        shared,
        contracted,
        local_x,
    }
}

/// The workgroup memory values of the block, by hand, for each entry of [`FLAGS`], with 2
/// buffers, staged k-major for vector reads: `a` 8 lines of 132 (136 read transposed), `b` 8
/// lines of 128 (132 padded).
const BLOCK_VALUES: [u32; 4] = [
    2 * (8 * 132 + 8 * 128),
    2 * (8 * 136 + 8 * 128),
    2 * (8 * 132 + 8 * 132),
    2 * (8 * 136 + 8 * 132),
];

#[test]
fn the_block_module_has_the_arithmetic_of_the_wide_module_and_the_memory_of_its_layout() {
    let profile = profile(49_152);
    let mut wrong = Vec::new();
    let mut compared = 0;
    for (m, k, n) in [
        (8192, 784, 4096),
        (1153, 65, 1921),
        (8191, 783, 255),
        (2049, 1023, 1000),
        (129, 64, 8191),
        (16_385, 72, 129),
        (8192, 8192, 4096),
    ] {
        let block_tile = MatMulTile::of_invocations(m, k, n, 1024);
        assert_eq!(block_tile.workgroup_size, 256, "[{m}, {k}] x [{k}, {n}]");
        assert_eq!(MatMulTile::of(m, k, n).cols_per_invocation, 8);
        for (flags, values) in FLAGS.iter().zip(BLOCK_VALUES) {
            let case = format!("[{m}, {k}] x [{k}, {n}] {flags:?}");
            let block = emit_module(&matmul(m, k, n, 1024, *flags), &profile)
                .unwrap_or_else(|e| panic!("{case}: block: {e}"));
            let wide = emit_module(&matmul(m, k, n, 128, *flags), &profile)
                .unwrap_or_else(|e| panic!("{case}: wide: {e}"));
            if let Err(e) = validate(&block.spirv) {
                wrong.push(format!("{case}: spirv-val: {e}"));
            }
            let (got, want) = (shape_of(&block.spirv), shape_of(&wide.spirv));
            compared += 1;
            if got.local_x != 256 || want.local_x != 128 {
                wrong.push(format!(
                    "{case}: LocalSize {} / {}",
                    got.local_x, want.local_x
                ));
            }
            if got.contracted != 0 {
                wrong.push(format!("{case}: {} contracted", got.contracted));
            }
            let expected_values = if k > 8 { values } else { values / 2 };
            if got.shared != [expected_values] {
                wrong.push(format!(
                    "{case}: workgroup memory {:?}, by hand {expected_values}",
                    got.shared
                ));
            }
            if got.arithmetic != want.arithmetic {
                let first = got
                    .arithmetic
                    .iter()
                    .zip(&want.arithmetic)
                    .position(|(g, w)| g != w);
                wrong.push(format!(
                    "{case}: arithmetic {} against {} ops of the wide tile, first difference {first:?}",
                    got.arithmetic.len(),
                    want.arithmetic.len()
                ));
            }
            let products = got
                .arithmetic
                .iter()
                .filter(|(op, a, b)| {
                    *op == OP_F_MUL && a == "load workgroup" && b == "load workgroup"
                })
                .count();
            if products < 64 {
                wrong.push(format!("{case}: {products} products from workgroup memory"));
            }
        }
    }
    eprintln!("{compared} block modules compared with the wide module");
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn the_padded_and_the_unpadded_square_module_have_the_same_arithmetic() {
    let (m, k, n) = (8193, 300, 100);
    let mut wrong = Vec::new();
    for (flags, padded, plain) in [
        (
            [true, false],
            2 * (64 * 20 + 16 * 64),
            2 * (64 * 16 + 16 * 64),
        ),
        (
            [false, true],
            2 * (64 * 16 + 16 * 66),
            2 * (64 * 16 + 16 * 64),
        ),
        (
            [true, true],
            2 * (64 * 20 + 16 * 66),
            2 * (64 * 16 + 16 * 64),
        ),
    ] {
        let kernel = matmul(m, k, n, 128, flags);
        let tile = kernel.matmul_tile().expect("a tile");
        assert_eq!(
            (tile.cols, tile.cols_per_invocation, tile.depth),
            (64, 4, 16)
        );
        let wide = emit_module(&kernel, &profile(49_152)).expect("emits at 48 KiB");
        let narrow = emit_module(&kernel, &profile(16_384)).expect("emits at 16 KiB");
        let as_is = emit_module(&matmul(m, k, n, 128, [false, false]), &profile(49_152))
            .expect("emits as is");
        let (w, s, base) = (
            shape_of(&wide.spirv),
            shape_of(&narrow.spirv),
            shape_of(&as_is.spirv),
        );
        for (name, shape, values) in [("48 KiB", &w, padded), ("16 KiB", &s, plain)] {
            if shape.shared != [values] {
                wrong.push(format!(
                    "{flags:?} {name}: {:?}, by hand {values}",
                    shape.shared
                ));
            }
            if shape.arithmetic != base.arithmetic || shape.contracted != 0 {
                wrong.push(format!(
                    "{flags:?} {name}: arithmetic differs from the module as is"
                ));
            }
            if let Err(e) = validate(if name == "48 KiB" {
                &wide.spirv
            } else {
                &narrow.spirv
            }) {
                wrong.push(format!("{flags:?} {name}: spirv-val {e}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn a_module_that_takes_a_value_from_the_profiles_workgroup_memory_records_it_in_its_ir_hash() {
    let mut wrong = Vec::new();
    // The precedent: the loss sum's chunk.
    let loss = common::loss_sum_kernel(100_000);
    let (l48, l16) = (
        emit_module(&loss, &profile(49_152)).expect("loss at 48 KiB"),
        emit_module(&loss, &profile(16_384)).expect("loss at 16 KiB"),
    );
    assert_ne!(
        l48.spirv, l16.spirv,
        "the loss sum module takes its chunk from the profile"
    );
    assert_ne!(
        l48.cache_key.ir_hash, l16.cache_key.ir_hash,
        "and its ir_hash records the chunk"
    );
    // The matmul reading `a` transposed at the square tile.
    let kernel = matmul(8193, 300, 100, 128, [true, false]);
    let (m48, m16) = (
        emit_module(&kernel, &profile(49_152)).expect("matmul at 48 KiB"),
        emit_module(&kernel, &profile(16_384)).expect("matmul at 16 KiB"),
    );
    if m48.spirv != m16.spirv && m48.cache_key == m16.cache_key {
        wrong.push(format!(
            "two modules of {} words and {} words under one cache key, ir_hash {}",
            m48.spirv.len(),
            m16.spirv.len(),
            m48.cache_key.ir_hash.0
        ));
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
