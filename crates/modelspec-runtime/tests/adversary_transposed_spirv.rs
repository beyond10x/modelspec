//! Adversary cases for story:matmul-reads-transposed-operand, without a device: the module of a
//! matmul that reads an operand transposed against the module of the same shape that reads it
//! as is (DESIGN.md § 5.2: "each value goes to the line and place in its line where the operand
//! read as is puts it in workgroup memory, so the products read the same values, and every
//! product and the order of each sum are those of the stored transpose").
//!
//! For every shape, tile family and flag combination:
//!
//! - the arithmetic of the two modules is the same sequence: every `OpFMul`, `OpFAdd` and
//!   `GLSL.std.450 Fma` (DESIGN.md § 12 ADR-009) in module order, each with the kind of
//!   instruction that defines each operand (a workgroup memory load, a function-variable load, an
//!   `OpFMul`, an `OpFAdd`, an `Fma`), and the workgroup memory loads the products read use the
//!   same address expression shapes;
//! - every `OpFMul`, `OpFAdd` and `Fma` result is decorated `NoContraction` in both;
//! - the workgroup memory array of the plain module has the length of the tile's unpadded
//!   layout, and that of the transposed module the length of the tile's layout for its flags
//!   on the profile ([`MatMulTile::layout_within`], its strides padded against bank conflicts
//!   when the padded layout fits), within the profile's workgroup memory;
//! - the transposed module passes `spirv-val`.
#![cfg(feature = "vulkan")]

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    Epilogue, Kernel, KernelKind, KernelModuleId, MatMulTile, Operand,
};
use modelspec_core::ir::tensor::{DType, TensorType};
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::spirv::validate::validate;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

const OP_EXT_INST: u32 = 12;
const GLSL_FMA: u32 = 50;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_DECORATE: u32 = 71;
const OP_F_ADD: u32 = 129;
const OP_F_MUL: u32 = 133;
const NO_CONTRACTION: u32 = 42;
const WORKGROUP: u32 = 4;
const FUNCTION: u32 = 7;

/// `(m, k, n)`: one and two rows per invocation, the square and the wide tile, each with the
/// partial blocks and partial tiles of `k` the brief names (1, 7, 63, 65, 127, 129).
const SHAPES: [(u64, u64, u64); 16] = [
    (1, 1, 1),
    (1, 7, 129),
    (7, 1, 63),
    (63, 65, 127),
    (129, 63, 65),
    (127, 129, 1),
    (65, 127, 129),
    (2049, 129, 7),
    (2049, 65, 13),
    (4097, 7, 63),
    (1031, 63, 1029),
    (1031, 1, 1029),
    (1025, 65, 1031),
    (1025, 129, 1031),
    (1088, 300, 1040),
    (784, 8192, 4096),
];

fn profile() -> DeviceProfile {
    DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile")
}

fn matmul(m: u64, k: u64, n: u64, epilogue: Option<Epilogue>, transposed: [bool; 2]) -> Kernel {
    let tile = MatMulTile::of(m, k, n);
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

/// `(opcode, operand words)` of each instruction after the header.
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

/// What one module's arithmetic and workgroup memory look like.
#[derive(Debug, PartialEq, Eq)]
struct Shape {
    /// Per `OpFMul` / `OpFAdd` / `Fma` in module order: the opcode (`OpExtInst` for `Fma`) and
    /// the kind of each operand.
    arithmetic: Vec<(u32, Vec<String>)>,
    /// The lengths of the workgroup memory arrays.
    shared: Vec<u32>,
}

fn shape_of(module: &[u32]) -> (Shape, usize, usize) {
    let instructions = instructions(module);
    let mut defined: BTreeMap<u32, (u32, &[u32])> = BTreeMap::new();
    let mut variables: BTreeMap<u32, u32> = BTreeMap::new();
    let mut constants: BTreeMap<u32, u32> = BTreeMap::new();
    let mut pointers: BTreeMap<u32, u32> = BTreeMap::new();
    let mut arrays: BTreeMap<u32, u32> = BTreeMap::new();
    let mut no_contraction = BTreeSet::new();
    for &(op, words) in &instructions {
        match op {
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
            OP_TYPE_ARRAY => {
                arrays.insert(words[0], words[2]);
            }
            OP_DECORATE if words[1] == NO_CONTRACTION => {
                no_contraction.insert(words[0]);
            }
            OP_LOAD | OP_ACCESS_CHAIN | OP_F_ADD | OP_F_MUL => {
                defined.insert(words[1], (op, words));
            }
            OP_EXT_INST if words[3] == GLSL_FMA => {
                defined.insert(words[1], (op, words));
            }
            _ => {}
        }
    }
    let kind = |id: u32| -> String {
        match defined.get(&id) {
            Some((OP_F_MUL, _)) => "fmul".to_owned(),
            Some((OP_F_ADD, _)) => "fadd".to_owned(),
            Some((OP_EXT_INST, _)) => "fma".to_owned(),
            Some((OP_LOAD, words)) => {
                let pointer = words[2];
                match defined.get(&pointer) {
                    Some((OP_VARIABLE, v)) if v[2] == FUNCTION => "load function".to_owned(),
                    Some((OP_ACCESS_CHAIN, chain)) => {
                        let base = chain[2];
                        let class = variables.get(&base).copied();
                        format!(
                            "load {} via {} indices",
                            match class {
                                Some(WORKGROUP) => "workgroup",
                                Some(FUNCTION) => "function",
                                Some(_) => "other",
                                None => "unknown",
                            },
                            chain.len() - 3
                        )
                    }
                    _ => "load other".to_owned(),
                }
            }
            _ => "other".to_owned(),
        }
    };
    let mut arithmetic = Vec::new();
    let (mut ops, mut decorated) = (0, 0);
    for &(op, words) in &instructions {
        let operands = match op {
            OP_F_ADD | OP_F_MUL => &words[2..4],
            OP_EXT_INST if words[3] == GLSL_FMA => &words[4..7],
            _ => continue,
        };
        ops += 1;
        if no_contraction.contains(&words[1]) {
            decorated += 1;
        }
        arithmetic.push((op, operands.iter().map(|&id| kind(id)).collect()));
    }
    let mut shared = Vec::new();
    for &(op, words) in &instructions {
        if op == OP_VARIABLE && words[2] == WORKGROUP {
            let array = pointers[&words[0]];
            let length = arrays
                .get(&array)
                .map(|len| constants[len])
                .expect("workgroup memory is an array");
            shared.push(length);
        }
    }
    (Shape { arithmetic, shared }, ops, decorated)
}

#[test]
fn a_transposed_operand_changes_no_arithmetic_and_stages_into_the_layout_of_its_flags() {
    let profile = profile();
    let mut wrong = Vec::new();
    let mut families = BTreeSet::new();
    let mut compared = 0;
    for (m, k, n) in SHAPES {
        let tile = MatMulTile::of(m, k, n);
        families.insert((tile.rows, tile.cols_per_invocation));
        for epilogue in [
            None,
            Some(Epilogue::Relu {
                writes_pre_activation: true,
            }),
        ] {
            let plain = matmul(m, k, n, epilogue, [false, false]);
            let base = emit_module(&plain, &profile).expect("the plain kernel emits");
            let (want, ops, decorated) = shape_of(&base.spirv);
            assert_eq!(
                ops, decorated,
                "[{m}, {k}] x [{k}, {n}] plain: NoContraction"
            );
            let products = want
                .arithmetic
                .iter()
                .filter(|(op, operands)| {
                    (*op == OP_F_MUL || *op == OP_EXT_INST)
                        && operands[..2]
                            .iter()
                            .all(|kind| kind.starts_with("load workgroup"))
                })
                .count();
            assert!(
                products as u64 >= k.min(tile.depth.into()),
                "[{m}, {k}] x [{k}, {n}]: the products read workgroup memory ({products})"
            );
            for transposed in [[true, false], [false, true], [true, true]] {
                let case = format!("[{m}, {k}] x [{k}, {n}] {tile} {epilogue:?} {transposed:?}");
                let kernel = matmul(m, k, n, epilogue, transposed);
                let module = match emit_module(&kernel, &profile) {
                    Ok(module) => module,
                    Err(e) => {
                        wrong.push(format!("{case}: does not emit: {e}"));
                        continue;
                    }
                };
                compared += 1;
                if let Err(e) = validate(&module.spirv) {
                    wrong.push(format!("{case}: spirv-val: {e}"));
                }
                let (got, ops, decorated) = shape_of(&module.spirv);
                if ops != decorated {
                    wrong.push(format!(
                        "{case}: {decorated} of {ops} arithmetic results NoContraction"
                    ));
                }
                let plain_values = tile.layout([false, false]).shared_values();
                let layout = tile.layout_within(transposed, profile.max_shared_memory_bytes);
                if want.shared != [plain_values]
                    || got.shared != [layout.shared_values()]
                    || layout.shared_bytes() > profile.max_shared_memory_bytes
                {
                    wrong.push(format!(
                        "{case}: workgroup memory {:?} against {:?}, layout {layout:?}",
                        got.shared, want.shared
                    ));
                }
                if got.arithmetic != want.arithmetic {
                    let first = got
                        .arithmetic
                        .iter()
                        .zip(&want.arithmetic)
                        .position(|(g, w)| g != w);
                    wrong.push(format!(
                        "{case}: arithmetic differs ({} against {} ops), first at {first:?}",
                        got.arithmetic.len(),
                        want.arithmetic.len()
                    ));
                }
            }
        }
    }
    eprintln!("{compared} transposed modules compared; tile families {families:?}");
    assert_eq!(
        families,
        BTreeSet::from([(1, 1), (2, 1), (8, 4), (8, 8)]),
        "every tile family is exercised"
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}
