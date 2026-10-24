//! Adversary cases for `story:matmul-shape-aware-tiling`, without a device: the accumulation
//! order of a register-blocked matmul module read from its SPIR-V, over shapes at the edges of
//! the block rule.
//!
//! For every `OpStore` of an `OpFAdd` or `OpFMul` result into a function variable (an
//! accumulator), the chain of `OpFAdd`s it stores is followed back to where it starts: the
//! `k = 0` product (`OpFMul`) on the first tile, or a load of the same variable on a later one,
//! never a constant (a sum started from zero). The products of one chain read workgroup memory
//! at one `a` start and one `b` start plus constants that ascend by one value of `k` each (by
//! 1 for `a`, by the `b` stride for `b`), so each tile adds its products in ascending `k`. Each
//! accumulator has exactly one opening chain, and there are `rows · cols_per_invocation` of
//! them. Every module passes `spirv-val` with and without the three float-control modes, and
//! every `OpFAdd`, `OpFSub` and `OpFMul` is `NoContraction`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Epilogue, MatMulTile};
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::spirv::validate::validate;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

const OP_CONSTANT: u32 = 43;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_I_ADD: u32 = 128;
const OP_F_ADD: u32 = 129;
const OP_F_MUL: u32 = 133;

/// Register-blocked shapes at the edges of the rule.
const SHAPES: [(u64, u64, u64); 22] = [
    (8192, 64, 64),
    (8129, 65, 65),
    (4097, 63, 127),
    (64, 64, 16_384),
    (1, 64, 16_385),
    (1, 1, 8192),
    (2, 2, 8191),
    (512, 64, 2048),
    (448, 64, 2048),
    (512, 63, 2048),
    (512, 65, 2048),
    (512, 72, 2048),
    (513, 127, 2040),
    (8192, 15, 64),
    (8192, 16, 64),
    (8192, 17, 64),
    (8192, 33, 64),
    (8192, 100, 64),
    (8192, 784, 64),
    (784, 1000, 1024),
    (784, 8192, 1024),
    (8192, 784, 4096),
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

fn profile(float_controls: bool) -> DeviceProfile {
    let mut profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile");
    if float_controls {
        profile.shader_rounding_mode_rte_f32 = Some(true);
        profile.shader_denorm_preserve_f32 = Some(true);
        profile.shader_signed_zero_inf_nan_preserve_f32 = Some(true);
    }
    profile
}

/// A workgroup memory read: the start id it adds a constant to, and the constant.
fn shared_read(defs: &BTreeMap<u32, (u32, Vec<u32>)>, value: u32) -> Result<(u32, u32), String> {
    let (op, w) = &defs[&value];
    if *op != OP_LOAD {
        return Err(format!("product operand %{value} is op {op}, not a load"));
    }
    let (op, chain) = &defs[&w[3]];
    if *op != OP_ACCESS_CHAIN {
        return Err(format!("load %{value} reads op {op}, not an access chain"));
    }
    let index = *chain.last().expect("an index");
    let (op, add) = &defs[&index];
    if *op != OP_I_ADD {
        return Err(format!("index %{index} is op {op}, not an add"));
    }
    match defs.get(&add[4]) {
        Some((OP_CONSTANT, c)) => Ok((add[3], c[3])),
        _ => Err(format!("index %{index} adds no constant")),
    }
}

/// Every break of the accumulation order in `module`, and the accumulators it found.
fn order_breaks(module: &[u32], tile: &MatMulTile) -> (Vec<String>, usize) {
    let instructions = common::words::instructions(module);
    let mut defs: BTreeMap<u32, (u32, Vec<u32>)> = BTreeMap::new();
    for (op, w) in &instructions {
        if [
            OP_CONSTANT,
            OP_LOAD,
            OP_ACCESS_CHAIN,
            OP_I_ADD,
            OP_F_ADD,
            OP_F_MUL,
        ]
        .contains(op)
        {
            defs.insert(w[2], (*op, w.to_vec()));
        }
    }
    let b_step = if tile.cols_per_invocation == 1 {
        1
    } else {
        tile.b_stride()
    };
    let mut wrong = Vec::new();
    let mut openings: BTreeMap<u32, usize> = BTreeMap::new();
    let mut accumulators = BTreeSet::new();
    for (op, w) in &instructions {
        if *op != OP_STORE {
            continue;
        }
        let (var, value) = (w[1], w[2]);
        if !matches!(defs.get(&value), Some((OP_F_ADD | OP_F_MUL, _))) {
            continue;
        }
        accumulators.insert(var);
        let mut products = Vec::new();
        let mut at = value;
        while let Some((OP_F_ADD, add)) = defs.get(&at) {
            products.push(add[4]);
            at = add[3];
        }
        match defs.get(&at) {
            Some((OP_F_MUL, _)) => {
                products.push(at);
                *openings.entry(var).or_default() += 1;
            }
            Some((OP_LOAD, load)) if load[3] == var => {}
            Some((OP_LOAD, load)) => wrong.push(format!(
                "the chain stored to %{var} continues the sum of %{}",
                load[3]
            )),
            other => wrong.push(format!(
                "the chain stored to %{var} starts from op {:?}, not the k = 0 product",
                other.map(|(op, _)| op)
            )),
        }
        products.reverse();
        let mut reads = Vec::new();
        for &product in &products {
            match defs.get(&product) {
                Some((OP_F_MUL, mul)) => {
                    match (shared_read(&defs, mul[3]), shared_read(&defs, mul[4])) {
                        (Ok(a), Ok(b)) => reads.push((a, b)),
                        (a, b) => {
                            wrong.push(format!("chain of %{var}: product %{product}: {a:?} {b:?}"))
                        }
                    }
                }
                other => wrong.push(format!(
                    "chain of %{var}: the term %{product} is op {:?}, not a product",
                    other.map(|(op, _)| op)
                )),
            }
        }
        if let Some(&((a_start, a0), (b_start, b0))) = reads.first() {
            for (p, &((a_at, a), (b_at, b))) in reads.iter().enumerate() {
                let p = p as u32;
                if a_at != a_start || b_at != b_start || a != a0 + p || b != b0 + p * b_step {
                    wrong.push(format!(
                        "chain of %{var}: product {p} reads a %{a_at}+{a}, b %{b_at}+{b}; \
                         expected a %{a_start}+{}, b %{b_start}+{}",
                        a0 + p,
                        b0 + p * b_step
                    ));
                    break;
                }
            }
            if a0 != 0 || b0 != 0 {
                wrong.push(format!(
                    "chain of %{var}: the first product of a tile reads a +{a0}, b +{b0}"
                ));
            }
        }
    }
    for &var in &accumulators {
        let opened = openings.get(&var).copied().unwrap_or(0);
        if opened != 1 {
            wrong.push(format!("accumulator %{var} is opened {opened} times"));
        }
    }
    (wrong, accumulators.len())
}

#[test]
fn every_register_blocked_chain_starts_at_the_k0_product_and_ascends_in_k() {
    let mut wrong = Vec::new();
    let mut modules = 0;
    for (m, k, n) in SHAPES {
        let tile = MatMulTile::of(m, k, n);
        assert!(
            tile.cols_per_invocation >= 4,
            "[{m}, {k}] x [{k}, {n}]: {tile} is register-blocked"
        );
        for epilogue in EPILOGUES {
            for float_controls in [false, true] {
                let what = format!(
                    "[{m}, {k}] x [{k}, {n}] {epilogue:?} float controls {float_controls}, {tile}"
                );
                let kernel = common::matmul_kernel_with(m, k, n, epilogue);
                let module = emit_module(&kernel, &profile(float_controls))
                    .unwrap_or_else(|error| panic!("{what}: {error}"))
                    .spirv;
                if let Err(error) = validate(&module) {
                    wrong.push(format!("{what}: spirv-val: {error}"));
                }
                let modes = common::words::float_control_modes(&module).len();
                if modes != if float_controls { 3 } else { 0 } {
                    wrong.push(format!("{what}: {modes} float-control modes"));
                }
                let (missing, _) = common::words::uncontracted_arithmetic(&module);
                if !missing.is_empty() {
                    wrong.push(format!("{what}: no NoContraction on {missing:?}"));
                }
                let (breaks, accumulators) = order_breaks(&module, &tile);
                let expected = (tile.rows * tile.cols_per_invocation) as usize;
                if accumulators != expected {
                    wrong.push(format!(
                        "{what}: {accumulators} accumulators, expected {expected}"
                    ));
                }
                wrong.extend(breaks.into_iter().map(|b| format!("{what}: {b}")));
                modules += 1;
            }
        }
    }
    eprintln!("register blocked: {modules} modules read for their accumulation order");
    assert!(
        wrong.is_empty(),
        "{} breaks, first {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(20)]
    );
}

#[test]
fn the_order_check_finds_a_chain_started_from_zero() {
    // The checker on a module it must refuse: the opening `OpFMul` of one accumulator chain
    // rewritten in place into an `OpFAdd` of a float zero constant and the product's `a` load.
    let (m, k, n) = (8192, 16, 64);
    let tile = MatMulTile::of(m, k, n);
    let kernel = common::matmul_kernel_with(m, k, n, None);
    let mut module = emit_module(&kernel, &profile(false))
        .expect("the kernel emits")
        .spirv;
    let (clean, _) = order_breaks(&module, &tile);
    assert!(clean.is_empty(), "{clean:#?}");
    let instructions: Vec<(usize, u32, Vec<u32>)> = {
        let mut out = Vec::new();
        let mut at = 5;
        while at < module.len() {
            let length = (module[at] >> 16) as usize;
            out.push((at, module[at] & 0xffff, module[at..at + length].to_vec()));
            at += length;
        }
        out
    };
    let zero = instructions
        .iter()
        .find(|(_, op, w)| *op == OP_CONSTANT && w.len() == 4 && w[3] == 0)
        .map(|(_, _, w)| w[2])
        .expect("a zero constant");
    let (at, _, _) = instructions
        .iter()
        .find(|(_, op, _)| *op == OP_F_MUL)
        .expect("a product");
    module[*at] = (module[*at] & 0xffff_0000) | OP_F_ADD;
    module[*at + 3] = zero;
    let (breaks, _) = order_breaks(&module, &tile);
    assert!(
        !breaks.is_empty(),
        "a chain whose first term adds a constant passes the check"
    );
}
