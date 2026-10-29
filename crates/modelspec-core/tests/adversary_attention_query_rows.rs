//! Adversary cases of story:attention-fewer-query-rows on the type checker and the Vulkan plan:
//! `attention(q, k, v, key_mask, heads = A)` with `q` of `[B * L, H]` against `k`, `v` of
//! `[B * S, H]` and a `[B, S]` key mask, `1 <= L <= S`.
//!
//! Each malformed call written in `.mspec` is refused with exactly one diagnostic of the right
//! code, never a panic; each well-formed one (L = 1, 2, S - 1, S) compiles for Vulkan, and its
//! four attention kernels run one workgroup per sequence, head and tile of the rows they write,
//! within the 16 KiB of workgroup memory every device has.

#![allow(clippy::type_complexity)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use modelspec_core::artifact::Backend;
use modelspec_core::compile;
use modelspec_core::ir::kernel::Kernel;
use modelspec_core::ir::tensor::{Dim, TensorOp, TensorType};
use modelspec_core::ops::{OpError, infer};
use modelspec_core::schedule::{
    ATTENTION_WORKGROUP, MAX_ATTENTION_KEYS, attention_tile, attention_workgroup_values,
    attention_workgroups,
};

/// A program attending `xq · w_q` (`[nq, 4]`) to `x · w_k`, `x · w_v` (`[nk, wk]`, `[nk, wv]`)
/// under a `[b, s]` key mask with `heads` heads.
fn source(nq: u64, nk: u64, b: u64, s: u64, wk: u64, wv: u64, heads: u64) -> String {
    format!(
        "model Attend {{
    param w_q: Tensor<f32, [4, 4]> = xavier_uniform();
    param w_k: Tensor<f32, [4, {wk}]> = xavier_uniform();
    param w_v: Tensor<f32, [4, {wv}]> = xavier_uniform();

    fn forward(xq: Tensor<f32, [{nq}, 4]>, x: Tensor<f32, [{nk}, 4]>, key_mask: Tensor<f32, [{b}, {s}]>) -> Tensor<f32, [{nq}, 4]> {{
        return attention(matmul(xq, w_q), matmul(x, w_k), matmul(x, w_v), key_mask, heads = {heads});
    }}
}}

train Fit {{
    model = Attend;
    input xq: Tensor<f32, [{nq}, 4]>;
    input x: Tensor<f32, [{nk}, 4]>;
    input key_mask: Tensor<f32, [{b}, {s}]>;
    target y: Tensor<f32, [{nq}, 4]>;
    loss = mse(model.forward(xq, x, key_mask), y);
    optimizer = sgd(lr = 0.1);
    steps = 5;
    seed = 7;
    device = cpu;
}}
"
    )
}

/// The diagnostics `compile_hir` gives `src`, each its code and its message followed by its
/// detail lines; a panic turned into a failure message.
fn diagnostics(label: &str, src: &str) -> Result<Vec<(String, String)>, String> {
    match catch_unwind(AssertUnwindSafe(|| compile::compile_hir("adv.mspec", src))) {
        Err(_) => Err(format!("{label}: the checker panicked")),
        Ok(Ok(_)) => Ok(Vec::new()),
        Ok(Err(found)) => Ok(found
            .iter()
            .map(|d| {
                let details: Vec<&str> = d.details.iter().map(|x| x.text.as_str()).collect();
                (
                    d.code.as_str().to_owned(),
                    format!("{} | {}", d.message, details.join(" | ")),
                )
            })
            .collect()),
    }
}

/// Every malformed call is one diagnostic of its code: `E-TYPE-003` for a `q` whose rows the
/// mask's `B` does not divide, divides into more than `S` or into 0, a key mask whose `B * S` is
/// not the rows of `k`, a `v` of other rows than `k`, a `k` of another width; `E-TYPE-020` for
/// heads that do not divide `H`. The shape message names the new rule.
#[test]
fn adv_qrows_every_malformed_call_is_one_diagnostic_of_its_code() {
    // (label, nq, nk, b, s, wk, wv, heads, code)
    let cases: [(&str, u64, u64, u64, u64, u64, u64, u64, &str); 9] = [
        (
            "q rows 5 not a multiple of B 2",
            5,
            6,
            2,
            3,
            4,
            4,
            2,
            "E-TYPE-003",
        ),
        (
            "q rows 7 not a multiple of B 2",
            7,
            8,
            2,
            4,
            4,
            4,
            2,
            "E-TYPE-003",
        ),
        ("L 4 > S 3", 8, 6, 2, 3, 4, 4, 2, "E-TYPE-003"),
        ("L 2 > S 1", 4, 2, 2, 1, 4, 4, 2, "E-TYPE-003"),
        (
            "q rows 1 for B 2 (L 0 and a half)",
            1,
            6,
            2,
            3,
            4,
            4,
            2,
            "E-TYPE-003",
        ),
        (
            "mask B*S 6 against k rows 8",
            2,
            8,
            2,
            3,
            4,
            4,
            2,
            "E-TYPE-003",
        ),
        (
            "k width 8 against q width 4",
            2,
            6,
            2,
            3,
            8,
            8,
            2,
            "E-TYPE-003",
        ),
        (
            "v width 8 against k width 4",
            2,
            6,
            2,
            3,
            4,
            8,
            2,
            "E-TYPE-003",
        ),
        ("heads 3 for H 4 at L 1", 2, 6, 2, 3, 4, 4, 3, "E-TYPE-020"),
    ];
    let mut wrong = Vec::new();
    for (label, nq, nk, b, s, wk, wv, heads, code) in cases {
        let src = source(nq, nk, b, s, wk, wv, heads);
        match diagnostics(label, &src) {
            Err(panic) => wrong.push(panic),
            Ok(found) => {
                let codes: Vec<&str> = found.iter().map(|(c, _)| c.as_str()).collect();
                if codes != [code] {
                    wrong.push(format!("{label}: {found:?}, want one {code}"));
                } else if code == "E-TYPE-003" && !found[0].1.contains("1 <= L <= S") {
                    wrong.push(format!("{label}: message {:?} misses the rule", found[0].1));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// `q` of 0 rows (`L = 0`) is refused by the checker, with some diagnostic and no panic.
#[test]
fn adv_qrows_zero_query_rows_are_refused_without_a_panic() {
    let src = source(0, 6, 2, 3, 4, 4, 2);
    let found = diagnostics("q rows 0", &src).unwrap_or_else(|p| panic!("{p}"));
    assert!(!found.is_empty(), "q of 0 rows checks clean");
}

/// `infer` refuses, as a shape mismatch against `q` or a heads error, every operand set outside
/// the rule, including a key mask of 0 rows (`B = 0`, which must not divide by zero) and of 0
/// keys, a rank-1 `k`, and a gradient `G` of `k`'s shape for `AttentionGradQ`.
#[test]
fn adv_qrows_infer_refuses_every_shape_outside_the_rule_without_a_panic() {
    let f = |shape: &[u64]| TensorType::f32(shape);
    let ops = [
        TensorOp::Attention { heads: 2 },
        TensorOp::AttentionGradQ { heads: 2 },
        TensorOp::AttentionGradK { heads: 2 },
        TensorOp::AttentionGradV { heads: 2 },
    ];
    // (label, q, k, v, mask, g)
    let cases: Vec<(
        &str,
        TensorType,
        TensorType,
        TensorType,
        TensorType,
        TensorType,
    )> = vec![
        (
            "B 0",
            f(&[2, 4]),
            f(&[0, 4]),
            f(&[0, 4]),
            f(&[0, 3]),
            f(&[2, 4]),
        ),
        (
            "S 0",
            f(&[2, 4]),
            f(&[0, 4]),
            f(&[0, 4]),
            f(&[2, 0]),
            f(&[2, 4]),
        ),
        (
            "q 0 rows",
            f(&[0, 4]),
            f(&[6, 4]),
            f(&[6, 4]),
            f(&[2, 3]),
            f(&[0, 4]),
        ),
        (
            "L > S",
            f(&[8, 4]),
            f(&[6, 4]),
            f(&[6, 4]),
            f(&[2, 3]),
            f(&[8, 4]),
        ),
        (
            "q not B * L",
            f(&[3, 4]),
            f(&[6, 4]),
            f(&[6, 4]),
            f(&[2, 3]),
            f(&[3, 4]),
        ),
        (
            "rank-1 k",
            f(&[2, 4]),
            f(&[24]),
            f(&[24]),
            f(&[2, 3]),
            f(&[2, 4]),
        ),
        (
            "mask rank 1",
            f(&[2, 4]),
            f(&[6, 4]),
            f(&[6, 4]),
            f(&[6]),
            f(&[2, 4]),
        ),
        (
            "G of k's shape",
            f(&[2, 4]),
            f(&[6, 4]),
            f(&[6, 4]),
            f(&[2, 3]),
            f(&[6, 4]),
        ),
        (
            "symbolic q rows",
            TensorType {
                dtype: modelspec_core::ir::tensor::DType::F32,
                shape: vec![Dim::Symbol(0), Dim::Static(4)],
            },
            f(&[6, 4]),
            f(&[6, 4]),
            f(&[2, 3]),
            f(&[2, 4]),
        ),
    ];
    let mut wrong = Vec::new();
    for (label, q, k, v, mask, g) in &cases {
        for op in &ops {
            let arity = if matches!(op, TensorOp::Attention { .. }) {
                4
            } else {
                5
            };
            if arity == 4 && *label == "G of k's shape" {
                continue;
            }
            let operands = [q, k, v, mask, g];
            let got = catch_unwind(AssertUnwindSafe(|| infer(op, &operands[..arity], None)));
            match got {
                Err(_) => wrong.push(format!("{label} {}: infer panicked", op.name())),
                Ok(Ok(ty)) => wrong.push(format!("{label} {}: accepted as {ty}", op.name())),
                Ok(Err(OpError::ShapeMismatch { .. } | OpError::Rank { .. })) => {}
                Ok(Err(other)) => {
                    wrong.push(format!("{label} {}: refused as {other:?}", op.name()))
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// The attention kernels of a training step compiled for Vulkan, by op.
fn attention_kernels(src: &str) -> Vec<Kernel> {
    let artifact = compile::compile_artifact("adv.mspec", src, Backend::Vulkan)
        .unwrap_or_else(|d| panic!("compiles for vulkan: {d:#?}"));
    artifact
        .execution
        .kernels
        .iter()
        .filter(|k| k.op.starts_with("Attention"))
        .cloned()
        .collect()
}

/// `L = 1, 2, S - 1, S` over `S = 3, 37, 129, 512` with 3 sequences compile for Vulkan, and each
/// attention kernel has `B · heads · ceil(R / attention_tile(R, S))` workgroups of
/// `ATTENTION_WORKGROUP`, `R` the rows it writes per sequence (`L` for `Attention` and
/// `AttentionGradQ`, `S` for the key gradients), and its result type is `q`'s or `k`'s.
#[test]
fn adv_qrows_vulkan_plan_tiles_the_rows_each_kernel_writes() {
    let b = 3u64;
    let mut wrong = Vec::new();
    for s in [3u64, 37, 129, 512] {
        for l in [1, 2, s - 1, s] {
            if l == 0 {
                continue;
            }
            let src = source(b * l, b * s, b, s, 4, 4, 2);
            let kernels = attention_kernels(&src);
            let ops: Vec<&str> = kernels.iter().map(|k| k.op).collect();
            for want in [
                "Attention",
                "AttentionGradQ",
                "AttentionGradK",
                "AttentionGradV",
            ] {
                if !ops.contains(&want) {
                    wrong.push(format!("S {s} L {l}: no {want} kernel in {ops:?}"));
                }
            }
            for k in &kernels {
                let rows = match k.op {
                    "AttentionGradK" | "AttentionGradV" => s,
                    _ => l,
                };
                let want_groups = attention_workgroups(b, rows, s, 2).unwrap();
                let groups: u64 = k.workgroups.iter().map(|&g| u64::from(g)).product();
                if groups < want_groups || k.workgroup_size != ATTENTION_WORKGROUP {
                    wrong.push(format!(
                        "S {s} L {l} {}: {groups} groups of {}, want {want_groups} of {ATTENTION_WORKGROUP}",
                        k.op, k.workgroup_size
                    ));
                }
                let result_rows = match k.result.shape[0] {
                    Dim::Static(n) => n,
                    Dim::Symbol(_) => 0,
                };
                let want_rows = if matches!(k.op, "AttentionGradK" | "AttentionGradV") {
                    b * s
                } else {
                    b * l
                };
                if result_rows != want_rows {
                    wrong.push(format!(
                        "S {s} L {l} {}: result rows {result_rows}, want {want_rows}",
                        k.op
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// For every `S` of 1 to `MAX_ATTENTION_KEYS` and every `R` of 1 to `S`, the tile is a power of
/// two of at most `R` and `S`, at least 1, and its workgroup memory is within 16 KiB; with
/// `R = S` it is what every kernel took before the story (`attention_tile(S, S)` is the tile
/// of `S` rows), and a smaller `R` never takes a larger tile.
#[test]
fn adv_qrows_every_tile_fits_16_kib_and_never_grows_with_fewer_rows() {
    let mut wrong = Vec::new();
    for s in 1..=MAX_ATTENTION_KEYS {
        let full = attention_tile(s, s);
        for r in 1..=s {
            let t = attention_tile(r, s);
            let bytes = attention_workgroup_values(t, s) * 4;
            if t == 0 || !t.is_power_of_two() || t > r || t > s || bytes > 16 * 1024 || t > full {
                wrong.push(format!(
                    "S {s} R {r}: tile {t}, {bytes} bytes, full tile {full}"
                ));
            }
            if wrong.len() > 8 {
                break;
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
