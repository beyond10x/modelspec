//! Adversary cases for the compiler side of `optimizer = muon(...)` (story:muon-optimizer,
//! adversary pass 1): the split rule of § What this story decides on models the unit's tests do
//! not build, the per-parameter scale constants, the printed rule of each update against the
//! step's own state, and the printer's round trip.

use std::collections::BTreeSet;

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::print::print_step;
use modelspec_core::ir::tensor::{ParamId, TensorOp};
use modelspec_core::ir::training::{Subgraph, muon_params};
use modelspec_core::syntax::{parse, print};

const MUON: &str = "muon(lr = 0.02, momentum = 0.95, weight_decay = 0.1, ns_steps = 5, warmup_steps = 2, adam_lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)";

fn file(
    params: &str,
    inputs: &str,
    forward_args: &str,
    ret: &str,
    body: &str,
    loss: &str,
) -> String {
    format!(
        "model M {{
{params}
    fn forward({forward_args}) -> {ret} {{
        return {body};
    }}
}}

train Fit {{
    model = M;
{inputs}
    loss = {loss};
    optimizer = {MUON};
    steps = 10;
    seed = 7;
    device = cpu;
}}
"
    )
}

fn compiled(source: &str) -> CompiledTraining {
    compile_training("adversary_muon.mspec", source)
        .unwrap_or_else(|d| panic!("compiles: {d:?}\n{source}"))
}

fn param(compiled: &CompiledTraining, path: &str) -> ParamId {
    compiled
        .forward
        .program
        .params
        .iter()
        .find(|(_, decl)| decl.path == path)
        .map(|(&id, _)| id)
        .unwrap_or_else(|| panic!("`{path}` is declared"))
}

/// The rule each parameter took, from the step's state: `muon` with a buffer, `adamw` with
/// moments, `none` without an update; and the same from the printed `update` line.
fn rules(c: &CompiledTraining, path: &str) -> (&'static str, String) {
    let id = param(c, path);
    let o = &c.training.optimizer;
    let state = match (
        o.updates.contains_key(&id),
        o.buffers.contains_key(&id),
        o.moments.contains_key(&id),
    ) {
        (true, true, false) => "muon",
        (true, false, true) => "adamw",
        (false, false, false) => "none",
        other => panic!("{path}: update, buffer, moments {other:?}"),
    };
    let printed = print_step(&c.training);
    let suffix = format!(" update @{path} ");
    let shown = printed
        .lines()
        .filter(|l| l.starts_with("output %"))
        .find_map(|l| l.split_once(&suffix).map(|(_, k)| k.to_owned()))
        .unwrap_or_else(|| "none".to_owned());
    (state, shown)
}

/// Hypothesis 2. A matrix read by `embedding` *and* by a `matmul` (an untied table that is also
/// a weight) takes AdamW: any `embedding` read excludes it. The dense weight beside it takes
/// Muon, and `ir --stage training` says the same as the step.
#[test]
fn adv_muon_split_a_table_also_read_by_matmul_takes_adamw() {
    let c = compiled(&file(
        "    param table: Tensor<f32, [16, 16]> = xavier_uniform();\n    param w: Tensor<f32, [16, 8]> = xavier_uniform();",
        "    input ids: Tensor<i32, [4]>;\n    target y: Tensor<f32, [4, 8]>;",
        "ids: Tensor<i32, [4]>",
        "Tensor<f32, [4, 8]>",
        "matmul(matmul(embedding(table, ids), table), w)",
        "mse(model.forward(ids), y)",
    ));
    assert_eq!(rules(&c, "M.table"), ("adamw", "adamw".to_owned()));
    assert_eq!(rules(&c, "M.w"), ("muon", "muon".to_owned()));
    assert_eq!(
        muon_params(&c.training.training),
        BTreeSet::from([param(&c, "M.w")])
    );
}

/// Hypothesis 2. A tied decoder read through `transpose(table)` takes AdamW with its table.
#[test]
fn adv_muon_split_a_tied_decoder_takes_adamw() {
    let c = compiled(&file(
        "    param table: Tensor<f32, [16, 8]> = xavier_uniform();\n    param w: Tensor<f32, [8, 8]> = xavier_uniform();",
        "    input ids: Tensor<i32, [4]>;\n    target y: Tensor<f32, [4, 16]>;",
        "ids: Tensor<i32, [4]>",
        "Tensor<f32, [4, 16]>",
        "matmul(matmul(embedding(table, ids), w), transpose(table))",
        "mse(model.forward(ids), y)",
    ));
    assert_eq!(rules(&c, "M.table"), ("adamw", "adamw".to_owned()));
    assert_eq!(rules(&c, "M.w"), ("muon", "muon".to_owned()));
}

/// Hypothesis 2. A rank-3 parameter takes AdamW (rank 1 is the bias of the runtime cases); a rank-2 parameter the loss does
/// not read takes neither, and has no buffer, no moment and no update.
#[test]
fn adv_muon_split_rank_three_and_an_unused_matrix() {
    let c = compiled(&file(
        "    param w3: Tensor<f32, [2, 3, 4]> = zeros();\n    param unused: Tensor<f32, [3, 3]> = xavier_uniform();",
        "    input x: Tensor<f32, [2, 3, 4]>;\n    target y: Tensor<f32, [2, 3, 4]>;",
        "x: Tensor<f32, [2, 3, 4]>",
        "Tensor<f32, [2, 3, 4]>",
        "add(x, w3)",
        "mse(model.forward(x), y)",
    ));
    assert_eq!(rules(&c, "M.w3"), ("adamw", "adamw".to_owned()));
    assert_eq!(rules(&c, "M.unused"), ("none", "none".to_owned()));
    assert!(c.training.optimizer.muon_lr_t.is_none());
    let printed = print_step(&c.training);
    assert!(!printed.contains("optimizer.b@"), "{printed}");
    assert!(!printed.contains("muon_lr_t"), "{printed}");
}

/// Hypothesis 1 and 2. Two Muon matrices of different shapes each read their own scale constant
/// `0.2 * sqrt(max(rows, cols))`, computed in `f64` and rounded once, in the `Optimizer`
/// subgraph: `[6, 4]` gives `0.2 * sqrt(6)`, `[4, 9]` gives `0.6`.
#[test]
fn adv_muon_each_matrix_reads_its_own_scale_rounded_once_from_f64() {
    let c = compiled(&file(
        "    param w1: Tensor<f32, [6, 4]> = xavier_uniform();\n    param w2: Tensor<f32, [4, 9]> = xavier_uniform();",
        "    input x: Tensor<f32, [8, 6]>;\n    target y: Tensor<f32, [8, 9]>;",
        "x: Tensor<f32, [8, 6]>",
        "Tensor<f32, [8, 9]>",
        "matmul(matmul(x, w1), w2)",
        "mse(model.forward(x), y)",
    ));
    assert_eq!(rules(&c, "M.w1"), ("muon", "muon".to_owned()));
    assert_eq!(rules(&c, "M.w2"), ("muon", "muon".to_owned()));
    let program = &c.training.training;
    let scalars: Vec<u32> = program
        .program
        .nodes
        .iter()
        .filter(|n| program.subgraphs.get(&n.id) == Some(&Subgraph::Optimizer))
        .filter_map(|n| match n.op {
            TensorOp::Scalar(v) => Some(v.to_bits()),
            _ => None,
        })
        .collect();
    for n in [6u32, 9] {
        let once = (0.2f64 * f64::from(n).sqrt()) as f32;
        assert_eq!(
            scalars.iter().filter(|&&b| b == once.to_bits()).count(),
            1,
            "scale of max {n}: {once:?} once among {:?}",
            scalars
                .iter()
                .map(|&b| f32::from_bits(b))
                .collect::<Vec<_>>()
        );
    }
    let transposes = program
        .program
        .nodes
        .iter()
        .filter(|n| program.subgraphs.get(&n.id) == Some(&Subgraph::Optimizer))
        .filter(|n| n.op == TensorOp::Transpose)
        .count();
    assert_eq!(
        transposes,
        (5 + 2) + 5,
        "the tall w1 takes one transpose per Newton-Schulz step and two around them, the wide \
         w2 one per step"
    );
}

/// Hypothesis 6. The printer writes a `muon(...)` call that parses back to the same tree.
#[test]
fn adv_muon_printer_round_trips_a_muon_call() {
    let source = file(
        "    param w: Tensor<f32, [4, 3]> = xavier_uniform();",
        "    input x: Tensor<f32, [8, 4]>;\n    target y: Tensor<f32, [8, 3]>;",
        "x: Tensor<f32, [8, 4]>",
        "Tensor<f32, [8, 3]>",
        "matmul(x, w)",
        "mse(model.forward(x), y)",
    );
    let mut first = parse("a.mspec", &source).unwrap_or_else(|d| panic!("{d:?}"));
    let printed = print(&first);
    assert!(printed.contains("optimizer = muon(lr = "), "{printed}");
    let mut second = parse("b.mspec", &printed).unwrap_or_else(|d| panic!("{d:?}\n{printed}"));
    first.erase_spans();
    second.erase_spans();
    assert_eq!(first, second, "printed:\n{printed}");
    compiled(&printed);
}
