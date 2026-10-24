//! Adversary cases for story:matmul-reads-transposed-operand, the fold rule of
//! `schedule::schedule` (DESIGN.md § 3.1): a `transpose` whose every reader is a `matmul`, that
//! is not the loss or the prediction, and whose operand is not a parameter updated before one of
//! its readers, dispatches nothing; each reader reads the transpose's operand in place, flagged
//! in `Kernel::transposed`.
//!
//! Every source here is audited the same way against its own Training IR, not against the
//! kernel record: each `Transpose` node is either a `Transpose` dispatch whose matmul readers
//! read its output unflagged, or folded, in which case every reader is a matmul, the node is
//! neither the loss nor the prediction, and every reader's step reads the operand's view at that
//! position with the flag set. No matmul carries a flag without a folded transpose behind it,
//! every plan passes `verify_plan`, and every dispatch has a count whose bytes are its views'.
//!
//! The sources reach the exclusions the implementor's cases do not: a prediction that is a
//! transpose read only by a matmul of the loss, a transpose read by two matmuls one of which
//! has the `relu` epilogue, a chain of transposes, both operands of one matmul folded, and a
//! transpose read by an `add`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::kernel::{
    BufferView, DispatchCount, ExecutionPlan, Kernel, KernelKind, MatMulTile, PlanOp, PlanStep,
};
use modelspec_core::ir::print::print_kernel;
use modelspec_core::ir::tensor::{TensorNode, TensorOp, ValueId};
use modelspec_core::ir::verify::verify_plan;
use modelspec_core::schedule::schedule;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A transpose of the input read by two matmuls and by an `add`: it stays a dispatch.
pub const ADD_READER: &str = "model AddReader {
    param w: Tensor<f32, [10, 3]> = xavier_uniform();

    fn forward(p: Tensor<f32, [10, 12]>) -> Tensor<f32, [12, 3]> {
        let t = transpose(p);
        return add(matmul(t, w), matmul(add(t, t), w));
    }
}

train Fit {
    model = AddReader;
    input p: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [12, 3]>;
    loss = mse(model.forward(p), y);
    optimizer = sgd(lr = 0.05);
    steps = 4;
    seed = 11;
    device = vulkan(0);
}
";

/// The prediction is a transpose, and the only op of the step that reads it is a matmul in the
/// loss: the exclusion of the prediction is the only thing that keeps it a dispatch.
pub const PREDICTION_TRANSPOSE: &str = "model PredT {
    param w: Tensor<f32, [12, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [10, 12]>) -> Tensor<f32, [3, 10]> {
        return transpose(matmul(x, w));
    }
}

train Fit {
    model = PredT;
    input x: Tensor<f32, [10, 12]>;
    input z: Tensor<f32, [10, 4]>;
    target y: Tensor<f32, [3, 4]>;
    loss = mse(matmul(model.forward(x), z), y);
    optimizer = sgd(lr = 0.05);
    steps = 4;
    seed = 12;
    device = vulkan(0);
}
";

/// `transpose(v)` of a parameter read by two matmuls, one with the `relu` epilogue that keeps
/// its pre-activation: folded into both.
pub const TWO_READERS: &str = "model TwoReaders {
    param v: Tensor<f32, [4, 12]> = xavier_uniform();

    fn forward(a: Tensor<f32, [10, 12]>, b: Tensor<f32, [10, 12]>) -> Tensor<f32, [10, 4]> {
        let t = transpose(v);
        return add(relu(matmul(a, t)), matmul(b, t));
    }
}

train Fit {
    model = TwoReaders;
    input a: Tensor<f32, [10, 12]>;
    input b: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [10, 4]>;
    loss = mse(model.forward(a, b), y);
    optimizer = sgd(lr = 0.05);
    steps = 4;
    seed = 13;
    device = vulkan(0);
}
";

/// Three transposes in a chain before a matmul, and the gradient's transpose of the last.
pub const CHAIN: &str = "model Chain {
    param w: Tensor<f32, [10, 3]> = xavier_uniform();

    fn forward(a: Tensor<f32, [10, 12]>) -> Tensor<f32, [12, 3]> {
        return matmul(transpose(transpose(transpose(a))), w);
    }
}

train Fit {
    model = Chain;
    input a: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [12, 3]>;
    loss = mse(model.forward(a), y);
    optimizer = sgd(lr = 0.05);
    steps = 4;
    seed = 14;
    device = vulkan(0);
}
";

/// A matmul of two transposed inputs: both operands folded into one kernel.
pub fn both(m: u64, k: u64, n: u64, out: u64) -> String {
    format!(
        "model Both {{
    param w: Tensor<f32, [{n}, {out}]> = xavier_uniform();

    fn forward(a: Tensor<f32, [{k}, {m}]>, b: Tensor<f32, [{n}, {k}]>) -> Tensor<f32, [{m}, {out}]> {{
        return matmul(matmul(transpose(a), transpose(b)), w);
    }}
}}

train Fit {{
    model = Both;
    input a: Tensor<f32, [{k}, {m}]>;
    input b: Tensor<f32, [{n}, {k}]>;
    target y: Tensor<f32, [{m}, {out}]>;
    loss = mse(model.forward(a, b), y);
    optimizer = sgd(lr = 0.001);
    steps = 2;
    seed = 15;
    device = vulkan(0);
}}
"
    )
}

/// One logical matmul shape `[12, 10] x [10, 3]` read once from a stored parameter and once from
/// a folded transpose of another.
pub const SAME_SHAPE: &str = "model SameShape {
    param w: Tensor<f32, [10, 3]> = xavier_uniform();
    param v: Tensor<f32, [3, 10]> = xavier_uniform();

    fn forward(p: Tensor<f32, [12, 10]>, q: Tensor<f32, [12, 10]>) -> Tensor<f32, [12, 3]> {
        return add(matmul(p, transpose(v)), matmul(q, w));
    }
}

train Fit {
    model = SameShape;
    input p: Tensor<f32, [12, 10]>;
    input q: Tensor<f32, [12, 10]>;
    target y: Tensor<f32, [12, 3]>;
    loss = mse(model.forward(p, q), y);
    optimizer = sgd(lr = 0.05);
    steps = 4;
    seed = 16;
    device = vulkan(0);
}
";

fn training(name: &str, source: &str) -> CompiledTraining {
    compile_training(name, source).unwrap_or_else(|d| panic!("{name} compiles: {d:#?}"))
}

fn plan_of(name: &str, compiled: &CompiledTraining) -> ExecutionPlan {
    let plan = schedule(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
    )
    .unwrap_or_else(|e| panic!("{name} schedules: {e}"));
    verify_plan(&plan).unwrap_or_else(|e| panic!("{name}: the plan verifies: {e}"));
    plan
}

fn kernel_of<'a>(plan: &'a ExecutionPlan, step: &PlanStep) -> &'a Kernel {
    plan.kernel(step.kernel_module_id.expect("a dispatch names its kernel"))
        .expect("the plan has the kernel")
}

/// What one source's plan does with each of its transposes.
#[derive(Debug, Default)]
struct Audit {
    kept: Vec<ValueId>,
    folded: Vec<ValueId>,
    flagged_operands: usize,
}

/// The view a step of `plan` reads for `value` of `nodes`, following folded transposes to
/// their operand.
fn view_of(
    plan: &ExecutionPlan,
    nodes: &BTreeMap<ValueId, &TensorNode>,
    value: ValueId,
) -> Option<BufferView> {
    let node = nodes[&value];
    match &node.op {
        TensorOp::Input { name } => {
            return plan.data.iter().find(|d| &d.name == name).map(|d| d.view);
        }
        TensorOp::ParamRead { param } => {
            return plan
                .parameters
                .iter()
                .find(|p| p.param == *param)
                .map(|p| p.views[0]);
        }
        _ => {}
    }
    if node.op == TensorOp::Relu {
        let operand = node.operands[0];
        if let Some(step) = plan.step.iter().find(|s| {
            s.op == PlanOp::Dispatch
                && s.value == Some(operand)
                && kernel_of(plan, s).epilogue.is_some()
        }) {
            return step.writes.last().copied();
        }
    }
    if let Some(step) = plan
        .step
        .iter()
        .rev()
        .find(|s| s.op == PlanOp::Dispatch && s.value == Some(value))
    {
        return step.writes.first().copied();
    }
    if node.op == TensorOp::Transpose {
        return view_of(plan, nodes, node.operands[0]);
    }
    None
}

fn audit(name: &str, source: &str) -> (Audit, ExecutionPlan, CompiledTraining) {
    let compiled = training(name, source);
    let plan = plan_of(name, &compiled);
    let program = &compiled.training.training.program;
    let nodes: BTreeMap<ValueId, &TensorNode> = program.nodes.iter().map(|n| (n.id, n)).collect();
    let position: BTreeMap<ValueId, usize> = program
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id, i))
        .collect();
    let bound = [compiled.training.training.loss, compiled.forward.prediction];
    let readers = |v: ValueId| -> Vec<&TensorNode> {
        program
            .nodes
            .iter()
            .filter(|n| n.operands.contains(&v))
            .collect()
    };
    let step_of = |v: ValueId| -> Option<&PlanStep> {
        plan.step
            .iter()
            .find(|s| s.op == PlanOp::Dispatch && s.value == Some(v))
    };
    let mut audit = Audit::default();
    for node in program.nodes.iter().filter(|n| n.op == TensorOp::Transpose) {
        let kept = step_of(node.id);
        let readers = readers(node.id);
        match kept {
            Some(step) => {
                assert_eq!(
                    kernel_of(&plan, step).kind,
                    KernelKind::Transpose,
                    "{name}: {:?} is dispatched as a transpose",
                    node.id
                );
                for reader in readers.iter().filter(|r| r.op == TensorOp::MatMul) {
                    let rs = step_of(reader.id).expect("a matmul reader is dispatched");
                    let kernel = kernel_of(&plan, rs);
                    for (p, _) in reader
                        .operands
                        .iter()
                        .enumerate()
                        .filter(|(_, o)| **o == node.id)
                    {
                        assert!(
                            !kernel.transposed[p],
                            "{name}: {:?} reads the stored transpose {:?} flagged at {p}",
                            reader.id, node.id
                        );
                        assert_eq!(
                            rs.reads[p], step.writes[0],
                            "{name}: {:?} reads the stored transpose {:?} at {p}",
                            reader.id, node.id
                        );
                    }
                }
                audit.kept.push(node.id);
            }
            None => {
                assert!(
                    !bound.contains(&node.id),
                    "{name}: the loss or prediction {:?} has no dispatch",
                    node.id
                );
                assert!(
                    !readers.is_empty() && readers.iter().all(|r| r.op == TensorOp::MatMul),
                    "{name}: folded {:?} is read by {:?}",
                    node.id,
                    readers.iter().map(|r| &r.op).collect::<Vec<_>>()
                );
                let operand = node.operands[0];
                if let TensorOp::ParamRead { param } = &nodes[&operand].op
                    && let Some(update) = compiled.training.optimizer.updates.get(param)
                {
                    for reader in &readers {
                        assert!(
                            position[&reader.id] < position[update],
                            "{name}: {:?} reads {param:?} in place after its update",
                            reader.id
                        );
                    }
                }
                let source_view = view_of(&plan, &nodes, operand)
                    .unwrap_or_else(|| panic!("{name}: {operand:?} has a view"));
                for reader in &readers {
                    let rs = step_of(reader.id).expect("a matmul reader is dispatched");
                    let kernel = kernel_of(&plan, rs);
                    for (p, _) in reader
                        .operands
                        .iter()
                        .enumerate()
                        .filter(|(_, o)| **o == node.id)
                    {
                        assert!(
                            kernel.transposed[p],
                            "{name}: {:?} reads folded {:?} unflagged at {p}: {}",
                            reader.id,
                            node.id,
                            kernel.signature()
                        );
                        assert_eq!(
                            rs.reads[p], source_view,
                            "{name}: {:?} reads the operand of folded {:?} in place at {p}",
                            reader.id, node.id
                        );
                    }
                }
                audit.folded.push(node.id);
            }
        }
    }
    for step in plan.step.iter().filter(|s| s.op == PlanOp::Dispatch) {
        let kernel = kernel_of(&plan, step);
        if kernel.kind != KernelKind::MatMul {
            assert_eq!(
                kernel.transposed,
                [false, false],
                "{name}: a {:?} kernel carries a flag",
                kernel.kind
            );
        }
        let value = step.value.expect("a dispatch names its value");
        if kernel.kind == KernelKind::MatMul {
            let reader = nodes[&value];
            for (p, flag) in kernel.transposed.iter().enumerate() {
                let operand = nodes[&reader.operands[p]];
                if *flag {
                    audit.flagged_operands += 1;
                    assert_eq!(
                        operand.op,
                        TensorOp::Transpose,
                        "{name}: {:?} flags operand {p}, which is no transpose",
                        value
                    );
                    assert!(
                        step_of(operand.id).is_none(),
                        "{name}: {:?} flags operand {p}, whose transpose is dispatched",
                        value
                    );
                }
            }
        }
        match plan.dispatch_count(step) {
            Some(DispatchCount::Counted { bytes, .. }) => {
                let views: u64 = step
                    .reads
                    .iter()
                    .chain(&step.writes)
                    .map(|v| v.size_bytes)
                    .sum();
                assert_eq!(bytes, views, "{name}: bytes of {value:?}");
            }
            other => panic!("{name}: {value:?} is {other:?}"),
        }
    }
    (audit, plan, compiled)
}

fn mnist() -> String {
    std::fs::read_to_string(repo_root().join("examples/mnist_mlp.mspec")).expect("reads")
}

fn linear() -> String {
    std::fs::read_to_string(repo_root().join("examples/linear_regression.mspec")).expect("reads")
}

/// Every source above and both examples pass the audit, and each one exercises the fold as
/// stated: how many transposes stay, how many fold, how many matmul operands are flagged.
#[test]
fn every_transpose_is_a_dispatch_its_readers_read_or_folded_into_matmuls_that_read_its_operand() {
    let cases: Vec<(&str, String, usize, usize)> = vec![
        ("mnist", mnist(), 0, 3),
        ("linear", linear(), 0, 1),
        ("add_reader", ADD_READER.to_owned(), 1, 2),
        (
            "prediction_transpose",
            PREDICTION_TRANSPOSE.to_owned(),
            1,
            3,
        ),
        ("two_readers", TWO_READERS.to_owned(), 1, 3),
        ("chain", CHAIN.to_owned(), 3, 1),
        ("both", both(10, 12, 7, 5), 0, 3),
        ("same_shape", SAME_SHAPE.to_owned(), 1, 3),
    ];
    let mut report = Vec::new();
    let mut wrong = Vec::new();
    for (name, source, kept, folded) in cases {
        let (audit, _, _) = audit(name, &source);
        report.push(format!(
            "{name}: kept {} folded {} flagged operands {}",
            audit.kept.len(),
            audit.folded.len(),
            audit.flagged_operands
        ));
        if (audit.kept.len(), audit.folded.len()) != (kept, folded) {
            wrong.push(format!(
                "{name}: expected kept {kept} folded {folded}, {audit:?}"
            ));
        }
    }
    eprintln!("{report:#?}");
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Exclusion (b): the prediction `transpose(matmul(x, w))` is read in the step only by the
/// loss's matmul. It stays a dispatch, it writes the prediction view, the forward plan writes
/// that view, and no step reads the prediction's operand where the prediction should be.
#[test]
fn a_prediction_that_only_a_matmul_reads_stays_a_transpose_that_writes_the_prediction_view() {
    let (_, plan, compiled) = audit("prediction_transpose", PREDICTION_TRANSPOSE);
    let program = &compiled.training.training.program;
    let prediction = compiled.forward.prediction;
    let node = program.node(prediction).expect("the prediction is a node");
    assert_eq!(node.op, TensorOp::Transpose);
    let readers: Vec<&TensorOp> = program
        .nodes
        .iter()
        .filter(|n| n.operands.contains(&prediction))
        .map(|n| &n.op)
        .collect();
    assert_eq!(
        readers,
        [&TensorOp::MatMul],
        "only a matmul reads the prediction in the step"
    );
    for (which, steps) in [("step", &plan.step), ("forward", &plan.forward)] {
        let writer = steps
            .iter()
            .find(|s| s.op == PlanOp::Dispatch && s.writes.contains(&plan.prediction_view))
            .unwrap_or_else(|| panic!("the {which} plan writes the prediction view"));
        assert_eq!(writer.value, Some(prediction), "{which}");
        assert_eq!(
            kernel_of(&plan, writer).kind,
            KernelKind::Transpose,
            "{which}"
        );
    }
    let operand = node.operands[0];
    let operand_view = plan
        .step
        .iter()
        .find(|s| s.op == PlanOp::Dispatch && s.value == Some(operand))
        .map(|s| s.writes[0])
        .expect("matmul(x, w) is dispatched");
    assert_ne!(plan.prediction_view, operand_view);
}

/// Exclusion (e): a transpose read by two matmuls, one with the `relu` epilogue that keeps its
/// pre-activation, is folded into both; both kernels flag `b`, both read `v` in place.
#[test]
fn a_transpose_read_by_two_matmuls_one_with_an_epilogue_folds_into_both() {
    let (audit, plan, _) = audit("two_readers", TWO_READERS);
    let v = plan.parameters[0].views[0];
    let readers: Vec<&Kernel> = plan
        .step
        .iter()
        .filter(|s| s.op == PlanOp::Dispatch && s.reads.get(1) == Some(&v))
        .map(|s| kernel_of(&plan, s))
        .filter(|k| k.kind == KernelKind::MatMul)
        .collect();
    assert_eq!(readers.len(), 2, "{audit:?}");
    assert!(readers.iter().all(|k| k.transposed == [false, true]));
    assert_eq!(readers.iter().filter(|k| k.epilogue.is_some()).count(), 1);
}

/// Both operands of one matmul folded: one kernel flags `[true, true]` and reads both inputs in
/// place, at the one-row tile, at the wide register-blocked tile and at the square one.
#[test]
fn a_matmul_of_two_transposed_inputs_flags_both_operands_at_every_tile() {
    for (m, k, n, family) in [
        (10, 12, 7, (1, 1)),
        (1024, 96, 1024, (8, 8)),
        (1031, 7, 1029, (8, 4)),
    ] {
        let name = format!("both {m} {k} {n}");
        let (_, plan, _) = audit(&name, &both(m, k, n, 3));
        let a = plan.data.iter().find(|d| d.name == "a").unwrap().view;
        let b = plan.data.iter().find(|d| d.name == "b").unwrap().view;
        let step = plan
            .step
            .iter()
            .find(|s| s.op == PlanOp::Dispatch && s.reads.len() == 2 && s.reads == [a, b])
            .unwrap_or_else(|| panic!("{name}: one dispatch reads a and b in place"));
        let kernel = kernel_of(&plan, step);
        assert_eq!(kernel.transposed, [true, true], "{name}");
        let (tm, tk, tn) = Kernel::matmul_shape(&kernel.operands).expect("a matmul shape");
        assert_eq!((tm, tk, tn), (m, k, n), "{name}: the logical shape");
        let tile = MatMulTile::of(tm, tk, tn);
        assert_eq!(kernel.workgroup_size, tile.workgroup_size, "{name}");
        assert_eq!(
            (tile.rows, tile.cols_per_invocation),
            family,
            "{name}: {tile}"
        );
    }
}

/// Signature separation: one logical shape `[12, 10] x [10, 3]`, read from a folded transpose
/// and from a stored parameter, gives two kernels with different ids, and the printed line of each
/// shows ` transposed` after exactly the operand that is.
#[test]
fn one_shape_read_transposed_and_as_is_is_two_kernels_and_two_printed_lines() {
    let (_, plan, _) = audit("same_shape", SAME_SHAPE);
    let forward: Vec<&Kernel> = plan
        .kernels
        .iter()
        .filter(|k| {
            k.kind == KernelKind::MatMul
                && k.operands.first().map(ToString::to_string).as_deref()
                    == Some("Tensor<f32, [12, 10]>")
                && k.operands.get(1).map(ToString::to_string).as_deref()
                    == Some("Tensor<f32, [10, 3]>")
        })
        .collect();
    assert_eq!(forward.len(), 2, "{forward:#?}");
    assert_ne!(forward[0].id, forward[1].id);
    let flags: Vec<[bool; 2]> = forward.iter().map(|k| k.transposed).collect();
    assert!(
        flags.contains(&[false, true]) && flags.contains(&[false, false]),
        "{flags:?}"
    );
    let printed = print_kernel(&plan);
    for kernel in forward {
        let line = printed
            .lines()
            .find(|l| l.starts_with("kernel ") && l.contains(&kernel.id.to_string()))
            .expect("each kernel has a line");
        let expected = if kernel.transposed[1] {
            "(Tensor<f32, [12, 10]>, Tensor<f32, [10, 3]> transposed) -> Tensor<f32, [12, 3]>"
        } else {
            "(Tensor<f32, [12, 10]>, Tensor<f32, [10, 3]>) -> Tensor<f32, [12, 3]>"
        };
        assert!(line.contains(expected), "{line}");
    }
}
