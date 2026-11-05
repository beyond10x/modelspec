//! `story:vulkan-adamw-one-dispatch-per-parameter` (DESIGN.md § 5.2, § 5.3): the AdamW update of
//! one parameter is one `sgd_update` dispatch of [`ADAMW_OP`].
//!
//! - In the step plan of `examples/bert_tiny.mspec` each of the 41 parameters has one such
//!   dispatch, which reads `p`, `g`, `m`, `v`, `lr_t`, `c1` and `c2` and writes the parameter and
//!   both moments into the views it read them from; the plan has no `Sqrt` and no `Div` kernel,
//!   and no barrier stands between two updates.
//! - An update whose ops are not the rule's, or one of whose inner values has a second reader,
//!   keeps the dispatches of the sixteen-op lowering, `Sqrt` and `Div` among them.
//! - Both blocked key-side attention gradients of the example run 256 workgroups
//!   (`story:vulkan-attention-blocked-key-tiles`).

use std::path::{Path, PathBuf};

use modelspec_core::artifact::Backend;
use modelspec_core::compile::{CompiledTraining, compile_artifact, compile_training};
use modelspec_core::ir::kernel::{
    ADAMW_BUFFERS, ADAMW_OP, ADAMW_OPERANDS, DispatchCount, ExecutionPlan, Kernel, KernelKind,
    Operand, PlanOp,
};
use modelspec_core::ir::print::print_kernel;
use modelspec_core::ir::tensor::{TensorNode, TensorOp, ValueId};
use modelspec_core::ir::verify::verify_plan;
use modelspec_core::schedule::schedule;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn bert_tiny() -> ExecutionPlan {
    let source = std::fs::read_to_string(repo_root().join("examples/bert_tiny.mspec"))
        .expect("the example reads");
    match compile_artifact("bert_tiny.mspec", &source, Backend::Vulkan) {
        Ok(artifact) => artifact.execution,
        Err(diagnostics) => panic!("compile_artifact refused the example: {diagnostics:#?}"),
    }
}

/// `examples/linear_regression.mspec` with one AdamW-updated parameter.
fn linear() -> CompiledTraining {
    let source = std::fs::read_to_string(repo_root().join("examples/linear_regression.mspec"))
        .expect("the example reads")
        .replace(
            "sgd(lr = 0.1)",
            "adamw(lr = 0.1, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, \
             warmup_steps = 10)",
        );
    compile_training("adamw.mspec", &source).expect("it compiles")
}

fn plan_of(compiled: &CompiledTraining) -> ExecutionPlan {
    schedule(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
    )
    .expect("the step schedules")
}

fn ops(plan: &ExecutionPlan) -> Vec<&'static str> {
    plan.kernels.iter().map(|kernel| kernel.op).collect()
}

/// The positions in `plan.step` of the dispatches of an AdamW update kernel.
fn updates(plan: &ExecutionPlan) -> Vec<usize> {
    plan.step
        .iter()
        .enumerate()
        .filter(|(_, step)| {
            step.op == PlanOp::Dispatch
                && step
                    .kernel_module_id
                    .and_then(|id| plan.kernel(id))
                    .is_some_and(Kernel::is_adamw)
        })
        .map(|(at, _)| at)
        .collect()
}

#[test]
fn bert_tiny_updates_each_parameter_with_one_dispatch_that_writes_three_views_in_place() {
    let plan = bert_tiny();
    verify_plan(&plan).unwrap_or_else(|error| panic!("the plan verifies: {error}"));
    let updates = updates(&plan);
    assert_eq!(updates.len(), 41, "one update per parameter");
    let ops = ops(&plan);
    assert!(
        !ops.contains(&"Sqrt") && !ops.contains(&"Div"),
        "no kernel of the sixteen-op lowering: {ops:?}"
    );
    let mut written = Vec::new();
    for &at in &updates {
        let step = &plan.step[at];
        let kernel = plan
            .kernel(step.kernel_module_id.expect("a dispatch names its module"))
            .expect("the plan holds the kernel");
        assert_eq!(kernel.kind, KernelKind::SgdUpdate, "{}", kernel.signature());
        assert_eq!(kernel.op, ADAMW_OP);
        assert_eq!(kernel.operands.len(), ADAMW_OPERANDS);
        assert!(
            kernel.operands[..ADAMW_BUFFERS]
                .iter()
                .all(|operand| matches!(operand, Operand::Buffer(_)))
                && kernel.operands[ADAMW_BUFFERS..]
                    .iter()
                    .all(|operand| matches!(operand, Operand::Immediate(_))),
            "{}",
            kernel.signature()
        );
        assert_eq!(kernel.written(), 3);
        assert_eq!(kernel.id, Kernel::module_id(&kernel.signature()));
        assert_eq!(step.reads.len(), ADAMW_BUFFERS, "p, g, m, v, lr_t, c1, c2");
        // The parameter and both moments are written where they are read.
        assert_eq!(
            step.writes,
            vec![step.reads[0], step.reads[2], step.reads[3]],
            "step {at}"
        );
        // The three rank-0 inputs are the same for every parameter.
        assert_eq!(
            step.reads[4..],
            plan.step[updates[0]].reads[4..],
            "step {at}"
        );
        assert!(
            matches!(
                plan.dispatch_count(step),
                Some(DispatchCount::Counted { flop, bytes }) if flop > 0 && bytes > 0
            ),
            "{:?}",
            plan.dispatch_count(step)
        );
        written.extend(step.writes.iter().copied());
    }
    // No two updates write one view, so none waits for another.
    let distinct: std::collections::BTreeSet<String> =
        written.iter().map(|view| format!("{view:?}")).collect();
    assert_eq!(distinct.len(), written.len(), "123 written views");
    let (first, last) = (updates[0], updates[updates.len() - 1]);
    let between: Vec<PlanOp> = plan.step[first..=last].iter().map(|step| step.op).collect();
    assert!(
        between.iter().all(|op| *op == PlanOp::Dispatch),
        "no barrier between two updates: {between:?}"
    );
    assert_eq!(last - first + 1, updates.len(), "the updates are adjacent");
    let printed = print_kernel(&plan);
    assert!(
        printed
            .lines()
            .any(|line| line.starts_with("kernel ") && line.contains(" sgd_update AdamW ")),
        "{printed}"
    );
}

#[test]
fn the_linear_example_has_one_update_kernel_and_no_root_or_quotient_kernel() {
    let plan = plan_of(&linear());
    verify_plan(&plan).unwrap_or_else(|error| panic!("the plan verifies: {error}"));
    assert_eq!(updates(&plan).len(), 1);
    let ops = ops(&plan);
    assert!(!ops.contains(&"Sqrt") && !ops.contains(&"Div"), "{ops:?}");
}

/// The node of the update and the ids of `scaled` (`lr_t * step`) and `direction`
/// (`m_hat / denominator`) of the one parameter of [`linear`].
fn rule_of(compiled: &CompiledTraining) -> (ValueId, ValueId, ValueId) {
    let program = &compiled.training.training.program;
    let update = *compiled
        .training
        .optimizer
        .updates
        .values()
        .next()
        .expect("one update");
    let node = |id: ValueId| program.node(id).expect("the node is defined");
    let scaled = node(update).operands[1];
    let step = node(scaled).operands[1];
    let direction = node(step).operands[0];
    assert_eq!(node(direction).op, TensorOp::Div);
    (update, scaled, direction)
}

#[test]
fn an_update_whose_ops_are_not_the_rule_keeps_the_sixteen_op_lowering() {
    let mut compiled = linear();
    let (_, scaled, _) = rule_of(&compiled);
    // `step * lr_t` in place of `lr_t * step`: the same value, another op order.
    let nodes = &mut compiled.training.training.program.nodes;
    let node = nodes
        .iter_mut()
        .find(|node| node.id == scaled)
        .expect("the node is defined");
    assert_eq!(node.op, TensorOp::Mul);
    node.operands.swap(0, 1);
    let plan = plan_of(&compiled);
    verify_plan(&plan).unwrap_or_else(|error| panic!("the plan verifies: {error}"));
    assert!(updates(&plan).is_empty());
    let ops = ops(&plan);
    assert!(ops.contains(&"Sqrt") && ops.contains(&"Div"), "{ops:?}");
}

#[test]
fn an_update_whose_inner_value_has_a_second_reader_keeps_the_sixteen_op_lowering() {
    let mut compiled = linear();
    let (_, _, direction) = rule_of(&compiled);
    let program = &mut compiled.training.training.program;
    let ty = program.node(direction).expect("defined").ty.clone();
    let id = ValueId(
        program
            .nodes
            .iter()
            .map(|node| node.id.0)
            .max()
            .unwrap_or(0)
            + 1,
    );
    program.nodes.push(TensorNode {
        id,
        op: TensorOp::Add,
        operands: vec![direction, direction],
        ty,
        source_span: None,
    });
    let plan = plan_of(&compiled);
    assert!(updates(&plan).is_empty());
    let ops = ops(&plan);
    assert!(ops.contains(&"Sqrt") && ops.contains(&"Div"), "{ops:?}");
}

#[test]
fn bert_tiny_key_side_attention_gradients_run_one_workgroup_per_tile_of_32_keys() {
    let plan = bert_tiny();
    let mut found = 0;
    for step in plan.step.iter().filter(|step| step.op == PlanOp::Dispatch) {
        let kernel = plan
            .kernel(step.kernel_module_id.expect("a dispatch names its module"))
            .expect("the plan holds the kernel");
        if matches!(kernel.op, "AttentionGradK" | "AttentionGradV") {
            found += 1;
            // 32 sequences, 2 heads, 128 / 32 tiles.
            assert_eq!(step.workgroups, Some([256, 1, 1]), "{}", kernel.signature());
            assert!(kernel.softmax_stats.is_some(), "{}", kernel.signature());
        }
    }
    assert_eq!(found, 4, "two layers, two key-side gradients each");
}
