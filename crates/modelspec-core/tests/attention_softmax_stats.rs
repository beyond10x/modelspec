//! The softmax statistics the attention kernels share (story:vulkan-attention-softmax-stats-reused,
//! DESIGN.md § 3.1, § 5.2): in the step plan of `examples/bert_tiny.mspec` each `Attention`
//! dispatch writes the maximum and sum of each query head's softmax into a scratch buffer an
//! `Allocate` makes, the `AttentionGradQ` dispatch of the same operands writes the row dot of its
//! gradient beside them, and `AttentionGradK` and `AttentionGradV` read the buffer instead of
//! recomputing the three values in every key invocation. The buffer is a plan buffer like any
//! other, and the plan verifier refuses a backward dispatch that reads it before the dispatches
//! of its recording that write it.

use std::path::{Path, PathBuf};

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile;
use modelspec_core::ir::kernel::{BufferView, ExecutionPlan, PlanOp, PlanStep};
use modelspec_core::ir::verify::verify_plan;

const EXAMPLE: &str = "examples/bert_tiny.mspec";

/// Rows and heads of the example's attention calls: 4,096 rows of 2 heads, three statistics each.
const STATISTICS: u64 = 4096 * 2 * 3;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn artifact() -> CompiledArtifact {
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example reads");
    match compile::compile_artifact("bert_tiny.mspec", &source, Backend::Vulkan) {
        Ok(artifact) => artifact,
        Err(diagnostics) => panic!("compile_artifact refused the example: {diagnostics:#?}"),
    }
}

/// The op of the kernel a Dispatch step runs; `None` for every other step.
fn op_of<'p>(plan: &'p ExecutionPlan, step: &PlanStep) -> Option<&'p str> {
    let kernel = plan.kernel(step.kernel_module_id?)?;
    Some(kernel.op)
}

/// One attention call's dispatches in `steps`, by position: the forward, and the three
/// gradients that read the forward's `q`, `k`, `v` and key mask.
struct Site {
    forward: usize,
    grad_q: Option<usize>,
    grad_k: Option<usize>,
    grad_v: Option<usize>,
}

fn sites(plan: &ExecutionPlan, steps: &[PlanStep]) -> Vec<Site> {
    let at = |op: &str, operands: &[BufferView]| {
        steps
            .iter()
            .position(|step| op_of(plan, step) == Some(op) && step.reads.get(..4) == Some(operands))
    };
    steps
        .iter()
        .enumerate()
        .filter(|(_, step)| op_of(plan, step) == Some("Attention"))
        .map(|(forward, step)| {
            let operands = &step.reads[..4];
            Site {
                forward,
                grad_q: at("AttentionGradQ", operands),
                grad_k: at("AttentionGradK", operands),
                grad_v: at("AttentionGradV", operands),
            }
        })
        .collect()
}

/// The statistics view the forward dispatch of `site` writes before its result.
fn statistics(steps: &[PlanStep], site: &Site) -> BufferView {
    let writes = &steps[site.forward].writes;
    assert_eq!(
        writes.len(),
        2,
        "the forward attention dispatch writes its softmax statistics, then its result: {writes:?}"
    );
    writes[0]
}

/// Acceptance 4: each attention call of the example has its own statistics buffer of
/// `[rows · heads, 3]` `f32`, a plan buffer made by an `Allocate` before the forward dispatch in
/// both plans; the forward writes it, `AttentionGradQ` writes it before its result, and
/// `AttentionGradK` and `AttentionGradV` read it after their five buffer operands.
#[test]
fn bert_tiny_backward_attention_reads_the_statistics_the_forward_and_grad_q_write() {
    let artifact = artifact();
    let plan = &artifact.execution;
    let step_sites = sites(plan, &plan.step);
    assert_eq!(step_sites.len(), 2, "two attention calls");
    let mut buffers = Vec::new();
    for site in &step_sites {
        let stats = statistics(&plan.step, site);
        assert_eq!(stats.element_count, STATISTICS, "{stats}");
        assert_eq!(
            plan.buffers[stats.buffer as usize],
            STATISTICS * 4,
            "{stats}"
        );
        let allocate = plan.step[..site.forward].iter().position(|step| {
            step.op == PlanOp::Allocate && step.writes.iter().any(|v| v.buffer == stats.buffer)
        });
        assert!(
            allocate.is_some(),
            "an Allocate makes {stats} before the forward"
        );
        let (grad_q, grad_k, grad_v) = (
            site.grad_q.expect("an AttentionGradQ dispatch"),
            site.grad_k.expect("an AttentionGradK dispatch"),
            site.grad_v.expect("an AttentionGradV dispatch"),
        );
        assert!(
            site.forward < grad_q && grad_q < grad_k,
            "forward, GradQ, then GradK"
        );
        let q = &plan.step[grad_q];
        assert_eq!(
            q.writes.len(),
            2,
            "GradQ writes the row dots, then its result"
        );
        assert_eq!(
            q.writes[0], stats,
            "GradQ writes the forward's statistics buffer"
        );
        for (name, at) in [("AttentionGradK", grad_k), ("AttentionGradV", grad_v)] {
            let reads = &plan.step[at].reads;
            assert_eq!(
                reads.len(),
                6,
                "{name} reads q, k, v, the mask, G and the statistics"
            );
            assert_eq!(
                reads[5], stats,
                "{name} reads the forward's statistics buffer"
            );
        }
        buffers.push(stats.buffer);
    }
    assert_ne!(
        buffers[0], buffers[1],
        "each call has its own statistics buffer"
    );

    let forward_sites = sites(plan, &plan.forward);
    assert_eq!(forward_sites.len(), 2, "the forward plan runs both calls");
    for site in &forward_sites {
        let stats = statistics(&plan.forward, site);
        assert!(
            plan.forward[..site.forward]
                .iter()
                .any(|step| step.op == PlanOp::Allocate
                    && step.writes.iter().any(|v| v.buffer == stats.buffer)),
            "the forward plan allocates {stats} before its forward dispatch"
        );
    }
    verify_plan(plan).expect("the plan verifies");
}

/// Acceptance 4: the plan verifier refuses a step plan in which a backward attention dispatch
/// reads the statistics buffer before the dispatches of its recording write it: the forward
/// writes no statistics (GradQ reads them unwritten), GradQ writes no row dots (GradK reads them
/// unwritten), or no `Allocate` made the buffer.
#[test]
fn the_plan_verifier_refuses_statistics_read_before_they_are_written() {
    let artifact = artifact();
    let plan = &artifact.execution;
    let site = &sites(plan, &plan.step)[0];
    let stats = statistics(&plan.step, site);
    let (grad_q, grad_k) = (site.grad_q.unwrap(), site.grad_k.unwrap());
    let refusal = |mutate: &dyn Fn(&mut ExecutionPlan)| {
        let mut broken = plan.clone();
        mutate(&mut broken);
        match verify_plan(&broken) {
            Ok(()) => panic!("the verifier accepts the broken plan"),
            Err(error) => error.to_string(),
        }
    };

    let no_forward_write = refusal(&|p| {
        p.step[site.forward].writes.remove(0);
    });
    assert!(
        no_forward_write.contains(&format!("step {grad_q} "))
            && no_forward_write.contains("softmax statistics")
            && no_forward_write.contains(&format!("b{}", stats.buffer)),
        "{no_forward_write}"
    );

    let no_row_dot = refusal(&|p| {
        p.step[grad_q].writes.remove(0);
    });
    assert!(
        no_row_dot.contains(&format!("step {grad_k} "))
            && no_row_dot.contains("row dot")
            && no_row_dot.contains(&format!("b{}", stats.buffer)),
        "{no_row_dot}"
    );

    let not_allocated = refusal(&|p| {
        for step in p.step.iter_mut().filter(|s| s.op == PlanOp::Allocate) {
            step.writes.retain(|v| v.buffer != stats.buffer);
        }
    });
    assert!(
        not_allocated.contains("Allocate") && not_allocated.contains(&format!("b{}", stats.buffer)),
        "{not_allocated}"
    );
}
