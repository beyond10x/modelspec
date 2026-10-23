//! The plan verifier's work on an untrusted artifact is bounded by the artifact's size
//! (story:plan-verifier-bounded-on-untrusted-artifact, DESIGN.md § 7.1). The reader refuses a plan
//! of more than [`MAX_PLAN_STEPS`] steps, [`MAX_PLAN_DEPENDENCIES`] dependencies or
//! [`MAX_PLAN_VIEWS`] step views with E-ARTIFACT-003 before the verifier runs, and a plan at those
//! limits, in the shape that makes each of the verifier's passes do the most work, is verified in
//! under 10 s and 1 GiB.
//!
//! Each case runs in a child process of this test binary, one at a time. The parent kills the
//! child when its resident memory passes 1 GiB or it is still running after 10 s, so a verifier
//! whose work grows faster than the plan fails the case instead of exhausting the machine.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use modelspec_core::artifact::{
    self, Backend, MAX_PLAN_DEPENDENCIES, MAX_PLAN_STEPS, MAX_PLAN_VIEWS,
};
use modelspec_core::ir::kernel::{
    BufferView, ExecutionPlan, Kernel, KernelKind, KernelModuleId, PlanOp, PlanStep,
};
use modelspec_core::ir::tensor::DType;
use modelspec_core::ir::verify::{PlanName, PlanVerifyError, verify_plan};
use serde_json::{Value, json};

/// The bound of the story's acceptance: wall time and peak resident memory of one case.
const SECONDS: u64 = 10;
const RESIDENT_KIB: u64 = 1 << 20;

/// Set in a child process to the name of the case it runs.
const CASE: &str = "MODELSPEC_PLAN_BOUNDS_CASE";

/// The prefix of the line a child prints: seconds, peak resident KiB, then the outcome.
const RESULT: &str = "plan-bounds-result";

/// One child at a time, so that no case's time includes another's.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// What a child reported.
struct Bounded {
    outcome: String,
    seconds: f64,
    peak_kib: u64,
}

fn in_child() -> bool {
    std::env::var_os(CASE).is_some()
}

/// A `/proc/<pid>/status` field in KiB, or 0 where it cannot be read.
fn status_kib(pid: u32, field: &str) -> u64 {
    std::fs::read_to_string(format!("/proc/{pid}/status"))
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with(field))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|kib| kib.parse().ok())
        })
        .unwrap_or(0)
}

/// In the parent, holds the other cases off until the guard drops, so that no case's input is
/// built while another case's child is timed; in a child, `None`.
fn serial() -> Option<MutexGuard<'static, ()>> {
    (!in_child()).then(|| ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// In the parent, runs the test `case` in a child process of this binary, kills it past the
/// bound and returns its report. In that child, runs `work`, prints the report and returns
/// `None`.
fn bounded(case: &str, work: impl FnOnce() -> String) -> Option<Bounded> {
    if std::env::var(CASE).as_deref() == Ok(case) {
        let start = Instant::now();
        let outcome = work();
        let seconds = start.elapsed().as_secs_f64();
        let peak = status_kib(std::process::id(), "VmHWM:");
        println!("{RESULT} {seconds:.3} {peak} {outcome}");
        return None;
    }
    let mut child = Command::new(std::env::current_exe().expect("the test binary"))
        .args(["--exact", case, "--nocapture", "--test-threads=1"])
        .env(CASE, case)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the test binary runs");
    let start = Instant::now();
    let mut peak = 0;
    let stopped = loop {
        if child.try_wait().expect("the child is waited on").is_some() {
            break None;
        }
        let resident = status_kib(child.id(), "VmRSS:");
        peak = peak.max(resident);
        if resident > RESIDENT_KIB {
            break Some(format!(
                "killed: resident memory {resident} KiB, past 1 GiB, after {:.1} s",
                start.elapsed().as_secs_f64()
            ));
        }
        if start.elapsed() >= Duration::from_secs(SECONDS) {
            break Some(format!(
                "killed: still running after {SECONDS} s, resident peak {peak} KiB"
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    if stopped.is_some() {
        let _ = child.kill();
    }
    let output = child.wait_with_output().expect("the child's output");
    if let Some(why) = stopped {
        panic!("{case}: {why}");
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The report can follow the harness's own `test <name> ... ` on the same line.
    let Some(line) = stdout
        .lines()
        .find_map(|line| Some(line.split_once(RESULT)?.1))
    else {
        panic!(
            "{case}: the child reported nothing ({}):\n{stdout}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    };
    let mut fields = line.trim_start().splitn(3, ' ');
    let seconds = fields.next().and_then(|s| s.parse().ok()).expect("seconds");
    let peak_kib = fields.next().and_then(|s| s.parse().ok()).expect("peak");
    let outcome = fields.next().unwrap_or_default().to_owned();
    eprintln!("{case}: {seconds:.3} s, peak resident {peak_kib} KiB: {outcome}");
    assert!(
        seconds < SECONDS as f64,
        "{case}: {seconds:.3} s, not under {SECONDS} s"
    );
    assert!(
        peak_kib < RESIDENT_KIB,
        "{case}: peak resident {peak_kib} KiB, not under 1 GiB"
    );
    Some(Bounded {
        outcome,
        seconds,
        peak_kib,
    })
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("plan_verifier_bounds")
        .join(name)
}

/// A fresh Cpu artifact of the linear-regression example in `dir`, with `steps` as its
/// `field` (`plan` or `forward_plan`), re-sealed so that it passes E-ARTIFACT-002.
fn cpu_artifact_with(dir: &Path, field: &str, steps: impl FnOnce(&Value) -> Vec<Value>) {
    if dir.is_dir() {
        std::fs::remove_dir_all(dir).expect("the old test dir is removed");
    }
    let source = std::fs::read_to_string(root().join("examples/linear_regression.mspec")).unwrap();
    let compiled =
        modelspec_core::compile::compile_artifact("linear_regression.mspec", &source, Backend::Cpu)
            .unwrap_or_else(|d| panic!("{d:?}"));
    artifact::write(dir, &compiled, &[]).expect("written");
    let path = dir.join(artifact::MANIFEST_FILE);
    let mut manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let view = manifest["loss_view"].clone();
    manifest[field] = Value::Array(steps(&view));
    std::fs::write(&path, serde_json::to_string(&manifest).unwrap()).unwrap();
    artifact::seal(dir).expect("re-sealed");
    let bytes = std::fs::metadata(&path).unwrap().len();
    assert!(
        bytes <= artifact::MAX_MANIFEST_BYTES,
        "the manifest has {bytes} bytes, more than the reader parses"
    );
}

fn wire_step(op: &str, depends_on: impl IntoIterator<Item = usize>, writes: Vec<Value>) -> Value {
    json!({
        "op": op,
        "kernel_module_id": null,
        "workgroups": null,
        "reads": [],
        "writes": writes,
        "depends_on": depends_on.into_iter().collect::<Vec<_>>(),
        "value": null,
    })
}

/// `n` steps: an Allocate, Barriers each on the one before it, and the fence.
fn wire_chain(n: usize) -> Vec<Value> {
    let mut steps = vec![wire_step("Allocate", [], Vec::new())];
    steps.extend((1..n - 1).map(|i| wire_step("Barrier", [i - 1], Vec::new())));
    steps.push(wire_step("Barrier", [], Vec::new()));
    steps
}

/// `n` steps with `dependencies` in all: Allocates, then Barriers each on up to 17 of the steps
/// just before it, and the fence.
fn wire_dependencies(n: usize, dependencies: usize) -> Vec<Value> {
    let mut left = dependencies;
    let mut counts = vec![0; n];
    for i in (1..n - 1).rev() {
        counts[i] = left.min(i).min(17);
        left -= counts[i];
    }
    assert_eq!(left, 0, "{n} steps hold {dependencies} dependencies");
    let mut steps: Vec<Value> = counts[..n - 1]
        .iter()
        .enumerate()
        .map(|(i, &k)| match k {
            0 => wire_step("Allocate", [], Vec::new()),
            k => wire_step("Barrier", i - k..i, Vec::new()),
        })
        .collect();
    steps.push(wire_step("Barrier", [], Vec::new()));
    steps
}

/// Allocates of up to 17 copies of `view` each, `views` in all, and the fence.
fn wire_views(views: usize, view: &Value) -> Vec<Value> {
    let mut steps = Vec::new();
    let mut left = views;
    while left > 0 {
        let k = left.min(17);
        steps.push(wire_step("Allocate", [], vec![view.clone(); k]));
        left -= k;
    }
    steps.push(wire_step("Barrier", [], Vec::new()));
    steps
}

/// The code and the invariant of a read, or `read`.
fn read_outcome(dir: &Path) -> String {
    match artifact::read(dir) {
        Ok(_) => "read".to_owned(),
        Err(d) => {
            let invariant = d
                .details
                .iter()
                .find(|detail| detail.label.as_deref() == Some("invariant"))
                .map_or("", |detail| detail.text.as_str());
            format!("{} {invariant}", d.code.as_str())
        }
    }
}

/// The outcomes of reading each of `dirs`, joined by ` | `.
fn read_outcomes(dirs: &[PathBuf]) -> String {
    dirs.iter()
        .map(|dir| read_outcome(dir))
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Acceptance 1: a re-sealed artifact whose step plan is a 300,000-step dependency chain (about
/// 59 MB of manifest, under the 64 MiB the reader parses) is refused in under 10 s and 1 GiB. The
/// verifier before this story kept every step's transitive dependencies as a set, about 45 * 10^9
/// entries for this plan.
#[test]
fn a_re_sealed_300000_step_chain_is_refused_in_under_10_s_and_1_gib() {
    const CASE_NAME: &str = "a_re_sealed_300000_step_chain_is_refused_in_under_10_s_and_1_gib";
    let _serial = serial();
    let dir = scratch("chain-300000");
    if !in_child() {
        cpu_artifact_with(&dir, "plan", |_| wire_chain(300_000));
    }
    let Some(run) = bounded(CASE_NAME, || read_outcome(&dir)) else {
        return;
    };
    assert_eq!(
        run.outcome,
        format!("E-ARTIFACT-003 plan has at most {MAX_PLAN_STEPS} steps (it has 300000)"),
        "after {:.3} s, {} KiB",
        run.seconds,
        run.peak_kib
    );
}

/// For the step plan and the forward plan: a plan of exactly [`MAX_PLAN_STEPS`] steps passes the
/// size check (the verifier then refuses it for another reason) and one more step is refused.
#[test]
fn a_plan_one_step_past_the_limit_is_refused_before_the_verifier() {
    const CASE_NAME: &str = "a_plan_one_step_past_the_limit_is_refused_before_the_verifier";
    let _serial = serial();
    let dirs: Vec<PathBuf> = ["plan", "forward_plan"]
        .into_iter()
        .flat_map(|field| {
            [MAX_PLAN_STEPS, MAX_PLAN_STEPS + 1].map(|n| {
                let dir = scratch(&format!("steps-{field}-{n}"));
                if !in_child() {
                    cpu_artifact_with(&dir, field, |_| wire_chain(n));
                }
                dir
            })
        })
        .collect();
    let Some(run) = bounded(CASE_NAME, || read_outcomes(&dirs)) else {
        return;
    };
    let past = MAX_PLAN_STEPS + 1;
    let outcomes: Vec<&str> = run.outcome.split(" | ").collect();
    assert_eq!(outcomes.len(), 4, "{}", run.outcome);
    for (outcome, field) in [(outcomes[0], "plan"), (outcomes[2], "forward_plan")] {
        assert!(
            !outcome.contains("at most"),
            "{field} at the limit: {outcome}"
        );
    }
    assert_eq!(
        outcomes[1],
        format!("E-ARTIFACT-003 plan has at most {MAX_PLAN_STEPS} steps (it has {past})")
    );
    assert_eq!(
        outcomes[3],
        format!("E-ARTIFACT-003 forward_plan has at most {MAX_PLAN_STEPS} steps (it has {past})")
    );
}

/// For the step plan and the forward plan: [`MAX_PLAN_STEPS`] steps holding exactly
/// [`MAX_PLAN_DEPENDENCIES`] dependencies pass the size check, and one more is refused.
#[test]
fn a_plan_one_dependency_past_the_limit_is_refused_before_the_verifier() {
    const CASE_NAME: &str = "a_plan_one_dependency_past_the_limit_is_refused_before_the_verifier";
    let _serial = serial();
    let dirs: Vec<PathBuf> = ["plan", "forward_plan"]
        .into_iter()
        .flat_map(|field| {
            [MAX_PLAN_DEPENDENCIES, MAX_PLAN_DEPENDENCIES + 1].map(|d| {
                let dir = scratch(&format!("dependencies-{field}-{d}"));
                if !in_child() {
                    cpu_artifact_with(&dir, field, |_| wire_dependencies(MAX_PLAN_STEPS, d));
                }
                dir
            })
        })
        .collect();
    let Some(run) = bounded(CASE_NAME, || read_outcomes(&dirs)) else {
        return;
    };
    let past = MAX_PLAN_DEPENDENCIES + 1;
    let outcomes: Vec<&str> = run.outcome.split(" | ").collect();
    assert_eq!(outcomes.len(), 4, "{}", run.outcome);
    for (outcome, field) in [(outcomes[0], "plan"), (outcomes[2], "forward_plan")] {
        assert!(
            !outcome.contains("at most"),
            "{field} at the limit: {outcome}"
        );
    }
    assert_eq!(
        outcomes[1],
        format!(
            "E-ARTIFACT-003 plan has at most {MAX_PLAN_DEPENDENCIES} dependencies (it has {past})"
        )
    );
    assert_eq!(
        outcomes[3],
        format!(
            "E-ARTIFACT-003 forward_plan has at most {MAX_PLAN_DEPENDENCIES} dependencies (it has \
             {past})"
        )
    );
}

/// For the step plan and the forward plan: steps holding exactly [`MAX_PLAN_VIEWS`] reads and
/// writes pass the size check, and one more is refused.
#[test]
fn a_plan_one_view_past_the_limit_is_refused_before_the_verifier() {
    const CASE_NAME: &str = "a_plan_one_view_past_the_limit_is_refused_before_the_verifier";
    let _serial = serial();
    let dirs: Vec<PathBuf> = ["plan", "forward_plan"]
        .into_iter()
        .flat_map(|field| {
            [MAX_PLAN_VIEWS, MAX_PLAN_VIEWS + 1].map(|v| {
                let dir = scratch(&format!("views-{field}-{v}"));
                if !in_child() {
                    cpu_artifact_with(&dir, field, |view| wire_views(v, view));
                }
                dir
            })
        })
        .collect();
    let Some(run) = bounded(CASE_NAME, || read_outcomes(&dirs)) else {
        return;
    };
    let past = MAX_PLAN_VIEWS + 1;
    let outcomes: Vec<&str> = run.outcome.split(" | ").collect();
    assert_eq!(outcomes.len(), 4, "{}", run.outcome);
    for (outcome, field) in [(outcomes[0], "plan"), (outcomes[2], "forward_plan")] {
        assert!(
            !outcome.contains("at most"),
            "{field} at the limit: {outcome}"
        );
    }
    assert_eq!(
        outcomes[1],
        format!("E-ARTIFACT-003 plan has at most {MAX_PLAN_VIEWS} views (it has {past})")
    );
    assert_eq!(
        outcomes[3],
        format!("E-ARTIFACT-003 forward_plan has at most {MAX_PLAN_VIEWS} views (it has {past})")
    );
}

/// The execution plan of the MNIST example compiled for Vulkan: its forward plan and kernels
/// stay, and each case below replaces its step plan.
fn mnist() -> ExecutionPlan {
    let source = std::fs::read_to_string(root().join("examples/mnist_mlp.mspec")).unwrap();
    modelspec_core::compile::compile_artifact("mnist_mlp.mspec", &source, Backend::Vulkan)
        .unwrap_or_else(|d| panic!("{d:?}"))
        .execution
}

fn step(op: PlanOp, depends_on: Vec<usize>) -> PlanStep {
    PlanStep {
        op,
        kernel_module_id: None,
        workgroups: None,
        reads: Vec::new(),
        writes: Vec::new(),
        depends_on,
        value: None,
    }
}

fn dispatch(
    kernel: &Kernel,
    depends_on: Vec<usize>,
    reads: Vec<BufferView>,
    writes: Vec<BufferView>,
) -> PlanStep {
    PlanStep {
        kernel_module_id: Some(kernel.id),
        workgroups: Some(kernel.workgroups),
        reads,
        writes,
        ..step(PlanOp::Dispatch, depends_on)
    }
}

/// One f32 at the start of buffer `buffer`.
fn view(buffer: u32) -> BufferView {
    BufferView {
        buffer,
        dtype: DType::F32,
        element_count: 1,
        offset: 0,
        size_bytes: 4,
        alignment: 4,
    }
}

/// A new 4-byte buffer of `plan`.
fn buffer(plan: &mut ExecutionPlan) -> u32 {
    plan.buffers.push(4);
    u32::try_from(plan.buffers.len() - 1).unwrap()
}

/// An elementwise kernel of `plan`.
fn elementwise(plan: &ExecutionPlan) -> Kernel {
    plan.kernels
        .iter()
        .find(|k| k.kind == KernelKind::Elementwise)
        .expect("an elementwise kernel")
        .clone()
}

/// `template` as `kind`, with an id no compiled kernel has.
fn kernel_as(template: &Kernel, kind: KernelKind, n: u32) -> Kernel {
    let mut kernel = template.clone();
    kernel.kind = kind;
    let mut id = [0xa5; 16];
    id[12..].copy_from_slice(&n.to_le_bytes());
    kernel.id = KernelModuleId(id);
    kernel
}

/// Every plan below passes each pass of the verifier and fails only its last check: the step
/// plan does not write the loss view.
fn passes_every_pass() -> String {
    format!(
        "{:?}",
        Err::<(), _>(PlanVerifyError::BoundViewNotWritten {
            name: "loss".to_owned(),
            plan: PlanName::Step,
        })
    )
}

fn verify_outcome(plan: &ExecutionPlan) -> String {
    format!("{:?}", verify_plan(plan))
}

fn assert_sizes(plan: &ExecutionPlan) {
    let steps = plan.step.len();
    let dependencies: usize = plan.step.iter().map(|s| s.depends_on.len()).sum();
    let views: usize = plan
        .step
        .iter()
        .map(|s| s.reads.len() + s.writes.len())
        .sum();
    assert!(steps <= MAX_PLAN_STEPS, "{steps} steps");
    assert!(
        dependencies <= MAX_PLAN_DEPENDENCIES,
        "{dependencies} dependencies"
    );
    assert!(views <= MAX_PLAN_VIEWS, "{views} views");
}

/// [`MAX_PLAN_STEPS`] steps, each Barrier on the 16 steps before it: the transitive closure of
/// every step is most of the plan, and the old verifier merged it once per dependency.
#[test]
fn dense_dependencies_at_the_limits_verify_in_bounds() {
    const CASE_NAME: &str = "dense_dependencies_at_the_limits_verify_in_bounds";
    let _serial = serial();
    let mut plan = mnist();
    let n = MAX_PLAN_STEPS;
    let mut steps = vec![step(PlanOp::Allocate, Vec::new())];
    steps.extend((1..n - 1).map(|i| step(PlanOp::Barrier, (i.saturating_sub(16)..i).collect())));
    steps.push(step(PlanOp::Barrier, Vec::new()));
    plan.step = steps;
    assert_sizes(&plan);
    let Some(run) = bounded(CASE_NAME, || verify_outcome(&plan)) else {
        return;
    };
    assert_eq!(run.outcome, passes_every_pass());
}

/// Allocates of 16 new buffers each, up to [`MAX_PLAN_VIEWS`]: the old verifier compared each
/// allocated buffer with every view of every earlier step.
#[test]
fn allocates_at_the_limits_verify_in_bounds() {
    const CASE_NAME: &str = "allocates_at_the_limits_verify_in_bounds";
    let _serial = serial();
    let mut plan = mnist();
    let mut steps = Vec::new();
    for _ in 0..MAX_PLAN_VIEWS / 16 - 1 {
        let mut allocate = step(PlanOp::Allocate, Vec::new());
        allocate.writes = (0..16).map(|_| view(buffer(&mut plan))).collect();
        steps.push(allocate);
    }
    steps.push(step(PlanOp::Barrier, Vec::new()));
    plan.step = steps;
    assert_sizes(&plan);
    let Some(run) = bounded(CASE_NAME, || verify_outcome(&plan)) else {
        return;
    };
    assert_eq!(run.outcome, passes_every_pass());
}

/// Dispatches that each read and write one buffer, a Barrier between each two: every pair
/// conflicts, and the old verifier searched the closure for a Barrier between each pair.
#[test]
fn conflicting_dispatches_at_the_limits_verify_in_bounds() {
    const CASE_NAME: &str = "conflicting_dispatches_at_the_limits_verify_in_bounds";
    let _serial = serial();
    let mut plan = mnist();
    let kernel = kernel_as(&elementwise(&plan), KernelKind::Elementwise, 0);
    plan.kernels.push(kernel.clone());
    let x = buffer(&mut plan);
    let n = MAX_PLAN_STEPS;
    let mut steps = Vec::new();
    for i in 0..n - 1 {
        let after = if i == 0 { Vec::new() } else { vec![i - 1] };
        steps.push(if i % 2 == 0 {
            dispatch(&kernel, after, vec![view(x)], vec![view(x)])
        } else {
            step(PlanOp::Barrier, after)
        });
    }
    steps.push(step(PlanOp::Barrier, Vec::new()));
    plan.step = steps;
    assert_sizes(&plan);
    let Some(run) = bounded(CASE_NAME, || verify_outcome(&plan)) else {
        return;
    };
    assert_eq!(run.outcome, passes_every_pass());
}

/// One stage 1 and then stage 2 Dispatches up to [`MAX_PLAN_STEPS`], with 4096 other kernels
/// listed before theirs: the old verifier looked each step's kernel up in the list, and searched
/// back over every earlier step for each stage 2's stage 1.
#[test]
fn stage_2_dispatches_at_the_limits_verify_in_bounds() {
    const CASE_NAME: &str = "stage_2_dispatches_at_the_limits_verify_in_bounds";
    let _serial = serial();
    let mut plan = mnist();
    let template = elementwise(&plan);
    let compiled = std::mem::take(&mut plan.kernels);
    plan.kernels = (1..=4096)
        .map(|n| kernel_as(&template, KernelKind::Elementwise, n))
        .collect();
    plan.kernels.extend(compiled);
    let stage1 = kernel_as(&template, KernelKind::ReduceStage1, 5000);
    let stage2 = kernel_as(&template, KernelKind::ReduceStage2, 5001);
    plan.kernels.extend([stage1.clone(), stage2.clone()]);
    let partials = buffer(&mut plan);
    let mut allocate = step(PlanOp::Allocate, Vec::new());
    allocate.writes.push(view(partials));
    let mut steps = vec![
        allocate,
        dispatch(&stage1, vec![0], Vec::new(), vec![view(partials)]),
        step(PlanOp::Barrier, vec![1]),
    ];
    for _ in 3..MAX_PLAN_STEPS - 1 {
        let out = buffer(&mut plan);
        steps.push(dispatch(
            &stage2,
            vec![2],
            vec![view(partials)],
            vec![view(out)],
        ));
    }
    steps.push(step(PlanOp::Barrier, Vec::new()));
    plan.step = steps;
    assert_sizes(&plan);
    let Some(run) = bounded(CASE_NAME, || verify_outcome(&plan)) else {
        return;
    };
    assert_eq!(run.outcome, passes_every_pass());
}

/// 2048 Dispatches reading a parameter (its view last of 16 reads), a Barrier on all of them,
/// then SGD updates writing the parameter through 16 views each, a Barrier after each: the old
/// verifier searched every earlier step for a reader per update view, every earlier step for an
/// update per parameter read, and the closure for a Barrier per conflicting pair.
#[test]
fn parameter_updates_at_the_limits_verify_in_bounds() {
    const CASE_NAME: &str = "parameter_updates_at_the_limits_verify_in_bounds";
    let _serial = serial();
    let mut plan = mnist();
    let parameter = plan.parameters[0].views[0].buffer;
    let reader = kernel_as(&elementwise(&plan), KernelKind::Elementwise, 0);
    let update = kernel_as(&elementwise(&plan), KernelKind::SgdUpdate, 1);
    plan.kernels.extend([reader.clone(), update.clone()]);
    let mut reads: Vec<BufferView> = (0..15).map(|_| view(buffer(&mut plan))).collect();
    reads.push(view(parameter));
    let readers = 2048;
    let mut steps: Vec<PlanStep> = (0..readers)
        .map(|_| dispatch(&reader, Vec::new(), reads.clone(), Vec::new()))
        .collect();
    steps.push(step(PlanOp::Barrier, (0..readers).collect()));
    while steps.len() < MAX_PLAN_STEPS - 2 {
        let barrier = steps.len() - 1;
        steps.push(dispatch(
            &update,
            vec![barrier],
            Vec::new(),
            vec![view(parameter); 16],
        ));
        steps.push(step(PlanOp::Barrier, vec![barrier + 1]));
    }
    steps.push(step(PlanOp::Barrier, Vec::new()));
    plan.step = steps;
    assert_sizes(&plan);
    let Some(run) = bounded(CASE_NAME, || verify_outcome(&plan)) else {
        return;
    };
    assert_eq!(run.outcome, passes_every_pass());
}
