//! Adversary cases for story:plan-verifier-bounded-on-untrusted-artifact (DESIGN.md § 7.1): an
//! untrusted artifact or plan whose reading or verifying was once outside the story's bound of
//! 10 s and 1 GiB, or outside the bound DESIGN.md states for the verifier, kept so that it stays
//! inside.
//!
//! The harness is the one of `plan_verifier_bounds.rs`: each case runs in a child process of this
//! test binary, one at a time, and the parent kills the child when its resident memory passes
//! 1 GiB or it is still running after 10 s.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use modelspec_core::artifact::{
    self, Backend, MAX_MANIFEST_BYTES, MAX_PLAN_DEPENDENCIES, MAX_PLAN_STEPS, MAX_PLAN_VIEWS,
};
use modelspec_core::ir::kernel::{BufferView, ExecutionPlan, PlanOp, PlanStep};
use modelspec_core::ir::tensor::DType;
use modelspec_core::ir::verify::{PlanName, PlanVerifyError, verify_plan};
use serde_json::{Value, json};

/// The bound of the story's acceptance: wall time and peak resident memory of one case.
const SECONDS: u64 = 10;
const RESIDENT_KIB: u64 = 1 << 20;

/// Set in a child process to the name of the case it runs.
const CASE: &str = "MODELSPEC_ADVERSARY_BOUNDS_CASE";

/// The prefix of the line a child prints: seconds, peak resident KiB, then the outcome.
const RESULT: &str = "adversary-bounds-result";

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
        .join("adversary_plan_verifier_bounds")
        .join(name)
}

/// A fresh Cpu artifact of the linear-regression example in `dir`, and its manifest file.
fn cpu_artifact(dir: &Path) -> PathBuf {
    if dir.is_dir() {
        std::fs::remove_dir_all(dir).expect("the old test dir is removed");
    }
    let source = std::fs::read_to_string(root().join("examples/linear_regression.mspec")).unwrap();
    let compiled =
        modelspec_core::compile::compile_artifact("linear_regression.mspec", &source, Backend::Cpu)
            .unwrap_or_else(|d| panic!("{d:?}"));
    artifact::write(dir, &compiled, &[]).expect("written");
    dir.join(artifact::MANIFEST_FILE)
}

fn wire_step(
    op: &str,
    depends_on: impl IntoIterator<Item = usize>,
    reads: Vec<Value>,
    writes: Vec<Value>,
) -> Value {
    json!({
        "op": op,
        "kernel_module_id": null,
        "workgroups": null,
        "reads": reads,
        "writes": writes,
        "depends_on": depends_on.into_iter().collect::<Vec<_>>(),
        "value": null,
    })
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

/// The value of the first line `"<key>": "<value>",` of the pretty manifest at `path`.
fn pretty_field(path: &Path, key: &str) -> String {
    use std::io::BufRead;
    let prefix = format!("\"{key}\": \"");
    let file = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    file.lines()
        .map(Result::unwrap)
        .find_map(|line| {
            let value = line.trim_start().strip_prefix(&prefix)?;
            Some(value.split('"').next()?.to_owned())
        })
        .unwrap_or_else(|| panic!("{key} in {}", path.display()))
}

/// A Cpu artifact in `dir` whose step plan is an Allocate, a Barrier and the fence, the Barrier
/// on step 0 as many times as fill the manifest to [`MAX_MANIFEST_BYTES`] written compactly
/// (`0,` each), and sealed. Returns how many dependencies it has.
fn dependency_padded_artifact(dir: &Path) -> usize {
    const MARK: usize = 987_654_321;
    let path = cpu_artifact(dir);
    let mut manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let old_hash = manifest["artifact_hash"].as_str().unwrap().to_owned();
    let old_id = manifest["artifact_id"].as_str().unwrap().to_owned();
    manifest["plan"] = json!([
        wire_step("Allocate", [], Vec::new(), Vec::new()),
        wire_step("Barrier", [MARK], Vec::new(), Vec::new()),
        wire_step("Barrier", [], Vec::new(), Vec::new()),
    ]);
    let text = serde_json::to_string(&manifest).unwrap();
    let mark = MARK.to_string();
    assert_eq!(text.matches(&mark).count(), 1);
    let rest = text.len() - mark.len();
    let limit = usize::try_from(MAX_MANIFEST_BYTES).unwrap();
    // `0,` per dependency, the last without its comma.
    let dependencies = (limit - rest).div_ceil(2);
    let padding = "0,".repeat(dependencies - 1) + "0";
    let mut text = text.replacen(&mark, &padding, 1);
    drop(padding);
    std::fs::write(&path, &text).unwrap();
    // The hash is over the manifest's content, not its layout: seal, which writes it pretty and
    // too large to read, then put the sealed hash and id into the compact text.
    artifact::seal(dir).expect("re-sealed");
    let new_hash = pretty_field(&path, "artifact_hash");
    let new_id = pretty_field(&path, "artifact_id");
    for (key, old, new) in [
        ("artifact_hash", &old_hash, &new_hash),
        ("artifact_id", &old_id, &new_id),
    ] {
        let from = format!("\"{key}\":\"{old}\"");
        assert_eq!(text.matches(&from).count(), 1, "{key} in the compact text");
        text = text.replacen(&from, &format!("\"{key}\":\"{new}\""), 1);
    }
    std::fs::write(&path, &text).unwrap();
    let bytes = std::fs::metadata(&path).unwrap().len();
    assert!(
        bytes <= MAX_MANIFEST_BYTES,
        "the manifest has {bytes} bytes, more than the reader parses"
    );
    dependencies
}

/// Bounds: a sealed artifact.json of at most [`MAX_MANIFEST_BYTES`] whose step plan is three
/// steps, one of them a Barrier on step 0 about 33.5 million times, so that the plan is far past
/// [`MAX_PLAN_DEPENDENCIES`] and the reader refuses it with E-ARTIFACT-003. The reader counts the
/// plan's arrays as it parses the manifest and keeps no step past a limit; it once parsed the
/// whole file into a `serde_json::Value` first, about 32 bytes per 2-byte `0,`, and reached the
/// refusal only past 1 GiB.
#[test]
fn a_manifest_of_padded_dependencies_is_refused_in_under_10_s_and_1_gib() {
    const CASE_NAME: &str = "a_manifest_of_padded_dependencies_is_refused_in_under_10_s_and_1_gib";
    let _serial = serial();
    let dir = scratch("dependency-padded");
    let dependencies = if in_child() {
        0
    } else {
        dependency_padded_artifact(&dir)
    };
    let Some(run) = bounded(CASE_NAME, || read_outcome(&dir)) else {
        return;
    };
    assert_eq!(
        run.outcome,
        format!(
            "E-ARTIFACT-003 plan has at most {MAX_PLAN_DEPENDENCIES} dependencies (it has \
             {dependencies})"
        ),
        "after {:.3} s, {} KiB",
        run.seconds,
        run.peak_kib
    );
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

/// The execution plan of the MNIST example compiled for Vulkan.
fn mnist() -> ExecutionPlan {
    let source = std::fs::read_to_string(root().join("examples/mnist_mlp.mspec")).unwrap();
    modelspec_core::compile::compile_artifact("mnist_mlp.mspec", &source, Backend::Vulkan)
        .unwrap_or_else(|d| panic!("{d:?}"))
        .execution
}

fn assert_sizes(steps: &[PlanStep]) {
    let dependencies: usize = steps.iter().map(|s| s.depends_on.len()).sum();
    let views: usize = steps.iter().map(|s| s.reads.len() + s.writes.len()).sum();
    assert!(steps.len() <= MAX_PLAN_STEPS, "{} steps", steps.len());
    assert!(
        dependencies <= MAX_PLAN_DEPENDENCIES,
        "{dependencies} dependencies"
    );
    assert!(views <= MAX_PLAN_VIEWS, "{views} views");
}

/// The MNIST plans with a step plan of two Copy steps, the second ordered after the first with
/// no Barrier between, then Barriers up to [`MAX_PLAN_STEPS`]: the first writes 32,767 views of
/// a buffer `c` and one of `b`, the second reads 32,767 views of `a` and one of `b`, the views
/// of `b` first if `b_first` and last otherwise. Returns the plan and `b`.
fn unsynced_copy_pair(b_first: bool) -> (ExecutionPlan, u32) {
    let mut plan = mnist();
    let a = buffer(&mut plan);
    let b = buffer(&mut plan);
    let c = buffer(&mut plan);
    let half = MAX_PLAN_VIEWS / 2;
    let mut first = step(PlanOp::Copy, Vec::new());
    first.writes = vec![view(c); half - 1];
    let mut second = step(PlanOp::Copy, vec![0]);
    second.reads = vec![view(a); half - 1];
    let at = if b_first { 0 } else { half - 1 };
    first.writes.insert(at, view(b));
    second.reads.insert(at, view(b));
    let mut steps = vec![first, second];
    steps.extend((2..MAX_PLAN_STEPS - 1).map(|i| step(PlanOp::Barrier, vec![i - 1])));
    steps.push(step(PlanOp::Barrier, Vec::new()));
    assert_sizes(&steps);
    plan.step = steps;
    (plan, b)
}

fn unsynced(b: u32) -> String {
    let unsynced = PlanVerifyError::UnsyncedAccess {
        plan: PlanName::Step,
        step: 1,
        earlier: 0,
        buffer: b,
    };
    format!("{:?}", Err::<(), _>(unsynced))
}

/// DESIGN.md § 7.1 states how long the plans of 4,096 steps in the shapes that load a pass most
/// take to verify in a debug build, 0.162 s at most. [`unsynced_copy_pair`] with the views of `b`
/// last: the access pass finds the unsynced pair from its bit sets, then names the view with
/// `conflict`, which looks the buffers of the earlier step up in sets; it once compared each read
/// with each write of the earlier step, 32,767 * 32,768 comparisons, and took about 3.4 s. This
/// asserts 2 s, more than ten times the slowest of those plans.
#[test]
fn an_unsynced_copy_pair_at_the_view_limit_verifies_in_under_2_s() {
    const CASE_NAME: &str = "an_unsynced_copy_pair_at_the_view_limit_verifies_in_under_2_s";
    let _serial = serial();
    let (plan, b) = unsynced_copy_pair(false);
    let Some(run) = bounded(CASE_NAME, || format!("{:?}", verify_plan(&plan))) else {
        return;
    };
    assert_eq!(run.outcome, unsynced(b));
    assert!(
        run.seconds < 2.0,
        "{:.3} s, peak resident {} KiB: not under 2 s, ten times the slowest plan of DESIGN.md § 7.1",
        run.seconds,
        run.peak_kib
    );
}

/// The control of [`an_unsynced_copy_pair_at_the_view_limit_verifies_in_under_2_s`]: the same
/// plan with the views of `b` first, so that `conflict` matches the first view it looks at. It
/// verifies in under 0.2 s, so what the other case takes beyond it is `conflict`'s.
#[test]
fn the_same_pair_naming_its_conflict_first_verifies_in_under_0_2_s() {
    const CASE_NAME: &str = "the_same_pair_naming_its_conflict_first_verifies_in_under_0_2_s";
    let _serial = serial();
    let (plan, b) = unsynced_copy_pair(true);
    let Some(run) = bounded(CASE_NAME, || format!("{:?}", verify_plan(&plan))) else {
        return;
    };
    assert_eq!(run.outcome, unsynced(b));
    assert!(run.seconds < 0.2, "{:.3} s", run.seconds);
}
