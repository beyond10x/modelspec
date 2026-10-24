//! Adversary pass on `story:fast-math-mode` (DESIGN.md § 7.1, § 8, § 12 ADR-008), no Vulkan
//! instance: every run sets `MODELSPEC_VULKAN_LIBRARY` to a path that does not exist and every
//! Vulkan compile names the RTX 3090 fixture profile.
//!
//! - Attack 4, hash separation: the artifacts `compile` writes for one source with and without
//!   `--fast-math` (the MNIST example and the `mlp_sweep` source at b8192 h4096) have different
//!   hashes and ids; their manifests are equal once `fast_math`, the hashes, the ids and each
//!   module's byte length and blake3 are set aside, so every kernel cache key differs only in
//!   `fast_math`; and every module file is the other's less its `NoContraction` decorations.
//! - Attack 3, tampering, on real modules: `fast_math` flipped in the manifest or in one kernel
//!   cache key is refused with E-ARTIFACT-002 when not resealed and E-ARTIFACT-003 when resealed.
//! - Attack 5, the usage errors: `--fast-math` without `--backend` and on `infer` exit 2 and
//!   write nothing.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::artifact;
use serde_json::Value;

const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";
const OP_DECORATE: u32 = 71;
const DECORATION_NO_CONTRACTION: u32 = 42;

const SWEEP_B8192_H4096: &str = "model SweepMlp {
    param w1: Tensor<f32, [784, 4096]> = xavier_uniform();
    param w2: Tensor<f32, [4096, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8192, 784]>)
        -> Tensor<f32, [8192, 10]> {
        return matmul(relu(matmul(x, w1)), w2);
    }
}

train Fit {
    model = SweepMlp;
    input x: Tensor<f32, [8192, 784]>;
    target y: Tensor<f32, [8192, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 220;
    seed = 42;
    device = vulkan(0);
}
";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture() -> String {
    root()
        .join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json")
        .display()
        .to_string()
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_fastmath_cli")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    std::fs::create_dir_all(dir.parent().unwrap()).expect("the temp dir is writable");
    dir
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .env("MODELSPEC_VULKAN_LIBRARY", NO_LOADER)
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The source file of `which`: the MNIST example or the sweep source written to scratch.
fn source(which: &str) -> String {
    match which {
        "mnist" => root()
            .join("examples/mnist_mlp.mspec")
            .display()
            .to_string(),
        "sweep" => {
            let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("adversary_fastmath_cli");
            std::fs::create_dir_all(&dir).unwrap();
            let file = dir.join("sweep_b8192_h4096.mspec");
            std::fs::write(&file, SWEEP_B8192_H4096).unwrap();
            file.display().to_string()
        }
        _ => unreachable!(),
    }
}

fn compile(which: &str, name: &str, extra: &[&str]) -> PathBuf {
    let out = scratch(name);
    let (file, shown, profile) = (source(which), out.display().to_string(), fixture());
    let mut args = vec![
        "compile",
        &file,
        "--backend",
        "vulkan",
        "--output",
        &shown,
        "--device-profile",
        &profile,
    ];
    args.extend_from_slice(extra);
    let output = modelspec(&args);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{args:?}: {}",
        text(&output.stderr)
    );
    out
}

fn manifest(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("artifact.json")).unwrap()).unwrap()
}

fn words(bytes: &[u8]) -> Vec<u32> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "whole words");
    chunks.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

fn without_no_contraction(module: &[u32]) -> Vec<u32> {
    let mut out = module[..5].to_vec();
    let mut at = 5;
    while at < module.len() {
        let length = (module[at] >> 16) as usize;
        let instruction = &module[at..at + length];
        let dropped =
            module[at] & 0xffff == OP_DECORATE && instruction[2] == DECORATION_NO_CONTRACTION;
        if !dropped {
            out.extend_from_slice(instruction);
        }
        at += length;
    }
    out
}

/// `manifest` with `fast_math`, the hashes, the ids and each module's byte length and blake3
/// set aside.
fn neutral(mut manifest: Value) -> Value {
    for field in ["artifact_hash", "artifact_id", "fast_math"] {
        manifest[field] = Value::Null;
    }
    for kernel in manifest["kernels"].as_array_mut().unwrap() {
        kernel["cache_key"]["fast_math"] = Value::Null;
        kernel["byte_length"] = Value::Null;
        kernel["blake3"] = Value::Null;
    }
    manifest
}

#[test]
fn fast_math_and_deterministic_artifacts_differ_only_in_fast_math_and_no_contraction() {
    for which in ["mnist", "sweep"] {
        let off_dir = compile(which, &format!("{which}-off"), &[]);
        let on_dir = compile(which, &format!("{which}-on"), &["--fast-math"]);
        let (off, on) = (manifest(&off_dir), manifest(&on_dir));
        assert_eq!(off["fast_math"], false, "{which}");
        assert_eq!(on["fast_math"], true, "{which}");
        assert_ne!(off["artifact_hash"], on["artifact_hash"], "{which}");
        assert_ne!(off["artifact_id"], on["artifact_id"], "{which}");
        let kernels = on["kernels"].as_array().unwrap();
        assert!(!kernels.is_empty());
        for (k_off, k_on) in off["kernels"].as_array().unwrap().iter().zip(kernels) {
            assert_eq!(k_off["cache_key"]["fast_math"], false);
            assert_eq!(k_on["cache_key"]["fast_math"], true);
        }
        assert_eq!(
            neutral(off.clone()),
            neutral(on.clone()),
            "{which}: the manifests differ in more than fast_math"
        );
        let mut differing = 0;
        for kernel in kernels {
            let id = kernel["kernel_module_id"].as_str().unwrap();
            let file = format!("kernels/{id}.spv");
            let a = words(&std::fs::read(off_dir.join(&file)).unwrap());
            let b = words(&std::fs::read(on_dir.join(&file)).unwrap());
            assert_eq!(
                b,
                without_no_contraction(&a),
                "{which} {id}: the fast_math module is not the deterministic one less NoContraction"
            );
            assert!(
                without_no_contraction(&b) == b,
                "{which} {id}: NoContraction left"
            );
            if a != b {
                differing += 1;
            }
        }
        assert!(
            differing > 0,
            "{which}: no module changed under --fast-math"
        );
        let names = |dir: &Path| {
            let mut v: Vec<_> = std::fs::read_dir(dir.join("kernels"))
                .unwrap()
                .map(|e| e.unwrap().file_name())
                .collect();
            v.sort();
            v
        };
        assert_eq!(
            names(&off_dir),
            names(&on_dir),
            "{which}: the same module files"
        );
        eprintln!(
            "{which}: {} kernels, {differing} module files differ",
            kernels.len()
        );
    }
}

fn edit(dir: &Path, change: impl FnOnce(&mut Value)) {
    let path = dir.join("artifact.json");
    let mut m: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    change(&mut m);
    std::fs::write(&path, serde_json::to_string_pretty(&m).unwrap()).unwrap();
}

fn refusal(dir: &Path) -> String {
    match artifact::read(dir) {
        Ok(_) => panic!(
            "the reader accepted the tampered artifact {}",
            dir.display()
        ),
        Err(d) => format!("{} {d}", d.code.as_str()),
    }
}

/// A named change to a manifest.
type Tamper = (&'static str, fn(&mut Value));

#[test]
fn a_flipped_fast_math_is_refused_with_or_without_a_new_seal() {
    let cases: [Tamper; 3] = [
        ("manifest", |m| m["fast_math"] = Value::Bool(false)),
        ("first-key", |m| {
            m["kernels"][0]["cache_key"]["fast_math"] = Value::Bool(false)
        }),
        ("last-key", |m| {
            let last = m["kernels"].as_array().unwrap().len() - 1;
            m["kernels"][last]["cache_key"]["fast_math"] = Value::Bool(false)
        }),
    ];
    for (name, change) in cases {
        let dir = compile("mnist", &format!("tamper-{name}"), &["--fast-math"]);
        artifact::read(&dir).expect("the untampered fast_math artifact reads");
        edit(&dir, change);
        let unsealed = refusal(&dir);
        assert!(
            unsealed.starts_with("E-ARTIFACT-002"),
            "{name} unsealed: {unsealed}"
        );
        artifact::seal(&dir).expect("resealed");
        let resealed = refusal(&dir);
        assert!(
            resealed.starts_with("E-ARTIFACT-003") && resealed.contains("fast_math"),
            "{name} resealed: {resealed}"
        );
    }
    let dir = compile("mnist", "tamper-deterministic", &[]);
    edit(&dir, |m| m["fast_math"] = Value::Bool(true));
    artifact::seal(&dir).expect("resealed");
    let resealed = refusal(&dir);
    assert!(resealed.starts_with("E-ARTIFACT-003"), "{resealed}");
}

/// A `fast_math` artifact relabelled deterministic in the manifest and in every key, resealed:
/// its modules carry no `NoContraction`, so it is not the deterministic artifact of its source.
/// Red on the unit: the reader accepts it (finding ADV-FM-1; nothing executes an artifact
/// directory, and the seal does not authenticate, `artifact::seal`).
#[test]
#[ignore = "ADV-FM-1: the reader accepts a fast_math artifact relabelled deterministic and resealed"]
fn a_fast_math_artifact_relabelled_deterministic_throughout_is_refused() {
    let dir = compile("mnist", "relabelled", &["--fast-math"]);
    edit(&dir, |m| {
        m["fast_math"] = Value::Bool(false);
        for kernel in m["kernels"].as_array_mut().unwrap() {
            kernel["cache_key"]["fast_math"] = Value::Bool(false);
        }
    });
    artifact::seal(&dir).expect("resealed");
    let refused = refusal(&dir);
    assert!(refused.starts_with("E-ARTIFACT-003"), "{refused}");
}

#[test]
fn fast_math_without_a_backend_or_on_infer_is_a_usage_error_that_writes_nothing() {
    let file = source("mnist");
    let out = scratch("no-backend");
    let shown = out.display().to_string();
    let output = modelspec(&["compile", &file, "--output", &shown, "--fast-math"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("--backend"), "{stderr}");
    assert!(!out.exists(), "nothing is written");

    let output = modelspec(&["train", &file, "--fast-math"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("--backend"), "{stderr}");
    assert!(output.stdout.is_empty());

    let output = modelspec(&[
        "infer",
        &file,
        "--backend",
        "vulkan",
        "--checkpoint",
        "/nonexistent/x.msckpt",
        "--fast-math",
    ]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("--fast-math"), "{stderr}");

    let output = modelspec(&[
        "compile",
        &file,
        "--backend",
        "cpu",
        "--output",
        &shown,
        "--fast-math",
        "--device-profile",
        &fixture(),
    ]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(!out.exists(), "nothing is written");
}
