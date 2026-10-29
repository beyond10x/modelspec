//! `story:matmul-reads-transposed-operand`, acceptance `same-bits`, for the kernels the change
//! does not fold into: every kernel of the step and forward plans of the MNIST and linear
//! regression examples and of the `mlp_sweep` sources at b1000 h1024, b8192 h1024 and b8192
//! h4096 whose signature the plan before the change also had emits the SPIR-V words it emitted
//! then. So each matmul that reads no operand transposed runs the module it ran before, and so
//! does every elementwise kernel that fuses no chain (story:fuse-elementwise-chains-and-gradient-sums).
//!
//! `tests/fixtures/untransposed_modules.txt` holds one line per kernel signature of those plans
//! at the base commit, with the blake3 of the words `emit` gives, recorded with
//! `MODELSPEC_SAME_BITS_RECORD=1`. The hashes of the signatures that still occur were re-recorded
//! once at story:deterministic-fma, whose fused multiply-add changed the matmul and loss-sum
//! words (DESIGN.md § 12 ADR-009), and the loss-sum hashes alone at
//! story:vulkan-loss-sum-overflow-parity, which adds a sum of each difference scaled by the
//! reciprocal of the element count, written where the unscaled sum overflows (DESIGN.md § 5.4,
//! revision 0.75). Needs no device.
#![cfg(feature = "vulkan")]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use modelspec_core::artifact::Backend;
use modelspec_core::compile::compile_artifact;
use modelspec_core::ir::kernel::{Kernel, KernelKind};
use modelspec_runtime::spirv::emit::emit;

const RECORD_ENV: &str = "MODELSPEC_SAME_BITS_RECORD";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/untransposed_modules.txt")
}

fn sweep(batch: u64, hidden: u64) -> String {
    format!(
        "model SweepMlp {{
    param w1: Tensor<f32, [784, {hidden}]> = xavier_uniform();
    param w2: Tensor<f32, [{hidden}, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [{batch}, 784]>)
        -> Tensor<f32, [{batch}, 10]> {{
        return matmul(relu(matmul(x, w1)), w2);
    }}
}}

train Fit {{
    model = SweepMlp;
    input x: Tensor<f32, [{batch}, 784]>;
    target y: Tensor<f32, [{batch}, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 220;
    seed = 42;
    device = vulkan(0);
}}
"
    )
}

/// Every kernel of the plans, by signature.
fn kernels() -> BTreeMap<String, Kernel> {
    let mut sources = Vec::new();
    for example in ["mnist_mlp", "linear_regression"] {
        let source = std::fs::read_to_string(repo_root().join(format!("examples/{example}.mspec")))
            .expect("the example reads");
        sources.push((example.to_owned(), source));
    }
    for (batch, hidden) in [(1000, 1024), (8192, 1024), (8192, 4096)] {
        sources.push((format!("sweep_b{batch}_h{hidden}"), sweep(batch, hidden)));
    }
    let mut kernels = BTreeMap::new();
    for (name, source) in sources {
        let plan = compile_artifact(&name, &source, Backend::Vulkan)
            .unwrap_or_else(|d| panic!("{name} compiles: {d:?}"))
            .execution;
        for kernel in plan.kernels {
            kernels.insert(kernel.signature(), kernel);
        }
    }
    kernels
}

fn words_hash(words: &[u32]) -> String {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    blake3::hash(&bytes).to_hex().to_string()
}

#[test]
fn every_kernel_signature_of_the_plan_before_the_change_emits_the_words_it_emitted_then() {
    let mut lines = String::new();
    let now: BTreeMap<String, String> = kernels()
        .into_iter()
        .map(|(signature, kernel)| {
            let words = emit(&kernel).unwrap_or_else(|e| panic!("{signature}: {e}"));
            (signature, words_hash(&words))
        })
        .collect();
    for (signature, hash) in &now {
        writeln!(lines, "{hash} {signature}").expect("a string");
    }
    if std::env::var(RECORD_ENV).as_deref() == Ok("1") {
        std::fs::write(fixture_path(), &lines).expect("the fixture is written");
        eprintln!(
            "recorded {} kernels in {}",
            now.len(),
            fixture_path().display()
        );
        return;
    }
    let recorded = std::fs::read_to_string(fixture_path()).expect("the fixture reads");
    let before: BTreeMap<&str, &str> = recorded
        .lines()
        .map(|line| {
            let (hash, signature) = line.split_once(' ').expect("hash and signature");
            (signature, hash)
        })
        .collect();
    // A fused elementwise chain (story:fuse-elementwise-chains-and-gradient-sums) is the other
    // kind of kernel new to the plans: its op names the chain, `Sub>Mul@0`.
    let new_kind = |signature: &str| {
        signature.contains(" transposed")
            || signature
                .strip_prefix("elementwise ")
                .and_then(|rest| rest.split(' ').next())
                .is_some_and(|op| op.contains('>'))
    };
    let mut wrong = Vec::new();
    let mut kept = BTreeSet::new();
    for (signature, hash) in &now {
        match before.get(signature.as_str()) {
            Some(was) if *was == hash => {
                kept.insert(signature.clone());
            }
            Some(was) => wrong.push(format!("{signature}: words {hash}, before {was}")),
            None => {
                assert!(
                    new_kind(signature),
                    "a kernel new to the plans reads an operand transposed or fuses an \
                     elementwise chain: {signature}"
                );
            }
        }
    }
    let untransposed = now.keys().filter(|s| !new_kind(s)).count();
    assert_eq!(
        kept.len(),
        untransposed,
        "every kernel that reads no operand transposed had its signature before"
    );
    let matmuls = kept
        .iter()
        .filter(|s| s.starts_with(&format!("{} ", KernelKind::MatMul)))
        .count();
    eprintln!(
        "{} kernels emit the words of the plan before the change, {matmuls} of them matmuls",
        kept.len()
    );
    assert!(matmuls >= 5, "matmuls are among them");
    assert!(wrong.is_empty(), "{wrong:#?}");
}
