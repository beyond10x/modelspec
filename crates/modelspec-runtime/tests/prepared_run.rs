//! story:cpu-prepared-training-step through the training loop: a CPU run verifies its training
//! step once and builds its tables once, whatever its number of steps.

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::verify;
use modelspec_runtime::cpu::interpreter;
use modelspec_runtime::train::optimizer;
use modelspec_runtime::train::runner;

const MNIST_MLP: &str = "examples/mnist_mlp.mspec";

fn compiled(steps: u64) -> CompiledTraining {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(MNIST_MLP);
    let source = std::fs::read_to_string(&path).expect("the example reads");
    let mut compiled = compile_training(MNIST_MLP, &source).expect("the example compiles");
    compiled.config.steps = steps;
    compiled
}

/// The verifier and table counters a run of `steps` steps moves, and its losses.
fn run_counts(steps: u64) -> ([u64; 5], Vec<f32>) {
    let compiled = compiled(steps);
    let read = || {
        [
            verify::counters::verify(),
            verify::counters::verify_step(),
            verify::counters::effects_in_order(),
            optimizer::counters::writes_table(),
            interpreter::counters::in_place_transposes(),
        ]
    };
    let before = read();
    let run = runner::run(&compiled).expect("the run finishes");
    let after = read();
    (std::array::from_fn(|i| after[i] - before[i]), run.losses)
}

/// verify-once-per-run and tables-once-per-run: a run of 5 steps calls each verifier and builds
/// each table as often as a run of 1 step, and the training step is verified once.
#[test]
fn a_run_verifies_its_step_and_builds_its_tables_once() {
    let (one, _) = run_counts(1);
    let (five, losses) = run_counts(5);
    assert_eq!(losses.len(), 6);
    let [_, verify_step, effects, writes, _] = five;
    assert_eq!(
        (verify_step, effects, writes),
        (1, 1, 1),
        "a run of 5 steps"
    );
    assert_eq!(five, one, "a run of 5 steps against a run of 1");
}
