//! `StepCompleted.parameters` of a device run without a device: the CPU run's [`Run`] stands in
//! for the device's, which has the same fields, and [`device_parameters`] must give at the first
//! step and at the post-loop step the `Value` entries the CPU run streams there
//! (approval-record:parameter-statistics-settlements, question 3: "the first step also carries
//! values" and "the final post-loop event always carries values"), and nothing at any other step.

use std::collections::BTreeMap;
use std::path::Path;

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::ParamId;
use modelspec_observatory::{ParameterStatistics, StatisticsSubject};
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::runner::{self, Observed, Run};

use super::{device_parameters, paths, statistics};

/// `examples/linear_regression.mspec` with a second parameter `LinearReg.u` the loss does not
/// read, compiled for `steps` steps.
fn compiled(steps: u64) -> CompiledTraining {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/linear_regression.mspec");
    let source = std::fs::read_to_string(&path).expect("the example is readable");
    let unread = source.replace(
        "    fn forward",
        "    param u: Tensor<f32, [2, 2]> = xavier_uniform();\n\n    fn forward",
    );
    assert_ne!(unread, source, "the example declares `fn forward`");
    let mut compiled = compile_training("unread.mspec", &unread).expect("the model compiles");
    compiled.config.steps = steps;
    compiled
}

/// Runs `compiled` on the CPU from `start` with `init`; the run and the `parameters` its every
/// step streams, by step: the CPU branch of `train` with every step sampled.
fn cpu_run(
    compiled: &CompiledTraining,
    start: u64,
    init: &BTreeMap<ParamId, HostTensor<f32>>,
) -> (Run, BTreeMap<u64, Vec<ParameterStatistics>>) {
    let params = paths(compiled);
    let mut streamed = BTreeMap::new();
    let observe = |o: &Observed<'_>| {
        streamed.insert(o.step, statistics(&params, o.values, o.gradients));
    };
    let run = runner::run_from(compiled, None, init, start, observe, |_, _| Ok(()))
        .expect("the CPU run finishes");
    (run, streamed)
}

/// The `Value` entries of `entries`, in their order.
fn values_of(entries: &[ParameterStatistics]) -> Vec<ParameterStatistics> {
    entries
        .iter()
        .filter(|entry| entry.subject == StatisticsSubject::Value)
        .cloned()
        .collect()
}

/// Checks `device_parameters` for every step of `run`, which ends at `end`, against the CPU
/// stream `streamed`: the CPU's `Value` entries at the first and the last step, one per declared
/// parameter in path order, and an empty list at every other step.
fn assert_device_steps(
    compiled: &CompiledTraining,
    run: &Run,
    streamed: &BTreeMap<u64, Vec<ParameterStatistics>>,
    end: u64,
) {
    let params = paths(compiled);
    let declared: Vec<&str> = params.keys().copied().collect();
    assert_eq!(declared, ["LinearReg.u", "LinearReg.w"]);
    assert_eq!(
        streamed.keys().copied().collect::<Vec<_>>(),
        (run.start_step..=end).collect::<Vec<_>>()
    );
    for (&k, cpu) in streamed {
        let device = device_parameters(&params, run, end, k);
        if k == run.start_step || k == end {
            assert_eq!(device, values_of(cpu), "step {k}: the CPU's Value entries");
            let listed: Vec<&str> = device.iter().map(|entry| entry.path.as_str()).collect();
            assert_eq!(listed, declared, "step {k}: every declared parameter once");
        } else {
            assert_eq!(device, Vec::new(), "step {k}: no statistics");
        }
    }
}

#[test]
fn a_device_run_streams_values_at_its_first_and_post_loop_steps_only() {
    let compiled = compiled(4);
    let (run, streamed) = cpu_run(&compiled, 0, &BTreeMap::new());
    assert_device_steps(&compiled, &run, &streamed, 4);
    let params = paths(&compiled);
    assert_eq!(
        device_parameters(&params, &run, 4, 0),
        statistics(&params, &run.initial, None),
        "step 0 is the values before the first update, the ones uploaded"
    );
    assert_eq!(
        device_parameters(&params, &run, 4, 4),
        statistics(&params, &run.state.values, None),
        "the post-loop step is the values after the last update, the ones read back"
    );
    assert_ne!(
        run.initial, run.state.values,
        "the run moved the parameters"
    );
}

#[test]
fn a_resumed_device_run_streams_the_checkpoint_values_at_its_first_step() {
    let compiled = compiled(3);
    let (before, _) = cpu_run(&compiled, 0, &BTreeMap::new());
    let (run, streamed) = cpu_run(&compiled, 3, &before.state.values);
    assert_eq!(run.initial, before.state.values);
    assert_device_steps(&compiled, &run, &streamed, 6);
}

#[test]
fn a_device_run_of_no_steps_streams_its_values_once() {
    let compiled = compiled(0);
    let (run, streamed) = cpu_run(&compiled, 0, &BTreeMap::new());
    assert_device_steps(&compiled, &run, &streamed, 0);
}
