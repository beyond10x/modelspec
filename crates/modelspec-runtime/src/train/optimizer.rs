//! SGD: its configuration and one training step on the CPU interpreter (DESIGN.md § 3.1, § 6).
//!
//! A step runs `&step.training.program` on the interpreter unchanged, reading the current
//! version of every parameter, then applies each update output as its parameter's next version,
//! in effect-token order. All writes come after the evaluation, so none overwrites a value a read
//! of the old version still needs (DESIGN.md § 4). A parameter without an update keeps its value
//! and its version.
//!
//! [`evaluate_step`] is the evaluation alone: it returns the loss, the gradients and the updates
//! of the old values and changes nothing, and [`Evaluated::apply`] writes the updates. Between
//! the two a caller can read the old values beside their gradients, as the observatory does
//! (story:observatory-parameter-statistics). [`sgd_step`] is both.
//!
//! Each of those is a run of one step. A run of many steps prepares the step once
//! ([`PreparedStep`], story:cpu-prepared-training-step): it verifies the step and builds its
//! tables and the interpreter's plan once, and each of its steps then reuses the storage of the
//! last one and moves its outputs instead of copying them, with the same bits as [`sgd_step`].

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::ParamId;
use modelspec_core::ir::training::{StateVersion, TrainConfig, TrainingStep};
use modelspec_core::ir::verify::{TrainingVerifyError, verify_step_writes};
use thiserror::Error;

use crate::cpu::interpreter::Prepared;
use crate::cpu::{HostTensor, RuntimeError};

/// The SGD configuration of a run, from its `train` block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SgdConfig {
    pub lr: f32,
    pub steps: u64,
    pub seed: u64,
}

impl SgdConfig {
    pub fn from_config(config: &TrainConfig) -> Self {
        Self {
            lr: config.lr,
            steps: config.steps,
            seed: config.seed,
        }
    }
}

/// The persistent parameter state of a run: each declared parameter's value and version.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamState {
    pub values: BTreeMap<ParamId, HostTensor<f32>>,
    pub versions: BTreeMap<ParamId, StateVersion>,
}

impl ParamState {
    /// The state before the first step: `values` at the versions `step` is compiled against.
    pub fn new(step: &TrainingStep, values: BTreeMap<ParamId, HostTensor<f32>>) -> Self {
        Self {
            values,
            versions: step.optimizer.versions.clone(),
        }
    }
}

/// Why a step failed.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum StepError {
    #[error("training step does not verify: {0}")]
    Verify(#[from] TrainingVerifyError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    /// The parameter is at the last version; no update can follow.
    #[error("parameter {param} has no version after {current}")]
    VersionExhausted {
        param: ParamId,
        current: StateVersion,
    },
}

/// What one step computed from the old parameter values.
#[derive(Clone, Debug, PartialEq)]
pub struct StepOutput {
    /// The loss of the old values.
    pub loss: f32,
    /// The gradient of the loss with respect to each parameter the loss reads, at the old values:
    /// the gradient the step's update applies. A parameter the loss does not read has none.
    pub gradients: BTreeMap<ParamId, HostTensor<f32>>,
}

/// A step evaluated on the old values and not yet applied ([`evaluate_step`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluated {
    pub loss: f32,
    pub gradients: BTreeMap<ParamId, HostTensor<f32>>,
    /// Each update and the parameter it is the next version of, in effect-token order.
    updates: Vec<(ParamId, HostTensor<f32>)>,
}

impl Evaluated {
    /// Writes each update as its parameter's next version, in effect-token order, and returns the
    /// loss and the gradients.
    pub fn apply(self, state: &mut ParamState) -> StepOutput {
        for (param, value) in self.updates {
            state.values.insert(param, value);
            let version = state.versions.entry(param).or_insert(StateVersion(0));
            version.0 += 1;
        }
        StepOutput {
            loss: self.loss,
            gradients: self.gradients,
        }
    }
}

/// Runs one step of `step` on `inputs` and `state`, applies its updates and returns the loss and
/// the gradients of the old parameter values: a run of one step of a [`PreparedStep`].
pub fn sgd_step(
    step: &TrainingStep,
    inputs: &BTreeMap<String, HostTensor<f32>>,
    state: &mut ParamState,
) -> Result<StepOutput, StepError> {
    Ok(evaluate_step(step, inputs, state)?.apply(state))
}

/// Evaluates one step of `step` on `inputs` and `state` and changes nothing: the step is refused
/// as [`sgd_step`] refuses it, before it runs, and otherwise holds the loss, the gradients and
/// the updates for [`Evaluated::apply`].
pub fn evaluate_step(
    step: &TrainingStep,
    inputs: &BTreeMap<String, HostTensor<f32>>,
    state: &ParamState,
) -> Result<Evaluated, StepError> {
    let mut prepared = PreparedStep::new(step)?;
    let loss = prepared.run(inputs, &BTreeMap::new(), state, false)?;
    let updates = prepared
        .writes
        .iter()
        .zip(&mut prepared.pending)
        .map(|(&(param, _), update)| (param, update.take().expect("a run holds every update")))
        .collect();
    Ok(Evaluated {
        loss,
        gradients: std::mem::take(&mut prepared.gradients),
        updates,
    })
}

/// Per-thread counts of the optimizer's work, for the tests of story:cpu-prepared-training-step.
#[doc(hidden)]
pub mod counters {
    use std::cell::Cell;

    thread_local! {
        pub(super) static WRITES_TABLE: Cell<u64> = const { Cell::new(0) };
    }

    /// Builds of the table of a step's writes on this thread: one per [`super::PreparedStep`].
    pub fn writes_table() -> u64 {
        WRITES_TABLE.get()
    }
}

/// What a training program's output is to a step.
#[derive(Clone, Copy, Debug)]
enum Role {
    Loss,
    Gradient(ParamId),
    /// The update of the write at this index of [`PreparedStep::writes`].
    Update(usize),
}

/// A training step verified and planned once, to be stepped any number of times: a run.
///
/// [`PreparedStep::new`] verifies the step and its program, orders its effects and builds the
/// table of its writes and the interpreter's plan; each step then runs the plan
/// ([`crate::cpu::interpreter::Prepared`]), refusing it as [`sgd_step`] refuses it. Each step
/// moves the updates into the state, and each parameter value they replace becomes the storage of
/// that parameter's next update; the gradients stay with the prepared step and are read through
/// [`PendingStep::gradients`].
pub struct PreparedStep<'s> {
    program: Prepared<'s, f32>,
    /// Each write's parameter and the output position of its update, in effect-token order.
    writes: Vec<(ParamId, usize)>,
    /// The role of each output position.
    roles: Vec<Role>,
    /// The outputs of the current run, between the interpreter and their roles.
    outputs: Vec<HostTensor<f32>>,
    /// The gradients of the last evaluation.
    gradients: BTreeMap<ParamId, HostTensor<f32>>,
    /// The updates of the last evaluation not yet applied, by index of [`PreparedStep::writes`].
    pending: Vec<Option<HostTensor<f32>>>,
}

impl<'s> PreparedStep<'s> {
    /// Verifies `step` and plans its runs; a step that does not verify is refused here, before
    /// anything runs.
    pub fn new(step: &'s TrainingStep) -> Result<Self, StepError> {
        let written = verify_step_writes(step)?;
        let program = &step.training.program;
        // Positions come from the verified output layout, never from a search by value: one
        // value may be listed more than once (two parameters sharing one gradient), and each
        // listing has its own position and its own tensor.
        let gradients = &step.training.gradients;
        let update_at: BTreeMap<ParamId, usize> = step
            .optimizer
            .updates
            .keys()
            .enumerate()
            .map(|(k, &param)| (param, 1 + gradients.len() + k))
            .collect();
        counters::WRITES_TABLE.set(counters::WRITES_TABLE.get() + 1);
        let writes: Vec<(ParamId, usize)> = written
            .into_iter()
            .map(|(param, value)| {
                let at = update_at[&param];
                debug_assert_eq!(program.outputs[at], value, "the verified layout");
                (param, at)
            })
            .collect();
        let mut roles: Vec<Option<Role>> = vec![None; program.outputs.len()];
        roles[0] = Some(Role::Loss);
        for (k, (&param, &value)) in gradients.iter().enumerate() {
            debug_assert_eq!(program.outputs[1 + k], value, "the verified layout");
            roles[1 + k] = Some(Role::Gradient(param));
        }
        for (k, &(_, at)) in writes.iter().enumerate() {
            roles[at] = Some(Role::Update(k));
        }
        let roles = roles
            .into_iter()
            .map(|role| role.expect("the verified layout gives every output a role"))
            .collect();
        Ok(Self {
            program: Prepared::verified(program),
            pending: vec![None; writes.len()],
            writes,
            roles,
            outputs: Vec::new(),
            gradients: BTreeMap::new(),
        })
    }

    /// Evaluates one step on `inputs` and `state` and changes nothing until the result is
    /// applied ([`PendingStep::apply`]); between the two the caller reads the old values beside
    /// their gradients.
    pub fn evaluate(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<f32>>,
        state: &ParamState,
    ) -> Result<PendingStep<'_, 's>, StepError> {
        self.evaluate_with_ids(inputs, &BTreeMap::new(), state)
    }

    /// [`Self::evaluate`] with `ids`, a tensor for exactly the program's `i32` inputs, by name, as
    /// [`crate::cpu::evaluate_with_ids`] takes them: the row ids and labels of a
    /// masked-language-model batch (story:bert-train-data-binding).
    pub fn evaluate_with_ids(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<f32>>,
        ids: &BTreeMap<String, HostTensor<i32>>,
        state: &ParamState,
    ) -> Result<PendingStep<'_, 's>, StepError> {
        let loss = self.run(inputs, ids, state, true)?;
        Ok(PendingStep {
            prepared: self,
            loss,
        })
    }

    /// Evaluates one step and applies it; returns its loss.
    pub fn step(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<f32>>,
        state: &mut ParamState,
    ) -> Result<f32, StepError> {
        Ok(self.evaluate(inputs, state)?.apply(state))
    }

    /// Runs the program on `inputs`, `ids` and `state` and sorts its outputs by role. With
    /// `again`, each gradient keeps a second buffer for the next run, which writes into it while
    /// the caller may still hold the first. A run without ids refills the output list of the
    /// last run; a run with ids gets a new list, its outputs still moved from storage.
    fn run(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<f32>>,
        ids: &BTreeMap<String, HostTensor<i32>>,
        state: &ParamState,
        again: bool,
    ) -> Result<f32, StepError> {
        for (k, &(_, at)) in self.writes.iter().enumerate() {
            if let Some(update) = self.pending[k].take() {
                self.program.recycle(at, update);
            }
        }
        for &(param, _) in &self.writes {
            let current = state
                .versions
                .get(&param)
                .copied()
                .unwrap_or(StateVersion(0));
            if current.0.checked_add(1).is_none() {
                return Err(StepError::VersionExhausted { param, current });
            }
        }
        if ids.is_empty() {
            self.program
                .run_into(inputs, &state.values, &mut self.outputs)?;
        } else {
            self.outputs = self.program.run_with_ids(inputs, ids, &state.values)?;
        }
        let mut loss = f32::NAN;
        for (position, output) in self.outputs.drain(..).enumerate() {
            match self.roles[position] {
                Role::Loss => {
                    loss = output.data()[0];
                    self.program.recycle(position, output);
                }
                Role::Gradient(param) => match self.gradients.insert(param, output) {
                    Some(old) => self.program.recycle(position, old),
                    None if again => self.program.reserve_output(position)?,
                    None => {}
                },
                Role::Update(k) => self.pending[k] = Some(output),
            }
        }
        Ok(loss)
    }
}

/// A step of a [`PreparedStep`] evaluated on the old values and not yet applied.
pub struct PendingStep<'a, 's> {
    prepared: &'a mut PreparedStep<'s>,
    loss: f32,
}

impl PendingStep<'_, '_> {
    /// The loss of the old values.
    pub fn loss(&self) -> f32 {
        self.loss
    }

    /// The gradient of the loss with respect to each parameter the loss reads, at the old values.
    pub fn gradients(&self) -> &BTreeMap<ParamId, HostTensor<f32>> {
        &self.prepared.gradients
    }

    /// Writes each update as its parameter's next version, in effect-token order, and returns
    /// the loss. Each value an update replaces becomes the storage of that parameter's next
    /// update.
    pub fn apply(self, state: &mut ParamState) -> f32 {
        let prepared = self.prepared;
        for (k, &(param, at)) in prepared.writes.iter().enumerate() {
            let value = prepared.pending[k]
                .take()
                .expect("an evaluated step holds every update");
            if let Some(old) = state.values.insert(param, value) {
                prepared.program.recycle(at, old);
            }
            let version = state.versions.entry(param).or_insert(StateVersion(0));
            version.0 += 1;
        }
        self.loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::evaluate;
    use modelspec_core::autodiff::differentiate;
    use modelspec_core::diag::SourceSpan;
    use modelspec_core::ir::tensor::{Initializer, TensorProgramBuilder, TensorType};
    use modelspec_core::ir::training::{Effect, EffectToken, attach_sgd};

    const LR: f32 = 0.1;
    const W: ParamId = ParamId(0);
    const U: ParamId = ParamId(1);

    fn tensor(shape: &[u64], data: &[f32]) -> HostTensor<f32> {
        HostTensor::new(shape.to_vec(), data.to_vec()).unwrap()
    }

    /// `mse(matmul(x, w), y)`; with `unused`, a parameter `LinearReg.u` nothing reads.
    fn linear_step(unused: bool) -> TrainingStep {
        let mut b = TensorProgramBuilder::new();
        let w = b
            .declare_param(
                "LinearReg.w",
                TensorType::f32(&[2, 1]),
                Initializer::XavierUniform,
            )
            .unwrap();
        if unused {
            b.declare_param(
                "LinearReg.u",
                TensorType::f32(&[2, 1]),
                Initializer::XavierUniform,
            )
            .unwrap();
        }
        let x = b.input("x", TensorType::f32(&[3, 2])).unwrap();
        let y = b.input("y", TensorType::f32(&[3, 1])).unwrap();
        let w = b.param_read(w).unwrap();
        let h = b.matmul(x, w).unwrap();
        let loss = b.mse(h, y).unwrap();
        let training = differentiate(&b.finish(), loss).unwrap();
        attach_sgd(training, LR).unwrap()
    }

    fn inputs() -> BTreeMap<String, HostTensor<f32>> {
        BTreeMap::from([
            (
                "x".into(),
                tensor(&[3, 2], &[1.0, 2.0, -1.0, 0.5, 0.25, -2.0]),
            ),
            ("y".into(), tensor(&[3, 1], &[1.0, -1.0, 0.5])),
        ])
    }

    fn values(unused: bool) -> BTreeMap<ParamId, HostTensor<f32>> {
        let mut v = BTreeMap::from([(W, tensor(&[2, 1], &[0.3, -0.7]))]);
        if unused {
            v.insert(U, tensor(&[2, 1], &[0.9, 0.1]));
        }
        v
    }

    #[test]
    fn sgd_config_reads_the_train_block() {
        let config = TrainConfig {
            steps: 500,
            lr: 0.1,
            seed: 42,
            inputs: vec![],
            targets: vec![],
            span: SourceSpan {
                file: "f.mspec".into(),
                line: 1,
                column: 1,
            },
        };
        assert_eq!(
            SgdConfig::from_config(&config),
            SgdConfig {
                lr: 0.1,
                steps: 500,
                seed: 42
            }
        );
    }

    /// One step writes `w - lr * grad`, computed in `f32` from the gradient of the old value,
    /// and moves `w` from version 0 to 1.
    #[test]
    fn a_step_writes_p_minus_lr_times_grad_as_the_next_version() {
        let step = linear_step(false);
        let mut state = ParamState::new(&step, values(false));
        assert_eq!(state.versions[&W], StateVersion(0));
        let outputs = evaluate(&step.training.program, &inputs(), &values(false)).unwrap();
        let loss = sgd_step(&step, &inputs(), &mut state).unwrap().loss;
        assert_eq!(loss, outputs[0].data()[0]);
        let g = outputs[1].data();
        let old = values(false)[&W].data().to_vec();
        let expected: Vec<f32> = old.iter().zip(g).map(|(&p, &g)| p - LR * g).collect();
        assert_eq!(state.values[&W].data(), &expected[..]);
        assert_ne!(state.values[&W].data(), &old[..]);
        assert_eq!(state.versions[&W], StateVersion(1));
    }

    #[test]
    fn each_step_advances_the_version_by_one_and_lowers_the_loss() {
        let step = linear_step(false);
        let mut state = ParamState::new(&step, values(false));
        let mut losses = Vec::new();
        for _ in 0..3 {
            losses.push(sgd_step(&step, &inputs(), &mut state).unwrap().loss);
        }
        assert_eq!(state.versions[&W], StateVersion(3));
        assert!(losses[0] > losses[1] && losses[1] > losses[2], "{losses:?}");
    }

    /// A parameter the loss does not read keeps its value and its version.
    #[test]
    fn a_step_leaves_an_unused_parameter_at_its_value_and_version() {
        let step = linear_step(true);
        let mut state = ParamState::new(&step, values(true));
        sgd_step(&step, &inputs(), &mut state).unwrap();
        sgd_step(&step, &inputs(), &mut state).unwrap();
        assert_eq!(state.values[&U], values(true)[&U]);
        assert_eq!(state.versions[&U], StateVersion(0));
        assert_eq!(state.versions[&W], StateVersion(2));
    }

    /// A step returns the gradient its update applies, the training program's output for it on
    /// the old values, and none for a parameter the loss does not read.
    #[test]
    fn a_step_returns_the_gradient_of_the_old_values() {
        let step = linear_step(true);
        let mut state = ParamState::new(&step, values(true));
        let outputs = evaluate(&step.training.program, &inputs(), &values(true)).unwrap();
        let gradient = step.training.gradients[&W];
        let position = step
            .training
            .program
            .outputs
            .iter()
            .position(|&o| o == gradient)
            .unwrap();
        let out = sgd_step(&step, &inputs(), &mut state).unwrap();
        assert_eq!(out.loss, outputs[0].data()[0]);
        assert_eq!(out.gradients[&W], outputs[position]);
        assert_eq!(out.gradients.keys().copied().collect::<Vec<_>>(), vec![W]);
        let g = out.gradients[&W].data();
        let expected: Vec<f32> = values(true)[&W]
            .data()
            .iter()
            .zip(g)
            .map(|(&p, &g)| p - LR * g)
            .collect();
        assert_eq!(
            state.values[&W].data(),
            &expected[..],
            "the update applied it"
        );
    }

    /// An evaluated step changes nothing until it is applied, and applying it is the step.
    #[test]
    fn an_evaluated_step_changes_nothing_until_it_is_applied() {
        let step = linear_step(false);
        let state = ParamState::new(&step, values(false));
        let evaluated = evaluate_step(&step, &inputs(), &state).unwrap();
        assert_eq!(state, ParamState::new(&step, values(false)));
        let mut applied = state.clone();
        let out = evaluated.apply(&mut applied);
        let mut stepped = state.clone();
        assert_eq!(sgd_step(&step, &inputs(), &mut stepped), Ok(out));
        assert_eq!(applied, stepped);
        assert_eq!(applied.versions[&W], StateVersion(1));
    }

    /// A step the verifier refuses is not run: the state is untouched.
    #[test]
    fn a_step_with_illegal_effects_is_refused_before_it_runs() {
        let mut step = linear_step(false);
        let update = step.optimizer.updates[&W];
        let effect = step.optimizer.effects[&update];
        step.optimizer.effects.insert(
            update,
            Effect {
                token: EffectToken(0),
                ..effect
            },
        );
        let mut state = ParamState::new(&step, values(false));
        let before = state.clone();
        let result = sgd_step(&step, &inputs(), &mut state);
        assert!(matches!(result, Err(StepError::Verify(_))), "{result:?}");
        assert_eq!(state, before);
    }

    /// At the last version a step is refused, not wrapped to version 0, and nothing is written.
    #[test]
    fn a_step_at_the_last_version_is_refused() {
        let step = linear_step(false);
        let mut state = ParamState::new(&step, values(false));
        state.versions.insert(W, StateVersion(u64::MAX));
        let before = state.clone();
        assert_eq!(
            sgd_step(&step, &inputs(), &mut state),
            Err(StepError::VersionExhausted {
                param: W,
                current: StateVersion(u64::MAX)
            })
        );
        assert_eq!(state, before);
    }
}
