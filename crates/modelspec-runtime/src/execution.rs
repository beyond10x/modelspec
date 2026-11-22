//! Repeated execution of verified tensor IR against application-provided tensors.
//! A plan can execute a prediction or a differentiated loss with its parameter gradients.

use std::collections::BTreeMap;

pub use modelspec_core::artifact::Backend;
use modelspec_core::ir::tensor::{DType, ParamId, TensorProgram};
use modelspec_core::ir::verify::{VerifyError, verify};
use thiserror::Error;

use crate::cpu::{HostTensor, RuntimeError, interpreter::Prepared};

/// A result keeps the element type and shape of its tensor IR output.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    F32(HostTensor<f32>),
    I32(HostTensor<i32>),
}

#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error(transparent)]
    Invalid(#[from] VerifyError),
    #[error(transparent)]
    Tensor(#[from] RuntimeError),
    #[error("native Vulkan execution: {0}")]
    Vulkan(String),
    #[error("{0}")]
    Target(modelspec_core::diag::Diagnostic),
    #[error("a physical device index applies only to the Vulkan backend")]
    CpuDevice,
    #[error("this ModelSpec runtime was built without the vulkan feature")]
    VulkanDisabled,
    #[error("this program returns integer tensors; use run_typed")]
    IntegerOutput,
    #[error("a tensor execution plan needs at least one output")]
    MissingOutput,
}

enum Target {
    Cpu(Box<Prepared<'static, f32>>),
    #[cfg(feature = "vulkan")]
    Vulkan(crate::vulkan::execution::DevicePlan),
}

/// An owned static-shape plan. Each call binds a batch and its parameters in memory.
/// Plans targeting Vulkan share a device. Each retains its compiled pipelines and buffers.
pub struct ExecutionPlan {
    target: Target,
    integer_output: bool,
    seed: Option<u64>,
}

impl ExecutionPlan {
    pub fn new(program: TensorProgram, backend: Backend) -> Result<Self, ExecutionError> {
        Self::new_on_device(program, backend, None)
    }

    /// Prepare the program on the selected Vulkan physical device (device zero by default).
    pub fn new_on_device(
        program: TensorProgram,
        backend: Backend,
        device: Option<usize>,
    ) -> Result<Self, ExecutionError> {
        verify(&program)?;
        if program.outputs.is_empty() {
            return Err(ExecutionError::MissingOutput);
        }
        let (program, _) = modelspec_core::passes::stop_gradient::eliminate(program);
        let integer_output = program.outputs.iter().any(|id| {
            program
                .node(*id)
                .is_some_and(|node| node.ty.dtype == DType::I32)
        });
        let target = match backend {
            Backend::Cpu if device.is_some() => return Err(ExecutionError::CpuDevice),
            Backend::Cpu => Target::Cpu(Box::new(Prepared::owned_verified(program))),
            #[cfg(feature = "vulkan")]
            Backend::Vulkan => Target::Vulkan(crate::vulkan::execution::DevicePlan::new(
                program,
                device.unwrap_or(0),
            )?),
            #[cfg(not(feature = "vulkan"))]
            Backend::Vulkan => return Err(ExecutionError::VulkanDisabled),
        };
        Ok(Self {
            target,
            integer_output,
            seed: None,
        })
    }

    /// Set the run key for dropout nodes. Their explicit integer inputs select each draw.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
    }

    /// Evaluate a program with floating-point outputs in their declared order.
    pub fn run(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<f32>>,
        ids: &BTreeMap<String, HostTensor<i32>>,
        params: &BTreeMap<ParamId, HostTensor<f32>>,
    ) -> Result<Vec<HostTensor<f32>>, ExecutionError> {
        if self.integer_output {
            return Err(ExecutionError::IntegerOutput);
        }
        self.run_typed(inputs, ids, params)?
            .into_iter()
            .map(|value| match value {
                Output::F32(tensor) => Ok(tensor),
                Output::I32(_) => Err(ExecutionError::IntegerOutput),
            })
            .collect()
    }

    /// Evaluate the declared outputs, including integer code indices, in their original order.
    pub fn run_typed(
        &mut self,
        inputs: &BTreeMap<String, HostTensor<f32>>,
        ids: &BTreeMap<String, HostTensor<i32>>,
        params: &BTreeMap<ParamId, HostTensor<f32>>,
    ) -> Result<Vec<Output>, ExecutionError> {
        match &mut self.target {
            Target::Cpu(plan) => {
                if let Some(seed) = self.seed {
                    plan.set_seed(seed);
                }
                Ok(plan.run_typed(inputs, ids, params)?)
            }
            #[cfg(feature = "vulkan")]
            Target::Vulkan(plan) => plan.run(inputs, ids, params, self.seed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelspec_core::autodiff::differentiate;
    use modelspec_core::ir::tensor::{Initializer, TensorOp, TensorProgramBuilder, TensorType};

    fn tensor<T: Copy>(shape: &[u64], values: &[T]) -> HostTensor<T> {
        HostTensor::new(shape.to_vec(), values.to_vec()).unwrap()
    }

    fn check_quantization_and_gradients(backend: Backend) {
        let mut b = TensorProgramBuilder::new();
        let weights = b
            .declare_param("Codec.codes", TensorType::f32(&[2, 2]), Initializer::Ones)
            .unwrap();
        let x = b.input("scores", TensorType::f32(&[4, 2])).unwrap();
        let passthrough = b.input("token", TensorType::i32(&[2])).unwrap();
        let table = b.param_read(weights).unwrap();
        let grouped = b.push(TensorOp::GroupSum { size: 2 }, &[x]).unwrap();
        let indices = b.push(TensorOp::RowArgMax, &[grouped]).unwrap();
        let selected = b.embedding(table, indices).unwrap();
        let decoded = b
            .push(TensorOp::RepeatRows { size: 2 }, &[selected])
            .unwrap();
        let loss = b.reduce_mean(decoded).unwrap();
        let bias = b.scalar(-0.75);
        let sum = b.reduce_sum(bias).unwrap();
        for output in [indices, decoded, table, x, passthrough, bias, sum, indices] {
            b.output(output).unwrap();
        }
        let ir = b.finish();
        let grads = differentiate(&ir, loss).unwrap().program;
        let mut forward = ExecutionPlan::new(ir, backend).unwrap();
        let mut backward = ExecutionPlan::new(grads, backend).unwrap();
        let scores = tensor(&[4, 2], &[1., 2., 0., 1., 4., 1., -1., 2.]);
        let token = tensor(&[2], &[i32::MIN, 0x7fc0_0001]);
        let inputs = BTreeMap::from([("scores".into(), scores.clone())]);
        let ids = BTreeMap::from([("token".into(), token.clone())]);
        for offset in [0., 0., 2.] {
            let a = 2. + offset;
            let codes = tensor(&[2, 2], &[a, 4., -1., 6.]);
            let params = BTreeMap::from([(weights, codes.clone())]);
            assert!(matches!(
                forward.run(&inputs, &ids, &params),
                Err(ExecutionError::IntegerOutput)
            ));
            let outputs = forward.run_typed(&inputs, &ids, &params).unwrap();
            let picked = Output::I32(tensor(&[2], &[1, 0]));
            assert_eq!(
                outputs,
                [
                    picked.clone(),
                    Output::F32(tensor(&[4, 2], &[-1., 6., -1., 6., a, 4., a, 4.])),
                    Output::F32(codes),
                    Output::F32(scores.clone()),
                    Output::I32(token.clone()),
                    Output::F32(HostTensor::scalar(-0.75)),
                    Output::F32(HostTensor::scalar(-0.75)),
                    picked,
                ]
            );
            let gradients = backward.run(&inputs, &ids, &params).unwrap();
            assert_eq!(
                gradients,
                [
                    HostTensor::scalar((a + 9.) / 4.),
                    tensor(&[2, 2], &[0.25; 4])
                ]
            );
        }
    }

    fn check_loss_bindings(backend: Backend) {
        let mut b = TensorProgramBuilder::new();
        let logits = b.input("logits", TensorType::f32(&[2, 2])).unwrap();
        let weights = b.input("weights", TensorType::f32(&[2])).unwrap();
        let labels = b.input("labels", TensorType::i32(&[2])).unwrap();
        let two = b.scalar(2.);
        let scaled = b.mul(weights, two).unwrap();
        let loss = b.cross_entropy(logits, labels, scaled).unwrap();
        b.output(loss).unwrap();
        let mut plan = ExecutionPlan::new(b.finish(), backend).unwrap();
        let params = BTreeMap::new();
        let mut floats = BTreeMap::from([
            ("logits".into(), tensor(&[2, 2], &[0.; 4])),
            ("weights".into(), tensor(&[2], &[-0.5, 1.])),
        ]);
        let ids = BTreeMap::from([("labels".into(), tensor(&[2], &[0, 1]))]);
        assert!(matches!(
            plan.run(&floats, &ids, &params),
            Err(ExecutionError::Tensor(RuntimeError::LossWeight {
                row: 0,
                ..
            }))
        ));
        floats.insert("weights".into(), tensor(&[2], &[0.5, 1.]));
        let bad = BTreeMap::from([("labels".into(), tensor(&[2], &[0, -1]))]);
        assert!(matches!(
            plan.run(&floats, &bad, &params),
            Err(ExecutionError::Tensor(RuntimeError::LabelOutOfRange {
                row: 1,
                ..
            }))
        ));
        let value = plan.run(&floats, &ids, &params).unwrap();
        assert_eq!(value.len(), 1);
        assert!((value[0].data()[0] - 2f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn cpu_in_memory_execution_reuses_plans_and_preserves_typed_outputs() {
        check_quantization_and_gradients(Backend::Cpu);
        check_loss_bindings(Backend::Cpu);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a Vulkan compute device"]
    fn vulkan_in_memory_execution_reuses_plans_and_preserves_typed_outputs() {
        check_quantization_and_gradients(Backend::Vulkan);
        check_loss_bindings(Backend::Vulkan);
    }
}
