//! The CPU backend: an interpreter of verified Tensor IR on host tensors, the reference oracle
//! for every other backend (DESIGN.md § 9). Training IR is not interpreted here: its effect
//! tokens and state versions are metadata, and the host runner applies parameter writes.

pub mod interpreter;
pub mod matmul;
pub mod tensor;

pub use interpreter::{
    Evaluation, RuntimeError, evaluate, evaluate_all, evaluate_all_with_ids, evaluate_seeded,
    evaluate_with_ids,
};
pub use tensor::{Element, HostTensor, TensorError};
