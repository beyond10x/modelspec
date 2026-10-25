//! `zeros()` and `ones()` parameters on the CPU interpreter (DESIGN.md § 2.3; coordinator
//! decision D33): every element is exactly `+0.0` or `1.0`, the parameter consumes no draw of
//! the parameter stream, and so it does not change the bits of any other parameter.

use std::collections::BTreeMap;

use modelspec_core::compile::compile_training;
use modelspec_core::ir::tensor::{Initializer, TensorProgram, TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::{HostTensor, evaluate};
use modelspec_runtime::train::init::{Stream, initialize};

/// A program whose outputs read the parameters `M.z: [5]` (`zeros`), `M.o: [2, 3]` (`ones`) and
/// `M.c: [2, 2, 2]` (`zeros`), in that order.
fn read_back_program() -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let params = [
        ("M.z", TensorType::f32(&[5]), Initializer::Zeros),
        ("M.o", TensorType::f32(&[2, 3]), Initializer::Ones),
        ("M.c", TensorType::f32(&[2, 2, 2]), Initializer::Zeros),
    ];
    for (path, ty, init) in params {
        let param = b.declare_param(path, ty, init).unwrap();
        let read = b.param_read(param).unwrap();
        b.output(read).unwrap();
    }
    b.finish()
}

#[test]
fn zeros_and_ones_parameters_read_back_as_all_0_and_all_1() {
    let program = read_back_program();
    for stream in [Stream::Init, Stream::Teacher] {
        let params = initialize(&program, 42, stream).unwrap();
        let out = evaluate(&program, &BTreeMap::new(), &params).unwrap();
        let bits = |t: &HostTensor<f32>| t.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(out[0].shape(), [5]);
        assert_eq!(bits(&out[0]), vec![0.0f32.to_bits(); 5], "zeros is +0.0");
        assert_eq!(out[1].shape(), [2, 3]);
        assert_eq!(bits(&out[1]), vec![1.0f32.to_bits(); 6]);
        assert_eq!(out[2].shape(), [2, 2, 2]);
        assert_eq!(bits(&out[2]), vec![0.0f32.to_bits(); 8]);
    }
}

const TRAIN: &str = "
train Fit {
    model = M;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 3;
    seed = 11;
    device = cpu;
}
";

/// `M.a` and `M.c` with the `zeros()` bias `M.b` declared between them.
const WITH_BIAS: &str = "model M {
    param a: Tensor<f32, [4, 3]> = xavier_uniform();
    param b: Tensor<f32, [3]> = zeros();
    param c: Tensor<f32, [3, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        return matmul(add_rows(matmul(x, a), b), c);
    }
}
";

/// The same `M.a` and `M.c` without `M.b`.
const WITHOUT_BIAS: &str = "model M {
    param a: Tensor<f32, [4, 3]> = xavier_uniform();
    param c: Tensor<f32, [3, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        return matmul(matmul(x, a), c);
    }
}
";

/// The initial bits of every parameter of `model`, by canonical path, on `stream`.
fn initial_bits(model: &str, stream: Stream) -> BTreeMap<String, Vec<u32>> {
    let compiled = compile_training("m.mspec", &format!("{model}{TRAIN}")).unwrap_or_else(|d| {
        panic!(
            "the model compiles:\n{}",
            d.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        )
    });
    let forward = &compiled.forward.program;
    let values = initialize(forward, compiled.config.seed, stream).unwrap();
    forward
        .params
        .iter()
        .map(|(id, decl)| {
            let bits = values[id].data().iter().map(|v| v.to_bits()).collect();
            (decl.path.clone(), bits)
        })
        .collect()
}

/// `M.b` takes the parameter id between `M.a` and `M.c`, but no draw: `M.c` keeps the bits it
/// has without `M.b`, on the parameter stream and on the teacher stream.
#[test]
fn a_zeros_parameter_between_two_xavier_parameters_does_not_change_their_bits() {
    for stream in [Stream::Init, Stream::Teacher] {
        let with = initial_bits(WITH_BIAS, stream);
        let without = initial_bits(WITHOUT_BIAS, stream);
        assert_eq!(with["M.b"], vec![0.0f32.to_bits(); 3]);
        assert_eq!(with["M.a"], without["M.a"], "{stream:?}");
        assert_eq!(with["M.c"], without["M.c"], "{stream:?}");
    }
}
