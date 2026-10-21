# ModelSpec

A declarative, statically typed language and a Rust compiler for differentiable tensor programs.
You write the model, the loss and the optimizer in a `.mspec` file; the compiler builds the whole
training step, derives its gradients, optimizes the graph and runs it on a GPU through Vulkan and
SPIR-V. No Python, PyTorch or CUDA.

**Status:** design only. There is no parser, compiler or runtime yet.

```modelspec
model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
```

## Where to read

| File | What it holds |
| --- | --- |
| [DESIGN.md](DESIGN.md) | The design: language, IR levels, autodiff, Vulkan backend, runtime, milestones M0–M4, ADRs |
| [examples/](examples/) | Target-syntax examples; they do not compile yet |
| [AGENTS.md](AGENTS.md) | Rules for agents working in this repository |

## Milestones

| Milestone | Delivers |
| --- | --- |
| M0 | Parser, type and shape checker, tensor SSA IR, CPU interpreter, reverse-mode autodiff, SGD |
| M1 | Rust Vulkan runtime and generated SPIR-V kernels: end-to-end f32 training on an RTX 3090 |
| M2 | Liveness, buffer reuse, fusion, tiled matmul, profiling |
| M3 | f16, AdamW, softmax, layer norm, attention, small GPT-like training |
| M4 | Cooperative-matrix Tensor Core path, mixed precision, autotuning |

## License

Apache 2.0, see [LICENSE](LICENSE).
