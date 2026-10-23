# ModelSpec

A declarative, statically typed language and a Rust compiler for differentiable tensor programs.
You write the model, the loss and the optimizer in a `.mspec` file; the compiler builds the whole
training step, derives its gradients, optimizes the graph and runs it on a GPU through Vulkan and
SPIR-V. No Python, PyTorch or CUDA.

**Status:** M0. `modelspec check`, `modelspec ir --stage tensor|training` and `modelspec train <file> --backend cpu` work on the M0 subset; the example below trains on the CPU reference. On `main`, not released yet, `modelspec train <file> --backend vulkan` trains on one Vulkan device within the GPU tolerance of the CPU run (M1).

**Releases:** [GitHub Releases](https://github.com/beyond10x/modelspec/releases) carry the `modelspec` binary for Linux x86_64 and its SHA-256 checksum; [CHANGELOG.md](CHANGELOG.md) lists what each version changed.

**Documentation:** <https://beyond10x.github.io/modelspec/>, with the [pitch deck](https://beyond10x.github.io/modelspec/pitch/) (source: [docs/pitch/index.html](docs/pitch/index.html)).

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

## Watch a run in the browser

On `main`, not released yet: the observatory UI in [ui/](ui/) draws the loss, and every metric a
run reports, per step, live or from a recording. Build it once with Node 22.18 or later and pnpm
([go-task](https://taskfile.dev) runs `pnpm install` and `pnpm build` in `ui/`):

```console
task ui
modelspec train examples/linear_regression.mspec --backend cpu --serve 127.0.0.1:8080 --ui ui/dist
```

and open <http://127.0.0.1:8080/>. To replay a recorded run instead, record it with
`--record run.jsonl` and serve it with `modelspec observe run.jsonl --serve 127.0.0.1:8080 --ui ui/dist`.
Both serve on this machine only and stop on Ctrl-C.

## Where to read

| File | What it holds |
| --- | --- |
| [DESIGN.md](DESIGN.md) | The design: language, IR levels, autodiff, Vulkan backend, runtime, milestones M0–M4, ADRs |
| [examples/](examples/) | `linear_regression.mspec` checks and trains; `tiny_mlp.mspec` is target syntax beyond M0 |
| [ui/](ui/) | The observatory UI (Vue 3, TypeScript, Vite) that `--ui` serves |
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
