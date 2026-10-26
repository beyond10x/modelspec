# ModelSpec

A declarative, statically typed language and a Rust compiler for differentiable tensor programs.
You write the model, the loss and the optimizer in a `.mspec` file; the compiler builds the whole
training step, derives its gradients, optimizes the graph and runs it on a GPU through Vulkan and
SPIR-V. No Python, PyTorch or CUDA.

**Status:** M1 in progress; the latest release is 0.18.0. `modelspec check`, `ir`, `compile`, `infer` and `train <file> --backend cpu|vulkan` work on the language subset the [documentation](https://beyond10x.github.io/modelspec/docs/concepts/language) describes; the example below trains on the CPU reference, and since 0.4.0 `--backend vulkan` trains on one Vulkan device within the GPU tolerance of the CPU run. The current milestone is a small BERT pre-trained from scratch: its operators (`embedding`, `add_rows`, `gelu`, `layer_norm`, `softmax`, `attention`, `cross_entropy` and `dropout`) run on the CPU reference, `modelspec data prepare`, `tokenizer` and `pack` turn a corpus into token sequences that `train --data` trains on with held-out evaluation (`--eval-every`), and since 0.17.0 every BERT operator and AdamW has a Vulkan kernel, so `train --backend vulkan --data` trains the hidden-32 BERT twin on the GPU. [Status](https://beyond10x.github.io/modelspec/docs/status) lists every capability with the test that holds it.

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

Since 0.7.0, the observatory UI in [ui/](ui/) draws the loss, and every metric a
run reports, per step, live or from a recording. It also draws, per parameter, the norm of its
values and of its gradient and their histograms at the first step, every `--statistics-every`
steps (default 100) and the last step of a CPU run, and those of its values at the first and the
last step of a Vulkan run. Build it once with Node 22.18 or later and pnpm ([go-task](https://taskfile.dev) runs `pnpm install` and `pnpm build` in `ui/`):

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
| [examples/](examples/) | `linear_regression.mspec` checks and trains; `mnist_mlp.mspec` is the MNIST benchmark model; `tiny_mlp.mspec` is target syntax beyond M0 |
| [benchmarks/runs.yaml](benchmarks/runs.yaml) | Every benchmark run measured so far, charted on the [Benchmarks](https://beyond10x.github.io/modelspec/docs/benchmarks) page |
| [ui/](ui/) | The observatory UI (Vue 3, TypeScript, Vite) that `--ui` serves |
| [AGENTS.md](AGENTS.md) | Rules for agents working in this repository |

## Milestones

| Milestone | Delivers |
| --- | --- |
| M0 | Parser, type and shape checker, tensor SSA IR, CPU interpreter, reverse-mode autodiff, SGD |
| M1 | Rust Vulkan runtime and generated SPIR-V kernels: end-to-end f32 training on an RTX 3090 |
| M2 | Liveness, buffer reuse, fusion, tiled matmul, profiling; fusion, the tiled matmul and profiling done early, in M1 |
| M3 | f16, AdamW, embeddings, softmax, layer norm, attention, GELU, small GPT-like training and small BERT-like masked-language-model pre-training; the BERT operators run on the CPU reference since 0.10.0 |
| M4 | Cooperative-matrix Tensor Core path, mixed precision, autotuning |

## License

Apache 2.0, see [LICENSE](LICENSE).
