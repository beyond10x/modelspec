---
title: Inspect a run
description: Print the IR, the kernels and plans of a training step, the compiled artifact and its SPIR-V, and the time of every GPU dispatch of a run, on the MNIST example.
---

# Inspect a run

This guide follows `examples/mnist_mlp.mspec` from its IR to the time each GPU dispatch takes. The
first three commands need only the `modelspec` binary. The fourth needs a binary built with the
`vulkan` feature and, with a device profile file, no GPU. The last needs that binary and a
Vulkan 1.2 device.

## 1. The Tensor IR

```shell
modelspec ir --stage tensor examples/mnist_mlp.mspec
```

prints the forward program after checking, one value per line:

```text
param @MnistMlp.w1: Tensor<f32, [784, 64]> = xavier_uniform
param @MnistMlp.w2: Tensor<f32, [64, 10]> = xavier_uniform
%0: Tensor<f32, [100, 784]> = Input "x"
%1: Tensor<f32, [100, 10]> = Input "y"
%2: Tensor<f32, [784, 64]> = ParamRead @MnistMlp.w1
%3: Tensor<f32, [64, 10]> = ParamRead @MnistMlp.w2
%4: Tensor<f32, [100, 64]> = MatMul %0, %2
%5: Tensor<f32, [100, 64]> = Relu %4
%6: Tensor<f32, [100, 10]> = MatMul %5, %3
%7: Tensor<f32, []> = Mse %6, %1
output %7
```

## 2. The training step

```shell
modelspec ir --stage training examples/mnist_mlp.mspec
```

prints the training step: the forward values, the loss, the gradients and the SGD updates, each
line tagged `forward`, `loss`, `gradient` or `optimizer`. This is the program the GPU runs once
per training step.

## 3. Kernels and plans

```shell
modelspec ir --stage kernel examples/mnist_mlp.mspec
```

prints the buffers, one `kernel` line per kernel with its content-hashed module id, and two plans.
The kernel lines name what each module computes:

```text
kernel k0 5666286c-c7a4-86c5-ba60-e9812c8b6812 matmul MatMul f32 (Tensor<f32, [100, 784]>, Tensor<f32, [784, 64]>) -> Tensor<f32, [100, 64]> epilogue Relu writes pre-activation workgroup_size 128 workgroups [52, 1, 1]
kernel k2 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 loss_sum Mse f32 (Tensor<f32, [100, 10]>, Tensor<f32, [100, 10]>) -> Tensor<f32, []> workgroup_size 64 workgroups [1, 1, 1]
```

`plan step` is the training step, numbered: `dispatch` steps run a kernel, `barrier` steps order
them. `plan forward` is the forward program alone; `train` does not run it. The profile below
names a dispatch by its position in `plan step`.

## 4. The compiled artifact

```shell
modelspec compile examples/mnist_mlp.mspec --backend vulkan --output build/mnist --device-profile crates/modelspec-runtime/tests/fixtures/rtx3090.device.json
```

writes the artifact directory `build/mnist`: `artifact.json`, the manifest with the parameters,
kernels and plans, and `kernels/<kernel_module_id>.spv`, one SPIR-V module per `kernel` line of
`--stage kernel`. `--device-profile` builds the modules for a device profile file, here the RTX
3090 profile the tests use, so no GPU or Vulkan loader is needed; without it `compile` builds for
device 0. Before it prints `artifact <artifact_id> <artifact_hash> kernels=14 build/mnist`, it
reads the directory back through the same reader that refuses an altered artifact with
`E-ARTIFACT-001` to `E-ARTIFACT-004`. Compiling the same file twice gives byte-equal directories.

`spirv-dis`, from SPIRV-Tools, shows one module, here `k2`, the `mse` loss sum:

```shell
spirv-dis build/mnist/kernels/d588cbc2-b3eb-8bf1-93a1-29d8680aba58.spv
```

```text
; SPIR-V
; Version: 1.5
; Generator: Google rspirv; 0
; Bound: 363
; Schema: 0
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %33 "main" %gl_GlobalInvocationID %gl_NumWorkGroups %gl_LocalInvocationID %25 %26 %27 %28 %32
               OpExecutionMode %33 LocalSize 64 1 1
```

## 5. The time of every dispatch

```shell
modelspec train examples/mnist_mlp.mspec --backend vulkan --steps 72000 --profile
```

trains as without `--profile`, with the same lines on stdout and the same single host wait. On
stderr it adds one line per dispatch site and then the counters. `--steps 72000` makes the run
long enough to time a warm GPU: the file's 1,800 steps finish in about 0.3 s, too short a run
to time a GPU whose clocks are not locked. This is one run on an RTX 3090 after a warm-up run
of 72,000 steps, its SM clock between 1,965 and 1,980 MHz in its 100 ms samples:

```text
profile dispatch step 0 5666286c-c7a4-86c5-ba60-e9812c8b6812 main calls 72000 total_us 602482.656 mean_us 8.368
profile dispatch step 2 4695d3c9-f7bc-8553-ad4b-8bda5f9af8bf main calls 72000 total_us 165234.080 mean_us 2.295
profile dispatch step 4 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 main calls 72000 total_us 283784.256 mean_us 3.941
profile dispatch step 5 2e1ec459-43bd-81d2-b9db-fa0f7fe3ae74 main calls 72000 total_us 134887.072 mean_us 1.873
profile dispatch step 7 3e9bd9e4-7689-8b30-9582-05282d19167c main calls 72000 total_us 129581.024 mean_us 1.800
profile dispatch step 8 00cb6768-fb04-8fde-a136-7b324efe8fbc main calls 72000 total_us 134726.656 mean_us 1.871
profile dispatch step 10 b9c2711f-8443-864a-86ef-f7493dde3311 main calls 72000 total_us 154567.232 mean_us 2.147
profile dispatch step 11 4ab19294-379e-8a14-b7ee-a9d23e2520e2 main calls 72000 total_us 144804.640 mean_us 2.011
profile dispatch step 13 f3113772-7e08-8d28-87c3-ca25c60a15d4 main calls 72000 total_us 179402.368 mean_us 2.492
profile dispatch step 14 ae896a79-8d47-8bca-807a-453bc394ae43 main calls 72000 total_us 135719.456 mean_us 1.885
profile dispatch step 15 cbd56b2b-4530-8c59-8969-a9eed9bc2f0c main calls 72000 total_us 220593.280 mean_us 3.064
profile dispatch step 17 d7d3124e-f2f6-83d4-a63c-166b318a5713 main calls 72000 total_us 347826.976 mean_us 4.831
profile dispatch step 19 88daaa23-8dc9-8bc4-ab05-dc5ecc946d99 main calls 72000 total_us 168548.448 mean_us 2.341
profile dispatch step 20 f509ed52-1817-8bb7-91f5-a998f93b9005 main calls 72000 total_us 133047.744 mean_us 1.848
profile dispatch final 0 5666286c-c7a4-86c5-ba60-e9812c8b6812 main calls 1 total_us 7.360 mean_us 7.360
profile dispatch final 2 4695d3c9-f7bc-8553-ad4b-8bda5f9af8bf main calls 1 total_us 2.048 mean_us 2.048
profile dispatch final 4 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 main calls 1 total_us 4.096 mean_us 4.096
profile dispatches 1008003
profile submits 282
profile host_waits 1
profile bytes_uploaded 520864
profile bytes_downloaded 491268
profile pipeline_creation_us 1573.877
profile compile_us 1876.371
profile steady_us_per_step 44.087
```

| Field | What it says |
|---|---|
| `step <position>` | a `dispatch` step of `plan step`, run once per training step: `calls` is the number of steps |
| `final <position>` | the loss of the final parameters, computed once after the last step by the steps of `plan step` up to the loss; `<position>` is the index in `plan step` |
| `<kernel_module_id> <entry>` | the module and its entry point; find the module id on a `kernel` line of `--stage kernel` |
| `total_us`, `mean_us` | GPU time from just before the dispatch starts to just after it ends, summed over the calls and per call |
| `steady_us_per_step` | GPU time per step from step 20 to the final loss, so the first 20 steps are left out as warm-up |
| `pipeline_creation_us`, `compile_us` | host time to build the pipelines and to compile the file and emit its kernels, outside the steady state |

Here step position 4, module `d588cbc2…`, is `k2`, the `mse` loss sum on one workgroup of 64
invocations: 3.9 µs of the 44.1 µs step. A device without timestamp queries prints
`profile kernel timings unavailable: no timestamp queries` in place of the dispatch lines and
still prints the counters.

The timestamps cost time themselves. On the same warm RTX 3090, 72,000 steps take 27.1 to
27.3 µs per step with `--stats` alone and 44.0 to 44.2 µs with `--profile` (three runs each,
SM clock between 1,860 and 1,980 MHz in 100 ms samples, 1,965 MHz in 182 of 192), so compare profiled runs with profiled runs, and use
`--stats` alone for the time per step.
