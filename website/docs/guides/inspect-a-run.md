---
title: Inspect a run
description: Print the IR, the kernels and plans of a training step, and the time of every GPU dispatch of a run, on the MNIST example.
---

# Inspect a run

This guide follows `examples/mnist_mlp.mspec` from its IR to the time each GPU dispatch takes. The
first three commands need only the `modelspec` binary. The last needs a binary built with the
`vulkan` feature and a Vulkan 1.2 device.

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
kernel k0 d68423b2-42d6-8f30-bd4a-492a35a66f06 matmul MatMul f32 (Tensor<f32, [100, 784]>, Tensor<f32, [784, 64]>) -> Tensor<f32, [100, 64]> workgroup_size 64 workgroups [100, 1, 1]
kernel k3 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 loss_sum Mse f32 (Tensor<f32, [100, 10]>, Tensor<f32, [100, 10]>) -> Tensor<f32, []> workgroup_size 64 workgroups [1, 1, 1]
```

`plan step` is the training step, numbered: `dispatch` steps run a kernel, `barrier` steps order
them. `plan forward` is the forward program alone; `train` does not run it. The profile below
names a dispatch by its position in `plan step`.

## 4. The time of every dispatch

```shell
modelspec train examples/mnist_mlp.mspec --backend vulkan --profile
```

trains as without `--profile`, with the same lines on stdout and the same single host wait. On
stderr it adds one line per dispatch site and then the counters. This is one run on an RTX 3090:

```text
profile dispatch step 0 d68423b2-42d6-8f30-bd4a-492a35a66f06 main calls 1800 total_us 59180.320 mean_us 32.878
profile dispatch step 2 b2485573-41c1-86a6-89c9-3ee7b0b0159b main calls 1800 total_us 5870.592 mean_us 3.261
profile dispatch step 4 cba0fd87-5af6-8f98-bc8b-1d9d5aee7e7f main calls 1800 total_us 9458.688 mean_us 5.255
profile dispatch step 6 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 main calls 1800 total_us 15709.184 mean_us 8.727
profile dispatch step 7 2e1ec459-43bd-81d2-b9db-fa0f7fe3ae74 main calls 1800 total_us 5951.488 mean_us 3.306
profile dispatch step 8 5bb7932a-569c-85fe-8946-5a688a1299e5 main calls 1800 total_us 6504.448 mean_us 3.614
profile dispatch step 10 e3ec6208-08d5-8b4c-a7a7-28be8470fecd main calls 1800 total_us 5875.712 mean_us 3.264
profile dispatch step 11 00cb6768-fb04-8fde-a136-7b324efe8fbc main calls 1800 total_us 5863.424 mean_us 3.257
profile dispatch step 13 1902a24e-aa7a-820c-b181-380008b822bd main calls 1800 total_us 7774.720 mean_us 4.319
profile dispatch step 14 4ab19294-379e-8a14-b7ee-a9d23e2520e2 main calls 1800 total_us 6295.232 mean_us 3.497
profile dispatch step 16 a94b7d6e-3815-8276-b1ca-b988735d6299 main calls 1800 total_us 11926.336 mean_us 6.626
profile dispatch step 17 ae896a79-8d47-8bca-807a-453bc394ae43 main calls 1800 total_us 5932.032 mean_us 3.296
profile dispatch step 18 cbd56b2b-4530-8c59-8969-a9eed9bc2f0c main calls 1800 total_us 10635.264 mean_us 5.908
profile dispatch step 20 433c82fb-dc3b-87cc-9f54-aaa99e09d5f7 main calls 1800 total_us 25694.208 mean_us 14.275
profile dispatch step 22 439f693b-8fb5-8576-af67-d32fa5e4ffd9 main calls 1800 total_us 6986.048 mean_us 3.881
profile dispatch step 24 3bc9dcb8-58e7-83b0-8cb6-bf00889c72a6 main calls 1800 total_us 7260.864 mean_us 4.034
profile dispatch step 25 2c3a7723-5b74-8340-98fa-af1482f88b90 main calls 1800 total_us 5666.816 mean_us 3.148
profile dispatch step 27 542d4968-c52b-82e3-b383-b5d27b90cf46 main calls 1800 total_us 5825.536 mean_us 3.236
profile dispatch final 0 d68423b2-42d6-8f30-bd4a-492a35a66f06 main calls 1 total_us 30.368 mean_us 30.368
profile dispatch final 2 b2485573-41c1-86a6-89c9-3ee7b0b0159b main calls 1 total_us 3.072 mean_us 3.072
profile dispatch final 4 cba0fd87-5af6-8f98-bc8b-1d9d5aee7e7f main calls 1 total_us 5.120 mean_us 5.120
profile dispatch final 6 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 main calls 1 total_us 8.192 mean_us 8.192
profile dispatches 32404
profile submits 8
profile host_waits 1
profile bytes_uploaded 520864
profile bytes_downloaded 210468
profile pipeline_creation_us 1523.308
profile compile_us 607.667
profile steady_us_per_step 120.349
```

| Field | What it says |
|---|---|
| `step <position>` | a `dispatch` step of `plan step`, run once per training step: `calls` is the number of steps |
| `final <position>` | the loss of the final parameters, computed once after the last step by the steps of `plan step` up to the loss; `<position>` is the index in `plan step` |
| `<kernel_module_id> <entry>` | the module and its entry point; find the module id on a `kernel` line of `--stage kernel` |
| `total_us`, `mean_us` | GPU time from just before the dispatch starts to just after it ends, summed over the calls and per call |
| `steady_us_per_step` | GPU time per step from step 20 to the final loss, so the first 20 steps are left out as warm-up |
| `pipeline_creation_us`, `compile_us` | host time to build the pipelines and to compile the file and emit its kernels, outside the steady state |

Here step position 6, module `d588cbc2…`, is `k3`, the `mse` loss sum on one workgroup of 64
invocations: 8.7 µs of the 120.3 µs step. A device without timestamp queries prints
`profile kernel timings unavailable: no timestamp queries` in place of the dispatch lines and
still prints the counters.

The timestamps cost time themselves. A step that takes about 68 µs without `--profile` takes about
91 µs with it on the same device, so compare profiled runs with profiled runs, and use
`--stats` alone for the time per step.
