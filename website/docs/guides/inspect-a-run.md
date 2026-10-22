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
modelspec train examples/mnist_mlp.mspec --backend vulkan --steps 72000 --profile
```

trains as without `--profile`, with the same lines on stdout and the same single host wait. On
stderr it adds one line per dispatch site and then the counters. `--steps 72000` makes the run
long enough to time a warm GPU: the file's 1,800 steps finish in about 0.4 s, and an RTX 3090
whose clocks are not locked stays near 780 MHz for a run that short. This is one run on an
RTX 3090 after a warm-up run of 72,000 steps, its SM clock between 780 and 1,920 MHz (mean
1,369 MHz in 100 ms samples):

```text
profile dispatch step 0 d68423b2-42d6-8f30-bd4a-492a35a66f06 main calls 72000 total_us 1658839.872 mean_us 23.039
profile dispatch step 2 b2485573-41c1-86a6-89c9-3ee7b0b0159b main calls 72000 total_us 174375.744 mean_us 2.422
profile dispatch step 4 cba0fd87-5af6-8f98-bc8b-1d9d5aee7e7f main calls 72000 total_us 263790.944 mean_us 3.664
profile dispatch step 6 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 main calls 72000 total_us 404889.152 mean_us 5.623
profile dispatch step 7 2e1ec459-43bd-81d2-b9db-fa0f7fe3ae74 main calls 72000 total_us 175582.112 mean_us 2.439
profile dispatch step 8 5bb7932a-569c-85fe-8946-5a688a1299e5 main calls 72000 total_us 155897.952 mean_us 2.165
profile dispatch step 10 e3ec6208-08d5-8b4c-a7a7-28be8470fecd main calls 72000 total_us 173608.672 mean_us 2.411
profile dispatch step 11 00cb6768-fb04-8fde-a136-7b324efe8fbc main calls 72000 total_us 173712.288 mean_us 2.413
profile dispatch step 13 1902a24e-aa7a-820c-b181-380008b822bd main calls 72000 total_us 238266.560 mean_us 3.309
profile dispatch step 14 4ab19294-379e-8a14-b7ee-a9d23e2520e2 main calls 72000 total_us 188538.624 mean_us 2.619
profile dispatch step 16 a94b7d6e-3815-8276-b1ca-b988735d6299 main calls 72000 total_us 324296.640 mean_us 4.504
profile dispatch step 17 ae896a79-8d47-8bca-807a-453bc394ae43 main calls 72000 total_us 176641.536 mean_us 2.453
profile dispatch step 18 cbd56b2b-4530-8c59-8969-a9eed9bc2f0c main calls 72000 total_us 294883.392 mean_us 4.096
profile dispatch step 20 433c82fb-dc3b-87cc-9f54-aaa99e09d5f7 main calls 72000 total_us 661017.984 mean_us 9.181
profile dispatch step 22 439f693b-8fb5-8576-af67-d32fa5e4ffd9 main calls 72000 total_us 208504.608 mean_us 2.896
profile dispatch step 24 3bc9dcb8-58e7-83b0-8cb6-bf00889c72a6 main calls 72000 total_us 216206.080 mean_us 3.003
profile dispatch step 25 2c3a7723-5b74-8340-98fa-af1482f88b90 main calls 72000 total_us 168432.032 mean_us 2.339
profile dispatch step 27 542d4968-c52b-82e3-b383-b5d27b90cf46 main calls 72000 total_us 172433.120 mean_us 2.395
profile dispatch final 0 d68423b2-42d6-8f30-bd4a-492a35a66f06 main calls 1 total_us 16.320 mean_us 16.320
profile dispatch final 2 b2485573-41c1-86a6-89c9-3ee7b0b0159b main calls 1 total_us 2.048 mean_us 2.048
profile dispatch final 4 cba0fd87-5af6-8f98-bc8b-1d9d5aee7e7f main calls 1 total_us 3.072 mean_us 3.072
profile dispatch final 6 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 main calls 1 total_us 4.096 mean_us 4.096
profile dispatches 1296004
profile submits 282
profile host_waits 1
profile bytes_uploaded 520864
profile bytes_downloaded 491268
profile pipeline_creation_us 1403.938
profile compile_us 595.761
profile steady_us_per_step 85.142
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
invocations: 5.6 µs of the 85.1 µs step. A device without timestamp queries prints
`profile kernel timings unavailable: no timestamp queries` in place of the dispatch lines and
still prints the counters.

The timestamps cost time themselves. On the same warm RTX 3090, 72,000 steps take 66.3 to
66.5 µs per step with `--stats` alone and 85.1 to 87.3 µs with `--profile` (three runs each,
SM clock between 780 and 1,950 MHz), so compare profiled runs with profiled runs, and use
`--stats` alone for the time per step.
