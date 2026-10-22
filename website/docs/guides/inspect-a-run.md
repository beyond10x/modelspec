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
kernel k0 0f20c50b-2526-8d85-ae0c-c3d5454139a3 matmul MatMul f32 (Tensor<f32, [100, 784]>, Tensor<f32, [784, 64]>) -> Tensor<f32, [100, 64]> epilogue Relu writes pre-activation workgroup_size 64 workgroups [100, 1, 1]
kernel k2 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 loss_sum Mse f32 (Tensor<f32, [100, 10]>, Tensor<f32, [100, 10]>) -> Tensor<f32, []> workgroup_size 64 workgroups [1, 1, 1]
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
long enough to time a warm GPU: the file's 1,800 steps finish in about 0.3 s, too short a run
to time a GPU whose clocks are not locked. This is one run on an RTX 3090 after a warm-up run
of 72,000 steps, its SM clock at 1,965 MHz in every 100 ms sample:

```text
profile dispatch step 0 0f20c50b-2526-8d85-ae0c-c3d5454139a3 main calls 72000 total_us 1321495.456 mean_us 18.354
profile dispatch step 2 cba0fd87-5af6-8f98-bc8b-1d9d5aee7e7f main calls 72000 total_us 210192.544 mean_us 2.919
profile dispatch step 4 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 main calls 72000 total_us 297924.512 mean_us 4.138
profile dispatch step 5 2e1ec459-43bd-81d2-b9db-fa0f7fe3ae74 main calls 72000 total_us 140991.520 mean_us 1.958
profile dispatch step 7 3e9bd9e4-7689-8b30-9582-05282d19167c main calls 72000 total_us 139825.600 mean_us 1.942
profile dispatch step 8 00cb6768-fb04-8fde-a136-7b324efe8fbc main calls 72000 total_us 143883.968 mean_us 1.998
profile dispatch step 10 1902a24e-aa7a-820c-b181-380008b822bd main calls 72000 total_us 198007.360 mean_us 2.750
profile dispatch step 11 4ab19294-379e-8a14-b7ee-a9d23e2520e2 main calls 72000 total_us 153805.984 mean_us 2.136
profile dispatch step 13 a94b7d6e-3815-8276-b1ca-b988735d6299 main calls 72000 total_us 242498.912 mean_us 3.368
profile dispatch step 14 ae896a79-8d47-8bca-807a-453bc394ae43 main calls 72000 total_us 141174.976 mean_us 1.961
profile dispatch step 15 cbd56b2b-4530-8c59-8969-a9eed9bc2f0c main calls 72000 total_us 224138.336 mean_us 3.113
profile dispatch step 17 433c82fb-dc3b-87cc-9f54-aaa99e09d5f7 main calls 72000 total_us 499021.088 mean_us 6.931
profile dispatch step 19 88daaa23-8dc9-8bc4-ab05-dc5ecc946d99 main calls 72000 total_us 180664.832 mean_us 2.509
profile dispatch step 20 f509ed52-1817-8bb7-91f5-a998f93b9005 main calls 72000 total_us 145400.128 mean_us 2.019
profile dispatch final 0 0f20c50b-2526-8d85-ae0c-c3d5454139a3 main calls 1 total_us 16.544 mean_us 16.544
profile dispatch final 2 cba0fd87-5af6-8f98-bc8b-1d9d5aee7e7f main calls 1 total_us 3.072 mean_us 3.072
profile dispatch final 4 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 main calls 1 total_us 4.096 mean_us 4.096
profile dispatches 1008003
profile submits 282
profile host_waits 1
profile bytes_uploaded 520864
profile bytes_downloaded 491268
profile pipeline_creation_us 1667.715
profile compile_us 536.249
profile steady_us_per_step 59.696
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
invocations: 4.1 µs of the 59.7 µs step. A device without timestamp queries prints
`profile kernel timings unavailable: no timestamp queries` in place of the dispatch lines and
still prints the counters.

The timestamps cost time themselves. On the same warm RTX 3090, 72,000 steps take 39.6 to
39.9 µs per step with `--stats` alone and 59.1 to 59.4 µs with `--profile` (three runs each,
SM clock between 1,950 and 1,965 MHz in 100 ms samples), so compare profiled runs with profiled runs, and use
`--stats` alone for the time per step.
