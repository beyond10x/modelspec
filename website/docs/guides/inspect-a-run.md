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
kernel k0 6bd32d9c-1c0c-8fd4-96b2-46d2c722adc9 matmul MatMul f32 (Tensor<f32, [100, 784]>, Tensor<f32, [784, 64]>) -> Tensor<f32, [100, 64]> epilogue Relu writes pre-activation workgroup_size 128 workgroups [52, 1, 1] tile workgroup_size 128 cols 16 rows 1 cols_per_invocation 1 depth 64 ahead 6 buffers 2 b_pad 4
kernel k2 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 loss_sum Mse f32 (Tensor<f32, [100, 10]>, Tensor<f32, [100, 10]>) -> Tensor<f32, []> workgroup_size 64 workgroups [1, 1, 1]
```

`plan step` is the training step, numbered: `dispatch` steps run a kernel, `barrier` steps order
them. `plan forward` is the forward program alone; `train` does not run it. The profile below
names a dispatch by its position in `plan step`. Each `dispatch` line carries, after its
workgroups, `flop <n> bytes <n>`: the arithmetic of the kernel and the bytes of the views it reads
and writes, by the counting rule of DESIGN.md § 3.1, or `uncounted` for a kernel the rule has no
count for. Step 0, `k0`, is `flop 10035200 bytes 565504`, the `2·m·k·n` of
`[100, 784] x [784, 64]`; the 10 dispatches of the step count 20,566,432 FLOP and 1,908,484 bytes.

## 4. The compiled artifact

```shell
modelspec compile examples/mnist_mlp.mspec --backend vulkan --output build/mnist --device-profile crates/modelspec-runtime/tests/fixtures/rtx3090.device.json
```

writes the artifact directory `build/mnist`: `artifact.json`, the manifest with the parameters,
kernels and plans, and `kernels/<kernel_module_id>.spv`, one SPIR-V module per `kernel` line of
`--stage kernel`. `--device-profile` builds the modules for a device profile file, here the RTX
3090 profile the tests use, so no GPU or Vulkan loader is needed; without it `compile` builds for
device 0. Before it prints `artifact <artifact_id> <artifact_hash> kernels=10 build/mnist`, it
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
to time a GPU whose clocks are not locked. This is one run on an RTX 3090 that had just run the
same command three times with `--stats`. Its SM clock was between 780 and 1,080 MHz in its 46
samples of 100 ms, below the 1,965 MHz of the runs in the last paragraph, so its times do not
compare with theirs:

```text
profile dispatch step 0 6bd32d9c-1c0c-8fd4-96b2-46d2c722adc9 main calls 72000 total_us 1077227.040 mean_us 14.961 flop 10035200 bytes 565504 gflop_per_s 670.735 gb_per_s 37.797
profile dispatch step 2 b2b11a63-ac18-8b96-9ec3-1751b6ebc039 main calls 72000 total_us 279154.240 mean_us 3.877 flop 128000 bytes 32160 gflop_per_s 33.014 gb_per_s 8.295
profile dispatch step 4 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 main calls 72000 total_us 547090.624 mean_us 7.598 flop 2000 bytes 8004 gflop_per_s 0.263 gb_per_s 1.053
profile dispatch step 5 09cd7ac3-53f2-8dbf-9ba7-b3ae566b111b main calls 72000 total_us 221928.928 mean_us 3.082 flop 2000 bytes 12000 gflop_per_s 0.649 gb_per_s 3.893
profile dispatch step 7 aa03322b-5cc2-8da1-afcf-82921498cd59 main calls 72000 total_us 256955.776 mean_us 3.569 flop 128000 bytes 32160 gflop_per_s 35.866 gb_per_s 9.011
profile dispatch step 8 41241727-be45-83a9-81e8-662471eaedc9 main calls 72000 total_us 314758.784 mean_us 4.372 flop 128000 bytes 32160 gflop_per_s 29.280 gb_per_s 7.356
profile dispatch step 10 ae896a79-8d47-8bca-807a-453bc394ae43 main calls 72000 total_us 224121.088 mean_us 3.113 flop 6400 bytes 76800 gflop_per_s 2.056 gb_per_s 24.672
profile dispatch step 12 c2dc253f-25aa-8b18-8282-fcea6ad8f491 main calls 72000 total_us 590253.824 mean_us 8.198 flop 10035200 bytes 539904 gflop_per_s 1224.108 gb_per_s 65.858
profile dispatch step 14 88daaa23-8dc9-8bc4-ab05-dc5ecc946d99 main calls 72000 total_us 277634.752 mean_us 3.856 flop 100352 bytes 602112 gflop_per_s 26.025 gb_per_s 156.148
profile dispatch step 15 f509ed52-1817-8bb7-91f5-a998f93b9005 main calls 72000 total_us 219640.544 mean_us 3.051 flop 1280 bytes 7680 gflop_per_s 0.420 gb_per_s 2.518
profile dispatch final 0 6bd32d9c-1c0c-8fd4-96b2-46d2c722adc9 main calls 1 total_us 12.288 mean_us 12.288 flop 10035200 bytes 565504 gflop_per_s 816.667 gb_per_s 46.021
profile dispatch final 2 b2b11a63-ac18-8b96-9ec3-1751b6ebc039 main calls 1 total_us 4.096 mean_us 4.096 flop 128000 bytes 32160 gflop_per_s 31.250 gb_per_s 7.852
profile dispatch final 4 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 main calls 1 total_us 6.144 mean_us 6.144 flop 2000 bytes 8004 gflop_per_s 0.326 gb_per_s 1.303
profile dispatches 720003
profile submits 282
profile host_waits 1
profile barriers_per_step 9
profile bytes_uploaded 520864
profile bytes_downloaded 491268
profile device_allocations 16
profile peak_device_bytes 2134704
profile pipeline_creation_us 1324.508
profile compile_us 1741.990
profile steady_us_per_step 59.444
profile step_total flop 20566432 bytes 1908484 gflop_per_s 345.983 gb_per_s 32.106 uncounted_sites 0
```

| Field | What it says |
|---|---|
| `step <position>` | a `dispatch` step of `plan step`, run once per training step: `calls` is the number of steps |
| `final <position>` | the loss of the final parameters, computed once after the last step by the steps of `plan step` up to the loss; `<position>` is the index in `plan step` |
| `<kernel_module_id> <entry>` | the module and its entry point; find the module id on a `kernel` line of `--stage kernel` |
| `total_us`, `mean_us` | GPU time from just before the dispatch starts to just after it ends, summed over the calls and per call |
| `steady_us_per_step` | GPU time per step from step 20 to the final loss, so the first 20 steps are left out as warm-up |
| `dispatches`, `submits`, `host_waits`, `bytes_uploaded`, `bytes_downloaded` | the run's counters, the same as the `--stats` lines: one host wait for the whole run |
| `barriers_per_step` | the pipeline barriers one training step records: the `barrier` steps of `plan step` and the two around the copy of the step's loss |
| `device_allocations`, `peak_device_bytes` | the `vkAllocateMemory` calls of the run, one per buffer, and the largest sum of the allocation sizes of its live buffers, as the device rounds them up |
| `pipeline_creation_us`, `compile_us` | host time to build the pipelines and to compile the file and emit its kernels, outside the steady state |
| `flop`, `bytes` | after `mean_us`: the counts of the dispatch on its `plan step` line of `--stage kernel` |
| `gflop_per_s`, `gb_per_s` | `flop` and `bytes` over `mean_us`; `unavailable` when the time is 0 |
| `uncounted` | in place of the four fields above, for a dispatch the counting rule has no count for |
| `step_total` | the last line: the `flop` and `bytes` of every counted `step` site, over `steady_us_per_step`, and `uncounted_sites`, the `step` sites it leaves out |

The step's 9 barriers are the 7 `barrier` steps of `plan step` and the 2 around the copy of its
loss, and the 16 allocations are the 13 plan buffers, the loss buffer and one staging buffer each
for the upload and the read-back. None of the counters depends on timing. The last line divides
the step's 20,566,432 FLOP and 1,908,484 bytes by the 59.444 µs step. Bytes are what the views
name, not DRAM traffic measured on the device.

Here step position 4, module `d588cbc2…`, is `k2`, the `mse` loss sum on one workgroup of 64
invocations: 7.6 µs of the 59.4 µs step. A device without timestamp queries prints
`profile kernel timings unavailable: no timestamp queries` in place of the dispatch lines and
still prints the counters.

The timestamps cost time themselves. On the same warm RTX 3090, 72,000 steps of the 14
dispatches the step had before matmuls read transposed operands in place took 27.1 to 27.3 µs
per step with `--stats` alone and 44.0 to 44.2 µs with `--profile` (three runs each, SM clock
between 1,860 and 1,980 MHz in 100 ms samples, 1,965 MHz in 182 of 192). The 11 dispatches the
step had before its loss gradient became one elementwise chain took 37.1 to 37.3 µs with
`--profile` (three runs, SM clock 1,965 MHz in all 90 samples). Compare profiled runs with
profiled runs, and use `--stats` alone for the time per step.
