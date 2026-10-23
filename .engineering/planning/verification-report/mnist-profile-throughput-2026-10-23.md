---
format: aep.planning-md/3
id: verification-report:mnist-profile-throughput-2026-10-23
kind: verification-report
status: draft
title: 'MNIST profile on the RTX 3090: 447.8 GFLOP/s, 56.6 GB/s per step, every site counted'
relations:
- reviews: story:profile-achieved-throughput
revision: 2
---
# Achieved GFLOP/s and GB/s of examples/mnist_mlp.mspec on the RTX 3090

One run on 2026-10-23 at 19:26:41Z, by the coordinating agent, with no warm-up run before it (approval-record:benchmark-wave-scope D14), from wave/benchmark at b2f9553, release build:

```console
modelspec train examples/mnist_mlp.mspec --backend vulkan --steps 72000 --profile --serve 127.0.0.1:8080 --ui ui/dist --record <file>
```

The run was served to the observatory UI in a browser on the same machine while it ran; the recording has 72003 lines.

## Host state

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, driver 610.57.04 (`nvidia-smi`) |
| other compute processes at start | none (`nvidia-smi --query-compute-apps`) |
| SM clock, 132 samples every 100 ms | 71 at 210 MHz, 20 at 1860 MHz, 41 at 1950 MHz; not locked |
| load average after the run | 15.15 (`/proc/loadavg`); shared workstation |

## Profile output (stderr, every `profile` line)

```text
profile dispatch step 0 ecca4089-63a1-8ae9-868c-1a08decd036d main calls 72000 total_us 635851.744 mean_us 8.831 flop 10035200 bytes 565504 gflop_per_s 1136.325 gb_per_s 64.034
profile dispatch step 2 00a652b2-efba-84ba-99a0-ab965096b910 main calls 72000 total_us 171039.584 mean_us 2.376 flop 128000 bytes 32160 gflop_per_s 53.882 gb_per_s 13.538
profile dispatch step 4 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 main calls 72000 total_us 294214.944 mean_us 4.086 flop 2000 bytes 8004 gflop_per_s 0.489 gb_per_s 1.959
profile dispatch step 5 2e1ec459-43bd-81d2-b9db-fa0f7fe3ae74 main calls 72000 total_us 143504.160 mean_us 1.993 flop 1000 bytes 12000 gflop_per_s 0.502 gb_per_s 6.021
profile dispatch step 7 3e9bd9e4-7689-8b30-9582-05282d19167c main calls 72000 total_us 136187.104 mean_us 1.891 flop 1000 bytes 8000 gflop_per_s 0.529 gb_per_s 4.229
profile dispatch step 8 00cb6768-fb04-8fde-a136-7b324efe8fbc main calls 72000 total_us 145561.408 mean_us 2.022 flop 0 bytes 5120 gflop_per_s 0.000 gb_per_s 2.533
profile dispatch step 10 781f88a2-e90a-8494-8ebd-b46bede8b3ea main calls 72000 total_us 161928.544 mean_us 2.249 flop 128000 bytes 32160 gflop_per_s 56.914 gb_per_s 14.300
profile dispatch step 11 4ab19294-379e-8a14-b7ee-a9d23e2520e2 main calls 72000 total_us 151004.032 mean_us 2.097 flop 0 bytes 51200 gflop_per_s 0.000 gb_per_s 24.413
profile dispatch step 13 88914743-2fe6-8086-8432-75c855f4ac12 main calls 72000 total_us 189477.856 mean_us 2.632 flop 128000 bytes 32160 gflop_per_s 48.639 gb_per_s 12.221
profile dispatch step 14 ae896a79-8d47-8bca-807a-453bc394ae43 main calls 72000 total_us 142820.672 mean_us 1.984 flop 6400 bytes 76800 gflop_per_s 3.226 gb_per_s 38.717
profile dispatch step 15 cbd56b2b-4530-8c59-8969-a9eed9bc2f0c main calls 72000 total_us 231914.272 mean_us 3.221 flop 0 bytes 627200 gflop_per_s 0.000 gb_per_s 194.720
profile dispatch step 17 73027c05-b462-8722-968d-e4181ae7e1f1 main calls 72000 total_us 338923.744 mean_us 4.707 flop 10035200 bytes 539904 gflop_per_s 2131.849 gb_per_s 114.696
profile dispatch step 19 88daaa23-8dc9-8bc4-ab05-dc5ecc946d99 main calls 72000 total_us 180167.552 mean_us 2.502 flop 100352 bytes 602112 gflop_per_s 40.103 gb_per_s 240.621
profile dispatch step 20 f509ed52-1817-8bb7-91f5-a998f93b9005 main calls 72000 total_us 139422.080 mean_us 1.936 flop 1280 bytes 7680 gflop_per_s 0.661 gb_per_s 3.966
profile dispatch final 0 ecca4089-63a1-8ae9-868c-1a08decd036d main calls 1 total_us 7.296 mean_us 7.296 flop 10035200 bytes 565504 gflop_per_s 1375.439 gb_per_s 77.509
profile dispatch final 2 00a652b2-efba-84ba-99a0-ab965096b910 main calls 1 total_us 2.048 mean_us 2.048 flop 128000 bytes 32160 gflop_per_s 62.500 gb_per_s 15.703
profile dispatch final 4 d588cbc2-b3eb-8bf1-93a1-29d8680aba58 main calls 1 total_us 4.096 mean_us 4.096 flop 2000 bytes 8004 gflop_per_s 0.488 gb_per_s 1.954
profile dispatches 1008003
profile submits 282
profile host_waits 1
profile bytes_uploaded 520864
profile bytes_downloaded 491268
profile pipeline_creation_us 867361.161
profile compile_us 322529.036
profile steady_us_per_step 45.925
profile step_total flop 20566432 bytes 2600004 gflop_per_s 447.828 gb_per_s 56.614 uncounted_sites 0
```

## Reading

- Every site is counted: `uncounted_sites 0`; per step 20566432 FLOP and 2600004 bytes, 447.828 GFLOP/s and 56.614 GB/s over the steady 45.925 µs per step.
- The matmul site `step 0` reaches 1136.325 GFLOP/s; the small elementwise and reduction sites stay under 54 GFLOP/s.
- The steady time per step is 45.925 µs against 44.797 µs in website/docs/guides/inspect-a-run.md:136, measured after a 72000-step warm-up run: 2.5 % slower with no warm-up run and the UI served. The three sites compared agree within 3 % (step 0: 8.831 against 8.603 µs, step 2: 2.376 against 2.335, step 4: 4.086 against 4.075).
