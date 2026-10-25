# Block kernel diagnosis, `[8192, 784] x [784, 4096]` (story:matmul-block-kernel-throughput)

Device: NVIDIA GeForce RTX 3090, driver 610.57.04, clocks never locked. Source: the `mlp_sweep`
step at batch 8192, hidden 4096, 220 steps (`<scratch>/mlp_sweep_b8192_h4096.mspec`, the source
`benches/mlp_sweep.rs` formats). Every figure is the median of 3 interleaved rounds of
`modelspec train --backend vulkan --profile` (`mean_us` of the site), each run with no other
compute app on the GPU (`# gpu-apps-before []` in every `.txt`, the 1 s `nvidia-smi
--query-compute-apps` samples in every `.apps.csv`), temperature and SM clock sampled before,
after and every 500 ms during each run (`.gpu.csv`). The block kernel is run against the FMA
kernels of ADR-009, as coordinator decision D27, 2026-10-24, `.engineering/waves/perf-1.md`
requires.

Site 0 is `[8192, 784] x [784, 4096]` (the forward matmul); site 14 is
`[784, 8192]ᵀ x [8192, 4096]` (the weight gradient, `a` read transposed).

## Directories

| directory | what | binaries |
|---|---|---|
| `diag1/` | the shipped block and the 64×128 tile, each with one ablation per hypothesis | `<scratch>/modelspec-probe1`: base plus env-selected ablations |
| `diag2/` | the vector-read block against the shipped block and the 64×128 tile, and its ablations | `<scratch>/modelspec-probe1`, `<scratch>/modelspec-vec1` |
| `pad/` | the vector-read block with and without its 4-value `a` padding | `<scratch>/modelspec-wired`, `<scratch>/modelspec-nopad` |
| `faster/` | `block-faster`: the vector-read block against the 64×128 tile, deterministic and `fast_math` | `<scratch>/modelspec-vec1` |
| `smalln/` | sites 2, 9 and 10 (`n = 10`): `Fma`, no `Fma`, unroll 2 and 8 | `<scratch>/modelspec-probe2` |
| `samebits/` | `train --backend vulkan` stdout sha256, base against the final tree, four sources | `<scratch>/modelspec-base`, `<scratch>/modelspec-final` |
| `final/` | `block-faster` and `throughput-reported` on the final tree against the base | `<scratch>/modelspec-base`, `<scratch>/modelspec-final` |
| `sweep/` | `mlp_sweep` at 8192:4096, 8192:1024 and 1000:1024 | the bench of the base and of the final tree |

The probe binaries select an ablation with `MODELSPEC_PROBE_ABLATE` (and the tile it applies
to with `MODELSPEC_PROBE_ABLATE_WG`), schedule for the RTX 3090 profile with
`MODELSPEC_PROBE_PROFILE`, read the vector layout with scalar loads with
`MODELSPEC_PROBE_SCALAR`, emit a multiply and an add in place of each `Fma` with
`MODELSPEC_PROBE_NOFMA`, and set the tile-loop unroll with `MODELSPEC_PROBE_UNROLL`. None of that
code is in the tree; `<scratch>/emit.rs.vec1` keeps the ablations. Ablations that change the
arithmetic (`noload`, `sametile`, `onceb`, `oncea`) are timing probes only; `sametile` exits 1 on
the divergence check, as expected.

## Pipeline statistics, tile against block

From the `profile pipeline` lines of `diag1/r1-wide.txt`, `diag1/r1-block.txt`,
`diag2/r1-block-vec.txt`, `diag2/r1-block-vec-scalar.txt` and `pad/r1-block-vec-nopad.txt`
(`VK_KHR_pipeline_executable_properties`, executable 0 "CS", subgroup size 32). The driver
reports Local Memory Size as 2^36 plus the bytes; the column shows the bytes. "declared" is the
workgroup memory the module declares (`MatMulTile::layout`).

| kernel | site | Register Count | Binary Size | Shared Memory Size | declared | spilled to shared | Local Memory |
|---|---|---|---|---|---|---|---|
| 64×128 tile (128 invocations) | 0 | 168 | 75,904 | 12,288 | 12,288 | 0 | 0 |
| 64×128 tile | 14 | 168 | 92,672 | 14,336 | 14,336 | 0 | 0 |
| shipped block (256 invocations) | 0 | 128 | 76,800 | 35,840 | 16,384 | 19,456 (19 floats per invocation) | 0 |
| shipped block | 14 | 128 | 93,696 | 48,128 | 20,480 | 27,648 (27 floats) | 16 |
| vector-read block | 0 | 128 | 70,272 | 16,640 | 16,640 | 0 | 0 |
| vector-read block | 14 | 128 | 87,552 | 30,208 | 16,896 | 13,312 (13 floats) | 0 |
| vector layout, scalar reads (probe) | 0 | 255 | 92,544 | 27,904 | 16,640 | 11,264 | 0 |
| vector-read block without `a` padding (probe) | 0 | 128 | 70,272 | 16,384 | 16,384 | 0 | 0 |

At 256 invocations the driver holds the block at 128 registers (two workgroups, 65,536
registers, per SM) and moves what does not fit into shared memory. Every `diag1` ablation of
the block keeps 128 registers; `noload` and `oncea` bring its shared memory back to the
declared 16,384 bytes. The 64×128 tile's ablations `noload`, `sametile` and `oncea` take it
from 168 to 128 registers.

## Diagnosis of the block at site 0

| hypothesis | probe | figure (site 0, µs; site 14 in brackets) | verdict |
|---|---|---|---|
| registers or spills | statistics above; the block's ablations | shipped block 128 registers and 19,456 B spilled to shared memory at site 0, 27,648 B at site 14; the 64×128 tile spills nothing | a cause: the spill adds shared-memory stores and loads to the inner loop of a kernel already bound by shared-memory reads (last row) |
| workgroup-memory bank conflicts | `pad/`: the vector-read block with `a` lines of 128 (no padding) against 132; the unit test `the_block_stages_k_major_aligned_for_vector_reads_without_bank_conflicts` counts conflicts by address | padded 2,614.5 (2,798.0), unpadded 2,722.1 (2,756.9): +4.1 % at site 0 without the padding; site 14 reads `a` transposed with lines of 136 in both | a small term, removed by the padding; not the block's bound |
| occupancy | statistics: registers per workgroup against the SM's 65,536 (published, not measured) | block 128 × 256 = 32,768 registers, 2 workgroups (16 warps) per SM; tile 168 × 128 = 21,504, 3 workgroups (12 warps) | not the bound: the block keeps more warps resident than the tile (inferred from the statistics; no occupancy counter is read) |
| global load width | `noload` (no global reads, constants staged) and `sametile` (every tile reads `k` tile 0) | shipped block 3,089.6 → noload 2,410.0 (-22 %), sametile 2,529.6; tile 2,949.5 → 2,354.6 (-20 %), sametile 2,456.3; vector-read block 2,686.5 → 2,265.6 (-16 %) | a remaining term of the same size for both tiles, not what makes the block slower |
| barrier count per `k` tile | `nobarrier` (the one barrier per tile removed) | shipped block 3,089.6 → 2,980.4 (-3.5 %) (3,767.8 → 3,464.6); tile 2,949.5 → 2,888.5 (-2.1 %); vector-read block 2,686.5 → 2,535.6 (-5.6 %) | minor |
| bounds checks in the inner loop | `nopick` (the select that zeroes an out-of-range read removed) | shipped block 3,089.6 → 3,497.5 (rounds 3,539 / 3,497 / 3,127) | not the bound: removing it does not make the block faster |
| scalar workgroup-memory read issue | `onceb` / `oncea` (`b` or `a` read from workgroup memory once per tile and reused) | shipped block 3,089.6 → onceb 2,294.1 (-26 %), oncea 2,492.8 (-19 %); tile 2,949.5 → 2,402.4 (-19 %), 2,588.3 (-12 %) | the bound: 16 scalar `LDS` per `k` for 64 `Fma` per invocation, plus the spill traffic, is the largest term and larger in the block |

The bound: the block issues one scalar workgroup-memory load per factor (8 of `a` and 8 of `b`
per `k` for 64 `Fma`), and at the 128 registers the driver gives 256 invocations it spills 19
floats per invocation into the same shared memory, so its inner loop issues more shared-memory
instructions per `Fma` than the 64×128 tile at 168 registers. Halving global traffic per FLOP
does not help a kernel bound there.

The fix (`MatMulTile::vector_layout`, the vector path of `matmul` in `emit.rs`): each invocation
of the block takes runs of 4 contiguous columns and 4 contiguous rows, both operands are staged
`k`-major with every line start a multiple of 4 values (`a` lines of 132, or 136 read
transposed; `b` lines of 128, or 132 read transposed), workgroup memory is declared as a `vec4`
array, and per `k` an invocation reads 2 `vec4` of `a` and 2 of `b` (4 loads for 64 `Fma`). The
products, their order and the accumulator per output element are those of the scalar block and
of the 64×128 tile, so the bits are unchanged (`samebits/`, and
`vulkan_matmul_plan_sites_same_bits`).

| variant | site 0 | site 14 | steady µs per step | SM MHz | raw |
|---|---|---|---|---|---|
| 64×128 tile | 2,959.5 | 3,710.9 | 8,294.7 | 1,800 | `diag2/r*-wide.txt` |
| shipped block | 3,117.9 | 3,741.7 | 8,490.8 | 1,815 | `diag2/r*-block.txt` |
| vector-read block | 2,686.5 | 2,878.1 | 7,183.1 | 1,755 | `diag2/r*-block-vec.txt` |
| vector layout, scalar reads | 4,577.9 | 5,020.8 | 11,141.6 | 1,890 | `diag2/r*-block-vec-scalar.txt` |
| vector-read block, `noload` | 2,265.6 | 2,288.8 | 6,179.7 | 1,800 | `diag2/r*-block-vec-noload.txt` |
| vector-read block, `nobarrier` | 2,535.6 | 2,652.6 | 6,824.1 | 1,710 | `diag2/r*-block-vec-nobarrier.txt` |

The scalar-read probe of the same layout reaches 255 registers and runs 70 % slower: the
vector reads, not the layout alone, take the block off its bound. What is left at site 0 is
global reads (`noload` -16 %) and the barrier (-5.6 %). Site 14 still spills 13,312 B (13 floats
per invocation); that is the lead for a later change there.

## The final tree against the base

`final/`: `train --backend vulkan --profile` of the base binary (92ba28b, the plan for every
device, the 64×128 tile at sites 0 and 14) and of the final tree (the plan for the device, the
vector-read block at sites 0 and 14), interleaved, deterministic and `--fast-math`. Medians of
3 rounds of 220 steps; `mean_us` and GFLOP/s per matmul site.

| site | shape | base | final | base `fast_math` | final `fast_math` |
|---|---|---|---|---|---|
| 0 | `[8192, 784] x [784, 4096]` | 2,853.5 µs, 18,438 GFLOP/s | 2,600.0 µs, 20,236 | 2,919.8 µs, 18,020 | 2,606.1 µs, 20,189 |
| 2 | `[8192, 4096] x [4096, 10]` | 266.6, 2,517 | 267.5, 2,509 | 263.9, 2,543 | 262.5, 2,556 |
| 9 | `[8192, 10] x [10, 4096]ᵀ` | 161.6, 4,154 | 160.1, 4,191 | 166.8, 4,024 | 158.3, 4,238 |
| 10 | `[4096, 8192]ᵀ x [8192, 10]` | 317.7, 2,112 | 316.3, 2,122 | 297.2, 2,258 | 312.9, 2,145 |
| 14 | `[784, 8192]ᵀ x [8192, 4096]` | 3,573.6, 14,723 | 2,766.6, 19,018 | 3,588.2, 14,663 | 2,767.5, 19,011 |
| steady step | | 8,019.3 µs | 6,968.7 µs | 8,022.7 µs | 6,973.3 µs |

Site 0 rounds: base 2,837.9 / 2,853.5 / 2,883.3, final 2,591.2 / 2,600.0 / 2,608.1; with
`fast_math` base 3,209.9 / 2,919.8 / 2,868.5, final 2,579.0 / 2,606.1 / 2,617.9.

`sweep/`: the `mlp_sweep` bench of the base tree (`git archive` of 92ba28b, built in
`<scratch>/base-tree`) and of the final tree, interleaved, 3 rounds, each the median of the
bench's 3 timed runs of at least 6 s; the figure is the median of the 3 rounds. PyTorch eager
CUDA: 7,290.5, 1,932.1 and 555.1 µs per step (the story's figures).

| point | base | final | base `fast_math` | final `fast_math` | final / PyTorch | final `fast_math` / PyTorch |
|---|---|---|---|---|---|---|
| b8192 h4096 | 8,721.2 µs | 7,504.0 µs (rounds 7,208.0 / 7,504.0 / 8,540.1) | 8,726.1 | 7,658.9 | 1.029 | 1.051 |
| b8192 h1024 | 2,600.8 | 2,535.5 | 2,593.8 | 2,588.6 | 1.312 | 1.340 |
| b1000 h1024 | 369.6 | 368.5 | 373.6 | 375.3 | 0.664 | 0.676 |

The GPU reached 85 to 90 °C during the sweep and its SM clock under load fell from a median
of 1,785 to 1,800 MHz in round 1 to 1,665 MHz in round 3 (`sweep/*.gpu.csv`), so the rounds
of one arm differ by up to 18 %; both arms alternate within each round.

## The earlier figures and the clock ramp

The figures in `benchmarks/raw/2026-10-24-fma/` were taken with the GPU idle between runs: in
`ab/r1-fma-det-profile-b8192-h4096.gpu.csv` the SM clock climbs from 780 MHz idle through 825,
975, 1,095, 1,290 to 1,440 MHz during the 2.5 s run. Here the runs follow each other and every
`.gpu.csv` shows 1,710 to 1,890 MHz under load. The same 64×128 kernel (`bf68bc3f…`) reads
5,211.8 µs at site 0 there (`fma-profile-b8192-h4096-r1.txt`) and 2,959.5 to 2,995.8 µs here.
Comparisons in this directory are interleaved, so both arms see the same clock; absolute
figures from the two directories should not be compared.

## Small-n diagnosis: sites 2, 9 and 10

Site 2 is `[8192, 4096] x [4096, 10]` (`f39a…`, the two-row tile), site 9 `[8192, 10] x [10, 4096]ᵀ`
(`545b…`), site 10 `[4096, 8192]ᵀ x [8192, 10]` (`66d0…`). All run the plan for every device
(these shapes take tiles of 128 invocations in both plans). Raw: `smalln/`.

| variant | site 2 | site 9 | site 10 | site 0 | site 14 | registers site 2 / 9 / 10 |
|---|---|---|---|---|---|---|
| `Fma` (shipped, unroll 4) | 282.5 | 165.2 | 305.4 | 2,984.9 | 3,698.1 | 48 / 64 / 53 |
| no `Fma` (multiply and add, the base before ADR-009) | 223.7 | 167.8 | 278.0 | 4,103.3 | 5,347.3 | 47 / 64 / 47 |
| unroll 2 | 229.6 | 167.2 | 272.9 | 2,998.2 | 3,789.5 | 49 / 64 / 47 |
| unroll 8 | 264.7 | 164.8 | 288.3 | 3,068.8 | 3,936.7 | 49 / 64 / 51 |

| hypothesis | probe | figure | verdict |
|---|---|---|---|
| register count | statistics of each variant | 48 against 47 at site 2, 53 against 47 at site 10; far below any occupancy limit | not the cause |
| issue of `Fma` on a dependent chain against independent multiplies | `MODELSPEC_PROBE_NOFMA` | site 2 282.5 → 223.7 (-21 %), site 10 305.4 → 278.0 (-9 %), site 9 unchanged; the large sites lose 37–45 % without `Fma` | the cause at sites 2 and 10, together with the unroll (next row) |
| unroll | `MODELSPEC_PROBE_UNROLL` 2 and 8 | unroll 2: site 2 229.6, site 10 272.9, as fast as no `Fma`; unroll 8 in between; unroll 2 costs +2.5 % at site 14 | the 4-tile unrolled loop with a dependent `Fma` chain is what regresses at `n = 10` |

Site 9 has not regressed: 165.2 µs with `Fma` against 167.8 without. The 223 → 297 µs in the
brief come from the clock-ramp runs above. No fix here: `MATMUL_UNROLL` is one constant for
every matmul, and unroll 2 slows the weight-gradient site by 2.5 %. The lead for the later story
is an unroll of 2 for the narrow tiles only (output width under 16).

## vulkaninfo

```
driverVersion     = 610.57.4.0 (2559459584)
deviceName        = NVIDIA GeForce RTX 3090
driverInfo        = 610.57.04
VK_KHR_pipeline_executable_properties         : extension revision 1
pipelineExecutableInfo = true
```

`inspect-device` prints no extension list, so the story's "`inspect-device` lists it" does not
hold; the extension was read with `vulkaninfo`.
