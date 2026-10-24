//! Adversary cases for `story:m1-loss-sum-staged`: the staged `mse` loss sum on shapes of rank 0,
//! 1, 3 and 4, at term counts that are not a multiple of 64 or of the chunk, and with device
//! profiles whose workgroup memory gives chunks of 1, 2, 63, 64, 65, 12,287 and 12,288 terms and
//! an odd byte count.
//!
//! The device case asserts the row-major order exactly. `OpFSub` and `OpFMul` are correctly
//! rounded, each later square is one `GLSL.std.450 Fma` (DESIGN.md § 12 ADR-009) that the CPU
//! computes with `mul_add`, all are decorated `NoContraction`, and the inputs have no
//! subnormal, so on a
//! device that rounds them to nearest even the one admissible result of the row-major order is
//! the CPU's, bit for bit. The inputs spread the terms over 2^46 in magnitude, so moving one term
//! changes the bits. The module declares `RoundingModeRTE` only in the bit-for-bit class
//! (DESIGN.md § 5.4); outside it, as on the RTX 3090, which reports no denorm preservation, the
//! case relies on the device's default rounding being to nearest even, and it runs only on a
//! device that reports `shaderRoundingModeRTEFloat32`. It is ignored by default; run it with
//! `cargo test -p modelspec-runtime --features vulkan --test adversary_loss_sum -- --ignored`.

#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::Draw;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Kernel, KernelKind, KernelModuleId, Operand};
use modelspec_core::ir::tensor::{DType, TensorType};
use modelspec_runtime::spirv::emit::{emit_loss_sum_undivided, emit_module};
use modelspec_runtime::spirv::validate::validate;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

/// Shapes of rank 0, 1, 3 and 4, none of the multi-term ones a multiple of 64 except where
/// named: 1, 127, 105, 1,122, 4,097, 12,285, 12,288, 12,289, 24,576 (= 2 × 12,288) and 24,639.
const SHAPES: [&[u64]; 11] = [
    &[],
    &[127],
    &[3, 5, 7],
    &[2, 33, 17],
    &[17, 241],
    &[7, 13, 135],
    &[4, 3072],
    &[12_289],
    &[2, 3, 4, 1024],
    &[3, 8213],
    &[1, 1, 1],
];

/// Workgroup memory a profile may report, in bytes: chunks of 1, 2, 63, 64, 65, 4,096, 12,287
/// and 12,288 terms, and an odd byte count above the RTX 3090's.
const BYTES: [u32; 9] = [4, 11, 252, 256, 260, 16_384, 49_148, 49_152, 65_539];

fn mse_kernel(shape: &[u64]) -> Kernel {
    let ty = if shape.is_empty() {
        TensorType::f32_scalar()
    } else {
        TensorType::f32(shape)
    };
    let mut kernel = Kernel {
        id: KernelModuleId([0; 16]),
        kind: KernelKind::LossSum,
        op: "Mse",
        dtype: DType::F32,
        operands: vec![Operand::Buffer(ty.clone()), Operand::Buffer(ty)],
        result: TensorType::f32_scalar(),
        reduces: None,
        epilogue: None,
        transposed: [false, false],
        workgroup_size: KernelKind::LossSum.workgroup_size(),
        workgroups: [1, 1, 1],
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

fn elements(shape: &[u64]) -> usize {
    shape.iter().product::<u64>() as usize
}

fn profile_with(base: &DeviceProfile, bytes: u32) -> DeviceProfile {
    let mut profile = base.clone();
    profile.max_shared_memory_bytes = bytes;
    profile
}

/// Inputs whose terms spread over 2^46: `a` is a draw scaled by `2^((7 i) mod 23 - 11)`, `b` a
/// second draw. No term is subnormal.
fn spread(draw: &mut Draw, n: usize) -> (Vec<f32>, Vec<f32>) {
    let a = draw.values(n as u64);
    let b = draw.values(n as u64);
    let a = a
        .iter()
        .enumerate()
        .map(|(i, &x)| x * 2f32.powi(((7 * i) % 23) as i32 - 11))
        .collect();
    (a, b)
}

/// The CPU interpreter's sum: the first term, then each next one in row-major order, its square
/// fused into the sum (DESIGN.md § 12 ADR-009).
fn cpu_sum(a: &[f32], b: &[f32]) -> f32 {
    let mut diffs = a.iter().zip(b).map(|(&x, &y)| x - y);
    let first = diffs.next().expect("one term at least");
    diffs.fold(first * first, |acc, d| d.mul_add(d, acc))
}

#[test]
fn every_staged_loss_sum_of_any_rank_and_chunk_passes_spirv_val_with_every_fop_uncontracted() {
    let base = DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile");
    let mut total = 0;
    for shape in SHAPES {
        let kernel = mse_kernel(shape);
        for bytes in BYTES {
            let profile = profile_with(&base, bytes);
            for module in [
                emit_module(&kernel, &profile).expect("the loss sum emits"),
                emit_loss_sum_undivided(&kernel, &profile).expect("the undivided sum emits"),
            ] {
                let what = format!("{shape:?}, {bytes} bytes");
                validate(&module.spirv).unwrap_or_else(|error| panic!("{what}: {error}"));
                let (missing, all) = common::words::uncontracted_arithmetic(&module.spirv);
                assert!(
                    all >= 2 && missing.is_empty(),
                    "{what}: {missing:?} of {all}"
                );
                total += 1;
            }
        }
    }
    eprintln!("adversary_loss_sum: {total} modules valid");
}

#[cfg(feature = "vulkan")]
mod device {
    use super::*;
    use common::{GUARD_WORDS, guard, guarded};
    use modelspec_runtime::spirv::emit::KernelModule;
    use modelspec_runtime::vulkan::memory::Buffer;
    use modelspec_runtime::vulkan::negotiate::negotiate;
    use modelspec_runtime::vulkan::sync::Barrier;
    use modelspec_runtime::vulkan::{Device, Vulkan};

    const UNWRITTEN: u32 = 0x7fc0_dead;

    fn run(
        device: &Device<'_>,
        kernel: &Kernel,
        module: &KernelModule,
        a: &[f32],
        b: &[f32],
    ) -> Vec<f32> {
        let uploads = [
            guarded(a),
            guarded(b),
            guarded(&[f32::from_bits(UNWRITTEN)]),
        ];
        let bytes = |values: &Vec<f32>| (values.len() * 4) as u64;
        let buffers: Vec<Buffer<'_>> = uploads
            .iter()
            .map(|values| device.device_local_buffer(bytes(values)).unwrap())
            .collect();
        let mut staging: Vec<Buffer<'_>> = uploads
            .iter()
            .map(|values| {
                let mut buffer = device.staging_buffer(bytes(values)).unwrap();
                buffer.write_f32(values).unwrap();
                buffer
            })
            .collect();
        let pipeline = device.kernel_pipeline(kernel, &module.spirv).unwrap();
        let mut recording = device.record().unwrap();
        for (from, to) in staging.iter().zip(&buffers) {
            recording.copy(from, to).unwrap();
        }
        recording.barrier(Barrier::TransferToCompute);
        let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
        recording
            .dispatch_kernel(&pipeline, kernel, &bound)
            .unwrap();
        recording.barrier(Barrier::ComputeToTransfer);
        recording.copy(&buffers[2], &staging[2]).unwrap();
        recording.barrier(Barrier::TransferToHost);
        let recorded = recording.finish().unwrap();
        device.submit(recorded).unwrap().wait().unwrap();
        staging[2].read_f32().unwrap()
    }

    #[test]
    #[ignore = "needs a Vulkan device"]
    fn the_staged_sum_equals_the_row_major_sum_bit_for_bit_on_a_round_to_nearest_device() {
        let vulkan = Vulkan::load().expect("the Vulkan loader opens");
        let candidates = vulkan.candidates().expect("the devices are probed");
        let profile = candidates[0].profile.clone();
        if profile.shader_rounding_mode_rte_f32 != Some(true) {
            eprintln!("adversary_loss_sum: device 0 reports no RTE for f32, nothing to check");
            return;
        }
        let negotiated = negotiate(&profile).expect("device 0 negotiates");
        let device = vulkan.create_device(0, &negotiated).unwrap();
        let mut draw = Draw::new(0x6164_7665_7273_6172);
        let mut wrong = Vec::new();
        let mut runs = 0;
        for shape in SHAPES {
            let n = elements(shape);
            let (a, b) = spread(&mut draw, n);
            assert!(a.iter().chain(&b).all(|v| !v.is_subnormal()));
            let expected = cpu_sum(&a, &b);
            let kernel = mse_kernel(shape);
            for bytes in BYTES {
                if bytes > profile.max_shared_memory_bytes {
                    continue;
                }
                if bytes < 16 && n > 4_200 {
                    continue;
                }
                let target = profile_with(&profile, bytes);
                let module = emit_loss_sum_undivided(&kernel, &target).unwrap();
                let out = run(&device, &kernel, &module, &a, &b);
                runs += 1;
                for i in 0..GUARD_WORDS {
                    if out[1 + i].to_bits() != guard(i) {
                        wrong.push(format!("{shape:?} {bytes} bytes: guard word {i}"));
                    }
                }
                if out[0].to_bits() != expected.to_bits() {
                    wrong.push(format!(
                        "{shape:?} ({n} terms), {bytes} bytes: gpu {:?} ({:#010x}) cpu {expected:?} \
                         ({:#010x})",
                        out[0],
                        out[0].to_bits(),
                        expected.to_bits()
                    ));
                }
            }
        }
        eprintln!(
            "adversary_loss_sum: {runs} device runs, {} differ",
            wrong.len()
        );
        assert!(wrong.is_empty(), "{wrong:#?}");
    }
}
