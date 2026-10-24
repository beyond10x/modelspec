//! `story:matmul-tile-from-device-invocation-limit`: the matmul tile takes its invocation budget
//! from the device profile.
//!
//! - `tile-from-profile`: with a limit of 128 the rule gives the tiles of the rule before the
//!   change for every shape sampled from `1..=10_000`; with 1024 an output of at least 128 blocks
//!   of 128 by 128, `n >= 128` and `k >= 64` takes the 128 by 128 block of 256 invocations at 8 by
//!   8 each, and its signature, printed tile and module id show it.
//! - `verifier-cap-from-profile`: the plan verifier refuses a workgroup over the profile's
//!   `max_workgroup_invocations` with `WorkgroupTooLarge` and accepts 256 under a profile that
//!   allows it; without a profile the cap is [`MAX_WORKGROUP_INVOCATIONS`].
//! - `workgroup-memory-bounded`: every tile a profile's plan stages fits its workgroup memory,
//!   and a profile too small for the block gets the tile of the Vulkan minimum, not a refusal.
//! - `default-unchanged`: a plan scheduled without a profile, or with a profile of 128
//!   invocations, is the plan of [`schedule`].

use std::path::{Path, PathBuf};

use modelspec_core::compile::compile_training;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    ExecutionPlan, Kernel, KernelKind, MATMUL_LARGE_BLOCK, MIN_SHARED_MEMORY_BYTES, MatMulLimits,
    MatMulTile,
};
use modelspec_core::ir::verify::{
    MAX_WORKGROUP_INVOCATIONS, PlanVerifyError, verify_plan, verify_plan_for,
};
use modelspec_core::schedule::{schedule, schedule_for};

const RTX3090: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../modelspec-runtime/tests/fixtures/rtx3090.device.json"
));

fn rtx3090() -> DeviceProfile {
    DeviceProfile::from_json(RTX3090).expect("the fixture is a valid profile")
}

/// The RTX 3090 profile with `invocations` and `shared_bytes` in place of its limits.
fn limited(invocations: u32, shared_bytes: u32) -> DeviceProfile {
    let mut profile = rtx3090();
    profile.max_workgroup_invocations = invocations;
    profile.max_shared_memory_bytes = shared_bytes;
    profile
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The source `benches/mlp_sweep.rs` compiles at batch `batch` and hidden width `hidden`.
fn sweep(batch: u64, hidden: u64) -> String {
    format!(
        "model SweepMlp {{
    param w1: Tensor<f32, [784, {hidden}]> = xavier_uniform();
    param w2: Tensor<f32, [{hidden}, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [{batch}, 784]>)
        -> Tensor<f32, [{batch}, 10]> {{
        return matmul(relu(matmul(x, w1)), w2);
    }}
}}

train Fit {{
    model = SweepMlp;
    input x: Tensor<f32, [{batch}, 784]>;
    target y: Tensor<f32, [{batch}, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 220;
    seed = 42;
    device = vulkan(0);
}}
"
    )
}

/// The MNIST and linear regression examples and the three `mlp_sweep` points of the story.
fn sources() -> Vec<(String, String)> {
    let mut sources = Vec::new();
    for example in ["mnist_mlp", "linear_regression"] {
        let path = repo_root().join(format!("examples/{example}.mspec"));
        sources.push((
            example.to_owned(),
            std::fs::read_to_string(path).expect("the example reads"),
        ));
    }
    for (batch, hidden) in [(1000, 1024), (8192, 1024), (8192, 4096)] {
        sources.push((format!("sweep_b{batch}_h{hidden}"), sweep(batch, hidden)));
    }
    sources
}

/// The plan of `source` without a profile ([`schedule`]) and with `profile` ([`schedule_for`]).
fn plans(
    name: &str,
    source: &str,
    profile: Option<&DeviceProfile>,
) -> (ExecutionPlan, ExecutionPlan) {
    let compiled = compile_training(name, source).unwrap_or_else(|d| panic!("{name}: {d:?}"));
    let default = schedule(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
    )
    .expect("the default schedule");
    let profiled = schedule_for(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
        profile.map(MatMulLimits::of),
    )
    .expect("the profile's schedule");
    (default, profiled)
}

/// The rule of `MatMulTile::of` at the base commit (6203b9b,
/// `crates/modelspec-core/src/ir/kernel.rs`), verbatim but for the names: the wide and square
/// register-blocked tiles, then one or two rows of one column per invocation.
fn base_of(m: u64, k: u64, n: u64) -> MatMulTile {
    let fitted = |mut tile: MatMulTile| {
        tile.depth = k.clamp(1, u64::from(tile.depth)) as u32;
        loop {
            tile.buffers = if u64::from(tile.depth) < k { 2 } else { 1 };
            if tile.depth == 1 || tile.shared_bytes() <= MIN_SHARED_MEMORY_BYTES {
                return tile;
            }
            tile.depth /= 2;
        }
    };
    // (workgroup_size, cols, rows, cols_per_invocation, depth, from_k, from_workgroups)
    for (workgroup_size, cols, rows, cols_per_invocation, depth, from_k, from_workgroups) in
        [(128, 128, 8, 8, 8, 64, 128), (128, 64, 8, 4, 16, 1, 128)]
    {
        let tile = MatMulTile {
            workgroup_size,
            cols,
            rows,
            cols_per_invocation,
            depth,
            ahead: 1,
            buffers: 1,
            b_pad: 0,
        };
        if n >= u64::from(cols) && k >= from_k && tile.workgroups(m, n) >= from_workgroups {
            return fitted(tile);
        }
    }
    let workgroup_size = 128;
    let cols = n.clamp(1, 16).next_power_of_two() as u32;
    let one_row = m.div_ceil(u64::from(workgroup_size / cols)) * n.div_ceil(u64::from(cols));
    let (rows, (depth, ahead)) = if one_row >= 256 {
        (2, (16, 2))
    } else {
        (1, (64, 6))
    };
    fitted(MatMulTile {
        workgroup_size,
        cols,
        rows,
        cols_per_invocation: 1,
        depth,
        ahead,
        buffers: 1,
        b_pad: 4,
    })
}

/// Whether `[m, k] x [k, n]` takes the 128 by 128 block under a limit of at least 256.
fn takes_block(m: u64, k: u64, n: u64) -> bool {
    n >= 128 && k >= 64 && m.div_ceil(128) * n.div_ceil(128) >= 128
}

/// The 128 by 128 block of `[m, k] x [k, n]`.
fn block_of(k: u64) -> MatMulTile {
    MatMulTile {
        workgroup_size: 256,
        cols: 128,
        rows: 8,
        cols_per_invocation: 8,
        depth: 8,
        ahead: 1,
        buffers: if k > 8 { 2 } else { 1 },
        b_pad: 0,
    }
}

/// `count` shapes drawn from `1..=10_000` by SplitMix64, then the boundaries of every rule.
fn shapes(count: usize) -> Vec<(u64, u64, u64)> {
    let mut state = 0x6d61_746d_756c_3235u64;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        1 + z % 10_000
    };
    let mut shapes: Vec<(u64, u64, u64)> = (0..count).map(|_| (next(), next(), next())).collect();
    let edges = [
        1, 2, 3, 7, 8, 9, 10, 15, 16, 17, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 784, 1000,
        1023, 1024, 1025, 1920, 2047, 2048, 2049, 4096, 8191, 8192, 8193, 10_000,
    ];
    for &m in &edges {
        for &k in &edges {
            for &n in &edges {
                shapes.push((m, k, n));
            }
        }
    }
    shapes
}

#[test]
fn with_a_limit_of_128_every_shape_takes_the_tile_of_the_rule_before_the_change() {
    let mut wrong = Vec::new();
    for (m, k, n) in shapes(200_000) {
        let before = base_of(m, k, n);
        let at_128 = MatMulTile::of_invocations(m, k, n, 128);
        let of = MatMulTile::of(m, k, n);
        let profiled = MatMulTile::for_profile(m, k, n, [false, false], &limited(128, 49_152));
        if at_128 != before || of != before || profiled != before {
            wrong.push(format!(
                "[{m}, {k}] x [{k}, {n}]: before {before}, limit 128 {at_128}, of {of}, \
                 128-invocation profile {profiled}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} shapes, first: {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(5)]
    );
}

#[test]
fn with_a_limit_of_1024_an_output_of_128_blocks_of_128_by_128_takes_the_256_invocation_block() {
    assert_eq!(MATMUL_LARGE_BLOCK.workgroup_size, 256);
    let mut wrong = Vec::new();
    let mut taken = 0;
    for (m, k, n) in shapes(200_000) {
        let tile = MatMulTile::of_invocations(m, k, n, 1024);
        let expected = if takes_block(m, k, n) {
            taken += 1;
            block_of(k)
        } else {
            base_of(m, k, n)
        };
        if tile != expected {
            wrong.push(format!(
                "[{m}, {k}] x [{k}, {n}]: {tile}, expected {expected}"
            ));
        }
        if MatMulTile::of_invocations(m, k, n, 255) != base_of(m, k, n) {
            wrong.push(format!(
                "[{m}, {k}] x [{k}, {n}]: a limit of 255 changed the tile"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} shapes, first: {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(5)]
    );
    assert!(taken > 1000, "{taken} sampled shapes take the block");

    let block = MatMulTile::of_invocations(8192, 784, 4096, 1024);
    assert_eq!(
        block.to_string(),
        "tile workgroup_size 256 cols 128 rows 8 cols_per_invocation 8 depth 8 ahead 1 \
         buffers 2 b_pad 0"
    );
    assert_eq!(block.rows_per_workgroup(), 128);
    assert_eq!(block.workgroups(8192, 4096), 64 * 32);
    assert_eq!(block.shared_bytes(), MIN_SHARED_MEMORY_BYTES);
    // 128 blocks of 128 by 128 exactly, and one block fewer.
    assert_eq!(
        MatMulTile::of_invocations(1024, 64, 2048, 1024),
        block_of(64)
    );
    assert_eq!(
        MatMulTile::of_invocations(1024, 64, 1920, 1024),
        base_of(1024, 64, 1920)
    );
    assert_eq!(
        MatMulTile::of_invocations(8192, 63, 4096, 1024),
        base_of(8192, 63, 4096)
    );
}

#[test]
fn a_profile_of_1024_invocations_schedules_the_block_and_its_signature_and_module_id_show_it() {
    let profile = rtx3090();
    assert_eq!(profile.max_workgroup_invocations, 1024);
    let (default, profiled) = plans("sweep_b8192_h4096", &sweep(8192, 4096), Some(&profile));
    let matmuls = |plan: &ExecutionPlan| -> Vec<Kernel> {
        plan.kernels
            .iter()
            .filter(|k| k.kind == KernelKind::MatMul)
            .cloned()
            .collect()
    };
    let (before, after) = (matmuls(&default), matmuls(&profiled));
    assert_eq!(before.len(), after.len());
    let mut blocks = 0;
    for (before, after) in before.iter().zip(&after) {
        let (m, k, n) = Kernel::matmul_shape(&after.operands).expect("a static matmul");
        let tile = after.matmul_tile().expect("a matmul has a tile");
        assert_eq!(after.id, Kernel::module_id(&after.signature()));
        assert_eq!(after.workgroup_size, tile.workgroup_size);
        assert_eq!(
            u64::from(after.workgroups.iter().product::<u32>()),
            tile.workgroups(m, n)
        );
        assert!(after.signature().ends_with(&format!(" {tile}")));
        if takes_block(m, k, n) {
            blocks += 1;
            assert_eq!(tile, block_of(k), "[{m}, {k}] x [{k}, {n}]");
            assert!(
                after
                    .signature()
                    .contains(" workgroup_size 256 workgroups "),
                "{}",
                after.signature()
            );
            assert_ne!(after.id, before.id, "the block has its own module id");
        } else {
            assert_eq!(after, before, "[{m}, {k}] x [{k}, {n}] keeps its kernel");
        }
    }
    // The forward matmul [8192, 784] x [784, 4096] and the weight gradient
    // [784, 8192] x [8192, 4096].
    assert_eq!(blocks, 2);
}

#[test]
fn the_plan_verifier_caps_workgroups_at_the_profile_limit_and_at_128_without_a_profile() {
    let profile = rtx3090();
    let (default, profiled) = plans("sweep_b8192_h4096", &sweep(8192, 4096), Some(&profile));
    assert_eq!(MAX_WORKGROUP_INVOCATIONS, 128);
    verify_plan(&default).expect("the default plan verifies without a profile");
    verify_plan_for(&default, None).expect("and with none");
    verify_plan_for(&profiled, Some(&profile)).expect("256 verifies under 1024");
    verify_plan_for(&profiled, Some(&limited(256, 49_152))).expect("and under 256");
    for (error, max) in [
        (verify_plan(&profiled), 128),
        (verify_plan_for(&profiled, None), 128),
        (verify_plan_for(&profiled, Some(&limited(255, 49_152))), 255),
        (verify_plan_for(&profiled, Some(&limited(128, 49_152))), 128),
    ] {
        match error {
            Err(PlanVerifyError::WorkgroupTooLarge {
                size: 256,
                max: found,
                ..
            }) => assert_eq!(found, max),
            other => panic!("not refused at {max}: {other:?}"),
        }
    }
    let message = verify_plan(&profiled).expect_err("refused").to_string();
    assert!(
        message.ends_with("256 invocations per workgroup, outside 1..=128"),
        "{message}"
    );

    // `LocalSize` x is bounded by `maxComputeWorkGroupSize[0]` as well, whose Vulkan minimum is
    // also 128: a profile that allows 1024 invocations but 128 along x schedules no block.
    let mut narrow = rtx3090();
    narrow.max_workgroup_size.x = 128;
    assert!(matches!(
        verify_plan_for(&profiled, Some(&narrow)),
        Err(PlanVerifyError::WorkgroupTooLarge {
            size: 256,
            max: 128,
            ..
        })
    ));
    let (default, narrowed) = plans("sweep_b8192_h4096", &sweep(8192, 4096), Some(&narrow));
    assert_eq!(narrowed, default);
    assert_eq!(
        MatMulTile::for_profile(8192, 784, 4096, [false, false], &narrow),
        MatMulTile::of(8192, 784, 4096)
    );
}

#[test]
fn every_tile_of_a_profile_plan_fits_its_workgroup_memory_and_a_smaller_one_takes_the_next_tile() {
    for (name, source) in sources() {
        let (_, profiled) = plans(&name, &source, Some(&rtx3090()));
        for kernel in profiled
            .kernels
            .iter()
            .filter(|k| k.kind == KernelKind::MatMul)
        {
            let tile = kernel.matmul_tile().expect("a matmul has a tile");
            assert!(
                tile.layout(kernel.transposed).shared_bytes() <= 49_152,
                "{name}: {}",
                kernel.signature()
            );
        }
    }
    let (m, k, n) = (8192, 784, 4096);
    for transposed in [[false, false], [true, false], [false, true]] {
        let block = MatMulTile::of_invocations(m, k, n, 1024);
        let needs = block.layout(transposed).shared_bytes();
        assert_eq!(
            MatMulTile::for_profile(m, k, n, transposed, &limited(1024, needs)),
            block,
            "{transposed:?}"
        );
        assert_eq!(
            MatMulTile::for_profile(m, k, n, transposed, &limited(1024, needs - 1)),
            MatMulTile::of(m, k, n),
            "{transposed:?}: a profile one byte short takes the tile of 128 invocations"
        );
    }
    // A profile with too little workgroup memory for the block schedules, with no refusal, and
    // each of its matmuls keeps the tile of the Vulkan minimum.
    let short = limited(1024, MATMUL_LARGE_BLOCK_BYTES - 1);
    let (default, profiled) = plans("sweep_b8192_h4096", &sweep(8192, 4096), Some(&short));
    verify_plan_for(&profiled, Some(&short)).expect("the plan verifies");
    assert_eq!(profiled, default);
}

/// The workgroup memory of the 128 by 128 block reading no operand transposed.
const MATMUL_LARGE_BLOCK_BYTES: u32 = MIN_SHARED_MEMORY_BYTES;

#[test]
fn without_a_profile_or_with_128_invocations_every_plan_is_the_plan_before_the_change() {
    for (name, source) in sources() {
        let (default, none) = plans(&name, &source, None);
        assert_eq!(none, default, "{name}: no profile");
        for shared in [MIN_SHARED_MEMORY_BYTES, 49_152] {
            let (_, minimum) = plans(&name, &source, Some(&limited(128, shared)));
            assert_eq!(minimum, default, "{name}: 128 invocations, {shared} bytes");
            verify_plan(&minimum).expect("verifies without a profile");
        }
    }
}
