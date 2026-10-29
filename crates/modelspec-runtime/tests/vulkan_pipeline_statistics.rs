//! `story:matmul-block-kernel-throughput`, acceptance `pipeline-statistics`: a profiled run
//! prints, for every kernel of its plan, the statistics `VK_KHR_pipeline_executable_properties`
//! reports for each executable of the kernel's pipeline (name, description and value, as the
//! driver reports them), and `profile pipeline statistics: not reported` on a device opened
//! without the extension.
//!
//! The first tests need no device: they check the lines [`Profile::report`] prints. The others
//! are ignored by default because they create a Vulkan instance, which `task ci` never does; run
//! them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_pipeline_statistics -- --ignored`;
//! every message of the Khronos validation layer fails them. They need a device that offers the
//! extension, as the RTX 3090 does (driver 610.57.04).
#![cfg(feature = "vulkan")]

use std::collections::BTreeSet;
use std::time::Duration;

use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_core::ir::kernel::{KernelKind, KernelModuleId};
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::vulkan::negotiate::{PIPELINE_EXECUTABLE_PROPERTIES, negotiate};
use modelspec_runtime::vulkan::pipeline::{
    KernelStatistics, PipelineExecutable, PipelineStatistic, PipelineStatistics, StatisticValue,
};
use modelspec_runtime::vulkan::profile::Profile;
use modelspec_runtime::vulkan::runner::{RunStats, train_profiled};
use modelspec_runtime::vulkan::{VALIDATION_ENV, Vulkan};

const LINEAR_REGRESSION: &str = "examples/linear_regression.mspec";
const MNIST_MLP: &str = "examples/mnist_mlp.mspec";

fn stats() -> RunStats {
    RunStats {
        steps: 3,
        host_waits: 1,
        submits: 1,
        dispatches: 7,
        dispatches_per_step: 2,
        barriers_per_step: 4,
        bytes_uploaded: 40,
        bytes_downloaded: 28,
        device_allocations: 5,
        peak_device_bytes: 1024,
        class: modelspec_runtime::vulkan::negotiate::NumericsClass::Tolerance,
        elapsed: Duration::from_micros(30),
        pipelines: 6,
        pipeline_creation: Duration::from_micros(1500),
    }
}

fn source(path: &str) -> String {
    let full = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path);
    std::fs::read_to_string(full).expect("the example reads")
}

/// The statistic of `executable` whose name contains `word`, ignoring case.
fn named<'a>(executable: &'a PipelineExecutable, word: &str) -> Option<&'a PipelineStatistic> {
    executable
        .statistics
        .iter()
        .find(|s| s.name.to_lowercase().contains(word))
}

fn positive(value: &StatisticValue) -> bool {
    match *value {
        StatisticValue::Int(v) => v > 0,
        StatisticValue::Uint(v) => v > 0,
        StatisticValue::Float(v) => v > 0.0,
        StatisticValue::Bool(_) => false,
    }
}

/// One line per executable and one per statistic, each naming its kernel, after the timing
/// lines and before the counters; the values print as the driver's format holds them.
#[test]
fn the_report_prints_every_statistic_of_every_kernel_before_the_counters() {
    let id = KernelModuleId([7; 16]);
    let profile = Profile {
        timings: None,
        pipeline_creation: Duration::from_micros(1),
        compile: Duration::from_micros(2),
        pipeline_statistics: PipelineStatistics::Reported(vec![KernelStatistics {
            kernel_module_id: id,
            executables: vec![PipelineExecutable {
                name: "Compute shader".to_owned(),
                description: "The compute stage".to_owned(),
                subgroup_size: 32,
                statistics: vec![
                    PipelineStatistic {
                        name: "Register Count".to_owned(),
                        description: "Registers per thread".to_owned(),
                        value: StatisticValue::Uint(168),
                    },
                    PipelineStatistic {
                        name: "Spills".to_owned(),
                        description: "Whether it spills".to_owned(),
                        value: StatisticValue::Bool(false),
                    },
                    PipelineStatistic {
                        name: "Delta".to_owned(),
                        description: "signed".to_owned(),
                        value: StatisticValue::Int(-3),
                    },
                    PipelineStatistic {
                        name: "Ratio".to_owned(),
                        description: "a float".to_owned(),
                        value: StatisticValue::Float(0.5),
                    },
                ],
            }],
        }]),
    };
    let report = profile.report(&stats());
    assert_eq!(
        report
            .lines()
            .take(6)
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        vec![
            "profile kernel timings unavailable: no timestamp queries".to_owned(),
            format!(
                "profile pipeline {id} executable 0 name \"Compute shader\" subgroup_size 32 \
                 description \"The compute stage\""
            ),
            format!(
                "profile pipeline {id} executable 0 statistic \"Register Count\" 168 \
                 description \"Registers per thread\""
            ),
            format!(
                "profile pipeline {id} executable 0 statistic \"Spills\" false \
                 description \"Whether it spills\""
            ),
            format!(
                "profile pipeline {id} executable 0 statistic \"Delta\" -3 description \"signed\""
            ),
            format!(
                "profile pipeline {id} executable 0 statistic \"Ratio\" 0.5 description \"a float\""
            ),
        ]
    );
    assert_eq!(
        report.lines().nth(6),
        Some("profile dispatches 7"),
        "{report}"
    );
}

#[test]
fn without_the_extension_the_report_says_pipeline_statistics_are_not_reported() {
    let profile = Profile {
        timings: None,
        pipeline_creation: Duration::from_micros(1),
        compile: Duration::from_micros(2),
        pipeline_statistics: PipelineStatistics::NotCaptured,
    };
    let report = profile.report(&stats());
    assert_eq!(
        report.lines().nth(1),
        Some("profile pipeline statistics: not reported"),
        "{report}"
    );
}

/// A statistics query that fails does not end the profiled run (adversary pass 1, the note on
/// `runner.rs`): [`PipelineStatistics::collect`], through which the runner reads the pipeline of
/// every kernel, keeps the first error as the reason and reads no further pipeline, and the
/// report prints `profile pipeline statistics: not reported (<reason>)` once, no statistic line,
/// and the run's counters after it. The error is injected here: no device at hand fails the
/// query, so the runner's own call on a device is not exercised by this test.
#[test]
fn a_failing_statistics_query_degrades_to_one_not_reported_line() {
    let executable = PipelineExecutable {
        name: "Compute shader".to_owned(),
        description: "The compute stage".to_owned(),
        subgroup_size: 32,
        statistics: Vec::new(),
    };
    let reason = "vkGetPipelineExecutableStatisticsKHR failed: ERROR_OUT_OF_HOST_MEMORY";
    let mut read = Vec::new();
    let statistics = PipelineStatistics::collect(
        [
            (KernelModuleId([7; 16]), Ok(Some(vec![executable.clone()]))),
            (KernelModuleId([8; 16]), Err(reason)),
            (KernelModuleId([9; 16]), Ok(Some(vec![executable.clone()]))),
        ]
        .into_iter()
        .inspect(|(id, _)| read.push(*id)),
    );
    assert_eq!(statistics, PipelineStatistics::Failed(reason.to_owned()));
    assert_eq!(
        read,
        [KernelModuleId([7; 16]), KernelModuleId([8; 16])],
        "no pipeline is read after the first error"
    );
    let profile = Profile {
        timings: None,
        pipeline_creation: Duration::from_micros(1),
        compile: Duration::from_micros(2),
        pipeline_statistics: statistics,
    };
    let report = profile.report(&stats());
    let pipeline: Vec<&str> = report
        .lines()
        .filter(|l| l.starts_with("profile pipeline "))
        .collect();
    assert_eq!(
        pipeline,
        [format!(
            "profile pipeline statistics: not reported ({reason})"
        )],
        "{report}"
    );
    assert_eq!(
        report.lines().nth(2),
        Some("profile dispatches 7"),
        "{report}"
    );

    let every = PipelineStatistics::collect([
        (
            KernelModuleId([7; 16]),
            Ok::<_, &str>(Some(vec![executable.clone()])),
        ),
        (KernelModuleId([8; 16]), Ok(None)),
    ]);
    assert_eq!(
        every.reported().map(|kernels| kernels
            .iter()
            .map(|k| (k.kernel_module_id, k.executables.len()))
            .collect::<Vec<_>>()),
        Some(vec![
            (KernelModuleId([7; 16]), 1),
            (KernelModuleId([8; 16]), 0)
        ])
    );
}

/// The pipeline of the forward matmul of the MNIST example, created on a device opened with the
/// extension, reports a positive register count, and its binary size; on a device opened without
/// it, no statistics.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_matmul_pipeline_reports_a_positive_register_count() {
    let artifact = compile_artifact(MNIST_MLP, &source(MNIST_MLP), Backend::Vulkan)
        .expect("the example compiles for Vulkan");
    let kernel = artifact
        .execution
        .kernels
        .iter()
        .find(|k| k.kind == KernelKind::MatMul)
        .expect("the plan has a matmul");

    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let device_profile = &candidates[0].profile;
    assert!(
        vulkan
            .offers_pipeline_statistics(0)
            .expect("the extensions are listed"),
        "device 0 offers {PIPELINE_EXECUTABLE_PROPERTIES} and its pipelineExecutableInfo feature"
    );
    let module = emit_module(kernel, device_profile).expect("the matmul emits");

    let plain = negotiate(device_profile).expect("device 0 negotiates");
    assert!(plain.extensions.is_empty(), "{plain:?}");
    let device = vulkan.create_device(0, &plain).expect("device 0 opens");
    assert!(!device.pipeline_statistics_enabled());
    let pipeline = device
        .kernel_pipeline(kernel, &module.spirv)
        .expect("the pipeline is created");
    assert_eq!(pipeline.statistics().expect("no query is made"), None);
    drop(pipeline);
    drop(device);

    let negotiated = plain.with_pipeline_statistics();
    assert_eq!(negotiated.extensions, [PIPELINE_EXECUTABLE_PROPERTIES]);
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    assert!(device.pipeline_statistics_enabled());
    assert_eq!(
        device.enabled_extensions(),
        [PIPELINE_EXECUTABLE_PROPERTIES]
    );
    let pipeline = device
        .kernel_pipeline(kernel, &module.spirv)
        .expect("the pipeline is created");
    let executables = pipeline
        .statistics()
        .expect("the statistics are read")
        .expect("the device reports statistics");
    drop(pipeline);
    drop(device);
    let messages = vulkan.take_validation_messages();
    assert!(
        messages.is_empty(),
        "{} validation messages: {messages:#?}",
        messages.len()
    );
    eprintln!("{executables:#?}");
    assert_eq!(executables.len(), 1, "one compute executable");
    let executable = &executables[0];
    assert_eq!(executable.subgroup_size, 32);
    let registers = named(executable, "register")
        .unwrap_or_else(|| panic!("no register count among {:#?}", executable.statistics));
    assert!(positive(&registers.value), "{registers:?}");
    let binary = named(executable, "binary")
        .unwrap_or_else(|| panic!("no binary size among {:#?}", executable.statistics));
    assert!(positive(&binary.value), "{binary:?}");
    assert!(
        named(executable, "local memory").is_some(),
        "no local memory size among {:#?}",
        executable.statistics
    );
}

/// A profiled run on a device opened with the extension carries the statistics of every kernel
/// of its plan, in plan order, and its report prints them; opened without it, the report says
/// they are not reported.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_profiled_run_reports_the_statistics_of_every_kernel_of_its_plan() {
    let text = source(LINEAR_REGRESSION);
    let compiled = compile_training(LINEAR_REGRESSION, &text).expect("the example compiles");
    let artifact = compile_artifact(LINEAR_REGRESSION, &text, Backend::Vulkan)
        .expect("the example compiles for Vulkan");
    let kernels: Vec<KernelModuleId> = artifact.execution.kernels.iter().map(|k| k.id).collect();

    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
    );
    let candidates = vulkan.candidates().expect("the devices are probed");
    let device_profile = &candidates[0].profile;
    let plain = negotiate(device_profile).expect("device 0 negotiates");

    let device = vulkan.create_device(0, &plain).expect("device 0 opens");
    let (_, stats, profile) = train_profiled(&device, device_profile, &plain, &compiled, &artifact)
        .expect("the profiled run finishes");
    drop(device);
    assert_eq!(profile.pipeline_statistics, PipelineStatistics::NotCaptured);
    assert!(
        profile
            .report(&stats)
            .lines()
            .any(|l| l == "profile pipeline statistics: not reported")
    );

    let negotiated = plain.with_pipeline_statistics();
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let (_, stats, profile) =
        train_profiled(&device, device_profile, &negotiated, &compiled, &artifact)
            .expect("the profiled run finishes");
    drop(device);
    let messages = vulkan.take_validation_messages();
    assert!(
        messages.is_empty(),
        "{} validation messages: {messages:#?}",
        messages.len()
    );
    let reported = profile
        .pipeline_statistics
        .reported()
        .expect("the device reports statistics");
    assert_eq!(
        reported
            .iter()
            .map(|k| k.kernel_module_id)
            .collect::<Vec<_>>(),
        kernels,
        "one entry per kernel of the plan, in plan order"
    );
    for kernel in reported {
        assert!(!kernel.executables.is_empty(), "{kernel:?}");
        for executable in &kernel.executables {
            let registers = named(executable, "register")
                .unwrap_or_else(|| panic!("no register count in {kernel:?}"));
            assert!(positive(&registers.value), "{kernel:?}");
        }
    }
    let report = profile.report(&stats);
    eprintln!("{report}");
    let printed: BTreeSet<String> = report
        .lines()
        .filter_map(|l| l.strip_prefix("profile pipeline "))
        .filter(|l| l.contains(" statistic \""))
        .map(|l| l.split(' ').next().unwrap_or_default().to_owned())
        .collect();
    let expected: BTreeSet<String> = kernels.iter().map(ToString::to_string).collect();
    assert_eq!(printed, expected, "{report}");
    assert!(!report.contains("not reported"), "{report}");
}
