//! `modelspec compile <file> --backend cpu|vulkan --output <dir> [--device-profile <json>]
//! [--fast-math]`:
//! compile a `.mspec` file into an artifact directory (DESIGN.md § 7.1, § 8), its
//! `artifact.json` and, for Vulkan, one `kernels/<kernel_module_id>.spv` per kernel.
//!
//! A Vulkan compile builds its modules for one device: the `DeviceProfile` JSON file named by
//! `--device-profile`, or without it device 0 of the Vulkan loader. With `--device-profile` the
//! plan is also scheduled within that profile's workgroup invocations and workgroup memory
//! (`modelspec_core::compile::compile_artifact_for`), the manifest records them as
//! `matmul_limits`, and the artifact is read back for that profile
//! (`modelspec_core::artifact::read_for_device`); without it the plan is the one for every
//! Vulkan device and the manifest has no `matmul_limits`. The profile is negotiated
//! before anything is written, so a device the runtime would refuse (E-TARGET-004) leaves the
//! output directory as it was. A profile file that cannot be read or breaks an invariant of
//! `spec/domains/device.yaml` exits 1 naming the file and the invariant. `--fast-math` sets the
//! artifact's `fast_math` (DESIGN.md § 12 ADR-008), recorded in the manifest, in every kernel
//! cache key and so in the artifact hash, and its modules carry no `NoContraction`.
//! `--device-profile` or `--fast-math` with `--backend cpu` is a usage error; a Cpu compile opens
//! no Vulkan library.
//!
//! What is written is read back through `modelspec_core::artifact::read` before the process
//! exits 0 and prints `artifact <artifact_id> <artifact_hash> kernels=<n> <dir>`. Two compiles of
//! one file on one profile write byte-equal trees.

use std::io::Write;
use std::path::PathBuf;

use modelspec_core::artifact::{self, Backend as Target};
use modelspec_core::compile;

/// Arguments of `modelspec compile`.
#[derive(clap::Args)]
pub struct Args {
    /// The .mspec file to compile.
    pub path: PathBuf,
    /// The backend the artifact is for.
    #[arg(long, value_enum)]
    pub backend: Backend,
    /// The artifact directory to write, created with its parents; files of an earlier artifact
    /// in it are overwritten.
    #[arg(long, value_name = "DIR")]
    pub output: PathBuf,
    /// With --backend vulkan, schedule the plan within the limits of the DeviceProfile JSON file
    /// PATH (the form `inspect-device --format json` prints: its workgroup invocations and
    /// workgroup memory, recorded in the manifest as matmul_limits) and build the kernel modules
    /// for it, instead of scheduling for every device and probing device 0.
    #[arg(long, value_name = "PATH")]
    pub device_profile: Option<PathBuf>,
    /// With --backend vulkan, compile with fast_math (DESIGN.md ADR-008): the manifest and every
    /// kernel cache key record it, and the float multiplies, adds and subtracts of the matmul,
    /// reduction, loss-sum and SGD kernels carry no NoContraction, so a driver may contract them
    /// into fused multiply-adds.
    #[arg(long)]
    pub fast_math: bool,
}

/// A backend `modelspec compile` can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    /// The CPU interpreter: an artifact without kernel modules.
    Cpu,
    /// Vulkan compute: an artifact with one SPIR-V module per kernel, in a build with the vulkan
    /// feature.
    Vulkan,
}

/// Runs `modelspec compile` and returns the process exit code.
pub fn run(args: Args, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    if args.backend == Backend::Cpu {
        let misplaced = [
            ("--device-profile", args.device_profile.is_some()),
            ("--fast-math", args.fast_math),
        ]
        .into_iter()
        .find_map(|(flag, given)| given.then_some(flag));
        if let Some(flag) = misplaced {
            let _ = writeln!(
                stderr,
                "error: the argument '{flag}' cannot be used with '--backend cpu'; it is \
                 available only with '--backend vulkan'"
            );
            return crate::EXIT_USAGE;
        }
    }
    #[cfg(not(feature = "vulkan"))]
    if args.backend == Backend::Vulkan {
        let _ = writeln!(
            stderr,
            "{}",
            modelspec_core::diag::Diagnostic::new(
                crate::train::E_TARGET_001,
                "the vulkan backend is not available in this build"
            )
            .with("backend", "vulkan")
            .with("available", "cpu")
        );
        return crate::EXIT_FAILURE;
    }
    let file = args.path.display().to_string();
    let source = match std::fs::read_to_string(&args.path) {
        Ok(source) => source,
        Err(error) => {
            let _ = writeln!(stderr, "modelspec compile {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    let target = match args.backend {
        Backend::Cpu => Target::Cpu,
        Backend::Vulkan => Target::Vulkan,
    };
    // The profile of `--device-profile`, read before the plan is scheduled within its limits.
    #[cfg(feature = "vulkan")]
    let scheduled_for = match args.device_profile.as_deref() {
        Some(path) if args.backend == Backend::Vulkan => match vulkan::profile(Some(path)) {
            Ok(profile) => Some(profile),
            Err(message) => {
                let _ = writeln!(stderr, "{message}");
                return crate::EXIT_FAILURE;
            }
        },
        _ => None,
    };
    #[cfg(not(feature = "vulkan"))]
    let scheduled_for: Option<modelspec_core::device::DeviceProfile> = None;
    let compiled =
        match compile::compile_artifact_for(&file, &source, target, scheduled_for.as_ref()) {
            Ok(mut compiled) => {
                compiled.fast_math = args.fast_math;
                compiled
            }
            Err(diagnostics) => {
                for diagnostic in &diagnostics {
                    let _ = writeln!(stderr, "{diagnostic}");
                }
                return crate::EXIT_FAILURE;
            }
        };
    let modules = match args.backend {
        Backend::Cpu => Vec::new(),
        #[cfg(not(feature = "vulkan"))]
        Backend::Vulkan => unreachable!("refused above in a build without the vulkan feature"),
        #[cfg(feature = "vulkan")]
        Backend::Vulkan => match vulkan::modules(&compiled, scheduled_for.as_ref()) {
            Ok(modules) => modules,
            Err(message) => {
                let _ = writeln!(stderr, "{message}");
                return crate::EXIT_FAILURE;
            }
        },
    };
    let out = args.output.display().to_string();
    if let Err(error) = artifact::write(&args.output, &compiled, &modules) {
        let _ = writeln!(stderr, "modelspec compile {file}: {error}");
        return crate::EXIT_FAILURE;
    }
    let read = match &scheduled_for {
        Some(profile) => artifact::read_for_device(&args.output, profile),
        None => artifact::read(&args.output),
    };
    match read {
        Ok(read) => {
            let _ = writeln!(
                stdout,
                "artifact {} {} kernels={} {out}",
                read.artifact_id.hyphenated(),
                read.artifact_hash.0,
                read.modules.len()
            );
            crate::EXIT_SUCCESS
        }
        Err(diagnostic) => {
            let _ = writeln!(stderr, "{diagnostic}");
            crate::EXIT_FAILURE
        }
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use std::path::Path;

    use modelspec_core::artifact::{CompiledArtifact, KernelCacheKey, KernelModuleFile};
    use modelspec_core::device::DeviceProfile;
    use modelspec_runtime::spirv::emit::emit_kernels;
    use modelspec_runtime::vulkan::Vulkan;
    use modelspec_runtime::vulkan::negotiate::negotiate;
    use modelspec_runtime::vulkan::select::select_device;

    /// The device profile of `path`, or of device 0 of the Vulkan loader; what fails is the
    /// line to print.
    pub(super) fn profile(path: Option<&Path>) -> Result<DeviceProfile, String> {
        match path {
            Some(path) => {
                let shown = path.display();
                let json = std::fs::read_to_string(path)
                    .map_err(|error| format!("modelspec compile: {shown}: {error}"))?;
                DeviceProfile::from_json(&json)
                    .map_err(|error| format!("modelspec compile: {shown}: {error}"))
            }
            None => {
                let vulkan = Vulkan::load().map_err(|d| d.to_string())?;
                let candidates = vulkan.candidates().map_err(|d| d.to_string())?;
                let index = select_device(&candidates, Some(0)).map_err(|d| d.to_string())?[0];
                Ok(candidates[index].profile.clone())
            }
        }
    }

    /// The kernel modules of `compiled` for the negotiated device profile: `scheduled_for`, the
    /// profile of `--device-profile`, or else that of device 0.
    pub(super) fn modules(
        compiled: &CompiledArtifact,
        scheduled_for: Option<&DeviceProfile>,
    ) -> Result<Vec<KernelModuleFile>, String> {
        let profile = match scheduled_for {
            Some(profile) => profile.clone(),
            None => profile(None)?,
        };
        negotiate(&profile).map_err(|d| d.to_string())?;
        let modules = emit_kernels(compiled, &profile).map_err(|error| {
            format!("modelspec compile: the kernel modules cannot be built: {error}")
        })?;
        Ok(modules
            .into_iter()
            .map(|module| {
                let spirv = module.bytes();
                let key = module.cache_key;
                KernelModuleFile {
                    kernel_module_id: module.kernel_module_id,
                    entry_point: module.entry_point,
                    cache_key: KernelCacheKey {
                        ir_hash: key.ir_hash.0,
                        shapes: key.shapes,
                        dtype: key.dtype,
                        fast_math: key.fast_math,
                        deterministic: key.deterministic,
                        vendor_id: key.vendor_id,
                        device_id: key.device_id,
                        driver_version: key.driver_version,
                        capabilities: key.capabilities,
                    },
                    spirv,
                }
            })
            .collect())
    }
}
