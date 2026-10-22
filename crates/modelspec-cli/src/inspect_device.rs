//! `modelspec inspect-device --backend vulkan [--device <index>] [--format text|json]`: probe
//! the Vulkan physical devices and print each one's `DeviceProfile` (`spec/domains/device.yaml`)
//! and its negotiation verdict (DESIGN.md § 5.1, § 6 "Device", § 8).
//!
//! Every value is the driver's own, never inferred from a product name. `--device <index>` picks
//! one device in the loader's enumeration order; without it every device is listed.
//!
//! - `--format text` prints, per device, one `<field>: <value>` line for each of the 24 fields
//!   besides the identity `device_profile_id`, each value as compact JSON, then `negotiation:
//!   ok` or `negotiation: refused (<code>)`. Without `--device` each device starts with
//!   `device: <index>` and devices are separated by a blank line.
//! - `--format json` prints the profile's JSON form (`modelspec_core::device::DeviceProfile`),
//!   or without `--device` a JSON array of every profile.
//!
//! A device that negotiation refuses is still printed; its diagnostic goes to stderr and the
//! process exits 1. A missing loader exits 1 with E-TARGET-002, no physical device or a
//! `--device` that names none, or names one without a compute queue family, exits 1 with
//! E-TARGET-003, and a build without the `vulkan` feature refuses the verb with E-TARGET-001.

use std::io::Write;

/// Arguments of `modelspec inspect-device`.
#[derive(clap::Args)]
pub struct Args {
    /// The backend whose devices are inspected.
    #[arg(long, value_enum)]
    pub backend: Backend,
    /// The index of one physical device, in the order the Vulkan loader lists them; without it
    /// every device is printed.
    #[arg(long, value_name = "INDEX")]
    pub device: Option<usize>,
    /// The output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    pub format: Format,
}

/// A backend `modelspec inspect-device` can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    /// Vulkan compute.
    Vulkan,
}

/// How `modelspec inspect-device` prints a profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    /// One `<field>: <value>` line per field, then the negotiation verdict.
    Text,
    /// The profile's JSON form, or a JSON array of profiles.
    Json,
}

/// Runs `modelspec inspect-device` and returns the process exit code.
#[cfg(not(feature = "vulkan"))]
pub fn run(args: Args, _stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    use modelspec_core::diag::Diagnostic;
    let Backend::Vulkan = args.backend;
    let _ = writeln!(
        stderr,
        "{}",
        Diagnostic::new(
            crate::train::E_TARGET_001,
            "the vulkan backend is not available in this build"
        )
        .with("backend", "vulkan")
    );
    crate::EXIT_FAILURE
}

/// Runs `modelspec inspect-device` and returns the process exit code.
#[cfg(feature = "vulkan")]
pub fn run(args: Args, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    use modelspec_runtime::vulkan::Vulkan;
    use modelspec_runtime::vulkan::negotiate::negotiate;
    use modelspec_runtime::vulkan::select::select_device;

    let Backend::Vulkan = args.backend;
    let probed = Vulkan::load().and_then(|vulkan| {
        let candidates = vulkan.candidates()?;
        let selected = select_device(&candidates, args.device)?;
        Ok((candidates, selected))
    });
    let (candidates, selected) = match probed {
        Ok(probed) => probed,
        Err(diagnostic) => {
            let _ = writeln!(stderr, "{diagnostic}");
            return crate::EXIT_FAILURE;
        }
    };
    let verdicts: Vec<_> = selected
        .iter()
        .map(|&index| {
            select_device(&candidates, Some(index)).and_then(|_| {
                negotiate(&candidates[index].profile)
                    .map_err(|refusal| refusal.with("device", index.to_string()))
            })
        })
        .collect();
    let mut out = String::new();
    match args.format {
        Format::Text => {
            for (position, (&index, verdict)) in selected.iter().zip(&verdicts).enumerate() {
                if args.device.is_none() {
                    if position > 0 {
                        out.push('\n');
                    }
                    out.push_str(&format!("device: {index}\n"));
                }
                for (name, value) in candidates[index].profile.fields() {
                    out.push_str(&format!("{name}: {value}\n"));
                }
                match verdict {
                    Ok(_) => out.push_str("negotiation: ok\n"),
                    Err(refusal) => {
                        out.push_str(&format!("negotiation: refused ({})\n", refusal.code));
                    }
                }
            }
        }
        Format::Json => match args.device {
            Some(_) => out.push_str(&candidates[selected[0]].profile.to_json()),
            None => {
                let profiles: Vec<_> = selected
                    .iter()
                    .map(|&index| &candidates[index].profile)
                    .collect();
                out.push_str(&serde_json::to_string_pretty(&profiles).expect("profiles serialize"));
                out.push('\n');
            }
        },
    }
    if let Err(error) = stdout
        .write_all(out.as_bytes())
        .and_then(|()| stdout.flush())
    {
        let _ = writeln!(stderr, "modelspec inspect-device: {error}");
        return crate::EXIT_FAILURE;
    }
    let mut status = crate::EXIT_SUCCESS;
    for refusal in verdicts.iter().filter_map(|verdict| verdict.as_ref().err()) {
        let _ = writeln!(stderr, "{refusal}");
        status = crate::EXIT_FAILURE;
    }
    status
}
