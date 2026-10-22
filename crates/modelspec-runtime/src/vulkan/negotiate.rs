//! Feature negotiation, in one place (DESIGN.md § 6 "Device").
//!
//! The runtime requires Vulkan API version 1.2, where float controls and SPIR-V 1.5 are core,
//! and enables no optional feature bit and no device extension in M1. f16, cooperative matrix
//! and the three float-control values are recorded in the profile, not required: a device that
//! lacks a float-control value runs in the tolerance class of DESIGN.md § 5.4. The compute queue
//! family, which a profile does not carry, is checked by [`super::select::select_device`].

use std::fmt;

use modelspec_core::device::DeviceProfile;
use modelspec_core::diag::{Code, Diagnostic};

/// The Vulkan device is below the Vulkan API version the runtime requires (1.2.0).
pub const E_TARGET_004: Code = Code::new("E-TARGET-004");

/// The Vulkan API version the runtime requires, packed as Vulkan packs it: 1.2.0.
pub const REQUIRED_API_VERSION: u32 = 1 << 22 | 2 << 12;

/// Which comparison with the CPU oracle a result on the device gets (DESIGN.md § 5.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericsClass {
    /// The device reports all three float-control values; results built only from correctly
    /// rounded instructions equal the CPU oracle bit for bit.
    BitForBit,
    /// The device lacks one of the three float-control values; every result is compared within
    /// tolerance.
    Tolerance,
}

impl fmt::Display for NumericsClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::BitForBit => "bit-for-bit",
            Self::Tolerance => "tolerance",
        })
    }
}

/// What a logical device is created with: the API version it targets and exactly the optional
/// feature bits and device extensions it enables, plus the numerics class of the device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Negotiated {
    pub api_version: u32,
    /// Optional `VkPhysicalDeviceFeatures` bits by Vulkan member name; empty in M1.
    pub features: Vec<String>,
    /// Device extensions by Vulkan name; empty in M1.
    pub extensions: Vec<String>,
    pub numerics: NumericsClass,
}

/// `major.minor.patch` of a packed Vulkan version.
pub fn format_version(packed: u32) -> String {
    format!(
        "{}.{}.{}",
        (packed >> 22) & 0x7f,
        (packed >> 12) & 0x3ff,
        packed & 0xfff
    )
}

/// The E-TARGET-004 refusal of a device that reports `reported`.
pub fn below_required(reported: u32) -> Diagnostic {
    Diagnostic::new(
        E_TARGET_004,
        format!(
            "the Vulkan device reports API version {}, below the required {}",
            format_version(reported),
            format_version(REQUIRED_API_VERSION)
        ),
    )
    .with("required", format_version(REQUIRED_API_VERSION))
    .with("reported", format_version(reported))
}

/// Negotiates the device of `profile`: the set its logical device is created with, or the
/// refusal of a device below [`REQUIRED_API_VERSION`].
pub fn negotiate(profile: &DeviceProfile) -> Result<Negotiated, Diagnostic> {
    let reported = profile.vulkan_api_version & !(0x7 << 29);
    if reported < REQUIRED_API_VERSION {
        return Err(below_required(profile.vulkan_api_version));
    }
    let stated = |value: Option<bool>| value == Some(true);
    let numerics = if stated(profile.shader_rounding_mode_rte_f32)
        && stated(profile.shader_denorm_preserve_f32)
        && stated(profile.shader_signed_zero_inf_nan_preserve_f32)
    {
        NumericsClass::BitForBit
    } else {
        NumericsClass::Tolerance
    };
    Ok(Negotiated {
        api_version: REQUIRED_API_VERSION,
        features: Vec::new(),
        extensions: Vec::new(),
        numerics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelspec_core::device::DeviceProfile;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/rtx3090.device.json"
    ));

    fn fixture() -> DeviceProfile {
        DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile")
    }

    #[test]
    fn negotiate_on_the_fixture_returns_the_required_set() {
        let negotiated = negotiate(&fixture()).unwrap();
        assert_eq!(negotiated.api_version, (1 << 22) | (2 << 12));
        assert_eq!(negotiated.api_version, REQUIRED_API_VERSION);
        assert!(negotiated.features.is_empty());
        assert!(negotiated.extensions.is_empty());
    }

    #[test]
    fn negotiate_refuses_vulkan_1_1_with_e_target_004_naming_both_versions() {
        let mut profile = fixture();
        profile.vulkan_api_version = (1 << 22) | (1 << 12);
        let refusal = negotiate(&profile).unwrap_err();
        assert_eq!(refusal.code.as_str(), "E-TARGET-004");
        assert!(refusal.message.contains("1.2.0"), "{refusal}");
        assert!(refusal.message.contains("1.1.0"), "{refusal}");
        assert!(refusal.to_string().starts_with("E-TARGET-004: "));
    }

    #[test]
    fn negotiate_accepts_exactly_1_2_0_and_refuses_the_patch_below_it() {
        let mut profile = fixture();
        profile.vulkan_api_version = REQUIRED_API_VERSION;
        assert!(negotiate(&profile).is_ok());
        profile.vulkan_api_version = REQUIRED_API_VERSION - 1;
        assert_eq!(
            negotiate(&profile).unwrap_err().code.as_str(),
            "E-TARGET-004"
        );
    }

    #[test]
    fn negotiate_reports_the_numerics_class_from_the_three_float_controls() {
        let mut profile = fixture();
        profile.shader_rounding_mode_rte_f32 = Some(true);
        profile.shader_denorm_preserve_f32 = Some(true);
        profile.shader_signed_zero_inf_nan_preserve_f32 = Some(true);
        assert_eq!(
            negotiate(&profile).unwrap().numerics,
            NumericsClass::BitForBit
        );
        for (which, value) in (0..3).flat_map(|which| [(which, Some(false)), (which, None)]) {
            let mut copy = profile.clone();
            match which {
                0 => copy.shader_rounding_mode_rte_f32 = value,
                1 => copy.shader_denorm_preserve_f32 = value,
                _ => copy.shader_signed_zero_inf_nan_preserve_f32 = value,
            }
            let negotiated = negotiate(&copy).expect("a missing float control is not required");
            assert_eq!(negotiated.numerics, NumericsClass::Tolerance, "{which}");
            assert_eq!(negotiated.numerics.to_string(), "tolerance");
        }
        assert_eq!(NumericsClass::BitForBit.to_string(), "bit-for-bit");
    }

    #[test]
    fn negotiate_formats_packed_versions() {
        assert_eq!(format_version(REQUIRED_API_VERSION), "1.2.0");
        assert_eq!(format_version((1 << 22) | (4 << 12) | 341), "1.4.341");
    }
}
