//! The Vulkan device layer (DESIGN.md § 5.1, § 6 "Device"), behind the `vulkan` feature.
//!
//! The loader is opened at run time through `ash` (no link-time dependency on `libvulkan`):
//! the library [`LIBRARY_ENV`] names, or the system loader. [`Vulkan::candidates`] probes every
//! physical device into a
//! [`DeviceProfile`](modelspec_core::device::DeviceProfile) and the index of its compute queue family;
//! [`select::select_device`] chooses among them and [`negotiate::negotiate`] decides what the
//! logical device is created with. [`Vulkan::create_device`] creates a compute-only logical
//! device (no surface, no swapchain) with exactly that set.

mod enabled;
pub mod negotiate;
mod probe;
pub mod select;

use std::ffi::{CString, c_char};

use ash::vk;
use modelspec_core::diag::{Code, Diagnostic};

use negotiate::{Negotiated, REQUIRED_API_VERSION};
use select::{Candidate, E_TARGET_003, select_device};

/// No Vulkan loader could be opened: the library `MODELSPEC_VULKAN_LIBRARY` names, or the
/// system's `libvulkan`, is missing or is not a Vulkan loader.
pub const E_TARGET_002: Code = Code::new("E-TARGET-002");

/// The environment variable that names the Vulkan loader to open instead of the system's.
pub const LIBRARY_ENV: &str = "MODELSPEC_VULKAN_LIBRARY";

/// The library the system loader is opened as when [`LIBRARY_ENV`] is not set.
const SYSTEM_LIBRARY: &str = "libvulkan.so.1";

/// An opened Vulkan loader and an instance created on it, with no layer and no extension.
pub struct Vulkan {
    instance: ash::Instance,
    entry: ash::Entry,
    /// The API version passed to `vkCreateInstance` in `VkApplicationInfo`.
    api_version: u32,
}

impl Vulkan {
    /// Opens the loader [`LIBRARY_ENV`] names, or the system's, and creates an instance that
    /// targets [`REQUIRED_API_VERSION`].
    pub fn load() -> Result<Self, Diagnostic> {
        let library = std::env::var_os(LIBRARY_ENV);
        let loaded = match &library {
            // SAFETY: opening the Vulkan loader runs its initialisers; no function loaded from
            // the entry is called after `Vulkan` drops it.
            Some(path) => unsafe { ash::Entry::load_from(path) },
            None => unsafe { ash::Entry::load() },
        };
        let shown = library
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_else(|| SYSTEM_LIBRARY.to_owned());
        let entry = loaded.map_err(|error| {
            Diagnostic::new(E_TARGET_002, "no Vulkan loader could be opened")
                .with("library", shown)
                .with("error", error.to_string())
        })?;
        let application = vk::ApplicationInfo::default()
            .application_name(c"modelspec")
            .engine_name(c"modelspec")
            .api_version(REQUIRED_API_VERSION);
        let info = vk::InstanceCreateInfo::default().application_info(&application);
        // SAFETY: `info` and what it points to outlive the call.
        let instance = unsafe { entry.create_instance(&info, None) }.map_err(|error| {
            Diagnostic::new(E_TARGET_003, "no Vulkan instance could be created")
                .with("error", error.to_string())
        })?;
        Ok(Self {
            instance,
            entry,
            api_version: application.api_version,
        })
    }

    /// Every physical device, in the order the loader enumerates them, probed into a
    /// [`Candidate`].
    pub fn candidates(&self) -> Result<Vec<Candidate>, Diagnostic> {
        // SAFETY: the instance is alive for the call.
        let devices = unsafe { self.instance.enumerate_physical_devices() }.map_err(|error| {
            Diagnostic::new(
                E_TARGET_003,
                "the Vulkan physical devices could not be listed",
            )
            .with("error", error.to_string())
        })?;
        devices
            .into_iter()
            .enumerate()
            .map(|(index, device)| probe::candidate(&self.entry, &self.instance, device, index))
            .collect()
    }

    /// Creates a compute-only logical device on physical device `index` of [`Self::candidates`],
    /// with one queue of its compute queue family and exactly the optional features and
    /// extensions `negotiated` names. A device that negotiation refuses (below
    /// [`REQUIRED_API_VERSION`]) is refused with its E-TARGET-004, whatever `negotiated` holds.
    pub fn create_device(
        &self,
        index: usize,
        negotiated: &Negotiated,
    ) -> Result<Device<'_>, Diagnostic> {
        assert!(
            negotiated.features.is_empty(),
            "M1 negotiation enables no optional feature"
        );
        let candidates = self.candidates()?;
        let selected = select_device(&candidates, Some(index))?;
        let candidate = &candidates[selected[0]];
        // The device itself must pass negotiation, whatever `negotiated` was derived from.
        negotiate::negotiate(&candidate.profile)
            .map_err(|refusal| refusal.with("device", index.to_string()))?;
        let queue_family = candidate
            .compute_queue_family
            .expect("select_device keeps only a device with a compute queue family");
        // SAFETY: the instance is alive for the call.
        let physical = unsafe { self.instance.enumerate_physical_devices() }
            .map_err(|error| {
                Diagnostic::new(
                    E_TARGET_003,
                    "the Vulkan physical devices could not be listed",
                )
                .with("error", error.to_string())
            })?
            .get(index)
            .copied()
            .ok_or_else(|| {
                Diagnostic::new(
                    E_TARGET_003,
                    format!("there is no Vulkan physical device {index}"),
                )
            })?;
        let extensions: Vec<CString> = negotiated
            .extensions
            .iter()
            .map(|name| CString::new(name.as_str()).expect("an extension name has no NUL"))
            .collect();
        let extension_pointers: Vec<*const c_char> =
            extensions.iter().map(|name| name.as_ptr()).collect();
        let features = vk::PhysicalDeviceFeatures::default();
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extension_pointers)
            .enabled_features(&features);
        // SAFETY: `physical` belongs to this instance; `info` and what it points to outlive the
        // call.
        let device =
            unsafe { self.instance.create_device(physical, &info, None) }.map_err(|error| {
                Diagnostic::new(
                    E_TARGET_003,
                    "the Vulkan logical device could not be created",
                )
                .with("error", error.to_string())
            })?;
        // SAFETY: `info` still points at `extensions`, `queues` and `features`, all alive here.
        let (enabled_extensions, enabled_features) = unsafe { enabled::of(&info) };
        Ok(Device {
            device,
            queue_family: queues[0].queue_family_index,
            api_version: self.api_version,
            enabled_extensions,
            enabled_features,
            _vulkan: self,
        })
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        // SAFETY: every `Device` borrows `self`, so none is alive here.
        unsafe { self.instance.destroy_instance(None) };
    }
}

/// A compute-only logical device and what it was created with.
pub struct Device<'a> {
    device: ash::Device,
    queue_family: u32,
    api_version: u32,
    enabled_extensions: Vec<String>,
    enabled_features: Vec<String>,
    _vulkan: &'a Vulkan,
}

impl Device<'_> {
    /// The device extensions passed to `vkCreateDevice`.
    pub fn enabled_extensions(&self) -> &[String] {
        &self.enabled_extensions
    }

    /// The optional `VkPhysicalDeviceFeatures` bits passed to `vkCreateDevice`, by member name.
    pub fn enabled_features(&self) -> &[String] {
        &self.enabled_features
    }

    /// The API version passed to `vkCreateInstance`, which the device was created under.
    pub fn api_version(&self) -> u32 {
        self.api_version
    }

    /// The queue family of the device's one queue.
    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        // SAFETY: nothing created from the device outlives it.
        unsafe { self.device.destroy_device(None) };
    }
}
