//! The Vulkan device layer (DESIGN.md § 5.1, § 6 "Device"), behind the `vulkan` feature.
//!
//! The loader is opened at run time through `ash` (no link-time dependency on `libvulkan`):
//! the library [`LIBRARY_ENV`] names, or the system loader. [`Vulkan::candidates`] probes every
//! physical device into a
//! [`DeviceProfile`](modelspec_core::device::DeviceProfile) and the index of its compute queue family;
//! [`select::select_device`] chooses among them and [`negotiate::negotiate`] decides what the
//! logical device is created with. [`Vulkan::create_device`] creates a compute-only logical
//! device (no surface, no swapchain) with exactly that set.
//!
//! On a [`Device`], [`memory`] allocates storage and staging buffers, [`pipeline`] turns one
//! SPIR-V module into a compute pipeline, and [`dispatch`] records copies, [`sync`] barriers and
//! dispatches into one command buffer, submits it and waits on its fence. Every host wait is
//! counted ([`Device::host_waits`]).
//!
//! With [`VALIDATION_ENV`] set to `1`, the instance is created with the Khronos validation layer
//! and a debug messenger that keeps every warning and error it reports
//! ([`Vulkan::take_validation_messages`]).

pub mod dispatch;
mod enabled;
pub mod memory;
pub mod negotiate;
pub mod pipeline;
mod probe;
pub mod select;
pub mod sync;

use std::cell::Cell;
use std::ffi::{CString, c_char, c_void};
use std::sync::Mutex;

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

/// The environment variable that, set to `1`, turns on the Khronos validation layer.
pub const VALIDATION_ENV: &str = "MODELSPEC_VK_VALIDATION";

/// The layer [`VALIDATION_ENV`] enables.
const VALIDATION_LAYER: &std::ffi::CStr = c"VK_LAYER_KHRONOS_validation";

/// What a run on a [`Device`] can fail with.
#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    /// No memory type of the device has the properties a buffer needs.
    #[error(
        "no memory type has properties {wanted:#x} among the types {types:#b} the buffer allows"
    )]
    NoMemoryType { wanted: u32, types: u32 },
    /// A buffer of zero bytes was asked for.
    #[error("a buffer of zero bytes was asked for")]
    ZeroSize,
    /// A buffer is larger than one storage buffer binding can address.
    #[error("{bytes} bytes exceed maxStorageBufferRange {max}")]
    TooLarge { bytes: u64, max: u64 },
    /// A device-local buffer was mapped.
    #[error("the buffer is not host visible")]
    NotHostVisible,
    /// A copy or write does not fit its destination.
    #[error("{needed} bytes do not fit a buffer of {size}")]
    TooShort { needed: u64, size: u64 },
    /// A dispatch binds a different number of buffers than its pipeline declares.
    #[error("the pipeline has {expected} bindings, the dispatch binds {got}")]
    Bindings { expected: u32, got: usize },
    /// A pipeline was asked for bindings other than the module's: its `Binding` decorations
    /// must be 0, 1, … once each without a gap, and every `DescriptorSet` decoration 0.
    #[error(
        "the SPIR-V module declares bindings {declared:?} in descriptor sets {sets:?}; the \
         pipeline was asked for {requested} bindings in set 0"
    )]
    ModuleBindings {
        declared: Vec<u32>,
        sets: Vec<u32>,
        requested: u32,
    },
    /// A dispatch reads or writes past the end of a bound buffer: a tensor binding needs
    /// `count` elements, a rank-0 broadcast binding one.
    #[error(
        "a dispatch of {count} elements needs {needed} elements at binding {binding}, which \
         holds {elements}"
    )]
    CountPastBuffer {
        count: u32,
        binding: usize,
        needed: u64,
        elements: u64,
    },
    /// A dispatch's workgroups cover fewer invocations than it has elements.
    #[error("{workgroups:?} workgroups of {size} cover {covered} invocations, fewer than {count}")]
    Workgroups {
        workgroups: [u32; 3],
        size: u32,
        covered: u64,
        count: u32,
    },
    /// A SPIR-V module declares no `LocalSize`.
    #[error("the SPIR-V module declares no LocalSize execution mode")]
    NoLocalSize,
    /// A Vulkan call failed.
    #[error("{call} failed: {result}")]
    Vulkan {
        call: &'static str,
        result: vk::Result,
    },
}

impl ExecError {
    /// The error of a failed Vulkan `call`.
    pub(crate) fn call(call: &'static str) -> impl FnOnce(vk::Result) -> Self {
        move |result| Self::Vulkan { call, result }
    }
}

/// The messages the validation layer reported, kept by the debug messenger.
type Messages = Mutex<Vec<String>>;

/// The debug messenger of a validated instance.
struct Validation {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
    messages: Box<Messages>,
}

/// Keeps each warning and error of the validation layer in the [`Messages`] at `user`.
unsafe extern "system" fn keep_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: `user` is the `Messages` box of a live `Validation`, and `data` is valid for the
    // call, as the messenger was created with both.
    let (messages, text) = unsafe {
        let text = data
            .as_ref()
            .and_then(|data| data.message_as_c_str())
            .map(|text| text.to_string_lossy().into_owned())
            .unwrap_or_default();
        (&*(user as *const Messages), text)
    };
    if let Ok(mut messages) = messages.lock() {
        let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            "error"
        } else {
            "warning"
        };
        messages.push(format!("{level}: {text}"));
    }
    vk::FALSE
}

/// An opened Vulkan loader and an instance created on it: with no layer and no extension, or
/// with the validation layer and `VK_EXT_debug_utils` when [`VALIDATION_ENV`] is `1`.
pub struct Vulkan {
    instance: ash::Instance,
    entry: ash::Entry,
    /// The API version passed to `vkCreateInstance` in `VkApplicationInfo`.
    api_version: u32,
    validation: Option<Validation>,
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
        let validated = std::env::var(VALIDATION_ENV).as_deref() == Ok("1");
        let layers = [VALIDATION_LAYER.as_ptr()];
        let extensions = [ash::ext::debug_utils::NAME.as_ptr()];
        let mut info = vk::InstanceCreateInfo::default().application_info(&application);
        if validated {
            info = info
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions);
        }
        // SAFETY: `info` and what it points to outlive the call.
        let instance = unsafe { entry.create_instance(&info, None) }.map_err(|error| {
            let refusal = Diagnostic::new(E_TARGET_003, "no Vulkan instance could be created")
                .with("error", error.to_string());
            if validated {
                refusal.with("layer", VALIDATION_LAYER.to_string_lossy())
            } else {
                refusal
            }
        })?;
        let mut vulkan = Self {
            instance,
            entry,
            api_version: application.api_version,
            validation: None,
        };
        if validated {
            vulkan.validation = Some(vulkan.messenger()?);
        }
        Ok(vulkan)
    }

    /// A debug messenger on the instance that keeps every warning and error.
    fn messenger(&self) -> Result<Validation, Diagnostic> {
        let loader = ash::ext::debug_utils::Instance::new(&self.entry, &self.instance);
        let messages: Box<Messages> = Box::default();
        let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(keep_message))
            .user_data(&*messages as *const Messages as *mut c_void);
        // SAFETY: the box the callback writes to is moved into the `Validation`, which destroys
        // the messenger before it drops the box.
        let messenger =
            unsafe { loader.create_debug_utils_messenger(&info, None) }.map_err(|error| {
                Diagnostic::new(
                    E_TARGET_003,
                    "the Vulkan validation messenger could not be created",
                )
                .with("error", error.to_string())
            })?;
        Ok(Validation {
            loader,
            messenger,
            messages,
        })
    }

    /// Whether the instance was created with the validation layer ([`VALIDATION_ENV`]).
    pub fn validation_enabled(&self) -> bool {
        self.validation.is_some()
    }

    /// The validation messages reported since the last call, oldest first; none without the
    /// layer.
    pub fn take_validation_messages(&self) -> Vec<String> {
        self.validation
            .as_ref()
            .map(|validation| {
                std::mem::take(
                    &mut *validation
                        .messages
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                )
            })
            .unwrap_or_default()
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
        // SAFETY: `physical` belongs to this instance.
        let (memory, limits) = unsafe {
            (
                self.instance
                    .get_physical_device_memory_properties(physical),
                self.instance
                    .get_physical_device_properties(physical)
                    .limits,
            )
        };
        Ok(Device {
            device,
            queue_family: queues[0].queue_family_index,
            api_version: self.api_version,
            enabled_extensions,
            enabled_features,
            memory,
            limits,
            host_waits: Cell::new(0),
            _vulkan: self,
        })
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        if let Some(validation) = self.validation.take() {
            // SAFETY: the messenger belongs to this instance and is destroyed once, before it.
            unsafe {
                validation
                    .loader
                    .destroy_debug_utils_messenger(validation.messenger, None)
            };
        }
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
    memory: vk::PhysicalDeviceMemoryProperties,
    limits: vk::PhysicalDeviceLimits,
    host_waits: Cell<u64>,
    _vulkan: &'a Vulkan,
}

impl Device<'_> {
    /// How many times the host has waited on this device's work: one per fence wait of a
    /// [`dispatch::Submitted`].
    pub fn host_waits(&self) -> u64 {
        self.host_waits.get()
    }

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
