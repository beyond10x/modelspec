//! The `VkPipelineCache` of a device run, kept on disk between runs (story:vulkan-pipeline-startup).
//!
//! A run creates its compute pipelines through one pipeline cache ([`PipelineCache`]), seeded from
//! the file of its device in the cache directory ([`directory`]) and written back after the
//! pipelines are made, so a later run on the same device and driver reads the compiled pipelines
//! instead of compiling them again. The file is named by the device UUID and the driver version
//! ([`file_name`]) and holds a header of its own before the driver's data ([`encode`]): a magic
//! word, the blake3 hash of the device's identity (vendor, device, UUID and driver version), the
//! length of the driver's data and its blake3 hash. A file whose header, identity, length or hash
//! does not match ([`decode`]) is ignored with a warning on stderr and the run starts with an
//! empty cache: the driver's data reaches `vkCreatePipelineCache` only when it is the data this
//! device and driver wrote. A cache changes no pipeline's code, so a run's bits are those of a run
//! without it.
//!
//! The directory is [`DIRECTORY_ENV`] when it is set and not empty, else
//! `$XDG_CACHE_HOME/modelspec/pipelines`, else `$HOME/.cache/modelspec/pipelines`; [`DIRECTORY_ENV`]
//! set to `off` keeps no file, and so does a home that is not set. The file is written to a
//! temporary file in the directory and renamed over the old one, so a reader sees a whole file or
//! the previous one. A file that cannot be written is a warning, never a failed run.

use std::path::{Path, PathBuf};

use ash::vk;
use modelspec_core::device::DeviceProfile;

use super::pipeline::{KernelPipelines, PipelineRequest};
use super::{Device, ExecError};

/// The environment variable that names the directory of the pipeline cache files, or `off`.
pub const DIRECTORY_ENV: &str = "MODELSPEC_PIPELINE_CACHE";

/// The first eight bytes of a pipeline cache file.
pub const MAGIC: [u8; 8] = *b"MSPLC001";

/// The bytes of the header before the driver's data: magic, identity hash, length, data hash.
pub const HEADER_BYTES: usize = 8 + 32 + 8 + 32;

/// The directory of the pipeline cache files, from the environment (the module documentation);
/// `None` when the run keeps no file.
pub fn directory() -> Option<PathBuf> {
    directory_from(
        std::env::var_os(DIRECTORY_ENV),
        std::env::var_os("XDG_CACHE_HOME"),
        std::env::var_os("HOME"),
    )
}

/// [`directory`] from the values of [`DIRECTORY_ENV`], `XDG_CACHE_HOME` and `HOME`.
pub fn directory_from(
    named: Option<std::ffi::OsString>,
    xdg_cache_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> Option<PathBuf> {
    let set = |value: Option<std::ffi::OsString>| value.filter(|value| !value.is_empty());
    if let Some(named) = set(named) {
        return (named != "off").then(|| PathBuf::from(named));
    }
    if let Some(cache) = set(xdg_cache_home) {
        return Some(PathBuf::from(cache).join("modelspec").join("pipelines"));
    }
    set(home).map(|home| PathBuf::from(home).join(".cache/modelspec/pipelines"))
}

/// What the cache of `profile`'s device depends on besides the SPIR-V: the device and the driver
/// that compiles it.
pub fn identity(profile: &DeviceProfile) -> String {
    format!(
        "modelspec-pipeline-cache/1 vendor {} device {} uuid {} driver {}",
        profile.vendor_id,
        profile.device_id,
        profile.device_uuid.as_deref().unwrap_or("none"),
        profile.driver_version
    )
}

/// The name of the cache file of `profile`'s device: its UUID (vendor and device id without
/// one) and the driver version.
pub fn file_name(profile: &DeviceProfile) -> String {
    let device = profile
        .device_uuid
        .clone()
        .unwrap_or_else(|| format!("{:08x}-{:08x}", profile.vendor_id, profile.device_id));
    format!("{device}-driver-{}.vkpc", profile.driver_version)
}

/// The bytes of a cache file of the device `identity` that holds the driver's `data`.
pub fn encode(identity: &str, data: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_BYTES + data.len());
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(blake3::hash(identity.as_bytes()).as_bytes());
    bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
    bytes.extend_from_slice(blake3::hash(data).as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

/// The driver's data in the cache file `bytes` of the device `identity`, or why the file is
/// ignored: too short, another format, another device or driver, or data that is not what the
/// header states.
pub fn decode<'b>(identity: &str, bytes: &'b [u8]) -> Result<&'b [u8], String> {
    if bytes.len() < HEADER_BYTES {
        return Err(format!(
            "{} bytes are shorter than the {HEADER_BYTES}-byte header",
            bytes.len()
        ));
    }
    if bytes[..8] != MAGIC {
        return Err("it is not a modelspec pipeline cache file".to_owned());
    }
    if bytes[8..40] != *blake3::hash(identity.as_bytes()).as_bytes() {
        return Err("it was written for another device or driver".to_owned());
    }
    let length = u64::from_le_bytes(bytes[40..48].try_into().expect("8 bytes"));
    let data = &bytes[HEADER_BYTES..];
    if data.len() as u64 != length {
        return Err(format!(
            "it holds {} bytes of data, its header states {length}",
            data.len()
        ));
    }
    if bytes[48..80] != *blake3::hash(data).as_bytes() {
        return Err("its data does not match the hash in its header".to_owned());
    }
    Ok(data)
}

/// A `VkPipelineCache` of a device and the file it was read from and is written to.
pub struct PipelineCache<'d> {
    device: &'d ash::Device,
    handle: vk::PipelineCache,
    file: Option<PathBuf>,
    identity: String,
    /// What was ignored or failed on the way, each a line for stderr.
    warnings: Vec<String>,
}

impl PipelineCache<'_> {
    /// The cache handle pipelines are created through.
    pub(crate) fn handle(&self) -> vk::PipelineCache {
        self.handle
    }

    /// The file the cache was read from and is written to, when the run keeps one.
    pub fn file(&self) -> Option<&Path> {
        self.file.as_deref()
    }

    /// The warnings so far: a file ignored, or one that could not be written.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The driver's current data of the cache.
    pub fn data(&self) -> Result<Vec<u8>, ExecError> {
        // SAFETY: the cache was created on `device`, which outlives it.
        unsafe { self.device.get_pipeline_cache_data(self.handle) }
            .map_err(ExecError::call("vkGetPipelineCacheData"))
    }

    /// Writes the cache to its file, when the run keeps one: a temporary file in the directory,
    /// renamed over the file. A failure is a warning, returned and kept.
    pub fn store(&mut self) -> Option<String> {
        let file = self.file.clone()?;
        let written = self
            .data()
            .map_err(|error| error.to_string())
            .and_then(|data| write_whole(&file, &encode(&self.identity, &data)));
        let warning = written.err().map(|reason| {
            format!(
                "warning: pipeline cache {} not written: {reason}",
                file.display()
            )
        });
        self.warnings.extend(warning.clone());
        warning
    }
}

impl Drop for PipelineCache<'_> {
    fn drop(&mut self) {
        // SAFETY: the cache was created on `device`; a pipeline created through it does not
        // need it any more.
        unsafe { self.device.destroy_pipeline_cache(self.handle, None) };
    }
}

/// Writes `bytes` to a temporary file beside `file` and renames it over `file`.
fn write_whole(file: &Path, bytes: &[u8]) -> Result<(), String> {
    let directory = file.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    let name = file
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temporary = directory.join(format!(".{name}.{}.tmp", std::process::id()));
    std::fs::write(&temporary, bytes)
        .and_then(|()| std::fs::rename(&temporary, file))
        .map_err(|error| {
            let _ = std::fs::remove_file(&temporary);
            error.to_string()
        })
}

/// The pipelines of `requests` for a run on `device`, whose profile is `profile`: through the
/// cache of [`directory`], on as many threads as the host has, one per distinct module
/// ([`Device::kernel_pipelines`]), and the cache written back. Each warning of the cache goes to
/// stderr. The duration is the whole of it, the cache read and written included: what `--stats`
/// prints as `pipeline creation us`.
pub fn make_pipelines<'d>(
    device: &'d Device<'_>,
    profile: &DeviceProfile,
    requests: &[PipelineRequest<'_>],
) -> Result<(KernelPipelines<'d>, std::time::Duration), ExecError> {
    let started = std::time::Instant::now();
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let mut cache = device.pipeline_cache(profile, directory().as_deref())?;
    let made = device.kernel_pipelines(requests, Some(&cache), threads)?;
    cache.store();
    for warning in cache.warnings() {
        eprintln!("{warning}");
    }
    Ok((made, started.elapsed()))
}

impl Device<'_> {
    /// A pipeline cache for `profile`'s device, seeded from its file in `directory` when that
    /// file is this device's and driver's (the module documentation), else empty, with a warning
    /// for a file that was there and ignored; without a directory, an empty cache kept in memory.
    pub fn pipeline_cache(
        &self,
        profile: &DeviceProfile,
        directory: Option<&Path>,
    ) -> Result<PipelineCache<'_>, ExecError> {
        let identity = identity(profile);
        let file = directory.map(|directory| directory.join(file_name(profile)));
        let mut warnings = Vec::new();
        let read = match &file {
            Some(file) => match std::fs::read(file) {
                Ok(bytes) => Some(bytes),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => {
                    warnings.push(format!(
                        "warning: pipeline cache {} ignored: {error}",
                        file.display()
                    ));
                    None
                }
            },
            None => None,
        };
        let initial: &[u8] = match (&file, &read) {
            (Some(file), Some(bytes)) => match decode(&identity, bytes) {
                Ok(data) => data,
                Err(reason) => {
                    warnings.push(format!(
                        "warning: pipeline cache {} ignored: {reason}",
                        file.display()
                    ));
                    &[]
                }
            },
            _ => &[],
        };
        let info = vk::PipelineCacheCreateInfo::default().initial_data(initial);
        // SAFETY: `info` and `initial` outlive the call; the cache is destroyed by its drop.
        let handle = unsafe { self.device.create_pipeline_cache(&info, None) }
            .map_err(ExecError::call("vkCreatePipelineCache"))?;
        Ok(PipelineCache {
            device: &self.device,
            handle,
            file,
            identity,
            warnings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_is_read_back_only_by_the_identity_that_wrote_it() {
        let data = b"driver data".to_vec();
        let file = encode("device a", &data);
        assert_eq!(decode("device a", &file), Ok(&data[..]));
        assert_eq!(
            decode("device b", &file),
            Err("it was written for another device or driver".to_owned())
        );
    }

    #[test]
    fn a_truncated_flipped_or_foreign_file_is_refused() {
        let file = encode("device a", b"driver data");
        assert!(decode("device a", &file[..HEADER_BYTES - 1]).is_err());
        assert!(decode("device a", &file[..file.len() - 1]).is_err());
        let mut flipped = file.clone();
        *flipped.last_mut().expect("data") ^= 1;
        assert_eq!(
            decode("device a", &flipped),
            Err("its data does not match the hash in its header".to_owned())
        );
        let mut foreign = file;
        foreign[..8].copy_from_slice(b"NOTMINE!");
        assert!(decode("device a", &foreign).is_err());
    }

    #[test]
    fn the_directory_is_the_named_one_else_the_user_cache_and_off_keeps_none() {
        let os = |text: &str| Some(std::ffi::OsString::from(text));
        assert_eq!(
            directory_from(os("/x/p"), os("/c"), os("/h")),
            Some(PathBuf::from("/x/p"))
        );
        assert_eq!(directory_from(os("off"), os("/c"), os("/h")), None);
        assert_eq!(
            directory_from(os(""), os("/c"), os("/h")),
            Some(PathBuf::from("/c/modelspec/pipelines"))
        );
        assert_eq!(
            directory_from(None, None, os("/h")),
            Some(PathBuf::from("/h/.cache/modelspec/pipelines"))
        );
        assert_eq!(directory_from(None, None, None), None);
    }
}
