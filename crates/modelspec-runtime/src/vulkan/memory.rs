//! Device memory: one allocation per buffer.
//!
//! A device-local buffer is a storage buffer that copies run into and out of; a staging buffer
//! is host-visible, host-coherent memory mapped for the buffer's whole life, which the host
//! writes inputs to and reads results from.
//!
//! Every `vkAllocateMemory` call and the free of its memory are counted for the thread that
//! made the buffer, which is the thread that frees it (a [`Buffer`] is not `Send`). A
//! [`MemorySpan`] reads them for a stretch of that thread, such as one device run: the
//! allocations made in it and the largest sum of the allocation sizes of its live buffers.

use std::cell::RefCell;
use std::marker::PhantomData;

use ash::vk;
use modelspec_core::ir::kernel::BufferView;

use super::{Device, ExecError};

/// The device memory counters of one thread.
#[derive(Default)]
struct Ledger {
    /// The `vkAllocateMemory` calls of the thread so far; the next allocation's serial.
    allocations: u64,
    /// The spans open on the thread, the outermost first.
    spans: Vec<Open>,
}

/// An open [`MemorySpan`]'s counts.
struct Open {
    /// The serial of the first allocation made in the span.
    first: u64,
    /// The allocation bytes of the span's allocations that are not freed.
    live: u64,
    /// The largest `live` so far.
    peak: u64,
}

thread_local! {
    static LEDGER: RefCell<Ledger> = RefCell::default();
}

/// Counts an allocation of `bytes` on this thread and returns its serial, for [`freed`].
fn allocated(bytes: u64) -> u64 {
    LEDGER.with_borrow_mut(|ledger| {
        let serial = ledger.allocations;
        ledger.allocations += 1;
        for open in &mut ledger.spans {
            open.live += bytes;
            open.peak = open.peak.max(open.live);
        }
        serial
    })
}

/// Counts the free of the allocation of `bytes` that [`allocated`] gave `serial`; nothing once
/// the thread's counters are destroyed as it exits, when no span is open any more.
fn freed(serial: u64, bytes: u64) {
    let _ = LEDGER.try_with(|ledger| {
        let mut ledger = ledger.borrow_mut();
        for open in ledger.spans.iter_mut().filter(|open| serial >= open.first) {
            open.live -= bytes;
        }
    });
}

/// The device memory this thread allocates from [`MemorySpan::start`] until the span is
/// dropped. Spans nest: one started inside another counts its own part, which the outer one
/// counts too. Dropping a span closes it and every span started after it.
pub struct MemorySpan {
    /// The span's place in the thread's open spans.
    index: usize,
    /// The serial of its first allocation.
    first: u64,
    /// Ties the span to the thread whose counters it reads.
    _thread: PhantomData<*const ()>,
}

impl MemorySpan {
    /// Opens a span on this thread.
    pub fn start() -> Self {
        LEDGER.with_borrow_mut(|ledger| {
            let first = ledger.allocations;
            ledger.spans.push(Open {
                first,
                live: 0,
                peak: 0,
            });
            MemorySpan {
                index: ledger.spans.len() - 1,
                first,
                _thread: PhantomData,
            }
        })
    }

    /// The `vkAllocateMemory` calls since the span started.
    pub fn allocations(&self) -> u64 {
        LEDGER.with_borrow(|ledger| ledger.allocations - self.first)
    }

    /// The largest sum of the allocation sizes of the span's allocations live at one time, in
    /// bytes.
    pub fn peak_bytes(&self) -> u64 {
        LEDGER.with_borrow(|ledger| ledger.spans.get(self.index).map_or(0, |open| open.peak))
    }
}

impl Drop for MemorySpan {
    fn drop(&mut self) {
        LEDGER.with_borrow_mut(|ledger| ledger.spans.truncate(self.index));
    }
}

/// A buffer and the memory bound to it.
pub struct Buffer<'d> {
    device: &'d ash::Device,
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: u64,
    /// The `allocationSize` of the memory's `vkAllocateMemory` call; 0 before it succeeds.
    allocation_bytes: u64,
    /// The serial [`allocated`] gave the memory.
    serial: u64,
    mapped: Option<*mut u8>,
}

impl Buffer<'_> {
    /// The size the buffer was asked for, in bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The size of the device memory allocated for the buffer, in bytes: at least
    /// [`Buffer::bytes`], as the device's memory requirements for the buffer round it.
    pub fn allocation_bytes(&self) -> u64 {
        self.allocation_bytes
    }

    /// Whether the buffer is a mapped staging buffer rather than a device-local one.
    pub fn is_staging(&self) -> bool {
        self.mapped.is_some()
    }

    /// Writes `values` at the start of a staging buffer.
    pub fn write_f32(&mut self, values: &[f32]) -> Result<(), ExecError> {
        let needed = std::mem::size_of_val(values) as u64;
        let at = self.mapped.ok_or(ExecError::NotHostVisible)?;
        if needed > self.bytes {
            return Err(ExecError::TooShort {
                needed,
                size: self.bytes,
            });
        }
        // SAFETY: the mapping covers `bytes` bytes, at least `needed`, and nothing else
        // references it while `self` is borrowed mutably.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), at, needed as usize) };
        Ok(())
    }

    /// Every whole `f32` of a staging buffer.
    ///
    /// It takes `&mut self` so that no read compiles while a recording or submission that copies
    /// into the buffer still borrows it: the read comes after [`Submitted::wait`] or its drop.
    ///
    /// ```no_run
    /// # use modelspec_runtime::vulkan::{Vulkan, negotiate::negotiate};
    /// # let vulkan = Vulkan::load().unwrap();
    /// # let profile = vulkan.candidates().unwrap()[0].profile.clone();
    /// # let device = vulkan.create_device(0, &negotiate(&profile).unwrap()).unwrap();
    /// let local = device.device_local_buffer(16).unwrap();
    /// let mut staging = device.staging_buffer(16).unwrap();
    /// let mut recording = device.record().unwrap();
    /// recording.copy(&local, &staging).unwrap();
    /// let submitted = device.submit(recording.finish().unwrap()).unwrap();
    /// submitted.wait().unwrap();
    /// let values = staging.read_f32().unwrap();
    /// ```
    ///
    /// Reading while the submission is pending does not compile:
    ///
    /// ```compile_fail,E0502
    /// # use modelspec_runtime::vulkan::{Vulkan, negotiate::negotiate};
    /// # let vulkan = Vulkan::load().unwrap();
    /// # let profile = vulkan.candidates().unwrap()[0].profile.clone();
    /// # let device = vulkan.create_device(0, &negotiate(&profile).unwrap()).unwrap();
    /// let local = device.device_local_buffer(16).unwrap();
    /// let mut staging = device.staging_buffer(16).unwrap();
    /// let mut recording = device.record().unwrap();
    /// recording.copy(&local, &staging).unwrap();
    /// let submitted = device.submit(recording.finish().unwrap()).unwrap();
    /// let values = staging.read_f32().unwrap();
    /// submitted.wait().unwrap();
    /// ```
    ///
    /// [`Submitted::wait`]: super::dispatch::Submitted::wait
    pub fn read_f32(&mut self) -> Result<Vec<f32>, ExecError> {
        let at = self.mapped.ok_or(ExecError::NotHostVisible)?;
        let count = (self.bytes / 4) as usize;
        let mut values = vec![0f32; count];
        // SAFETY: the mapping covers `bytes` bytes, at least `4 * count`.
        unsafe { std::ptr::copy_nonoverlapping(at, values.as_mut_ptr().cast::<u8>(), count * 4) };
        Ok(values)
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        // SAFETY: no submitted work uses the buffer any more: a `Submitted` waits before it is
        // dropped, and it borrows every buffer it was recorded with.
        unsafe {
            if self.mapped.is_some() {
                self.device.unmap_memory(self.memory);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
        if self.memory != vk::DeviceMemory::null() {
            freed(self.serial, self.allocation_bytes);
        }
    }
}

/// The index of the first memory type of `memory` that `allowed` (a bit per type) permits and
/// that has every property of `wanted`.
pub fn find_memory_type(
    memory: &vk::PhysicalDeviceMemoryProperties,
    allowed: u32,
    wanted: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..memory.memory_type_count.min(vk::MAX_MEMORY_TYPES as u32)).find(|&index| {
        allowed & (1 << index) != 0
            && memory.memory_types[index as usize]
                .property_flags
                .contains(wanted)
    })
}

impl Device<'_> {
    /// A device-local storage buffer of `bytes` bytes that copies can run into and out of.
    pub fn device_local_buffer(&self, bytes: u64) -> Result<Buffer<'_>, ExecError> {
        self.buffer(
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    /// The device-local buffer an `Allocate` step of the plan makes for `view`, such as the
    /// partials of a reduction's stage 1: `view.size_bytes` bytes, never mapped.
    pub fn scratch_buffer(&self, view: &BufferView) -> Result<Buffer<'_>, ExecError> {
        self.device_local_buffer(view.size_bytes)
    }

    /// A host-visible, host-coherent buffer of `bytes` bytes, mapped, for copies to and from
    /// device-local buffers.
    pub fn staging_buffer(&self, bytes: u64) -> Result<Buffer<'_>, ExecError> {
        self.buffer(
            bytes,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    fn buffer(
        &self,
        bytes: u64,
        usage: vk::BufferUsageFlags,
        wanted: vk::MemoryPropertyFlags,
    ) -> Result<Buffer<'_>, ExecError> {
        if bytes == 0 {
            return Err(ExecError::ZeroSize);
        }
        let max = u64::from(self.limits.max_storage_buffer_range);
        if bytes > max {
            return Err(ExecError::TooLarge { bytes, max });
        }
        let info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `info` outlives the call.
        let buffer = unsafe { self.device.create_buffer(&info, None) }
            .map_err(ExecError::call("vkCreateBuffer"))?;
        let mut owned = Buffer {
            device: &self.device,
            buffer,
            memory: vk::DeviceMemory::null(),
            bytes,
            allocation_bytes: 0,
            serial: 0,
            mapped: None,
        };
        // SAFETY: `buffer` was created on this device.
        let needs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let index = find_memory_type(&self.memory, needs.memory_type_bits, wanted).ok_or(
            ExecError::NoMemoryType {
                wanted: wanted.as_raw(),
                types: needs.memory_type_bits,
            },
        )?;
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(needs.size)
            .memory_type_index(index);
        // SAFETY: `allocation` outlives the call; the memory is freed by `owned`'s drop.
        owned.memory = unsafe { self.device.allocate_memory(&allocation, None) }
            .map_err(ExecError::call("vkAllocateMemory"))?;
        owned.allocation_bytes = needs.size;
        owned.serial = allocated(needs.size);
        // SAFETY: the memory is new, of a type the buffer allows and at least its size.
        unsafe { self.device.bind_buffer_memory(buffer, owned.memory, 0) }
            .map_err(ExecError::call("vkBindBufferMemory"))?;
        if wanted.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            // SAFETY: the memory is host visible and not mapped yet.
            let at = unsafe {
                self.device
                    .map_memory(owned.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }
            .map_err(ExecError::call("vkMapMemory"))?;
            owned.mapped = Some(at.cast::<u8>());
        }
        Ok(owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn properties(flags: &[vk::MemoryPropertyFlags]) -> vk::PhysicalDeviceMemoryProperties {
        let mut memory = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: flags.len() as u32,
            ..Default::default()
        };
        for (at, &property_flags) in flags.iter().enumerate() {
            memory.memory_types[at].property_flags = property_flags;
        }
        memory
    }

    #[test]
    fn find_memory_type_takes_the_first_allowed_type_with_every_wanted_property() {
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory = properties(&[
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            host,
            host | vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ]);
        assert_eq!(find_memory_type(&memory, 0b1111, host), Some(2));
        assert_eq!(find_memory_type(&memory, 0b1000, host), Some(3));
        assert_eq!(
            find_memory_type(&memory, 0b1111, vk::MemoryPropertyFlags::DEVICE_LOCAL),
            Some(0)
        );
        assert_eq!(find_memory_type(&memory, 0b0011, host), None);
        assert_eq!(find_memory_type(&memory, 0b1_0000, host), None);
    }

    /// story:vulkan-step-counters: a span counts the allocations made after it starts and the
    /// largest sum of their live sizes; an allocation from before it, freed during it, is not
    /// its own, and a span inside it counts its own part and adds to the outer one.
    #[test]
    fn a_span_counts_its_allocations_and_the_peak_of_their_live_bytes() {
        let earlier = allocated(100);
        let span = MemorySpan::start();
        assert_eq!((span.allocations(), span.peak_bytes()), (0, 0));
        let a = allocated(10);
        let b = allocated(20);
        freed(a, 10);
        let c = allocated(5);
        freed(earlier, 100);
        assert_eq!((span.allocations(), span.peak_bytes()), (3, 30));
        {
            let inner = MemorySpan::start();
            let d = allocated(40);
            assert_eq!((inner.allocations(), inner.peak_bytes()), (1, 40));
            freed(d, 40);
            assert_eq!((inner.allocations(), inner.peak_bytes()), (1, 40));
        }
        assert_eq!((span.allocations(), span.peak_bytes()), (4, 65));
        freed(b, 20);
        freed(c, 5);
        assert_eq!((span.allocations(), span.peak_bytes()), (4, 65));
    }
}
