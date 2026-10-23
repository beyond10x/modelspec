//! The room of a descriptor pool, on the host: the sets and storage-buffer descriptors it is
//! made with, and how many of each its sets have taken.
//!
//! A pool is created from its [`PoolRoom`] ([`pool_info`]), and a set is allocated from it only
//! with the [`Admitted`] that [`admit`] returns for that set, which nothing else can make; so a
//! set that does not fit is refused with [`ExecError::DescriptorPoolFull`] before the pool is
//! asked, on every driver, not only on one that accounts the pool exactly.

use std::cell::Cell;

use ash::vk;

use super::ExecError;

/// The sets and storage-buffer descriptors a descriptor pool is made with, and how many of each
/// its sets have taken.
#[derive(Debug)]
pub(crate) struct PoolRoom {
    sets: u32,
    descriptors: u32,
    bound_sets: Cell<u32>,
    bound: Cell<u32>,
}

impl PoolRoom {
    /// The room of a pool of `sets` sets and `descriptors` descriptors, at least one of each.
    pub(crate) fn new(sets: u32, descriptors: u32) -> Self {
        Self {
            sets: sets.max(1),
            descriptors: descriptors.max(1),
            bound_sets: Cell::new(0),
            bound: Cell::new(0),
        }
    }
}

/// One set of a pool, admitted by [`admit`]: the only way to allocate a set of a pool.
#[derive(Debug)]
pub(crate) struct Admitted {
    bindings: usize,
}

impl Admitted {
    /// The storage-buffer descriptors the set was admitted with.
    pub(crate) fn bindings(&self) -> usize {
        self.bindings
    }
}

/// Admits one set of `bindings` storage-buffer descriptors into `room`, or refuses with
/// [`ExecError::DescriptorPoolFull`] and takes nothing.
pub(crate) fn admit(room: &PoolRoom, bindings: usize) -> Result<Admitted, ExecError> {
    let (bound_sets, bound) = (room.bound_sets.get(), room.bound.get());
    let fits = u32::try_from(bindings)
        .ok()
        .and_then(|bindings| bound.checked_add(bindings))
        .filter(|&taken| taken <= room.descriptors && bound_sets < room.sets);
    let Some(taken) = fits else {
        return Err(ExecError::DescriptorPoolFull {
            sets: room.sets,
            descriptors: room.descriptors,
            bound_sets,
            bound,
            needed: bindings,
        });
    };
    room.bound_sets.set(bound_sets + 1);
    room.bound.set(taken);
    Ok(Admitted { bindings })
}

/// The pool sizes of `room`: its storage-buffer descriptors.
pub(crate) fn pool_sizes(room: &PoolRoom) -> [vk::DescriptorPoolSize; 1] {
    [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(room.descriptors)]
}

/// The create info of the pool of `room`, with `sizes` from [`pool_sizes`].
pub(crate) fn pool_info<'a>(
    room: &PoolRoom,
    sizes: &'a [vk::DescriptorPoolSize; 1],
) -> vk::DescriptorPoolCreateInfo<'a> {
    vk::DescriptorPoolCreateInfo::default()
        .max_sets(room.sets)
        .pool_sizes(sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal of `room` admitting one more set of `bindings`, its fields as one tuple.
    fn refusal(room: &PoolRoom, bindings: usize) -> (u32, u32, u32, u32, usize) {
        match admit(room, bindings) {
            Err(ExecError::DescriptorPoolFull {
                sets,
                descriptors,
                bound_sets,
                bound,
                needed,
            }) => (sets, descriptors, bound_sets, bound, needed),
            other => panic!("a set of {bindings} is refused, got {other:?}"),
        }
    }

    #[test]
    fn a_set_past_the_pools_descriptors_is_refused_by_name() {
        let room = PoolRoom::new(4, 5);
        assert_eq!(admit(&room, 2).expect("2 of 5 fit").bindings(), 2);
        admit(&room, 2).expect("4 of 5 fit");
        assert_eq!(refusal(&room, 2), (4, 5, 2, 4, 2));
        assert_eq!(
            admit(&room, 2).unwrap_err().to_string(),
            "the descriptor pool holds 4 sets and 5 storage-buffer descriptors; 2 sets of 4 are \
             bound, and one more set of 2 does not fit"
        );
        admit(&room, 1).expect("the refused sets took nothing, so 5 of 5 fit");
        assert_eq!(refusal(&room, 1), (4, 5, 3, 5, 1));
    }

    #[test]
    fn a_set_past_the_pools_sets_is_refused_by_name() {
        let room = PoolRoom::new(2, 100);
        admit(&room, 3).expect("the first set fits");
        admit(&room, 3).expect("the second set fits");
        assert_eq!(refusal(&room, 0), (2, 100, 2, 6, 0));
    }

    const STORAGE_BUFFER: i32 = vk::DescriptorType::STORAGE_BUFFER.as_raw();

    /// The pool `room` describes, read back from its create info: max sets, then each size.
    fn created(room: &PoolRoom) -> (u32, Vec<(i32, u32)>) {
        let sizes = pool_sizes(room);
        let info = pool_info(room, &sizes);
        let read = if info.p_pool_sizes.is_null() {
            &[][..]
        } else {
            // SAFETY: `p_pool_sizes` points at `pool_size_count` sizes of `sizes`, alive here.
            unsafe { std::slice::from_raw_parts(info.p_pool_sizes, info.pool_size_count as usize) }
        };
        (
            info.max_sets,
            read.iter()
                .map(|s| (s.ty.as_raw(), s.descriptor_count))
                .collect(),
        )
    }

    /// The pool is made with the room that admits its sets: exactly the sets and descriptors
    /// `admit` lets through, so admission and the pool never disagree.
    #[test]
    fn the_pool_is_created_with_the_room_that_admits_its_sets() {
        let room = PoolRoom::new(14, 39);
        assert_eq!(created(&room), (14, vec![(STORAGE_BUFFER, 39)]));
        for _ in 0..13 {
            admit(&room, 3).expect("13 sets of 3 fit 39");
        }
        assert_eq!(refusal(&room, 1), (14, 39, 13, 39, 1));
        admit(&room, 0).expect("the 14th set of none fits");
        assert_eq!(refusal(&room, 0), (14, 39, 14, 39, 0));
    }

    /// An empty room still makes a pool of one set of one descriptor, as Vulkan requires.
    #[test]
    fn an_empty_room_makes_a_pool_of_one_set_of_one_descriptor() {
        let room = PoolRoom::new(0, 0);
        assert_eq!(created(&room), (1, vec![(STORAGE_BUFFER, 1)]));
        admit(&room, 1).expect("one set of one fits");
        assert_eq!(refusal(&room, 1), (1, 1, 1, 1, 1));
    }
}
