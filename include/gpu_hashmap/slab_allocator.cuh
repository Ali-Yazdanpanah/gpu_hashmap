/**
 * @file slab_allocator.cuh
 * @brief Slab allocation structures for GPU: fixed-size blocks, atomic free list.
 *
 * Avoids fragmentation by allocating from a contiguous pool of equal-sized slots.
 * Allocation = atomic pop from free list; free = atomic push. No host involvement
 * per alloc after init.
 */

#ifndef GPU_HASHMAP_SLAB_ALLOCATOR_CUH
#define GPU_HASHMAP_SLAB_ALLOCATOR_CUH

#include "types.cuh"
#include <cstddef>
#include <cuda_runtime.h>

namespace gpu_hashmap {

/**
 * Device-side slab pool: contiguous slots, each slot index is either in the
 * free list or in use. free_list_head is the head of a linked list of free
 * slot indices (atomic for concurrent alloc).
 */
struct SlabDevice {
  SlotIndex* free_list_next;   ///< free_list_next[i] = next free slot after i (or unused)
  unsigned long long* free_list_head;  ///< atomic: head of free list (slot index)
  size_t capacity;             ///< number of slots
  size_t slot_size_bytes;       ///< bytes per slot (for future variable-sized slabs)
};

/**
 * Host-owned descriptor for the slab (allocates device memory at init).
 */
struct SlabAllocator {
  SlabDevice device;
  void* d_pool;                ///< raw device memory for free_list_next (and optional payload)
  size_t capacity;
  bool owns_memory;
};

/** Initialize slab on device: all slots in free list, capacity = num_slots. */
void slab_init(SlabAllocator* alloc, size_t num_slots, cudaStream_t stream = nullptr);

/** Release device memory. */
void slab_destroy(SlabAllocator* alloc);

/**
 * Allocate one slot from the slab (device). Returns kInvalidSlot if empty.
 * Thread-safe: CAS on free_list_head. Defined in header so callers (e.g. insert_kernel) compile it.
 */
__device__ __forceinline__ SlotIndex slab_alloc_device(SlabDevice const* alloc) {
  for (;;) {
    unsigned long long head = *alloc->free_list_head;
    if (head == kInvalidSlot)
      return kInvalidSlot;
    SlotIndex next = alloc->free_list_next[static_cast<SlotIndex>(head)];
    if (atomicCAS(alloc->free_list_head, head, static_cast<unsigned long long>(next)) == head)
      return static_cast<SlotIndex>(head);
  }
}

/**
 * Return a slot to the free list (device). Thread-safe.
 */
__device__ __forceinline__ void slab_free_device(SlabDevice* alloc, SlotIndex idx) {
  for (;;) {
    unsigned long long old_head = *alloc->free_list_head;
    alloc->free_list_next[idx] = static_cast<SlotIndex>(old_head);
    if (atomicCAS(alloc->free_list_head, old_head, idx) == old_head)
      break;
  }
}

} // namespace gpu_hashmap

#endif // GPU_HASHMAP_SLAB_ALLOCATOR_CUH
