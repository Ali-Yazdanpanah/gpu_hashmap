/**
 * @file types.cuh
 * @brief Fundamental types for the GPU hash map (keys, values, slab indices).
 */

#ifndef GPU_HASHMAP_TYPES_CUH
#define GPU_HASHMAP_TYPES_CUH

#include <cstdint>
#include <cstddef>

namespace gpu_hashmap {

using KeyType   = uint64_t;
using ValueType = uint64_t;

/** Index into the slab (node array). Use uint32_t for 4B; use uint64_t for >4G entries. */
using SlotIndex = uint32_t;

/** Sentinel for "no next node" and "invalid slot". */
constexpr SlotIndex kInvalidSlot = 0xFFFFFFFFu;

/** Maximum capacity when using SlotIndex = uint32_t. */
constexpr size_t kMaxSlots = 0xFFFFFFFFu;

/**
 * Where a table's bulk storage is allocated.
 *
 * This was previously fixed per table type -- the chained table always in mapped host
 * memory, the slab table always in device memory -- which fused two effects that any
 * comparison between them then could not separate: whether one scheme is better, and
 * whether reaching the table over PCIe is worse than reaching it in VRAM. Both tables
 * now accept a placement so each effect can be measured on its own.
 */
enum class TablePlacement {
  kMappedHost,  ///< cudaHostAlloc(Mapped): host resident, GPU reaches it over PCIe
  kDevice,      ///< cudaMalloc: device resident, no host-side view
};

} // namespace gpu_hashmap

#endif // GPU_HASHMAP_TYPES_CUH
