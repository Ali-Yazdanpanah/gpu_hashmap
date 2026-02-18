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

} // namespace gpu_hashmap

#endif // GPU_HASHMAP_TYPES_CUH
