/**
 * @file lookup_kernel.cuh
 * @brief Lookup kernel declaration for chained hash table (shared by API and heuristic).
 */

#ifndef GPU_HASHMAP_LOOKUP_KERNEL_CUH
#define GPU_HASHMAP_LOOKUP_KERNEL_CUH

#include "gpu_hashmap/hash_buckets.cuh"
#include <cstddef>
#include <cuda_runtime.h>

namespace gpu_hashmap {

/** Device kernel: batch lookup; writes value or 0xFFFFFFFFFFFFFFFFull if not found. */
__global__ void hash_map_lookup_kernel(HashTableDevice const* table,
                                       KeyType const* keys,
                                       ValueType* values, size_t n);

} // namespace gpu_hashmap

#endif // GPU_HASHMAP_LOOKUP_KERNEL_CUH
