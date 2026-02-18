/**
 * @file probe_depth.cuh
 * @brief Lookup kernel that records probe depth (number of chain steps) per query.
 */

#ifndef GPU_HASHMAP_ANALYSIS_PROBE_DEPTH_CUH
#define GPU_HASHMAP_ANALYSIS_PROBE_DEPTH_CUH

#include "../hash_buckets.cuh"
#include "../insert_kernel.cuh"
#include <cstddef>
#include <cuda_runtime.h>

namespace gpu_hashmap {
namespace analysis {

/** Lookup and write value + probe depth (number of nodes traversed). */
__global__ void lookup_with_probe_depth(HashTableDevice const* table,
                                        KeyType const* keys,
                                        ValueType* values,
                                        unsigned int* probe_depths,
                                        size_t n);

} // namespace analysis
} // namespace gpu_hashmap

#endif // GPU_HASHMAP_ANALYSIS_PROBE_DEPTH_CUH
