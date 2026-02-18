/**
 * @file insert_kernel.cuh
 * @brief Insertion kernel: parallel inserts with atomic updates, lock-free chain head.
 *
 * Each thread (or each element in a batch) inserts one key-value pair:
 * 1. Hash key to bucket index.
 * 2. Allocate a node from the slab (atomic pop).
 * 3. Fill node (key, value, next = current head).
 * 4. Atomic CAS to make the new node the bucket head (lock-free).
 * Duplicate keys are allowed (multiple entries in chain); lookup returns first match.
 */

#ifndef GPU_HASHMAP_INSERT_KERNEL_CUH
#define GPU_HASHMAP_INSERT_KERNEL_CUH

#include "hash_buckets.cuh"
#include <cstddef>
#include <cuda_runtime.h>

namespace gpu_hashmap {

/**
 * Insert a batch of (key, value) pairs into the hash table in parallel.
 * Uses atomic slab allocation and atomic CAS on bucket heads. No locks.
 *
 * @param table    Device view of the hash table (bucket_heads, nodes, slab).
 * @param keys     Device array of keys (length n).
 * @param values   Device array of values (length n).
 * @param n        Number of pairs to insert.
 */
void insert_batch(HashTableDevice const* table, KeyType const* keys,
                  ValueType const* values, size_t n, cudaStream_t stream = nullptr);

/** Same as insert_batch but uses warp aggregation to reduce atomic contention. */
void insert_batch_warp_aggregated(HashTableDevice const* table, KeyType const* keys,
                                  ValueType const* values, size_t n,
                                  cudaStream_t stream = nullptr);

/**
 * Device kernel: each thread inserts one (key, value) if thread index < n.
 */
__global__ void insert_kernel(HashTableDevice const* table, KeyType const* keys,
                              ValueType const* values, size_t n);

/**
 * Warp-aggregated insert: same-bucket threads in a warp elect a leader; only the
 * leader does atomicCAS; result is broadcast via __shfl_sync. Reduces global memory pressure.
 */
__global__ void insert_kernel_warp_aggregated(HashTableDevice const* table,
                                              KeyType const* keys,
                                              ValueType const* values, size_t n);

/** Simple hash: multiply-shift for power-of-two num_buckets. */
__device__ __host__ inline size_t hash_key(KeyType key, size_t num_buckets) {
  return static_cast<size_t>(key * 0x9e3779b97f4a7c15ull) & (num_buckets - 1);
}

} // namespace gpu_hashmap

#endif // GPU_HASHMAP_INSERT_KERNEL_CUH
