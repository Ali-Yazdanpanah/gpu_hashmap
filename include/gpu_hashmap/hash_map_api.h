/**
 * @file hash_map_api.h
 * @brief Host API: create/destroy hash table, insert batch, lookup (to be extended).
 */

#ifndef GPU_HASHMAP_HASH_MAP_API_H
#define GPU_HASHMAP_HASH_MAP_API_H

#include "gpu_hashmap/hash_buckets.cuh"
#include "gpu_hashmap/slab_allocator.cuh"
#include <cstddef>
#include <cuda_runtime.h>

namespace gpu_hashmap {

/** Create hash table: num_buckets (power of 2), capacity = max entries (slab size). */
void hash_map_create(HashTable* table, size_t num_buckets, size_t capacity,
                     cudaStream_t stream = nullptr);

void hash_map_destroy(HashTable* table);

/** Insert n (key, value) pairs from device arrays. */
void hash_map_insert_batch(HashTable* table, KeyType const* d_keys,
                           ValueType const* d_values, size_t n,
                           cudaStream_t stream = nullptr);

/** Insert using warp aggregation (fewer atomics, less contention). */
void hash_map_insert_batch_warp_aggregated(HashTable* table, KeyType const* d_keys,
                                           ValueType const* d_values, size_t n,
                                           cudaStream_t stream = nullptr);

/**
 * Build the table on the host from (key, value) pairs and upload to device.
 * Use for hybrid: insert on CPU, then GPU lookup only. n must be <= capacity.
 */
void hash_map_upload_from_host(HashTable* table, KeyType const* h_keys,
                               ValueType const* h_values, size_t n,
                               cudaStream_t stream = nullptr);

/** Lookup batch: host keys in, host results out. Standard path (cudaMemcpy H2D/D2H). */
void hash_map_lookup_batch_standard_copy(HashTable* table, KeyType const* h_keys,
                                         ValueType* h_results, size_t n,
                                         cudaStream_t stream = nullptr);

/**
 * Lookup batch: True Zero-Copy (in-place mapping).
 * Caller must provide keys and results in pinned memory (cudaHostAlloc with
 * cudaHostAllocMapped | cudaHostAllocPortable). The kernel writes results
 * directly into the mapped buffer; PCIe cache coherency allows the CPU to
 * read GPU output without any explicit memory migration. No std::memcpy.
 */
void hash_map_lookup_batch_zero_copy(HashTable* table, KeyType* h_pinned_keys,
                                     ValueType* h_pinned_results, size_t n,
                                     cudaStream_t stream = nullptr);

} // namespace gpu_hashmap

#endif // GPU_HASHMAP_HASH_MAP_API_H
