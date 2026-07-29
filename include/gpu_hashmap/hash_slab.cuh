/**
 * @file hash_slab.cuh
 * @brief Bucketed table with drop-on-overflow: each bucket is a slab of K key-value pairs.
 * A warp cooperatively searches for empty slots and performs one atomic per warp to claim
 * them. This is neither open addressing nor chaining: a key whose bucket is full is not
 * probed elsewhere and gets no overflow node, it is discarded (see slab_hash_insert_failure_count).
 */

#ifndef GPU_HASHMAP_HASH_SLAB_CUH
#define GPU_HASHMAP_HASH_SLAB_CUH

#include "types.cuh"
#include <cstddef>
#include <cuda_runtime.h>

namespace gpu_hashmap {

/** Slab size: 8 pairs = 128 bytes (two cache lines). Fits 4 for 64 bytes. */
constexpr int kSlabSize = 8;

/** Sentinel key for empty slot. */
constexpr KeyType kEmptyKey = 0xFFFFFFFFFFFFFFFFull;

/**
 * Device view: each bucket has kSlabSize slots. keys[bucket*K + slot], values[bucket*K + slot].
 * used_mask[bucket] = bits 0..K-1: 1 = slot used. One atomic per bucket for claiming.
 */
struct SlabHashTableDevice {
  KeyType* keys;                    ///< keys[bucket * kSlabSize + slot]
  ValueType* values;                ///< values[bucket * kSlabSize + slot]
  unsigned int* used_mask;         ///< atomic per bucket; bit i = slot i used
  size_t num_buckets;
  /** Atomic counter of inserts dropped because the target bucket was full.
   *  There is no overflow bucket, so without this the loss is invisible. */
  unsigned long long* insert_failures;
};

/** Host-owned slab hash table. */
struct SlabHashTable {
  SlabHashTableDevice device;
  KeyType* d_keys;
  ValueType* d_values;
  unsigned int* d_used_mask;
  unsigned long long* d_insert_failures;
  SlabHashTableDevice* d_device_table;
  size_t num_buckets;
  /* Host side of the mapped allocations; null under kDevice placement. Retained only
   * so destroy can release them with cudaFreeHost. */
  void* h_keys;
  void* h_values;
  void* h_used_mask;
  /** Where keys/values/used_mask live. Device by default; kMappedHost exists so the
   *  slab scheme can be measured under the same PCIe residency as the chained table,
   *  which is what separates "better scheme" from "better placement". */
  TablePlacement placement;
};

void slab_hash_create(SlabHashTable* table, size_t num_buckets, cudaStream_t stream = nullptr,
                      TablePlacement placement = TablePlacement::kDevice);
void slab_hash_destroy(SlabHashTable* table);

/**
 * Number of inserts dropped because the target bucket had no free slot. Each bucket
 * holds exactly kSlabSize entries and there is no overflow chain or probe to a
 * neighbouring bucket, so a non-zero count means those keys are absent from the
 * table and will not be found by lookup. Synchronizing read.
 */
unsigned long long slab_hash_insert_failure_count(SlabHashTable const* table);

/** Zero the insert-failure counter (call before a batch you want to audit). */
void slab_hash_reset_insert_failure_count(SlabHashTable* table);

/** Insert batch: warp-cooperative, one atomic per warp when claiming slots. */
void slab_hash_insert_batch(SlabHashTableDevice const* table, KeyType const* keys,
                             ValueType const* values, size_t n, cudaStream_t stream = nullptr);

/** Lookup: linear scan of the slab for the key. */
__global__ void slab_lookup_kernel(SlabHashTableDevice const* table,
                                   KeyType const* keys, ValueType* values, size_t n);

} // namespace gpu_hashmap

#endif // GPU_HASHMAP_HASH_SLAB_CUH
