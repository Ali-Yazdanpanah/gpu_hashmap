/**
 * @file hash_buckets.cuh
 * @brief Hash bucket and chain node structures for chained hashing on GPU.
 *
 * Each bucket holds an atomic head index into the slab. Nodes are slab slots
 * containing (key, value, next). Lock-free reads: traverse chain without
 * acquiring locks. Atomic updates: CAS on bucket head and node next pointers.
 */

#ifndef GPU_HASHMAP_HASH_BUCKETS_CUH
#define GPU_HASHMAP_HASH_BUCKETS_CUH

#include "types.cuh"
#include "slab_allocator.cuh"
#include <cstddef>

namespace gpu_hashmap {

/**
 * One node in a bucket chain. Stored in slab; next is slab index (kInvalidSlot = end).
 */
struct Node {
  KeyType key;
  ValueType value;
  SlotIndex next;
};

/**
 * Device view of the hash table: array of bucket heads (atomic), plus
 * pointers to the node storage (slab) and slab allocator for new nodes.
 */
struct HashTableDevice {
  unsigned long long* bucket_heads;  ///< atomic: per-bucket head slot index
  Node* nodes;                      ///< node array (slab storage); index = SlotIndex
  SlabDevice const* slab;           ///< for allocating new nodes
  size_t num_buckets;               ///< number of buckets (power of two for fast modulo)
  size_t capacity;                  ///< max nodes (slab capacity)
  /** Atomic counter of inserts that were abandoned (slab exhausted or CAS gave up).
   *  Without this an over-full table silently loses data and still reports fast times. */
  unsigned long long* insert_failures;
};

/**
 * Host-owned hash table. When using zero-copy (mapped memory):
 * - h_bucket_heads / h_nodes: host-side of cudaHostAlloc(Mapped); GPU accesses via d_*.
 * - d_bucket_heads / d_nodes: device pointers from cudaHostGetDevicePointer (for kernels).
 * - d_device_table is a device copy of the descriptor; pass to insert/lookup kernels.
 *
 * Under TablePlacement::kDevice the h_* pointers are null: the table has no host-side
 * view, so anything that walks it from the CPU (hash_map_upload_from_host, a manual
 * full-table migration) requires kMappedHost.
 */
struct HashTable {
  HashTableDevice device;           ///< host copy (device.bucket_heads/nodes = d_* for kernels)
  HashTableDevice* d_device_table;  ///< device pointer: pass to insert/lookup kernels
  void* h_bucket_heads;            ///< host pointer (mapped alloc); null if not using zero-copy
  void* h_nodes;                    ///< host pointer (mapped alloc)
  void* d_bucket_heads;             ///< device pointer (for kernels; may be from cudaHostGetDevicePointer)
  void* d_nodes;                    ///< device pointer (for kernels)
  SlabDevice* d_slab_device;        ///< device copy of SlabDevice (freed in destroy)
  unsigned long long* d_insert_failures;  ///< device counter behind device.insert_failures
  SlabAllocator* slab;
  size_t num_buckets;
  size_t capacity;
  TablePlacement placement;         ///< where bucket_heads/nodes live; set by hash_map_create

  /* Reusable staging buffers for the standard-copy lookup path. Kept across calls
   * so that allocation never lands inside a timed region. Grown on demand. */
  KeyType* d_scratch_keys;
  ValueType* d_scratch_values;
  size_t scratch_capacity;          ///< elements each scratch buffer can hold
};

} // namespace gpu_hashmap

#endif // GPU_HASHMAP_HASH_BUCKETS_CUH
