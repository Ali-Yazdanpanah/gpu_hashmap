/**
 * @file insert_kernel.cu
 * @brief Insert kernel: allocate node from slab, fill key/value/next, CAS into bucket head.
 */

#include "gpu_hashmap/insert_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                  cudaGetErrorString(err));                                   \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

namespace gpu_hashmap {

__global__ void insert_kernel(HashTableDevice const* table, KeyType const* keys,
                              ValueType const* values, size_t n) {
  /* Per-block batch allocation: one thread pulls a chunk from the global free list,
   * then all threads in the block take a slot from shared memory. Reduces global
   * atomic contention from N to (N / blockDim.x). */
  extern __shared__ SlotIndex block_slots[];

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  if (threadIdx.x == 0) {
    for (int k = 0; k < blockDim.x; ++k)
      block_slots[k] = slab_alloc_device(table->slab);
  }
  __syncthreads();

  SlotIndex slot = block_slots[threadIdx.x];
  if (slot == kInvalidSlot)
    return;

  KeyType key = keys[i];
  ValueType value = values[i];
  size_t b = hash_key(key, table->num_buckets);

  Node* node = &table->nodes[slot];
  node->key = key;
  node->value = value;

  for (;;) {
    unsigned long long old_head = table->bucket_heads[b];
    node->next = static_cast<SlotIndex>(old_head);
    __threadfence();
    if (atomicCAS(&table->bucket_heads[b], old_head, slot) == old_head)
      break;
  }
}

namespace {

__device__ __forceinline__ unsigned long long shfl_sync_ull(unsigned long long val, int src_lane) {
  unsigned lo = (unsigned)(val & 0xFFFFFFFFu);
  unsigned hi = (unsigned)(val >> 32);
  lo = __shfl_sync(0xFFFFFFFFu, lo, src_lane);
  hi = __shfl_sync(0xFFFFFFFFu, hi, src_lane);
  return (unsigned long long)lo | ((unsigned long long)hi << 32);
}

} // namespace

__global__ void insert_kernel_warp_aggregated(HashTableDevice const* table,
                                              KeyType const* keys,
                                              ValueType const* values, size_t n) {
  extern __shared__ SlotIndex block_slots[];
  const int lane_id = threadIdx.x % 32;
  const unsigned full_mask = 0xFFFFFFFFu;

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  if (threadIdx.x == 0) {
    for (int k = 0; k < blockDim.x; ++k)
      block_slots[k] = slab_alloc_device(table->slab);
  }
  __syncthreads();

  SlotIndex slot = block_slots[threadIdx.x];
  if (slot == kInvalidSlot)
    return;

  KeyType key = keys[i];
  ValueType value = values[i];
  size_t b = hash_key(key, table->num_buckets);

  Node* node = &table->nodes[slot];
  node->key = key;
  node->value = value;

  /* Find lanes in this warp that have the same bucket. */
  unsigned mask = 0u;
  for (int L = 0; L < 32; ++L)
    if (__shfl_sync(full_mask, (unsigned)b, L) == (unsigned)b)
      mask |= (1u << L);
  if (mask == 0u) return;

  int leader_lane = __ffs(mask) - 1;
  unsigned long long current_head = 0ull;
  if (lane_id == leader_lane)
    current_head = table->bucket_heads[b];
  current_head = shfl_sync_ull(current_head, leader_lane);

  /* Serialize inserts within the warp: each lane in the group does CAS in order, then broadcasts new head. */
  for (int lane = 0; lane < 32; ++lane) {
    if (!(mask & (1u << lane))) continue;
    if (lane_id == lane) {
      unsigned long long old_head = current_head;
      node->next = static_cast<SlotIndex>(old_head);
      __threadfence();
      atomicCAS(&table->bucket_heads[b], old_head, slot);
      current_head = slot;
    }
    current_head = shfl_sync_ull(current_head, lane);
  }
}

void insert_batch(HashTableDevice const* table, KeyType const* keys,
                  ValueType const* values, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  const int block = 256;
  size_t shared_bytes = block * sizeof(SlotIndex);
  insert_kernel<<<(n + block - 1) / block, block, shared_bytes, stream>>>(
      table, keys, values, n);
}

void insert_batch_warp_aggregated(HashTableDevice const* table, KeyType const* keys,
                                  ValueType const* values, size_t n,
                                  cudaStream_t stream) {
  if (n == 0) return;
  const int block = 256;
  size_t shared_bytes = block * sizeof(SlotIndex);
  insert_kernel_warp_aggregated<<<(n + block - 1) / block, block, shared_bytes, stream>>>(
      table, keys, values, n);
}

} // namespace gpu_hashmap
