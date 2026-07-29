/**
 * @file slab_hash.cu
 * @brief Slab hash table: create/destroy and warp-cooperative insert (one atomic per warp).
 */

#include "gpu_hashmap/hash_slab.cuh"
#include "gpu_hashmap/insert_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(err));                                  \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

namespace gpu_hashmap {

namespace {

__global__ void init_slab_table_kernel(KeyType* keys, ValueType* values,
                                       unsigned int* used_mask, size_t num_buckets) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = num_buckets * kSlabSize;
  if (i < total) {
    keys[i] = kEmptyKey;
    values[i] = 0;
  }
  if (i < num_buckets)
    used_mask[i] = 0;
}

/** Return a bitmask with the first 'count' zero bits set in ~used (within low K bits). */
__device__ __forceinline__ unsigned int select_first_n_bits(unsigned int used, int count) {
  unsigned int empty = ~used & ((1u << kSlabSize) - 1);
  unsigned int claim = 0u;
  for (int c = 0; c < count && empty; ++c) {
    int pos = __ffs(empty) - 1;
    claim |= (1u << pos);
    empty &= ~(1u << pos);
  }
  return claim;
}

/** Index of the r-th set bit in mask (r in 0..popc(mask)-1). */
__device__ __forceinline__ int rank_to_slot(unsigned int mask, int r) {
  for (int s = 0, count = 0; s < kSlabSize; ++s)
    if (mask & (1u << s)) {
      if (count == r) return s;
      ++count;
    }
  return 0;
}

/**
 * Claim any free slot in bucket `b` with a per-thread CAS and store the pair.
 * Returns false when every slot is taken; there is no overflow bucket, so a false
 * return means the key cannot be stored at all.
 */
__device__ __forceinline__ bool claim_slot_and_store(SlabHashTableDevice const* table,
                                                     size_t b, KeyType key, ValueType value) {
  /* Each losing CAS means some other thread set a bit, and there are only
   * kSlabSize bits, so kSlabSize attempts is enough to distinguish "lost a race"
   * from "genuinely full". */
  for (int attempt = 0; attempt < kSlabSize; ++attempt) {
    unsigned int old = *(volatile unsigned int*)(table->used_mask + b);
    unsigned int empty = ~old & ((1u << kSlabSize) - 1);
    if (empty == 0u) return false;
    int s = __ffs(empty) - 1;
    unsigned int bit = 1u << s;
    if (atomicCAS(table->used_mask + b, old, old | bit) == old) {
      size_t idx = b * kSlabSize + s;
      table->keys[idx] = key;
      table->values[idx] = value;
      return true;
    }
  }
  return false;
}

__device__ __forceinline__ void record_insert_failure(SlabHashTableDevice const* table) {
  if (table->insert_failures)
    atomicAdd(table->insert_failures, 1ull);
}

__global__ void slab_insert_kernel(SlabHashTableDevice const* table,
                                   KeyType const* keys, ValueType const* values, size_t n) {
  const int lane_id = threadIdx.x % 32;
  const unsigned full_mask = 0xFFFFFFFFu;

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  KeyType key = keys[i];
  ValueType value = values[i];
  size_t b = hash_key(key, table->num_buckets);

  /* Warp ballot: which lanes have the same bucket? */
  unsigned same_bucket_mask = 0u;
  for (int L = 0; L < 32; ++L)
    if (__shfl_sync(full_mask, (unsigned)b, L) == (unsigned)b)
      same_bucket_mask |= (1u << L);

  int need_count = __popc(same_bucket_mask);
  int rank = __popc(same_bucket_mask & ((1u << lane_id) - 1));
  int leader_lane = __ffs(same_bucket_mask) - 1;

  /* Leader loads used mask for the bucket. */
  unsigned int used = 0u;
  if (lane_id == leader_lane)
    used = *(volatile unsigned int*)(table->used_mask + b);
  used = __shfl_sync(full_mask, used, leader_lane);

  unsigned int empty_bits = ~used & ((1u << kSlabSize) - 1);
  int available = __popc(empty_bits);

  if (available >= need_count) {
    /* One atomic for the whole warp: claim need_count slots. */
    unsigned int claim_bits = select_first_n_bits(used, need_count);
    unsigned int old_used = used;
    int won = (lane_id == leader_lane) &&
              (atomicCAS(table->used_mask + b, old_used, old_used | claim_bits) == old_used) ? 1 : 0;
    claim_bits = __shfl_sync(full_mask, claim_bits, leader_lane);
    won = __shfl_sync(full_mask, won, leader_lane);
    if (won && (same_bucket_mask & (1u << lane_id))) {
      int slot = rank_to_slot(claim_bits, rank);
      size_t idx = b * kSlabSize + slot;
      table->keys[idx] = key;
      table->values[idx] = value;
    } else if (!won && (same_bucket_mask & (1u << lane_id))) {
      /* Warp-wide claim lost the race; fall back to a per-thread claim. */
      if (!claim_slot_and_store(table, b, key, value))
        record_insert_failure(table);
    }
  } else {
    /* Not enough free slots for the whole group: each thread claims individually. */
    if (!(same_bucket_mask & (1u << lane_id))) return;
    if (!claim_slot_and_store(table, b, key, value))
      record_insert_failure(table);
  }
}

} // namespace

void slab_hash_create(SlabHashTable* table, size_t num_buckets, cudaStream_t stream) {
  if (num_buckets == 0 || (num_buckets & (num_buckets - 1)) != 0) {
    std::fprintf(stderr, "slab_hash_create: num_buckets must be power of 2\n");
    std::abort();
  }
  size_t slab_entries = num_buckets * kSlabSize;
  CUDA_CHECK(cudaMalloc(&table->d_keys, slab_entries * sizeof(KeyType)));
  CUDA_CHECK(cudaMalloc(&table->d_values, slab_entries * sizeof(ValueType)));
  CUDA_CHECK(cudaMalloc(&table->d_used_mask, num_buckets * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&table->d_device_table, sizeof(SlabHashTableDevice)));
  CUDA_CHECK(cudaMalloc(&table->d_insert_failures, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(table->d_insert_failures, 0, sizeof(unsigned long long)));

  table->device.keys = table->d_keys;
  table->device.values = table->d_values;
  table->device.used_mask = table->d_used_mask;
  table->device.num_buckets = num_buckets;
  table->device.insert_failures = table->d_insert_failures;
  table->num_buckets = num_buckets;

  const int block = 256;
  size_t init_size = (slab_entries > num_buckets) ? slab_entries : num_buckets;
  init_slab_table_kernel<<<(init_size + block - 1) / block, block, 0, stream>>>(
      table->d_keys, table->d_values, table->d_used_mask, num_buckets);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaMemcpyAsync(table->d_device_table, &table->device, sizeof(SlabHashTableDevice),
                             cudaMemcpyHostToDevice, stream ? stream : (cudaStream_t)0));
  if (!stream)
    CUDA_CHECK(cudaDeviceSynchronize());
}

void slab_hash_destroy(SlabHashTable* table) {
  if (!table) return;
  if (table->d_device_table) CUDA_CHECK(cudaFree(table->d_device_table));
  if (table->d_used_mask) CUDA_CHECK(cudaFree(table->d_used_mask));
  if (table->d_values) CUDA_CHECK(cudaFree(table->d_values));
  if (table->d_keys) CUDA_CHECK(cudaFree(table->d_keys));
  if (table->d_insert_failures) CUDA_CHECK(cudaFree(table->d_insert_failures));
  table->d_device_table = nullptr;
  table->d_used_mask = nullptr;
  table->d_values = nullptr;
  table->d_keys = nullptr;
  table->d_insert_failures = nullptr;
  table->device.keys = nullptr;
  table->device.values = nullptr;
  table->device.used_mask = nullptr;
  table->device.insert_failures = nullptr;
  table->num_buckets = 0;
}

unsigned long long slab_hash_insert_failure_count(SlabHashTable const* table) {
  if (!table || !table->d_insert_failures) return 0;
  unsigned long long count = 0;
  CUDA_CHECK(cudaMemcpy(&count, table->d_insert_failures, sizeof(count),
                        cudaMemcpyDeviceToHost));
  return count;
}

void slab_hash_reset_insert_failure_count(SlabHashTable* table) {
  if (!table || !table->d_insert_failures) return;
  CUDA_CHECK(cudaMemset(table->d_insert_failures, 0, sizeof(unsigned long long)));
}

void slab_hash_insert_batch(SlabHashTableDevice const* table, KeyType const* keys,
                             ValueType const* values, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  const int block = 256;
  slab_insert_kernel<<<(n + block - 1) / block, block, 0, stream>>>(table, keys, values, n);
}

__global__ void slab_lookup_kernel(SlabHashTableDevice const* table,
                                   KeyType const* keys, ValueType* values, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  KeyType key = keys[i];
  size_t b = hash_key(key, table->num_buckets);
  for (int s = 0; s < kSlabSize; ++s) {
    size_t idx = b * kSlabSize + s;
    if (table->keys[idx] == key) {
      values[i] = table->values[idx];
      return;
    }
  }
  values[i] = 0xFFFFFFFFFFFFFFFFull;
}

} // namespace gpu_hashmap
