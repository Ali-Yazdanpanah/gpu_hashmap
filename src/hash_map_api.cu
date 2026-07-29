/**
 * @file hash_map_api.cu
 * @brief Hash table create/destroy and insert batch implementation.
 */

#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/insert_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

__global__ void init_bucket_heads_kernel(unsigned long long* heads, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    heads[i] = kInvalidSlot;
}

} // namespace

void hash_map_create(HashTable* table, size_t num_buckets, size_t capacity,
                    cudaStream_t stream) {
  if (num_buckets == 0 || (num_buckets & (num_buckets - 1)) != 0) {
    std::fprintf(stderr, "hash_map_create: num_buckets must be power of 2\n");
    std::abort();
  }
  if (capacity == 0 || capacity > kMaxSlots) {
    std::fprintf(stderr, "hash_map_create: invalid capacity %zu\n", capacity);
    std::abort();
  }

  table->slab = new SlabAllocator();
  slab_init(table->slab, capacity, stream);

  table->num_buckets = num_buckets;
  table->capacity = capacity;
  table->d_scratch_keys = nullptr;
  table->d_scratch_values = nullptr;
  table->scratch_capacity = 0;

  /* Zero-copy: mapped host memory so GPU accesses over PCIe; portable for multi-GPU. */
  const unsigned int flags = cudaHostAllocMapped | cudaHostAllocPortable;
  CUDA_CHECK(cudaHostAlloc(&table->h_bucket_heads, num_buckets * sizeof(unsigned long long), flags));
  CUDA_CHECK(cudaHostAlloc(&table->h_nodes, capacity * sizeof(Node), flags));
  CUDA_CHECK(cudaHostGetDevicePointer(&table->d_bucket_heads, table->h_bucket_heads, 0));
  CUDA_CHECK(cudaHostGetDevicePointer(&table->d_nodes, table->h_nodes, 0));

  table->device.bucket_heads = static_cast<unsigned long long*>(table->d_bucket_heads);
  table->device.nodes = static_cast<Node*>(table->d_nodes);
  table->device.num_buckets = num_buckets;
  table->device.capacity = capacity;

  const int block = 256;
  init_bucket_heads_kernel<<<(num_buckets + block - 1) / block, block, 0, stream>>>(
      table->device.bucket_heads, num_buckets);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaMalloc(&table->d_insert_failures, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(table->d_insert_failures, 0, sizeof(unsigned long long)));
  table->device.insert_failures = table->d_insert_failures;

  /* Slab and descriptor stay in device memory (small, high traffic). */
  CUDA_CHECK(cudaMalloc(&table->d_slab_device, sizeof(SlabDevice)));
  CUDA_CHECK(cudaMemcpy(table->d_slab_device, &table->slab->device, sizeof(SlabDevice),
                        cudaMemcpyHostToDevice));
  table->device.slab = table->d_slab_device;

  CUDA_CHECK(cudaMalloc(&table->d_device_table, sizeof(HashTableDevice)));
  CUDA_CHECK(cudaMemcpy(table->d_device_table, &table->device, sizeof(HashTableDevice),
                        cudaMemcpyHostToDevice));
}

void hash_map_destroy(HashTable* table) {
  if (!table) return;
  if (table->d_device_table)
    CUDA_CHECK(cudaFree(table->d_device_table));
  if (table->d_slab_device)
    CUDA_CHECK(cudaFree(table->d_slab_device));
  if (table->h_bucket_heads)
    CUDA_CHECK(cudaFreeHost(table->h_bucket_heads));
  if (table->h_nodes)
    CUDA_CHECK(cudaFreeHost(table->h_nodes));
  if (table->d_scratch_keys)
    CUDA_CHECK(cudaFree(table->d_scratch_keys));
  if (table->d_scratch_values)
    CUDA_CHECK(cudaFree(table->d_scratch_values));
  if (table->d_insert_failures)
    CUDA_CHECK(cudaFree(table->d_insert_failures));
  if (table->slab) {
    slab_destroy(table->slab);
    delete table->slab;
    table->slab = nullptr;
  }
  table->d_scratch_keys = nullptr;
  table->d_scratch_values = nullptr;
  table->scratch_capacity = 0;
  table->d_insert_failures = nullptr;
  table->device.insert_failures = nullptr;
  table->d_device_table = nullptr;
  table->d_slab_device = nullptr;
  table->h_bucket_heads = nullptr;
  table->h_nodes = nullptr;
  table->d_bucket_heads = nullptr;
  table->d_nodes = nullptr;
  table->device.bucket_heads = nullptr;
  table->device.nodes = nullptr;
  table->device.slab = nullptr;
  table->num_buckets = 0;
  table->capacity = 0;
}

void hash_map_insert_batch(HashTable* table, KeyType const* d_keys,
                           ValueType const* d_values, size_t n,
                           cudaStream_t stream) {
  insert_batch(table->d_device_table, d_keys, d_values, n, stream);
}

void hash_map_insert_batch_warp_aggregated(HashTable* table, KeyType const* d_keys,
                                           ValueType const* d_values, size_t n,
                                           cudaStream_t stream) {
  insert_batch_warp_aggregated(table->d_device_table, d_keys, d_values, n, stream);
}

unsigned long long hash_map_insert_failure_count(HashTable const* table) {
  if (!table || !table->d_insert_failures) return 0;
  unsigned long long count = 0;
  CUDA_CHECK(cudaMemcpy(&count, table->d_insert_failures, sizeof(count),
                        cudaMemcpyDeviceToHost));
  return count;
}

void hash_map_reset_insert_failure_count(HashTable* table) {
  if (!table || !table->d_insert_failures) return;
  CUDA_CHECK(cudaMemset(table->d_insert_failures, 0, sizeof(unsigned long long)));
}

void hash_map_upload_from_host(HashTable* table, KeyType const* h_keys,
                               ValueType const* h_values, size_t n,
                               cudaStream_t stream) {
  if (n == 0 || n > table->capacity) {
    std::fprintf(stderr, "hash_map_upload_from_host: n %zu invalid (capacity %zu)\n",
                 n, table->capacity);
    std::abort();
  }
  const size_t num_buckets = table->num_buckets;
  std::vector<unsigned long long> h_bucket_heads(num_buckets, kInvalidSlot);
  std::vector<Node> h_nodes(n);
  for (size_t i = 0; i < n; ++i) {
    KeyType key = h_keys[i];
    ValueType value = h_values[i];
    size_t b = hash_key(key, num_buckets);
    SlotIndex slot = static_cast<SlotIndex>(i);
    h_nodes[i].key = key;
    h_nodes[i].value = value;
    h_nodes[i].next = static_cast<SlotIndex>(h_bucket_heads[b]);
    h_bucket_heads[b] = slot;
  }
  /* Zero-copy: write directly to mapped host memory; GPU reads over PCIe. No cudaMemcpy. */
  std::memcpy(table->h_bucket_heads, h_bucket_heads.data(),
              num_buckets * sizeof(unsigned long long));
  std::memcpy(table->h_nodes, h_nodes.data(), n * sizeof(Node));
}

} // namespace gpu_hashmap
