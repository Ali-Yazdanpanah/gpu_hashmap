/**
 * @file lookup_api.cu
 * @brief Lookup kernel and Standard Copy / Zero-Copy path implementations.
 */

#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/insert_kernel.cuh"
#include "gpu_hashmap/lookup_kernel.cuh"
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

__global__ void hash_map_lookup_kernel(HashTableDevice const* table,
                                       KeyType const* keys,
                                       ValueType* values, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  KeyType key = keys[i];
  size_t b = hash_key(key, table->num_buckets);
  unsigned long long head = table->bucket_heads[b];
  while (head != kInvalidSlot) {
    Node const* node = &table->nodes[head];
    if (node->key == key) {
      values[i] = node->value;
      return;
    }
    head = node->next;
  }
  values[i] = 0xFFFFFFFFFFFFFFFFull;
}

namespace {

constexpr int kLookupBlock = 256;

inline void launch_lookup(HashTableDevice const* d_table, KeyType const* d_keys,
                         ValueType* d_values, size_t n, cudaStream_t stream) {
  hash_map_lookup_kernel<<<(n + kLookupBlock - 1) / kLookupBlock, kLookupBlock, 0, stream>>>(
      d_table, d_keys, d_values, n);
}

} // namespace

void hash_map_reserve_lookup_scratch(HashTable* table, size_t n) {
  if (!table || n <= table->scratch_capacity) return;
  if (table->d_scratch_keys) CUDA_CHECK(cudaFree(table->d_scratch_keys));
  if (table->d_scratch_values) CUDA_CHECK(cudaFree(table->d_scratch_values));
  table->d_scratch_keys = nullptr;
  table->d_scratch_values = nullptr;
  table->scratch_capacity = 0;
  CUDA_CHECK(cudaMalloc(&table->d_scratch_keys, n * sizeof(KeyType)));
  CUDA_CHECK(cudaMalloc(&table->d_scratch_values, n * sizeof(ValueType)));
  table->scratch_capacity = n;
}

void hash_map_lookup_batch_standard_copy(HashTable* table,
                                         KeyType const* h_keys,
                                         ValueType* h_results,
                                         size_t n,
                                         cudaStream_t stream) {
  if (n == 0) return;
  cudaStream_t s = stream ? stream : (cudaStream_t)0;
  hash_map_reserve_lookup_scratch(table, n);
  KeyType* d_keys = table->d_scratch_keys;
  ValueType* d_values = table->d_scratch_values;
  CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys, n * sizeof(KeyType), cudaMemcpyHostToDevice, s));
  launch_lookup(table->d_device_table, d_keys, d_values, n, s);
  CUDA_CHECK(cudaMemcpyAsync(h_results, d_values, n * sizeof(ValueType), cudaMemcpyDeviceToHost, s));
  if (!stream)
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* True Zero-Copy: caller provides pinned (mapped) buffers; kernel writes results
   in place. PCIe cache coherency lets the CPU read GPU output without explicit
   migration—no std::memcpy. Fair comparison: Standard path = Explicit Copying,
   Zero-Copy path = In-place Mapping. */
void hash_map_lookup_batch_zero_copy(HashTable* table,
                                     KeyType* h_pinned_keys,
                                     ValueType* h_pinned_results,
                                     size_t n,
                                     cudaStream_t stream) {
  if (n == 0) return;
  KeyType* d_keys = nullptr;
  ValueType* d_vals = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&d_keys, h_pinned_keys, 0));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_vals, h_pinned_results, 0));
  cudaStream_t s = stream ? stream : (cudaStream_t)0;
  launch_lookup(table->d_device_table, d_keys, d_vals, n, s);
  if (!stream)
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gpu_hashmap
