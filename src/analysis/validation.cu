/**
 * @file validation.cu
 * @brief Gold-standard CPU verification for GPU hash map results.
 */

#include "gpu_hashmap/analysis/validation.hpp"
#include "gpu_hashmap/insert_kernel.cuh"
#include <cuda_runtime.h>
#include <unordered_map>
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
namespace analysis {

namespace {

__global__ void lookup_kernel(HashTableDevice const* table, KeyType const* keys,
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

} // namespace

bool validate_against_cpu_gold(
    const std::vector<KeyType>& h_keys,
    const std::vector<ValueType>& h_values,
    const std::vector<KeyType>& h_lookup_keys,
    size_t num_buckets, size_t capacity,
    size_t* out_first_mismatch_index,
    ValueType* out_expected,
    ValueType* out_got) {
  const size_t n = h_keys.size();
  const size_t m = h_lookup_keys.size();

  std::unordered_map<uint64_t, uint64_t> gold;
  gold.reserve(n);
  for (size_t i = 0; i < n; ++i)
    gold[h_keys[i]] = h_values[i];

  std::vector<ValueType> gold_results(m);
  for (size_t i = 0; i < m; ++i) {
    auto it = gold.find(h_lookup_keys[i]);
    gold_results[i] = (it != gold.end()) ? it->second : 0xFFFFFFFFFFFFFFFFull;
  }

  HashTable table = {};
  hash_map_create(&table, num_buckets, capacity);
  KeyType* d_keys = nullptr;
  ValueType* d_values = nullptr;
  KeyType* d_lookup_keys = nullptr;
  ValueType* d_lookup_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(KeyType)));
  CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(ValueType)));
  CUDA_CHECK(cudaMalloc(&d_lookup_keys, m * sizeof(KeyType)));
  CUDA_CHECK(cudaMalloc(&d_lookup_out, m * sizeof(ValueType)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(KeyType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), n * sizeof(ValueType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lookup_keys, h_lookup_keys.data(), m * sizeof(KeyType), cudaMemcpyHostToDevice));

  hash_map_insert_batch(&table, d_keys, d_values, n);
  lookup_kernel<<<(m + 255) / 256, 256>>>(table.d_device_table, d_lookup_keys, d_lookup_out, m);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<ValueType> gpu_results(m);
  CUDA_CHECK(cudaMemcpy(gpu_results.data(), d_lookup_out, m * sizeof(ValueType), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_lookup_keys));
  CUDA_CHECK(cudaFree(d_lookup_out));
  hash_map_destroy(&table);

  for (size_t i = 0; i < m; ++i) {
    if (gpu_results[i] != gold_results[i]) {
      if (out_first_mismatch_index) *out_first_mismatch_index = i;
      if (out_expected) *out_expected = gold_results[i];
      if (out_got) *out_got = gpu_results[i];
      return false;
    }
  }
  return true;
}

} // namespace analysis
} // namespace gpu_hashmap
