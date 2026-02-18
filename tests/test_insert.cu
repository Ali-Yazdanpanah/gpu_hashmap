/**
 * @file test_insert.cu
 * @brief Test insert batch and verify a few lookups (read-only traverse).
 */

#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/insert_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                  cudaGetErrorString(err));                                   \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

namespace {

__global__ void lookup_kernel(gpu_hashmap::HashTableDevice const* table,
                              gpu_hashmap::KeyType const* keys,
                              gpu_hashmap::ValueType* values, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  gpu_hashmap::KeyType key = keys[i];
  size_t b = gpu_hashmap::hash_key(key, table->num_buckets);
  unsigned long long head = table->bucket_heads[b];
  while (head != gpu_hashmap::kInvalidSlot) {
    gpu_hashmap::Node const* node = &table->nodes[head];
    if (node->key == key) {
      values[i] = node->value;
      return;
    }
    head = node->next;
  }
  values[i] = 0xFFFFFFFFFFFFFFFFull;
}

} // namespace

int main() {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  const size_t num_buckets = 1024;
  const size_t capacity = 65536;
  const size_t n = 10000;

  gpu_hashmap::HashTable table = {};
  gpu_hashmap::hash_map_create(&table, num_buckets, capacity);

  std::vector<gpu_hashmap::KeyType> h_keys(n);
  std::vector<gpu_hashmap::ValueType> h_values(n);
  for (size_t i = 0; i < n; ++i) {
    h_keys[i] = i * 2;
    h_values[i] = i + 1000;
  }

  gpu_hashmap::KeyType* d_keys = nullptr;
  gpu_hashmap::ValueType* d_values = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), n * sizeof(gpu_hashmap::ValueType), cudaMemcpyHostToDevice));

  gpu_hashmap::hash_map_insert_batch(&table, d_keys, d_values, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<gpu_hashmap::KeyType> h_lookup_keys = { 0, 2, 100, 9999*2, 99999 };
  size_t n_lookup = h_lookup_keys.size();
  gpu_hashmap::ValueType* d_lookup_out = nullptr;
  gpu_hashmap::KeyType* d_lookup_keys = nullptr;
  CUDA_CHECK(cudaMalloc(&d_lookup_keys, n_lookup * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_lookup_out, n_lookup * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMemcpy(d_lookup_keys, h_lookup_keys.data(), n_lookup * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));

  lookup_kernel<<<1, 256>>>(table.d_device_table, d_lookup_keys, d_lookup_out, n_lookup);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<gpu_hashmap::ValueType> h_out(n_lookup);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_lookup_out, n_lookup * sizeof(gpu_hashmap::ValueType), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (size_t i = 0; i < n_lookup; ++i) {
    gpu_hashmap::KeyType k = h_lookup_keys[i];
    gpu_hashmap::ValueType expected = (k % 2 == 0 && k/2 < n) ? (k/2 + 1000) : 0xFFFFFFFFFFFFFFFFull;
    if (h_out[i] != expected) {
      std::fprintf(stderr, "lookup key %llu: got %llu expected %llu\n",
                   (unsigned long long)k, (unsigned long long)h_out[i], (unsigned long long)expected);
      ok = false;
    }
  }

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_lookup_keys));
  CUDA_CHECK(cudaFree(d_lookup_out));
  gpu_hashmap::hash_map_destroy(&table);

  if (ok)
    std::printf("PASS\n");
  else
    std::printf("FAIL\n");
  return ok ? 0 : 1;
}
