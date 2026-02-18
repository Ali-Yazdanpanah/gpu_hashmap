/**
 * @file test_slab.cu
 * @brief Test slab hash table: insert batch + lookup (bucketed linear probing).
 */

#include "gpu_hashmap/hash_slab.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
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

int main() {
  const size_t num_buckets = 1024;
  const size_t n = 4096;

  gpu_hashmap::SlabHashTable table = {};
  gpu_hashmap::slab_hash_create(&table, num_buckets);

  std::vector<gpu_hashmap::KeyType> h_keys(n), h_lookup_keys(n);
  std::vector<gpu_hashmap::ValueType> h_values(n), h_out(n);
  for (size_t i = 0; i < n; ++i) {
    h_keys[i] = i * 2;
    h_values[i] = i + 1000;
    h_lookup_keys[i] = i * 2;
  }

  gpu_hashmap::KeyType* d_keys = nullptr;
  gpu_hashmap::ValueType* d_values = nullptr;
  gpu_hashmap::KeyType* d_lookup_keys = nullptr;
  gpu_hashmap::ValueType* d_lookup_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMalloc(&d_lookup_keys, n * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_lookup_out, n * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), n * sizeof(gpu_hashmap::ValueType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lookup_keys, h_lookup_keys.data(), n * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));

  gpu_hashmap::slab_hash_insert_batch(table.d_device_table, d_keys, d_values, n);
  gpu_hashmap::slab_lookup_kernel<<<(n + 255) / 256, 256>>>(table.d_device_table, d_lookup_keys, d_lookup_out, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_lookup_out, n * sizeof(gpu_hashmap::ValueType), cudaMemcpyDeviceToHost));

  int errors = 0;
  for (size_t i = 0; i < n; ++i)
    if (h_out[i] != h_values[i]) ++errors;

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_lookup_keys));
  CUDA_CHECK(cudaFree(d_lookup_out));
  gpu_hashmap::slab_hash_destroy(&table);

  if (errors == 0)
    std::printf("PASS (slab hash)\n");
  else
    std::printf("FAIL (slab hash): %d errors\n", errors);
  return errors ? 1 : 0;
}
