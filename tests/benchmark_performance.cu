/**
 * @file benchmark_performance.cu
 * @brief Compares GPU (parallel) hash map vs CPU single-threaded (non-parallel).
 * GPU uses per-block batch allocation (less allocator contention) and amortized
 * setup: table and data stay on device across multiple insert/lookup batches.
 */

#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/insert_kernel.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
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

/* Soak transfer cost: create table and copy data once, run many batches, destroy once. */
constexpr int N_INSERT_BATCHES = 5;
constexpr int N_LOOKUP_BATCHES = 5;

void run_gpu(size_t num_buckets, size_t capacity,
             std::vector<gpu_hashmap::KeyType> const& h_keys,
             std::vector<gpu_hashmap::ValueType> const& h_values,
             std::vector<gpu_hashmap::KeyType> const& h_lookup_keys,
             double* out_insert_ms, double* out_lookup_ms) {
  const size_t n = h_keys.size();
  const size_t m = h_lookup_keys.size();
  const size_t insert_batch_size = n / N_INSERT_BATCHES;
  const size_t lookup_batch_size = m / N_LOOKUP_BATCHES;

  gpu_hashmap::HashTable table = {};
  gpu_hashmap::hash_map_create(&table, num_buckets, capacity);

  gpu_hashmap::KeyType* d_keys = nullptr;
  gpu_hashmap::ValueType* d_values = nullptr;
  gpu_hashmap::KeyType* d_lookup_keys = nullptr;
  gpu_hashmap::ValueType* d_lookup_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMalloc(&d_lookup_keys, m * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_lookup_out, m * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), n * sizeof(gpu_hashmap::ValueType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lookup_keys, h_lookup_keys.data(), m * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  /* Warmup: one insert batch, one lookup batch */
  gpu_hashmap::hash_map_insert_batch(&table, d_keys, d_values, insert_batch_size);
  lookup_kernel<<<(lookup_batch_size + 255) / 256, 256>>>(table.d_device_table, d_lookup_keys, d_lookup_out, lookup_batch_size);
  CUDA_CHECK(cudaDeviceSynchronize());

  /* Timed: multiple insert batches (table and data stay on device) */
  CUDA_CHECK(cudaEventRecord(start));
  for (int batch = 0; batch < N_INSERT_BATCHES; ++batch) {
    size_t offset = batch * insert_batch_size;
    gpu_hashmap::hash_map_insert_batch(&table, d_keys + offset, d_values + offset, insert_batch_size);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  *out_insert_ms = static_cast<double>(ms);

  /* Timed: multiple lookup batches */
  CUDA_CHECK(cudaEventRecord(start));
  for (int batch = 0; batch < N_LOOKUP_BATCHES; ++batch) {
    size_t offset = batch * lookup_batch_size;
    lookup_kernel<<<(lookup_batch_size + 255) / 256, 256>>>(table.d_device_table, d_lookup_keys + offset, d_lookup_out + offset, lookup_batch_size);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  *out_lookup_ms = static_cast<double>(ms);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_lookup_keys));
  CUDA_CHECK(cudaFree(d_lookup_out));
  gpu_hashmap::hash_map_destroy(&table);
}

void run_cpu(std::vector<gpu_hashmap::KeyType> const& h_keys,
             std::vector<gpu_hashmap::ValueType> const& h_values,
             std::vector<gpu_hashmap::KeyType> const& h_lookup_keys,
             double* out_insert_ms, double* out_lookup_ms) {
  const size_t n = h_keys.size();
  const size_t m = h_lookup_keys.size();

  std::unordered_map<uint64_t, uint64_t> map;
  map.reserve(n);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < n; ++i)
    map[h_keys[i]] = h_values[i];
  auto t1 = std::chrono::high_resolution_clock::now();
  *out_insert_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::vector<uint64_t> results(m);
  t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < m; ++i) {
    auto it = map.find(h_lookup_keys[i]);
    results[i] = (it != map.end()) ? it->second : 0xFFFFFFFFFFFFFFFFull;
  }
  t1 = std::chrono::high_resolution_clock::now();
  *out_lookup_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  (void)results;
}

} // namespace

int main() {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  const size_t num_buckets = 1 << 20;   /* 1M buckets */
  const size_t capacity = 4 * 1024 * 1024;  /* 4M entries */
  const size_t n = 2 * 1024 * 1024;     /* 2M inserts */
  const size_t m = 1 * 1024 * 1024;     /* 1M lookups */

  std::printf("GPU Hash Map vs CPU (single-threaded) performance\n");
  std::printf("  Inserts: %zu (%d batches)  Lookups: %zu (%d batches)  Buckets: %zu  Capacity: %zu\n",
              n, N_INSERT_BATCHES, m, N_LOOKUP_BATCHES, num_buckets, capacity);
  std::printf("  GPU: per-block batch alloc + amortized setup (table/data kept on device)\n\n");

  std::vector<gpu_hashmap::KeyType> h_keys(n);
  std::vector<gpu_hashmap::ValueType> h_values(n);
  std::vector<gpu_hashmap::KeyType> h_lookup_keys(m);
  for (size_t i = 0; i < n; ++i) {
    h_keys[i] = i * 3;
    h_values[i] = i + 1000;
  }
  for (size_t i = 0; i < m; ++i)
    h_lookup_keys[i] = (i * 7) % (n * 3);

  double gpu_insert_ms = 0, gpu_lookup_ms = 0;
  double cpu_insert_ms = 0, cpu_lookup_ms = 0;

  /* Warmup GPU */
  run_gpu(num_buckets, capacity, h_keys, h_values, h_lookup_keys, &gpu_insert_ms, &gpu_lookup_ms);

  /* Timed runs */
  run_gpu(num_buckets, capacity, h_keys, h_values, h_lookup_keys, &gpu_insert_ms, &gpu_lookup_ms);
  run_cpu(h_keys, h_values, h_lookup_keys, &cpu_insert_ms, &cpu_lookup_ms);

  std::printf("                    GPU (parallel)    CPU (single-thread)\n");
  std::printf("  Insert (ms total) %12.3f       %12.3f    speedup: %.2fx\n",
              gpu_insert_ms, cpu_insert_ms, cpu_insert_ms / (gpu_insert_ms + 1e-9));
  std::printf("  Lookup (ms total) %12.3f       %12.3f    speedup: %.2fx\n",
              gpu_lookup_ms, cpu_lookup_ms, cpu_lookup_ms / (gpu_lookup_ms + 1e-9));
  std::printf("  Total (ms)        %12.3f       %12.3f    speedup: %.2fx\n",
              gpu_insert_ms + gpu_lookup_ms, cpu_insert_ms + cpu_lookup_ms,
              (cpu_insert_ms + cpu_lookup_ms) / (gpu_insert_ms + gpu_lookup_ms + 1e-9));
  std::printf("  (GPU: %d insert + %d lookup batches, amortized setup; avg insert batch %.1f ms)\n",
              N_INSERT_BATCHES, N_LOOKUP_BATCHES, gpu_insert_ms / N_INSERT_BATCHES);

  return 0;
}
