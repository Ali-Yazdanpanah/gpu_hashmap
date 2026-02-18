/**
 * @file benchmark_hybrid.cu
 * @brief Hybrid: insert on CPU, upload once, then GPU lookup only.
 * Compares hybrid total time vs full CPU (insert + lookup).
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

constexpr int N_LOOKUP_BATCHES = 10;

void run_hybrid(size_t num_buckets, size_t capacity,
                std::vector<gpu_hashmap::KeyType> const& h_keys,
                std::vector<gpu_hashmap::ValueType> const& h_values,
                std::vector<gpu_hashmap::KeyType> const& h_lookup_keys,
                double* out_cpu_build_ms, double* out_upload_ms,
                double* out_gpu_lookup_ms) {
  const size_t n = h_keys.size();
  const size_t m = h_lookup_keys.size();
  const size_t lookup_batch_size = m / N_LOOKUP_BATCHES;

  gpu_hashmap::HashTable table = {};
  gpu_hashmap::hash_map_create(&table, num_buckets, capacity);

  /* CPU build (host layout) + upload (H2D) in one call */
  /*
   * we time that build separately. So build on host into temp arrays, then upload.
   * For simplicity we time hash_map_upload_from_host as a whole and report it as
   * "CPU build + upload". Alternatively we could expose a build-only step.
   * Here we time: (1) CPU build = populating host structures, (2) upload = cudaMemcpy.
   * hash_map_upload_from_host does both. So we'll report "upload" as the whole
   * hash_map_upload_from_host (build on CPU + copy to device). And compare to
   * "full CPU" which is unordered_map insert + lookup.
   * So hybrid path: create table, then "CPU build+upload" = hash_map_upload_from_host,
   * then GPU lookup batches. We can time upload separately by doing the build
   * manually and then cudaMemcpy - but that would duplicate logic. Simpler: report
   * "Hybrid: upload (CPU build + H2D) ms, GPU lookup ms, total ms" and
   * "Full CPU: insert ms, lookup ms, total ms". We time upload (build + H2D) and GPU lookup.
   */
  auto t_build_start = std::chrono::high_resolution_clock::now();
  gpu_hashmap::hash_map_upload_from_host(&table, h_keys.data(), h_values.data(), n);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t_build_end = std::chrono::high_resolution_clock::now();
  *out_cpu_build_ms = std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count();
  *out_upload_ms = 0.; /* included in cpu_build_ms for now; upload is the cudaMemcpy part */

  gpu_hashmap::KeyType* d_lookup_keys = nullptr;
  gpu_hashmap::ValueType* d_lookup_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_lookup_keys, m * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_lookup_out, m * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMemcpy(d_lookup_keys, h_lookup_keys.data(), m * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int batch = 0; batch < N_LOOKUP_BATCHES; ++batch) {
    size_t offset = batch * lookup_batch_size;
    lookup_kernel<<<(lookup_batch_size + 255) / 256, 256>>>(table.d_device_table, d_lookup_keys + offset, d_lookup_out + offset, lookup_batch_size);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  *out_gpu_lookup_ms = static_cast<double>(ms);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_lookup_keys));
  CUDA_CHECK(cudaFree(d_lookup_out));
  gpu_hashmap::hash_map_destroy(&table);
}

void run_full_cpu(std::vector<gpu_hashmap::KeyType> const& h_keys,
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
  const size_t num_buckets = 1 << 20;
  const size_t capacity = 4 * 1024 * 1024;
  const size_t n = 2 * 1024 * 1024;
  const size_t m = 1 * 1024 * 1024;

  std::printf("Hybrid benchmark: CPU insert + upload once, then GPU lookup only\n");
  std::printf("  Inserts: %zu  Lookups: %zu (%d GPU batches)  Buckets: %zu  Capacity: %zu\n\n",
              n, m, N_LOOKUP_BATCHES, num_buckets, capacity);

  std::vector<gpu_hashmap::KeyType> h_keys(n);
  std::vector<gpu_hashmap::ValueType> h_values(n);
  std::vector<gpu_hashmap::KeyType> h_lookup_keys(m);
  for (size_t i = 0; i < n; ++i) {
    h_keys[i] = i * 3;
    h_values[i] = i + 1000;
  }
  for (size_t i = 0; i < m; ++i)
    h_lookup_keys[i] = (i * 7) % (n * 3);

  double hybrid_build_ms = 0, hybrid_upload_ms = 0, hybrid_lookup_ms = 0;
  double cpu_insert_ms = 0, cpu_lookup_ms = 0;

  run_hybrid(num_buckets, capacity, h_keys, h_values, h_lookup_keys,
             &hybrid_build_ms, &hybrid_upload_ms, &hybrid_lookup_ms);
  run_full_cpu(h_keys, h_values, h_lookup_keys, &cpu_insert_ms, &cpu_lookup_ms);

  const double hybrid_total = hybrid_build_ms + hybrid_lookup_ms;
  const double cpu_total = cpu_insert_ms + cpu_lookup_ms;

  std::printf("  Hybrid (CPU build + upload, then GPU lookup)\n");
  std::printf("    CPU build + upload (ms)  %12.3f\n", hybrid_build_ms);
  std::printf("    GPU lookup total (ms)    %12.3f  (%d batches)\n", hybrid_lookup_ms, N_LOOKUP_BATCHES);
  std::printf("    Hybrid total (ms)       %12.3f\n", hybrid_total);
  std::printf("\n");
  std::printf("  Full CPU (insert + lookup)\n");
  std::printf("    CPU insert (ms)          %12.3f\n", cpu_insert_ms);
  std::printf("    CPU lookup (ms)          %12.3f\n", cpu_lookup_ms);
  std::printf("    CPU total (ms)           %12.3f\n", cpu_total);
  std::printf("\n");
  std::printf("  Comparison:  Hybrid total vs CPU total  =>  %.2fx %s\n",
              cpu_total / (hybrid_total + 1e-9),
              hybrid_total < cpu_total ? "(hybrid faster)" : "(CPU faster)");

  return 0;
}
