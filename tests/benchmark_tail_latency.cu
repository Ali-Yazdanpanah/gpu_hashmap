/**
 * @file benchmark_tail_latency.cu
 * @brief P99 tail latency: high-resolution timers over many batches to capture
 *        the distribution of lookup batch times (P50, P90, P99) and a histogram.
 *        Host-only logic (sort, percentile, CPU benchmark) lives in _impl.cpp
 *        to avoid nvcc + libstdc++ parameter-pack errors with std::function.
 */

#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/lookup_kernel.cuh"
#include "gpu_hashmap/hash_slab.cuh"
#include "benchmark_tail_latency_impl.h"
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

namespace {

constexpr size_t N_INSERT = 512 * 1024;
constexpr size_t N_LOOKUP_TOTAL = 256 * 1024;
constexpr size_t BATCH_SIZE = 4096;
constexpr int N_BATCHES = 500;  /* number of batches to get distribution */

static void run_gpu_tail_latency(gpu_hashmap::HashTable* table,
                                 gpu_hashmap::KeyType const* d_lookup_keys,
                                 gpu_hashmap::ValueType* d_lookup_out,
                                 std::vector<double>* out_batch_ms) {
  out_batch_ms->clear();
  out_batch_ms->reserve(N_BATCHES);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int b = 0; b < N_BATCHES; ++b) {
    size_t offset = (static_cast<size_t>(b) * BATCH_SIZE) % (N_LOOKUP_TOTAL - BATCH_SIZE + 1);
    CUDA_CHECK(cudaEventRecord(start));
    gpu_hashmap::hash_map_lookup_kernel<<<(BATCH_SIZE + 255) / 256, 256>>>(
        table->d_device_table, d_lookup_keys + offset, d_lookup_out + offset, BATCH_SIZE);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    out_batch_ms->push_back(static_cast<double>(ms));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

static void run_gpu_slab_tail_latency(gpu_hashmap::SlabHashTable* table,
                                      gpu_hashmap::KeyType const* d_lookup_keys,
                                      gpu_hashmap::ValueType* d_lookup_out,
                                      std::vector<double>* out_batch_ms) {
  out_batch_ms->clear();
  out_batch_ms->reserve(N_BATCHES);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int b = 0; b < N_BATCHES; ++b) {
    size_t offset = (static_cast<size_t>(b) * BATCH_SIZE) % (N_LOOKUP_TOTAL - BATCH_SIZE + 1);
    CUDA_CHECK(cudaEventRecord(start));
    gpu_hashmap::slab_lookup_kernel<<<(BATCH_SIZE + 255) / 256, 256>>>(
        table->d_device_table, d_lookup_keys + offset, d_lookup_out + offset, BATCH_SIZE);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    out_batch_ms->push_back(static_cast<double>(ms));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

} // namespace

int main() {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  const size_t num_buckets = 1 << 18;
  const size_t capacity = 2 * 1024 * 1024;

  std::printf("================================================================================\n");
  std::printf("  P99 Tail Latency Benchmark (high-resolution batch timings)\n");
  std::printf("  Batch size = %zu lookups, %d batches -> P50 / P90 / P99\n", BATCH_SIZE, N_BATCHES);
  std::printf("================================================================================\n\n");

  std::vector<gpu_hashmap::KeyType> h_keys(N_INSERT);
  std::vector<gpu_hashmap::ValueType> h_values(N_INSERT);
  std::vector<gpu_hashmap::KeyType> h_lookup_keys(N_LOOKUP_TOTAL);
  for (size_t i = 0; i < N_INSERT; ++i) {
    h_keys[i] = i * 3 + 1;
    h_values[i] = i + 1000;
  }
  for (size_t i = 0; i < N_LOOKUP_TOTAL; ++i)
    h_lookup_keys[i] = h_keys[i % N_INSERT];

  gpu_hashmap::HashTable table = {};
  gpu_hashmap::hash_map_create(&table, num_buckets, capacity);
  gpu_hashmap::hash_map_upload_from_host(&table, h_keys.data(), h_values.data(), N_INSERT);
  CUDA_CHECK(cudaDeviceSynchronize());

  gpu_hashmap::KeyType* d_lookup_keys = nullptr;
  gpu_hashmap::ValueType* d_lookup_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_lookup_keys, N_LOOKUP_TOTAL * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_lookup_out, N_LOOKUP_TOTAL * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMemcpy(d_lookup_keys, h_lookup_keys.data(),
                        N_LOOKUP_TOTAL * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));

  /* Warmup */
  gpu_hashmap::hash_map_lookup_kernel<<<(BATCH_SIZE + 255) / 256, 256>>>(
      table.d_device_table, d_lookup_keys, d_lookup_out, BATCH_SIZE);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> gpu_batch_ms;
  run_gpu_tail_latency(&table, d_lookup_keys, d_lookup_out, &gpu_batch_ms);

  double gpu_p50 = 0, gpu_p90 = 0, gpu_p99 = 0;
  tail_latency_impl::sort_and_report_percentiles(&gpu_batch_ms, &gpu_p50, &gpu_p90, &gpu_p99);

  std::printf("  GPU (chained lookup, 256 threads/block):\n");
  std::printf("    P50 (median)  %8.4f ms\n", gpu_p50);
  std::printf("    P90           %8.4f ms\n", gpu_p90);
  std::printf("    P99           %8.4f ms\n", gpu_p99);
  std::printf("    P99/P50       %8.2fx (tail spread)\n\n", gpu_p99 / (gpu_p50 + 1e-9));
  tail_latency_impl::print_histogram(gpu_batch_ms, 12);

  CUDA_CHECK(cudaFree(d_lookup_keys));
  CUDA_CHECK(cudaFree(d_lookup_out));
  gpu_hashmap::hash_map_destroy(&table);

  /* GPU Slab: same batch structure for tail latency */
  gpu_hashmap::SlabHashTable slab_table = {};
  gpu_hashmap::slab_hash_create(&slab_table, num_buckets);
  gpu_hashmap::KeyType* d_slab_keys = nullptr;
  gpu_hashmap::ValueType* d_slab_vals = nullptr;
  gpu_hashmap::KeyType* d_slab_lookup_keys = nullptr;
  gpu_hashmap::ValueType* d_slab_lookup_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_slab_keys, N_INSERT * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_slab_vals, N_INSERT * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMalloc(&d_slab_lookup_keys, N_LOOKUP_TOTAL * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_slab_lookup_out, N_LOOKUP_TOTAL * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMemcpy(d_slab_keys, h_keys.data(), N_INSERT * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_slab_vals, h_values.data(), N_INSERT * sizeof(gpu_hashmap::ValueType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_slab_lookup_keys, h_lookup_keys.data(), N_LOOKUP_TOTAL * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));
  gpu_hashmap::slab_hash_insert_batch(slab_table.d_device_table, d_slab_keys, d_slab_vals, N_INSERT);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> slab_batch_ms;
  run_gpu_slab_tail_latency(&slab_table, d_slab_lookup_keys, d_slab_lookup_out, &slab_batch_ms);
  double slab_p50 = 0, slab_p90 = 0, slab_p99 = 0;
  tail_latency_impl::sort_and_report_percentiles(&slab_batch_ms, &slab_p50, &slab_p90, &slab_p99);
  std::printf("  GPU Slab (batch latency, 256 threads/block):\n");
  std::printf("    P50 (median)  %8.4f ms\n", slab_p50);
  std::printf("    P90           %8.4f ms\n", slab_p90);
  std::printf("    P99           %8.4f ms\n\n", slab_p99);

  CUDA_CHECK(cudaFree(d_slab_keys));
  CUDA_CHECK(cudaFree(d_slab_vals));
  CUDA_CHECK(cudaFree(d_slab_lookup_keys));
  CUDA_CHECK(cudaFree(d_slab_lookup_out));
  gpu_hashmap::slab_hash_destroy(&slab_table);

  /* CPU: same batch structure for comparison (host-only impl to avoid nvcc/stdlib issues) */
  std::vector<uint64_t> h_keys_u(h_keys.begin(), h_keys.end());
  std::vector<uint64_t> h_vals_u(h_values.begin(), h_values.end());
  std::vector<uint64_t> h_look_u(h_lookup_keys.begin(), h_lookup_keys.end());
  std::vector<double> cpu_batch_ms;
  tail_latency_impl::run_cpu_tail_latency(h_keys_u, h_vals_u, h_look_u,
                                          BATCH_SIZE, N_BATCHES, &cpu_batch_ms);
  double cpu_p50 = 0, cpu_p90 = 0, cpu_p99 = 0;
  tail_latency_impl::sort_and_report_percentiles(&cpu_batch_ms, &cpu_p50, &cpu_p90, &cpu_p99);

  std::printf("\n  CPU (std::unordered_map, single-thread, same batch size):\n");
  std::printf("    P50 (median)  %8.4f ms\n", cpu_p50);
  std::printf("    P90           %8.4f ms\n", cpu_p90);
  std::printf("    P99           %8.4f ms\n", cpu_p99);
  std::printf("    P99/P50       %8.2fx\n\n", cpu_p99 / (cpu_p50 + 1e-9));
  tail_latency_impl::print_histogram(cpu_batch_ms, 12);

  std::printf("================================================================================\n");
  std::printf("  Tail latency benchmark done. Use P99/P50 to compare warp divergence / collision impact.\n");
  std::printf("================================================================================\n");
  return 0;
}
