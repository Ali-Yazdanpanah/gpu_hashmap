/**
 * @file benchmark_vs_cpu.cu
 * @brief Benchmark all GPU hash map variants against CPU (std::unordered_map).
 * Reports: CPU, GPU chained, GPU warp-aggregated, GPU slab, Hybrid (CPU build + GPU lookup).
 */

#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/hash_slab.cuh"
#include "gpu_hashmap/insert_kernel.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
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

__global__ void lookup_kernel_chained(gpu_hashmap::HashTableDevice const* table,
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

static int compare_double(const void* a, const void* b) {
  double x = *(const double*)a;
  double y = *(const double*)b;
  return (x > y) ? 1 : (x < y) ? -1 : 0;
}

/* Slab totals land near 1 ms, so a single timed run is mostly scheduler noise and
 * its "speedup vs CPU" swung between 62x and 208x across back-to-back runs. Report
 * the median of several reps instead; it also discards the cold first-touch outlier
 * on the mapped-memory paths. */
static double median_of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  qsort(v.data(), v.size(), sizeof(double), compare_double);
  const size_t k = v.size() / 2;
  return (v.size() % 2 != 0) ? v[k] : 0.5 * (v[k - 1] + v[k]);
}

static double percentile_from_sorted(const double* sorted, size_t len, double p) {
  if (len == 0) return 0.0;
  double idx = p * (len - 1) / 100.0;
  size_t i = (size_t)idx;
  if (i >= len - 1) return sorted[len - 1];
  double frac = idx - (double)i;
  return sorted[i] * (1.0 - frac) + sorted[i + 1] * frac;
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

void run_cpu_latency_distribution(std::vector<gpu_hashmap::KeyType> const& h_keys,
                                   std::vector<gpu_hashmap::ValueType> const& h_values,
                                   std::vector<gpu_hashmap::KeyType> const& h_lookup_keys,
                                   double* out_p50_us, double* out_p90_us, double* out_p99_us) {
  const size_t n = h_keys.size();
  const size_t m = h_lookup_keys.size();
  std::unordered_map<uint64_t, uint64_t> map;
  map.reserve(n);
  for (size_t i = 0; i < n; ++i)
    map[h_keys[i]] = h_values[i];

  std::vector<double> latencies_us(m);
  for (size_t i = 0; i < m; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto it = map.find(h_lookup_keys[i]);
    (void)(it != map.end() ? it->second : 0xFFFFFFFFFFFFFFFFull);
    auto t1 = std::chrono::high_resolution_clock::now();
    latencies_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
  }
  qsort(latencies_us.data(), m, sizeof(double), compare_double);
  *out_p50_us = percentile_from_sorted(latencies_us.data(), m, 50.0);
  *out_p90_us = percentile_from_sorted(latencies_us.data(), m, 90.0);
  *out_p99_us = percentile_from_sorted(latencies_us.data(), m, 99.0);
}

void run_gpu_chained(size_t num_buckets, size_t capacity,
                     std::vector<gpu_hashmap::KeyType> const& h_keys,
                     std::vector<gpu_hashmap::ValueType> const& h_values,
                     std::vector<gpu_hashmap::KeyType> const& h_lookup_keys,
                     double* out_insert_ms, double* out_lookup_ms,
                     unsigned long long* out_dropped) {
  const size_t n = h_keys.size();
  const size_t m = h_lookup_keys.size();
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

  CUDA_CHECK(cudaEventRecord(start));
  gpu_hashmap::hash_map_insert_batch(&table, d_keys, d_values, n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  *out_insert_ms = static_cast<double>(ms);
  *out_dropped = gpu_hashmap::hash_map_insert_failure_count(&table);

  CUDA_CHECK(cudaEventRecord(start));
  lookup_kernel_chained<<<(m + 255) / 256, 256>>>(table.d_device_table, d_lookup_keys, d_lookup_out, m);
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

void run_gpu_warp_agg(size_t num_buckets, size_t capacity,
                      std::vector<gpu_hashmap::KeyType> const& h_keys,
                      std::vector<gpu_hashmap::ValueType> const& h_values,
                      std::vector<gpu_hashmap::KeyType> const& h_lookup_keys,
                      double* out_insert_ms, double* out_lookup_ms,
                      unsigned long long* out_dropped) {
  const size_t n = h_keys.size();
  const size_t m = h_lookup_keys.size();
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

  CUDA_CHECK(cudaEventRecord(start));
  gpu_hashmap::hash_map_insert_batch_warp_aggregated(&table, d_keys, d_values, n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  *out_insert_ms = static_cast<double>(ms);
  *out_dropped = gpu_hashmap::hash_map_insert_failure_count(&table);

  CUDA_CHECK(cudaEventRecord(start));
  lookup_kernel_chained<<<(m + 255) / 256, 256>>>(table.d_device_table, d_lookup_keys, d_lookup_out, m);
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

void run_gpu_slab(size_t num_buckets,
                  std::vector<gpu_hashmap::KeyType> const& h_keys,
                  std::vector<gpu_hashmap::ValueType> const& h_values,
                  std::vector<gpu_hashmap::KeyType> const& h_lookup_keys,
                  double* out_insert_ms, double* out_lookup_ms,
                  unsigned long long* out_dropped) {
  const size_t n = h_keys.size();
  const size_t m = h_lookup_keys.size();
  gpu_hashmap::SlabHashTable table = {};
  gpu_hashmap::slab_hash_create(&table, num_buckets);

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

  CUDA_CHECK(cudaEventRecord(start));
  gpu_hashmap::slab_hash_insert_batch(table.d_device_table, d_keys, d_values, n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  *out_insert_ms = static_cast<double>(ms);
  *out_dropped = gpu_hashmap::slab_hash_insert_failure_count(&table);

  CUDA_CHECK(cudaEventRecord(start));
  gpu_hashmap::slab_lookup_kernel<<<(m + 255) / 256, 256>>>(table.d_device_table, d_lookup_keys, d_lookup_out, m);
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
  gpu_hashmap::slab_hash_destroy(&table);
}

void run_hybrid(size_t num_buckets, size_t capacity,
                std::vector<gpu_hashmap::KeyType> const& h_keys,
                std::vector<gpu_hashmap::ValueType> const& h_values,
                std::vector<gpu_hashmap::KeyType> const& h_lookup_keys,
                double* out_build_upload_ms, double* out_lookup_ms) {
  const size_t n = h_keys.size();
  const size_t m = h_lookup_keys.size();
  gpu_hashmap::HashTable table = {};
  gpu_hashmap::hash_map_create(&table, num_buckets, capacity);

  auto t0 = std::chrono::high_resolution_clock::now();
  gpu_hashmap::hash_map_upload_from_host(&table, h_keys.data(), h_values.data(), n);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  *out_build_upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  gpu_hashmap::KeyType* d_lookup_keys = nullptr;
  gpu_hashmap::ValueType* d_lookup_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_lookup_keys, m * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_lookup_out, m * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMemcpy(d_lookup_keys, h_lookup_keys.data(), m * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  lookup_kernel_chained<<<(m + 255) / 256, 256>>>(table.d_device_table, d_lookup_keys, d_lookup_out, m);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  *out_lookup_ms = static_cast<double>(ms);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_lookup_keys));
  CUDA_CHECK(cudaFree(d_lookup_out));
  gpu_hashmap::hash_map_destroy(&table);
}

} // namespace

int main() {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  const size_t num_buckets = 1 << 18;   /* 256K buckets */
  const size_t capacity = 2 * 1024 * 1024;
  const size_t n = 1024 * 1024;        /* 1M inserts */
  const size_t m = 512 * 1024;         /* 512K lookups */

  std::printf("Hash map benchmark vs CPU (std::unordered_map)\n");
  std::printf("  Inserts: %zu   Lookups: %zu   Buckets: %zu\n\n", n, m, num_buckets);

  /* Random keys, fixed seed. An arithmetic sequence such as i*3 is a degenerate
   * input here: hash_key multiplies by an odd constant and masks the low bits, so
   * i -> bucket is a bijection over each cycle and every bucket receives exactly
   * n / num_buckets keys. That perfect equidistribution understates chain depth,
   * collision handling, and slab-bucket overflow relative to any real workload. */
  std::vector<gpu_hashmap::KeyType> h_keys(n);
  std::vector<gpu_hashmap::ValueType> h_values(n);
  std::vector<gpu_hashmap::KeyType> h_lookup_keys(m);
  std::mt19937_64 rng(42);
  for (size_t i = 0; i < n; ++i) {
    h_keys[i] = rng();
    h_values[i] = i + 1000;
  }
  /* Lookup keys are drawn from the inserted set so every probe is a hit; this
   * measures find-success cost rather than a mix of hits and misses. */
  for (size_t i = 0; i < m; ++i)
    h_lookup_keys[i] = h_keys[(i * 7919) % n];

  double cpu_ins = 0, cpu_lup = 0;
  double gpu_chain_ins = 0, gpu_chain_lup = 0;
  double gpu_warp_ins = 0, gpu_warp_lup = 0;
  double gpu_slab_ins = 0, gpu_slab_lup = 0;
  double hybrid_build = 0, hybrid_lup = 0;
  unsigned long long chain_dropped = 0, warp_dropped = 0, slab_dropped = 0;

  const int reps = 5;
  std::vector<double> ins(reps), lup(reps);

  for (int r = 0; r < reps; ++r)
    run_cpu(h_keys, h_values, h_lookup_keys, &ins[r], &lup[r]);
  cpu_ins = median_of(ins);
  cpu_lup = median_of(lup);

  double cpu_p50_us = 0, cpu_p90_us = 0, cpu_p99_us = 0;
  run_cpu_latency_distribution(h_keys, h_values, h_lookup_keys, &cpu_p50_us, &cpu_p90_us, &cpu_p99_us);
  std::printf("  CPU per-find latency (µs):  P50= %8.2f   P90= %8.2f   P99= %8.2f\n", cpu_p50_us, cpu_p90_us, cpu_p99_us);
  std::printf("  Each row below is the median of %d runs.\n\n", reps);

  for (int r = 0; r < reps; ++r)
    run_gpu_chained(num_buckets, capacity, h_keys, h_values, h_lookup_keys, &ins[r], &lup[r], &chain_dropped);
  gpu_chain_ins = median_of(ins);
  gpu_chain_lup = median_of(lup);

  for (int r = 0; r < reps; ++r)
    run_gpu_warp_agg(num_buckets, capacity, h_keys, h_values, h_lookup_keys, &ins[r], &lup[r], &warp_dropped);
  gpu_warp_ins = median_of(ins);
  gpu_warp_lup = median_of(lup);

  for (int r = 0; r < reps; ++r)
    run_gpu_slab(num_buckets, h_keys, h_values, h_lookup_keys, &ins[r], &lup[r], &slab_dropped);
  gpu_slab_ins = median_of(ins);
  gpu_slab_lup = median_of(lup);

  for (int r = 0; r < reps; ++r)
    run_hybrid(num_buckets, capacity, h_keys, h_values, h_lookup_keys, &ins[r], &lup[r]);
  hybrid_build = median_of(ins);
  hybrid_lup = median_of(lup);

  const double cpu_total = cpu_ins + cpu_lup;
  const double gpu_chain_total = gpu_chain_ins + gpu_chain_lup;
  const double gpu_warp_total = gpu_warp_ins + gpu_warp_lup;
  const double gpu_slab_total = gpu_slab_ins + gpu_slab_lup;
  const double hybrid_total = hybrid_build + hybrid_lup;

  std::printf("  %-20s  %10s  %10s  %10s  %8s\n", "Approach", "Insert(ms)", "Lookup(ms)", "Total(ms)", "vs CPU");
  std::printf("  %-20s  %10.2f  %10.2f  %10.2f  %7.2fx\n",
              "CPU (unordered_map)", cpu_ins, cpu_lup, cpu_total, 1.0);
  std::printf("  %-20s  %10.2f  %10.2f  %10.2f  %7.2fx\n",
              "GPU chained", gpu_chain_ins, gpu_chain_lup, gpu_chain_total, cpu_total / (gpu_chain_total + 1e-9));
  std::printf("  %-20s  %10.2f  %10.2f  %10.2f  %7.2fx\n",
              "GPU warp-aggregated", gpu_warp_ins, gpu_warp_lup, gpu_warp_total, cpu_total / (gpu_warp_total + 1e-9));
  std::printf("  %-20s  %10.2f  %10.2f  %10.2f  %7.2fx\n",
              "GPU slab (8/bucket)", gpu_slab_ins, gpu_slab_lup, gpu_slab_total, cpu_total / (gpu_slab_total + 1e-9));
  std::printf("  %-20s  %10.2f  %10.2f  %10.2f  %7.2fx\n",
              "Hybrid (CPU+GPU lup)", hybrid_build, hybrid_lup, hybrid_total, cpu_total / (hybrid_total + 1e-9));

  /* A row that dropped inserts stored fewer than n keys, so its time covers less
   * work than the CPU baseline's and the speedup is not directly comparable. */
  std::printf("\n  Inserts not stored (of %zu requested):\n", n);
  std::printf("    %-20s  %10llu  (%.3f%%)\n", "GPU chained", chain_dropped,
              100.0 * chain_dropped / n);
  std::printf("    %-20s  %10llu  (%.3f%%)\n", "GPU warp-aggregated", warp_dropped,
              100.0 * warp_dropped / n);
  std::printf("    %-20s  %10llu  (%.3f%%)\n", "GPU slab (8/bucket)", slab_dropped,
              100.0 * slab_dropped / n);
  if (chain_dropped || warp_dropped || slab_dropped)
    std::printf("    NOTE: non-zero above means those keys are absent from the table.\n");

  return 0;
}
