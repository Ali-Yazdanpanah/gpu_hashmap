/**
 * @file benchmark_heuristic.cu
 * @brief Benchmark for the heuristic optimizer: warm-up, path choice, correctness, and sparsity-driven crossover.
 *
 * - Runs heuristic_warm_up to calibrate crossover from current PCIe/copy cost.
 * - Sweeps lookup batch sizes (below/at/above crossover) and reports chosen path + time.
 * - Verifies that heuristic path results match the other path (bit-perfect).
 * - Sparsity-Driven Crossover: massive table (~2GB or 80% VRAM) with sparse lookups (N=10,000).
 *   Total system latency: Standard = full table H2D + keys H2D + kernel + results D2H;
 *   Zero-Copy = kernel only (GPU fetches only needed buckets over PCIe). Crossover when
 *   Time(Full Table Copy) > Time(Sparse PCIe Stalls).
 */

#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/heuristic_lookup.h"
#include "gpu_hashmap/lookup_kernel.cuh"
#include "gpu_hashmap/hash_buckets.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

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
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

  const size_t num_buckets = 1 << 18;
  const size_t capacity = 2 * 1024 * 1024;
  const size_t n_insert = 512 * 1024;

  std::vector<gpu_hashmap::KeyType> h_keys(n_insert);
  std::vector<gpu_hashmap::ValueType> h_values(n_insert);
  for (size_t i = 0; i < n_insert; ++i) {
    h_keys[i] = i * 3 + 1;
    h_values[i] = i + 1000;
  }

  gpu_hashmap::HashTable table = {};
  hash_map_create(&table, num_buckets, capacity);
  hash_map_upload_from_host(&table, h_keys.data(), h_values.data(), n_insert);
  CUDA_CHECK(cudaDeviceSynchronize());

  gpu_hashmap::HeuristicState state = {};
  heuristic_init(&state);

  std::printf("================================================================================\n");
  std::printf("  Heuristic Optimizer Benchmark\n");
  std::printf("  Table: %zu buckets, %zu entries. Default crossover = %zu\n",
              num_buckets, n_insert, gpu_hashmap::kHeuristicDefaultCrossoverN);
  std::printf("================================================================================\n\n");

  /* ---------- Warm-up: calibrate crossover from current system ---------- */
  std::printf("--- Warm-up (measures PCIe/copy cost at 256k lookups) ---\n");
  heuristic_warm_up(&table, &state);
  std::printf("  After warm-up: crossover_n = %zu\n\n", state.crossover_n);

  /* Sweep: use pinned buffers so Zero-Copy path is true zero-copy (no memcpy at end). */
  const size_t max_sweep = 512 * 1024;
  const unsigned int pin_flags = cudaHostAllocMapped | cudaHostAllocPortable;
  gpu_hashmap::KeyType* h_pinned_keys = nullptr;
  gpu_hashmap::ValueType* h_pinned_results = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_pinned_keys, max_sweep * sizeof(gpu_hashmap::KeyType), pin_flags));
  CUDA_CHECK(cudaHostAlloc(&h_pinned_results, max_sweep * sizeof(gpu_hashmap::ValueType), pin_flags));

  const size_t sweep_sizes[] = { 64 * 1024, 128 * 1024, 256 * 1024, 384 * 1024, 512 * 1024 };
  const int num_sizes = (int)(sizeof(sweep_sizes) / sizeof(sweep_sizes[0]));

  std::printf("--- Lookup sweep (path chosen by heuristic) ---\n");
  std::printf("  %-12s  %-14s  %-10s  %s\n", "N", "Chosen path", "Time (ms)", "");
  std::printf("  %-12s  %-14s  %-10s  %s\n", "---", "---", "---", "---");

  for (int i = 0; i < num_sizes; ++i) {
    const size_t n = sweep_sizes[i];
    for (size_t j = 0; j < n; ++j) h_pinned_keys[j] = h_keys[j % n_insert];

    gpu_hashmap::LookupPath path = heuristic_choose_path(n, &state);
    auto t0 = std::chrono::steady_clock::now();
    if (path == gpu_hashmap::LookupPath::ZeroCopy)
      hash_map_lookup_batch_zero_copy(&table, h_pinned_keys, h_pinned_results, n);
    else
      hash_map_lookup_batch_standard_copy(&table, h_pinned_keys, h_pinned_results, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const char* path_str = (path == gpu_hashmap::LookupPath::ZeroCopy) ? "Zero-Copy" : "Standard Copy";
    std::printf("  %-12zu  %-14s  %-10.3f  (expected %s for n %s %zu)\n",
                n, path_str, ms,
                path_str,
                (n >= state.crossover_n) ? ">=" : "<",
                state.crossover_n);
  }
  std::printf("\n");

  /* Correctness: read directly from pinned buffer (true zero-copy; no memcpy). */
  std::printf("--- Correctness: heuristic path vs alternate path (read from pinned buffer) ---\n");
  const size_t check_n = 128 * 1024;
  gpu_hashmap::KeyType* h_check_keys = nullptr;
  gpu_hashmap::ValueType* h_result_a = nullptr;
  gpu_hashmap::ValueType* h_result_b = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_check_keys, check_n * sizeof(gpu_hashmap::KeyType), pin_flags));
  CUDA_CHECK(cudaHostAlloc(&h_result_a, check_n * sizeof(gpu_hashmap::ValueType), pin_flags));
  CUDA_CHECK(cudaHostAlloc(&h_result_b, check_n * sizeof(gpu_hashmap::ValueType), pin_flags));
  for (size_t i = 0; i < check_n; ++i) h_check_keys[i] = h_keys[i % n_insert];

  gpu_hashmap::LookupPath path_used = heuristic_choose_path(check_n, &state);
  if (path_used == gpu_hashmap::LookupPath::ZeroCopy) {
    hash_map_lookup_batch_zero_copy(&table, h_check_keys, h_result_a, check_n);
    hash_map_lookup_batch_standard_copy(&table, h_check_keys, h_result_b, check_n);
  } else {
    hash_map_lookup_batch_standard_copy(&table, h_check_keys, h_result_a, check_n);
    hash_map_lookup_batch_zero_copy(&table, h_check_keys, h_result_b, check_n);
  }
  size_t mismatches = 0;
  for (size_t i = 0; i < check_n; ++i) {
    if (h_result_a[i] != h_result_b[i]) ++mismatches;
  }
  if (mismatches)
    std::printf("  FAIL: %zu mismatches between heuristic path and alternate path.\n", mismatches);
  else
    std::printf("  PASS: Heuristic path and alternate path results match (bit-perfect, N=%zu).\n", check_n);
  CUDA_CHECK(cudaFreeHost(h_check_keys));
  CUDA_CHECK(cudaFreeHost(h_result_a));
  CUDA_CHECK(cudaFreeHost(h_result_b));
  std::printf("\n");

  /* One heuristic lookup batch: pinned buffers so Zero-Copy writes in place. */
  std::printf("--- One heuristic lookup batch (log message below) ---\n");
  const size_t demo_n = 300 * 1024;
  gpu_hashmap::KeyType* h_demo_keys = nullptr;
  gpu_hashmap::ValueType* h_demo_out = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_demo_keys, demo_n * sizeof(gpu_hashmap::KeyType), pin_flags));
  CUDA_CHECK(cudaHostAlloc(&h_demo_out, demo_n * sizeof(gpu_hashmap::ValueType), pin_flags));
  for (size_t i = 0; i < demo_n; ++i) h_demo_keys[i] = h_keys[i % n_insert];
  hash_map_lookup_batch_heuristic(&table, h_demo_keys, h_demo_out, demo_n, &state);
  CUDA_CHECK(cudaFreeHost(h_demo_keys));
  CUDA_CHECK(cudaFreeHost(h_demo_out));
  std::printf("\n");

  /* ---------- Sparsity-Driven Crossover: Massive Table + Sparse Lookups (N=10,000) ---------- */
  /*
   * Mathematical validation: crossover when
   *   Time(Full Table Copy) > Time(Sparse PCIe Stalls)
   * Standard path: copy entire table to VRAM, then 10K keys H2D, kernel, 10K results D2H.
   * Zero-Copy path: table stays in mapped memory; GPU fetches only buckets needed over PCIe.
   */
  std::printf("--- Sparsity-Driven Crossover (massive table + N=10,000 sparse lookups) ---\n");
  size_t free_vram = 0, total_vram = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
  const size_t two_gb = 2ULL * 1024 * 1024 * 1024;
  const size_t target_table_bytes = (two_gb < (size_t)((double)free_vram * 0.8)) ? two_gb : (size_t)((double)free_vram * 0.8);
  const size_t num_buckets_big = 1 << 21;
  const size_t capacity_big = (target_table_bytes - num_buckets_big * sizeof(unsigned long long))
      / sizeof(gpu_hashmap::Node);
  if (capacity_big > 1024 * 1024) {
    const size_t n_insert_big = (capacity_big / 40 > 2 * 1024 * 1024) ? (2 * 1024 * 1024) : (capacity_big / 40);
    const size_t table_bytes = num_buckets_big * sizeof(unsigned long long) + capacity_big * sizeof(gpu_hashmap::Node);
    std::printf("  Massive table: %zu buckets, capacity %zu (table size %.2f GB)\n",
                num_buckets_big, capacity_big, table_bytes / (1024.0 * 1024.0 * 1024.0));
    std::printf("  Filling %zu entries; sparse lookup N = 10,000\n", n_insert_big);

    gpu_hashmap::HashTable big_table = {};
    hash_map_create(&big_table, num_buckets_big, capacity_big);
    std::vector<gpu_hashmap::KeyType> h_big_keys(n_insert_big);
    std::vector<gpu_hashmap::ValueType> h_big_values(n_insert_big);
    for (size_t i = 0; i < n_insert_big; ++i) {
      h_big_keys[i] = i * 7 + 11;
      h_big_values[i] = i + 2000;
    }
    hash_map_upload_from_host(&big_table, h_big_keys.data(), h_big_values.data(), n_insert_big);
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t n_sparse = 10000;
    gpu_hashmap::KeyType* h_sparse_keys = nullptr;
    gpu_hashmap::ValueType* h_sparse_results = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_sparse_keys, n_sparse * sizeof(gpu_hashmap::KeyType), pin_flags));
    CUDA_CHECK(cudaHostAlloc(&h_sparse_results, n_sparse * sizeof(gpu_hashmap::ValueType), pin_flags));
    for (size_t i = 0; i < n_sparse; ++i)
      h_sparse_keys[i] = h_big_keys[i % n_insert_big];

    /* Standard path: full table H2D + keys H2D + kernel + results D2H */
    void* d_heads_copy = nullptr;
    void* d_nodes_copy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_heads_copy, num_buckets_big * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_nodes_copy, capacity_big * sizeof(gpu_hashmap::Node)));
    cudaEvent_t ev1, ev2;
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&ev2));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaMemcpy(d_heads_copy, big_table.h_bucket_heads, num_buckets_big * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes_copy, big_table.h_nodes, capacity_big * sizeof(gpu_hashmap::Node), cudaMemcpyHostToDevice));
    gpu_hashmap::HashTableDevice h_dev = {};
    h_dev.bucket_heads = static_cast<unsigned long long*>(d_heads_copy);
    h_dev.nodes = static_cast<gpu_hashmap::Node*>(d_nodes_copy);
    h_dev.slab = big_table.device.slab;
    h_dev.num_buckets = num_buckets_big;
    h_dev.capacity = capacity_big;
    gpu_hashmap::HashTableDevice* d_table_copy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_table_copy, sizeof(gpu_hashmap::HashTableDevice)));
    CUDA_CHECK(cudaMemcpy(d_table_copy, &h_dev, sizeof(gpu_hashmap::HashTableDevice), cudaMemcpyHostToDevice));
    gpu_hashmap::KeyType* d_sk = nullptr;
    gpu_hashmap::ValueType* d_sv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sk, n_sparse * sizeof(gpu_hashmap::KeyType)));
    CUDA_CHECK(cudaMalloc(&d_sv, n_sparse * sizeof(gpu_hashmap::ValueType)));
    CUDA_CHECK(cudaMemcpy(d_sk, h_sparse_keys, n_sparse * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));
    gpu_hashmap::hash_map_lookup_kernel<<<(n_sparse + 255) / 256, 256>>>(d_table_copy, d_sk, d_sv, n_sparse);
    CUDA_CHECK(cudaMemcpy(h_sparse_results, d_sv, n_sparse * sizeof(gpu_hashmap::ValueType), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev2));
    CUDA_CHECK(cudaEventSynchronize(ev2));
    float standard_with_migration_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&standard_with_migration_ms, ev1, ev2));
    std::printf("  Standard path (full table copy + lookup): %.3f ms\n", standard_with_migration_ms);
    CUDA_CHECK(cudaFree(d_table_copy));
    CUDA_CHECK(cudaFree(d_sk));
    CUDA_CHECK(cudaFree(d_sv));
    CUDA_CHECK(cudaFree(d_heads_copy));
    CUDA_CHECK(cudaFree(d_nodes_copy));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaEventDestroy(ev2));

    /* Zero-Copy path: no table copy; kernel only (mapped table + keys/results) */
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&ev2));
    CUDA_CHECK(cudaEventRecord(ev1));
    hash_map_lookup_batch_zero_copy(&big_table, h_sparse_keys, h_sparse_results, n_sparse);
    CUDA_CHECK(cudaEventRecord(ev2));
    CUDA_CHECK(cudaEventSynchronize(ev2));
    float zerocopy_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&zerocopy_ms, ev1, ev2));
    std::printf("  Zero-Copy path (no table migration):      %.3f ms\n", zerocopy_ms);
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaEventDestroy(ev2));

    heuristic_set_table_size(&state, table_bytes);
    std::printf("  Heuristic (table_size=%.2f GB, N=%zu): ", table_bytes / (1024.0 * 1024.0 * 1024.0), n_sparse);
    hash_map_lookup_batch_heuristic(&big_table, h_sparse_keys, h_sparse_results, n_sparse, &state);

    CUDA_CHECK(cudaFreeHost(h_sparse_keys));
    CUDA_CHECK(cudaFreeHost(h_sparse_results));
    hash_map_destroy(&big_table);
  } else {
    std::printf("  Skipped (insufficient VRAM for massive table).\n");
  }
  std::printf("\n");

  CUDA_CHECK(cudaFreeHost(h_pinned_keys));
  CUDA_CHECK(cudaFreeHost(h_pinned_results));
  hash_map_destroy(&table);

  std::printf("================================================================================\n");
  std::printf("  Heuristic benchmark done.\n");
  std::printf("================================================================================\n");

  return 0;
}
