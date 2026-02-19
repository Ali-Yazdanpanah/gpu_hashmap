/**
 * @file performance_validation_suite.cu
 * @brief Comprehensive performance analysis & validation suite for academic paper metrics.
 *
 * 1. Memory: Effective BW, load efficiency, occupancy
 * 2. Stress: Zipfian (hot-key), load factor sweep, probe depth, sparsity-driven crossover
 * 3. Multi-GPU: P2P vs host-staged, weak/strong scaling
 * 4. Theoretical: Measured vs peak bandwidth
 * 5. Validation: Gold standard CPU bit-perfect verification
 *
 * Sparsity-Driven Crossover (mathematical validation in 2d):
 *   Time(Full Table Copy) > Time(Sparse PCIe Stalls)
 * => Zero-Copy chosen to bypass massive table migration tax for sparse lookups.
 */

#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/heuristic_lookup.h"
#include "gpu_hashmap/insert_kernel.cuh"
#include "gpu_hashmap/lookup_kernel.cuh"
#include "gpu_hashmap/hash_buckets.cuh"
#include "gpu_hashmap/analysis/workloads.hpp"
#include "gpu_hashmap/analysis/metrics.hpp"
#include "gpu_hashmap/analysis/validation.hpp"
#include "gpu_hashmap/analysis/probe_depth.cuh"
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

__global__ void lookup_kernel_impl(gpu_hashmap::HashTableDevice const* table,
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

namespace {

void run_insert_lookup(gpu_hashmap::HashTable* table,
                       gpu_hashmap::KeyType const* d_keys,
                       gpu_hashmap::ValueType const* d_values,
                       size_t n,
                       gpu_hashmap::KeyType const* d_lookup_keys,
                       gpu_hashmap::ValueType* d_lookup_out,
                       size_t m,
                       float* out_insert_ms,
                       float* out_lookup_ms) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  gpu_hashmap::hash_map_insert_batch(table, d_keys, d_values, n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(out_insert_ms, start, stop));
  CUDA_CHECK(cudaEventRecord(start));
  lookup_kernel_impl<<<(m + 255) / 256, 256>>>(table->d_device_table, d_lookup_keys, d_lookup_out, m);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(out_lookup_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

} // namespace

int main() {
  using namespace gpu_hashmap;
  using namespace gpu_hashmap::analysis;

  /* Enable mapped host memory for zero-copy hash table (must be before other CUDA calls). */
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

  std::printf("================================================================================\n");
  std::printf("  CUDA Key-Value Store — Performance Analysis & Validation Suite\n");
  std::printf("================================================================================\n\n");

  char gpu_name[256];
  get_device_name(gpu_name, sizeof(gpu_name));
  std::printf("  Device: %s\n", gpu_name);
  double peak_bw = get_peak_memory_bandwidth_gbps();
  std::printf("  Peak memory bandwidth (theoretical): %.2f GB/s\n\n", peak_bw);

  const size_t num_buckets = 1 << 18;
  const size_t capacity = 2 * 1024 * 1024;
  const size_t n_default = 512 * 1024;
  const size_t m_lookup = 256 * 1024;

  std::vector<KeyType> h_keys(n_default), h_lookup_keys(m_lookup);
  std::vector<ValueType> h_values(n_default);
  fill_uniform_keys_values(h_keys, h_values, n_default, 42);
  for (size_t i = 0; i < m_lookup; ++i)
    h_lookup_keys[i] = h_keys[i % n_default];

  size_t mismatch_idx = 0;
  ValueType expected = 0, got = 0;
  bool gold_ok = true;

  // ---------- 1. Memory architecture metrics ----------
  std::printf("--------------------------------------------------------------------------------\n");
  std::printf("  1. MEMORY ARCHITECTURE METRICS\n");
  std::printf("--------------------------------------------------------------------------------\n");
  HashTable table = {};
  hash_map_create(&table, num_buckets, capacity);
  KeyType* d_keys = nullptr;
  ValueType* d_values = nullptr;
  KeyType* d_lookup_keys = nullptr;
  ValueType* d_lookup_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, n_default * sizeof(KeyType)));
  CUDA_CHECK(cudaMalloc(&d_values, n_default * sizeof(ValueType)));
  CUDA_CHECK(cudaMalloc(&d_lookup_keys, m_lookup * sizeof(KeyType)));
  CUDA_CHECK(cudaMalloc(&d_lookup_out, m_lookup * sizeof(ValueType)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n_default * sizeof(KeyType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), n_default * sizeof(ValueType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lookup_keys, h_lookup_keys.data(), m_lookup * sizeof(KeyType), cudaMemcpyHostToDevice));

  float insert_ms = 0, lookup_ms = 0;
  run_insert_lookup(&table, d_keys, d_values, n_default, d_lookup_keys, d_lookup_out, m_lookup, &insert_ms, &lookup_ms);

  size_t bytes_read_insert = n_default * (sizeof(KeyType) + sizeof(ValueType));
  size_t bytes_written_insert = n_default * (sizeof(KeyType) + sizeof(ValueType)) + capacity * sizeof(Node) / 2;
  size_t bytes_read_lookup = m_lookup * sizeof(KeyType);
  size_t bytes_written_lookup = m_lookup * sizeof(ValueType);
  double time_insert_sec = insert_ms / 1000.0;
  double time_lookup_sec = lookup_ms / 1000.0;
  double bw_insert = effective_bandwidth_gbps(bytes_read_insert, bytes_written_insert, time_insert_sec);
  double bw_lookup = effective_bandwidth_gbps(bytes_read_lookup, bytes_written_lookup, time_lookup_sec);
  double load_eff = estimate_load_efficiency_coalesced(n_default, sizeof(KeyType) + sizeof(ValueType), 32);

  std::printf("  Effective memory bandwidth (insert): %.2f GB/s\n", bw_insert);
  std::printf("  Effective memory bandwidth (lookup): %.2f GB/s\n", bw_lookup);
  std::printf("  Global memory load efficiency (estimated, coalesced): %.0f%%\n", load_eff * 100);
  int max_blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, (const void*)insert_kernel, 256, 256 * sizeof(SlotIndex));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int max_warps_sm = prop.maxThreadsPerMultiProcessor / 32;
  int warps_per_block = 256 / 32;
  double occ = static_cast<double>(max_blocks * warps_per_block) / max_warps_sm;
  if (occ > 1.0) occ = 1.0;
  std::printf("  Occupancy (insert kernel, 256 threads): %.2f (max %d blocks/SM)\n\n", occ, max_blocks);

  // ---------- 2a. Zipfian stress ----------
  std::printf("--------------------------------------------------------------------------------\n");
  std::printf("  2a. STRESS — Zipfian (hot-key) workload\n");
  std::printf("--------------------------------------------------------------------------------\n");
  const size_t zipf_keys = 100000;
  const size_t zipf_samples = 500000;
  std::vector<KeyType> zipf_k(zipf_samples), zipf_v(zipf_samples);
  for (size_t i = 0; i < zipf_samples; ++i) zipf_v[i] = i + 1000;
  double alphas[] = {0.5, 1.0, 1.5, 2.0};
  std::printf("  %-8s  %-10s  %-10s\n", "alpha", "Insert(ms)", "Lookup(ms)");
  for (double alpha : alphas) {
    ZipfianGenerator zipf(zipf_keys, alpha);
    zipf.generate(zipf_k, zipf_samples);
    KeyType* dz = nullptr;
    ValueType* vz = nullptr;
    CUDA_CHECK(cudaMalloc(&dz, zipf_samples * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&vz, zipf_samples * sizeof(ValueType)));
    CUDA_CHECK(cudaMemcpy(dz, zipf_k.data(), zipf_samples * sizeof(KeyType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vz, zipf_v.data(), zipf_samples * sizeof(ValueType), cudaMemcpyHostToDevice));
    HashTable tz = {};
    hash_map_create(&tz, num_buckets, capacity);
    KeyType* dlk = nullptr;
    ValueType* dlo = nullptr;
    CUDA_CHECK(cudaMalloc(&dlk, zipf_samples * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&dlo, zipf_samples * sizeof(ValueType)));
    CUDA_CHECK(cudaMemcpy(dlk, zipf_k.data(), zipf_samples * sizeof(KeyType), cudaMemcpyHostToDevice));
    cudaEvent_t e1, e2;
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventRecord(e1));
    hash_map_insert_batch(&tz, dz, vz, zipf_samples);
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaEventSynchronize(e2));
    float ins_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ins_ms, e1, e2));
    CUDA_CHECK(cudaEventRecord(e1));
    lookup_kernel_impl<<<(zipf_samples + 255) / 256, 256>>>(tz.d_device_table, dlk, dlo, zipf_samples);
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaEventSynchronize(e2));
    float lup_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&lup_ms, e1, e2));
    std::printf("  %-8.1f  %-10.2f  %-10.2f\n", alpha, ins_ms, lup_ms);
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2));
    CUDA_CHECK(cudaFree(dz));
    CUDA_CHECK(cudaFree(vz));
    CUDA_CHECK(cudaFree(dlk));
    CUDA_CHECK(cudaFree(dlo));
    hash_map_destroy(&tz);
  }
  std::printf("\n");

  // ---------- 2b. Load factor sweep ----------
  std::printf("--------------------------------------------------------------------------------\n");
  std::printf("  2b. LOAD FACTOR SWEEP (10%% to 99%%)\n");
  std::printf("--------------------------------------------------------------------------------\n");
  std::printf("  %-10s  %-12s  %-12s  %-10s\n", "LoadFactor", "Insert(ms)", "Lookup(ms)", "Throughput");
  for (double lf : {0.10, 0.30, 0.50, 0.70, 0.90, 0.99}) {
    size_t n_lf = static_cast<size_t>(capacity * lf);
    if (n_lf == 0) n_lf = 1;
    std::vector<KeyType> k_lf(n_lf);
    std::vector<ValueType> v_lf(n_lf);
    fill_uniform_keys_values(k_lf, v_lf, n_lf, static_cast<uint64_t>(lf * 1000));
    KeyType* dk = nullptr;
    ValueType* dv = nullptr;
    CUDA_CHECK(cudaMalloc(&dk, n_lf * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&dv, n_lf * sizeof(ValueType)));
    CUDA_CHECK(cudaMemcpy(dk, k_lf.data(), n_lf * sizeof(KeyType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, v_lf.data(), n_lf * sizeof(ValueType), cudaMemcpyHostToDevice));
    HashTable t_lf = {};
    hash_map_create(&t_lf, num_buckets, capacity);
    KeyType* dl_lf = nullptr;
    ValueType* do_lf = nullptr;
    CUDA_CHECK(cudaMalloc(&dl_lf, n_lf * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&do_lf, n_lf * sizeof(ValueType)));
    CUDA_CHECK(cudaMemcpy(dl_lf, k_lf.data(), n_lf * sizeof(KeyType), cudaMemcpyHostToDevice));
    float ins_lf = 0, lup_lf = 0;
    run_insert_lookup(&t_lf, dk, dv, n_lf, dl_lf, do_lf, n_lf, &ins_lf, &lup_lf);
    double throughput = (ins_lf + lup_lf) > 0 ? (n_lf * 2.0) / (ins_lf + lup_lf) * 1000.0 : 0;
    std::printf("  %-10.0f%%  %-12.2f  %-12.2f  %-10.0f ops/s\n", lf * 100, ins_lf, lup_lf, throughput);
    CUDA_CHECK(cudaFree(dk));
    CUDA_CHECK(cudaFree(dv));
    CUDA_CHECK(cudaFree(dl_lf));
    CUDA_CHECK(cudaFree(do_lf));
    hash_map_destroy(&t_lf);
  }
  std::printf("\n");

  // ---------- 2c. Probe depth ----------
  std::printf("--------------------------------------------------------------------------------\n");
  std::printf("  2c. PROBE DEPTH (avg chain steps per find)\n");
  std::printf("--------------------------------------------------------------------------------\n");
  unsigned int* d_depth = nullptr;
  CUDA_CHECK(cudaMalloc(&d_depth, m_lookup * sizeof(unsigned int)));
  analysis::lookup_with_probe_depth<<<(m_lookup + 255) / 256, 256>>>(
      table.d_device_table, d_lookup_keys, d_lookup_out, d_depth, m_lookup);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<unsigned int> h_depth(m_lookup);
  std::vector<ValueType> h_out(m_lookup);
  CUDA_CHECK(cudaMemcpy(h_depth.data(), d_depth, m_lookup * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_lookup_out, m_lookup * sizeof(ValueType), cudaMemcpyDeviceToHost));
  double sum_hit = 0, sum_miss = 0;
  int count_hit = 0, count_miss = 0;
  for (size_t i = 0; i < m_lookup; ++i) {
    if (h_out[i] != 0xFFFFFFFFFFFFFFFFull) {
      sum_hit += h_depth[i];
      count_hit++;
    } else {
      sum_miss += h_depth[i];
      count_miss++;
    }
  }
  std::printf("  Successful find — avg probe depth: %.3f (count %d)\n",
              count_hit > 0 ? sum_hit / count_hit : 0, count_hit);
  std::printf("  Unsuccessful find — avg probe depth: %.3f (count %d)\n\n",
              count_miss > 0 ? sum_miss / count_miss : 0, count_miss);
  CUDA_CHECK(cudaFree(d_depth));

  // ---------- 2d. Sparsity-Driven Crossover (massive table + N=10,000) ----------
  /*
   * Crossover occurs when: Time(Full Table Copy) > Time(Sparse PCIe Stalls).
   * Standard path: full table H2D + keys H2D + kernel + results D2H (Copy Tax).
   * Zero-Copy path: table in mapped memory; GPU fetches only needed buckets over PCIe.
   */
  std::printf("--------------------------------------------------------------------------------\n");
  std::printf("  2d. SPARSITY-DRIVEN CROSSOVER (Massive Table + Sparse Lookups)\n");
  std::printf("--------------------------------------------------------------------------------\n");
  size_t free_vram_s = 0, total_vram_s = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_vram_s, &total_vram_s));
  const size_t two_gb_s = 2ULL * 1024 * 1024 * 1024;
  const size_t target_bytes = (two_gb_s < (size_t)((double)free_vram_s * 0.8)) ? two_gb_s : (size_t)((double)free_vram_s * 0.8);
  const size_t nb_sparse = 1 << 21;
  const size_t cap_sparse = (target_bytes - nb_sparse * sizeof(unsigned long long)) / sizeof(gpu_hashmap::Node);
  if (cap_sparse > 1024 * 1024) {
    const size_t table_bytes = nb_sparse * sizeof(unsigned long long) + cap_sparse * sizeof(gpu_hashmap::Node);
    /* Cap n_fill so hash_map_upload_from_host stays fast; large capacity alone gives migration tax. */
    const size_t n_fill_cap = 128 * 1024;
    const size_t n_fill = (cap_sparse / 40 > n_fill_cap) ? n_fill_cap : (cap_sparse / 40);
    std::printf("  Massive table: %.2f GB; fill %zu entries; sparse N = 10,000\n",
                table_bytes / (1024.0 * 1024.0 * 1024.0), n_fill);

    gpu_hashmap::HashTable tab_sparse = {};
    hash_map_create(&tab_sparse, nb_sparse, cap_sparse);
    std::vector<KeyType> h_fill_k(n_fill), h_sparse_k(10000);
    std::vector<ValueType> h_fill_v(n_fill);
    fill_uniform_keys_values(h_fill_k, h_fill_v, n_fill, 12345);
    for (size_t i = 0; i < 10000; ++i) h_sparse_k[i] = h_fill_k[i % n_fill];
    hash_map_upload_from_host(&tab_sparse, h_fill_k.data(), h_fill_v.data(), n_fill);
    CUDA_CHECK(cudaDeviceSynchronize());

    const unsigned int pin_f = cudaHostAllocMapped | cudaHostAllocPortable;
    KeyType* h_pin_k = nullptr;
    ValueType* h_pin_v = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_pin_k, 10000 * sizeof(KeyType), pin_f));
    CUDA_CHECK(cudaHostAlloc(&h_pin_v, 10000 * sizeof(ValueType), pin_f));
    for (size_t i = 0; i < 10000; ++i) h_pin_k[i] = h_sparse_k[i];

    void* d_h = nullptr, *d_n = nullptr;
    CUDA_CHECK(cudaMalloc(&d_h, nb_sparse * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_n, cap_sparse * sizeof(gpu_hashmap::Node)));
    cudaEvent_t se1, se2;
    CUDA_CHECK(cudaEventCreate(&se1));
    CUDA_CHECK(cudaEventCreate(&se2));
    CUDA_CHECK(cudaEventRecord(se1));
    CUDA_CHECK(cudaMemcpy(d_h, tab_sparse.h_bucket_heads, nb_sparse * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n, tab_sparse.h_nodes, cap_sparse * sizeof(gpu_hashmap::Node), cudaMemcpyHostToDevice));
    gpu_hashmap::HashTableDevice h_dev = {};
    h_dev.bucket_heads = static_cast<unsigned long long*>(d_h);
    h_dev.nodes = static_cast<gpu_hashmap::Node*>(d_n);
    h_dev.slab = tab_sparse.device.slab;
    h_dev.num_buckets = nb_sparse;
    h_dev.capacity = cap_sparse;
    gpu_hashmap::HashTableDevice* d_tab = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tab, sizeof(gpu_hashmap::HashTableDevice)));
    CUDA_CHECK(cudaMemcpy(d_tab, &h_dev, sizeof(gpu_hashmap::HashTableDevice), cudaMemcpyHostToDevice));
    KeyType* d_k = nullptr;
    ValueType* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_k, 10000 * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&d_v, 10000 * sizeof(ValueType)));
    CUDA_CHECK(cudaMemcpy(d_k, h_pin_k, 10000 * sizeof(KeyType), cudaMemcpyHostToDevice));
    gpu_hashmap::hash_map_lookup_kernel<<<(10000 + 255) / 256, 256>>>(d_tab, d_k, d_v, 10000);
    CUDA_CHECK(cudaMemcpy(h_pin_v, d_v, 10000 * sizeof(ValueType), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(se2));
    CUDA_CHECK(cudaEventSynchronize(se2));
    float std_migration_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&std_migration_ms, se1, se2));
    std::printf("  Standard (full table copy + lookup): %.3f ms\n", std_migration_ms);
    CUDA_CHECK(cudaFree(d_tab));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(d_n));
    CUDA_CHECK(cudaEventDestroy(se1));
    CUDA_CHECK(cudaEventDestroy(se2));

    CUDA_CHECK(cudaEventCreate(&se1));
    CUDA_CHECK(cudaEventCreate(&se2));
    CUDA_CHECK(cudaEventRecord(se1));
    hash_map_lookup_batch_zero_copy(&tab_sparse, h_pin_k, h_pin_v, 10000);
    CUDA_CHECK(cudaEventRecord(se2));
    CUDA_CHECK(cudaEventSynchronize(se2));
    float zc_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&zc_ms, se1, se2));
    std::printf("  Zero-Copy (no table migration):      %.3f ms\n", zc_ms);
    CUDA_CHECK(cudaEventDestroy(se1));
    CUDA_CHECK(cudaEventDestroy(se2));

    gpu_hashmap::HeuristicState h_state = {};
    heuristic_init(&h_state);
    heuristic_set_table_size(&h_state, table_bytes);
    std::printf("  Heuristic: ");
    hash_map_lookup_batch_heuristic(&tab_sparse, h_pin_k, h_pin_v, 10000, &h_state);

    CUDA_CHECK(cudaFreeHost(h_pin_k));
    CUDA_CHECK(cudaFreeHost(h_pin_v));
    hash_map_destroy(&tab_sparse);
  } else {
    std::printf("  Skipped (insufficient VRAM for massive table).\n");
  }
  std::printf("\n");

  // ---------- 3. Multi-GPU (if available) ----------
  std::printf("--------------------------------------------------------------------------------\n");
  std::printf("  3. MULTI-GPU & DISTRIBUTED ANALYTICS\n");
  std::printf("--------------------------------------------------------------------------------\n");
  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  if (ndev < 2) {
    std::printf("  Skipped (single GPU). Need 2+ GPUs for P2P and scaling.\n\n");
  } else {
    int can_access_01 = 0, can_access_10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_01, 1, 0));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_10, 0, 1));
    if (!can_access_01 || !can_access_10) {
      std::printf("  Skipped (P2P not enabled between GPU 0 and 1). Run with cudaDeviceEnablePeerAccess or skip.\n\n");
    } else {
    const size_t transfer_size = 64 * 1024 * 1024;
    void* d0 = nullptr, *d1 = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d0, transfer_size));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&d1, transfer_size));
    cudaEvent_t pe1, pe2;
    CUDA_CHECK(cudaEventCreate(&pe1));
    CUDA_CHECK(cudaEventCreate(&pe2));
    CUDA_CHECK(cudaEventRecord(pe1));
    for (int rep = 0; rep < 10; ++rep)
      CUDA_CHECK(cudaMemcpyPeer(d1, 1, d0, 0, transfer_size));
    CUDA_CHECK(cudaEventRecord(pe2));
    CUDA_CHECK(cudaEventSynchronize(pe2));
    float p2p_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&p2p_ms, pe1, pe2));
    double p2p_gbps = (transfer_size * 10.0 / 1e9) / (p2p_ms / 1000.0);
    std::vector<char> host_buf(transfer_size);
    CUDA_CHECK(cudaEventRecord(pe1));
    for (int rep = 0; rep < 10; ++rep) {
      CUDA_CHECK(cudaSetDevice(0));
      CUDA_CHECK(cudaMemcpy(host_buf.data(), d0, transfer_size, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaSetDevice(1));
      CUDA_CHECK(cudaMemcpy(d1, host_buf.data(), transfer_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaEventRecord(pe2));
    CUDA_CHECK(cudaEventSynchronize(pe2));
    float staged_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&staged_ms, pe1, pe2));
    double staged_gbps = (transfer_size * 10.0 * 2 / 1e9) / (staged_ms / 1000.0);
    std::printf("  P2P throughput (10 x 64MB):     %.2f GB/s\n", p2p_gbps);
    std::printf("  Host-staged throughput (10 x):  %.2f GB/s\n", staged_gbps);

    // Strong scaling: fixed total work (n_strong), 1 GPU vs 2 GPUs
    const size_t n_strong = 256 * 1024;
    gpu_hashmap::HashTable t1 = {};
    CUDA_CHECK(cudaSetDevice(0));
    hash_map_create(&t1, num_buckets, capacity);
    std::vector<KeyType> hs_keys(n_strong), hs_look(n_strong);
    std::vector<ValueType> hs_vals(n_strong);
    fill_uniform_keys_values(hs_keys, hs_vals, n_strong, 99);
    for (size_t i = 0; i < n_strong; ++i) hs_look[i] = hs_keys[i % n_strong];
    KeyType *ds_k = nullptr, *ds_lk = nullptr;
    ValueType *ds_v = nullptr, *ds_out = nullptr;
    CUDA_CHECK(cudaMalloc(&ds_k, n_strong * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&ds_v, n_strong * sizeof(ValueType)));
    CUDA_CHECK(cudaMalloc(&ds_lk, n_strong * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&ds_out, n_strong * sizeof(ValueType)));
    CUDA_CHECK(cudaMemcpy(ds_k, hs_keys.data(), n_strong * sizeof(KeyType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds_v, hs_vals.data(), n_strong * sizeof(ValueType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds_lk, hs_look.data(), n_strong * sizeof(KeyType), cudaMemcpyHostToDevice));
    float strong_1gpu_ms = 0, strong_2gpu_ms = 0;
    run_insert_lookup(&t1, ds_k, ds_v, n_strong, ds_lk, ds_out, n_strong, &strong_1gpu_ms, &strong_2gpu_ms);
    float strong_1gpu_total = strong_1gpu_ms + strong_2gpu_ms;
    hash_map_destroy(&t1);
    gpu_hashmap::HashTable t2a = {}, t2b = {};
    CUDA_CHECK(cudaSetDevice(0));
    hash_map_create(&t2a, num_buckets, capacity);
    CUDA_CHECK(cudaSetDevice(1));
    hash_map_create(&t2b, num_buckets, capacity);
    size_t half = n_strong / 2;
    CUDA_CHECK(cudaSetDevice(0));
    KeyType *d2a_k = nullptr, *d2a_v = nullptr, *d2a_lk = nullptr, *d2a_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d2a_k, half * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&d2a_v, half * sizeof(ValueType)));
    CUDA_CHECK(cudaMalloc(&d2a_lk, half * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&d2a_out, half * sizeof(ValueType)));
    CUDA_CHECK(cudaMemcpy(d2a_k, hs_keys.data(), half * sizeof(KeyType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d2a_v, hs_vals.data(), half * sizeof(ValueType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d2a_lk, hs_look.data(), half * sizeof(KeyType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaSetDevice(1));
    KeyType *d2b_k = nullptr, *d2b_v = nullptr, *d2b_lk = nullptr, *d2b_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d2b_k, (n_strong - half) * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&d2b_v, (n_strong - half) * sizeof(ValueType)));
    CUDA_CHECK(cudaMalloc(&d2b_lk, (n_strong - half) * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&d2b_out, (n_strong - half) * sizeof(ValueType)));
    CUDA_CHECK(cudaMemcpy(d2b_k, hs_keys.data() + half, (n_strong - half) * sizeof(KeyType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d2b_v, hs_vals.data() + half, (n_strong - half) * sizeof(ValueType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d2b_lk, hs_look.data() + half, (n_strong - half) * sizeof(KeyType), cudaMemcpyHostToDevice));
    float i0 = 0, l0 = 0, i1 = 0, l1 = 0;
    run_insert_lookup(&t2a, d2a_k, d2a_v, half, d2a_lk, d2a_out, half, &i0, &l0);
    run_insert_lookup(&t2b, d2b_k, d2b_v, n_strong - half, d2b_lk, d2b_out, n_strong - half, &i1, &l1);
    float strong_2gpu_total = (i0 + l0) + (i1 + l1);
    double strong_eff = 100.0 * strong_1gpu_total / (2.0 * strong_2gpu_total + 1e-9);
    std::printf("  Strong scaling (fixed N=%zu): 1-GPU %.2f ms, 2-GPU %.2f ms -> Efficiency %.1f%%\n",
                n_strong, strong_1gpu_total, strong_2gpu_total, strong_eff);
    hash_map_destroy(&t2a);
    hash_map_destroy(&t2b);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(d2a_k)); CUDA_CHECK(cudaFree(d2a_v)); CUDA_CHECK(cudaFree(d2a_lk)); CUDA_CHECK(cudaFree(d2a_out));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(d2b_k)); CUDA_CHECK(cudaFree(d2b_v)); CUDA_CHECK(cudaFree(d2b_lk)); CUDA_CHECK(cudaFree(d2b_out));

    // Weak scaling: work per GPU constant (n_weak per GPU)
    const size_t n_weak = 128 * 1024;
    CUDA_CHECK(cudaSetDevice(0));
    gpu_hashmap::HashTable tw1 = {};
    hash_map_create(&tw1, num_buckets, capacity);
    std::vector<KeyType> hw_k(n_weak), hw_l(n_weak);
    std::vector<ValueType> hw_v(n_weak);
    fill_uniform_keys_values(hw_k, hw_v, n_weak, 77);
    for (size_t i = 0; i < n_weak; ++i) hw_l[i] = hw_k[i % n_weak];
    KeyType *dw_k = nullptr, *dw_lk = nullptr;
    ValueType *dw_v = nullptr, *dw_out = nullptr;
    CUDA_CHECK(cudaMalloc(&dw_k, n_weak * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&dw_v, n_weak * sizeof(ValueType)));
    CUDA_CHECK(cudaMalloc(&dw_lk, n_weak * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&dw_out, n_weak * sizeof(ValueType)));
    CUDA_CHECK(cudaMemcpy(dw_k, hw_k.data(), n_weak * sizeof(KeyType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw_v, hw_v.data(), n_weak * sizeof(ValueType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw_lk, hw_l.data(), n_weak * sizeof(KeyType), cudaMemcpyHostToDevice));
    float weak_1gpu_ms = 0, weak_1gpu_lk = 0;
    run_insert_lookup(&tw1, dw_k, dw_v, n_weak, dw_lk, dw_out, n_weak, &weak_1gpu_ms, &weak_1gpu_lk);
    float weak_1gpu_total = weak_1gpu_ms + weak_1gpu_lk;
    hash_map_destroy(&tw1);
    gpu_hashmap::HashTable tw2a = {}, tw2b = {};
    CUDA_CHECK(cudaSetDevice(0));
    hash_map_create(&tw2a, num_buckets, capacity);
    CUDA_CHECK(cudaSetDevice(1));
    hash_map_create(&tw2b, num_buckets, capacity);
    KeyType *dw2_k = nullptr, *dw2_lk = nullptr;
    ValueType *dw2_v = nullptr, *dw2_out = nullptr;
    CUDA_CHECK(cudaMalloc(&dw2_k, n_weak * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&dw2_v, n_weak * sizeof(ValueType)));
    CUDA_CHECK(cudaMalloc(&dw2_lk, n_weak * sizeof(KeyType)));
    CUDA_CHECK(cudaMalloc(&dw2_out, n_weak * sizeof(ValueType)));
    CUDA_CHECK(cudaMemcpy(dw2_k, hw_k.data(), n_weak * sizeof(KeyType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw2_v, hw_v.data(), n_weak * sizeof(ValueType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw2_lk, hw_l.data(), n_weak * sizeof(KeyType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaSetDevice(0));
    run_insert_lookup(&tw2a, dw_k, dw_v, n_weak, dw_lk, dw_out, n_weak, &i0, &l0);
    CUDA_CHECK(cudaSetDevice(1));
    run_insert_lookup(&tw2b, dw2_k, dw2_v, n_weak, dw2_lk, dw2_out, n_weak, &i1, &l1);
    float weak_2gpu_total = (i0 + l0 > i1 + l1) ? (i0 + l0) : (i1 + l1);
    double weak_eff = 100.0 * weak_1gpu_total / (weak_2gpu_total + 1e-9);
    std::printf("  Weak scaling (N per GPU=%zu): 1-GPU %.2f ms, 2-GPU max %.2f ms -> Efficiency %.1f%%\n",
                n_weak, weak_1gpu_total, weak_2gpu_total, weak_eff);
    hash_map_destroy(&tw2a);
    hash_map_destroy(&tw2b);
    CUDA_CHECK(cudaFree(dw2_k)); CUDA_CHECK(cudaFree(dw2_v)); CUDA_CHECK(cudaFree(dw2_lk)); CUDA_CHECK(cudaFree(dw2_out));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(dw_k)); CUDA_CHECK(cudaFree(dw_v)); CUDA_CHECK(cudaFree(dw_lk)); CUDA_CHECK(cudaFree(dw_out));
    CUDA_CHECK(cudaFree(ds_k)); CUDA_CHECK(cudaFree(ds_v)); CUDA_CHECK(cudaFree(ds_lk)); CUDA_CHECK(cudaFree(ds_out));
    std::printf("\n");
    CUDA_CHECK(cudaEventDestroy(pe1));
    CUDA_CHECK(cudaEventDestroy(pe2));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(d0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(d1));
    }
  }

  // ---------- 4. Theoretical comparison ----------
  std::printf("--------------------------------------------------------------------------------\n");
  std::printf("  4. THEORETICAL COMPARISON (measured vs peak bandwidth)\n");
  std::printf("--------------------------------------------------------------------------------\n");
  std::printf("  %-30s  %12s  %12s  %10s\n", "Operation", "Measured(GB/s)", "Peak(GB/s)", "Efficiency");
  std::printf("  %-30s  %12.2f  %12.2f  %9.1f%%\n", "Insert (effective)", bw_insert, peak_bw, 100.0 * bw_insert / (peak_bw + 1e-9));
  std::printf("  %-30s  %12.2f  %12.2f  %9.1f%%\n", "Lookup (effective)", bw_lookup, peak_bw, 100.0 * bw_lookup / (peak_bw + 1e-9));
  std::printf("\n");

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_lookup_keys));
  CUDA_CHECK(cudaFree(d_lookup_out));
  hash_map_destroy(&table);

  // ---------- 5. Validation: Gold standard (run last to avoid long CPU work before GPU metrics) ----------
  std::printf("--------------------------------------------------------------------------------\n");
  std::printf("  5. VALIDATION — Gold standard (bit-perfect vs std::unordered_map)\n");
  std::printf("--------------------------------------------------------------------------------\n");
  gold_ok = validate_against_cpu_gold(h_keys, h_values, h_lookup_keys,
                                      num_buckets, capacity,
                                      &mismatch_idx, &expected, &got);
  if (gold_ok)
    std::printf("  Result: PASS — All GPU insert/lookup results match CPU gold.\n");
  else
    std::printf("  Result: FAIL — First mismatch at index %zu (expected %llu, got %llu).\n",
                (size_t)mismatch_idx, (unsigned long long)expected, (unsigned long long)got);
  std::printf("\n");

  std::printf("================================================================================\n");
  std::printf("  Suite complete. Use Nsight Compute for exact load efficiency.\n");
  std::printf("================================================================================\n");

  return gold_ok ? 0 : 1;
}
