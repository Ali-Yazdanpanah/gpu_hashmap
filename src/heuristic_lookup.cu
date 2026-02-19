/**
 * @file heuristic_lookup.cu
 * @brief Heuristic warm-up (measure PCIe/copy cost) and lookup batch dispatcher.
 *
 * Sparsity-Driven Crossover (mathematical validation):
 * ---------------------------------
 * The crossover between Standard and Zero-Copy path occurs when:
 *
 *   Time(Full Table Copy) > Time(Sparse PCIe Stalls)
 *
 * - Standard path: migrates the entire table to VRAM (H2D), then copies keys H2D,
 *   runs kernel, copies results D2H. For a massive table (e.g. 2GB) and sparse
 *   lookups (e.g. N = 10,000), the one-time "Copy Tax" of the table dominates.
 * - Zero-Copy path: table remains in pinned host memory; the GPU fetches only
 *   the buckets needed for those 10,000 keys over PCIe. No full table migration.
 * So for sparse workloads on a massive table, Zero-Copy is optimal despite
 * slower per-access PCIe latency, because we avoid the full table migration cost.
 */

#include "gpu_hashmap/heuristic_lookup.h"
#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/lookup_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

namespace {

constexpr size_t kWarmUpProbeN = 256 * 1024;
constexpr float kSafetyMargin = 0.8f;  /* Only use Zero-Copy if ZeroCopy_Total < Standard_Total * 0.8 */

/* Total end-to-end time: H2D + Kernel + D2H. Uses pre-allocated device buffers (no alloc in timed region). */
void measure_standard_total(HashTable* table, KeyType const* h_keys, ValueType* h_results,
                            KeyType* d_keys, ValueType* d_vals, size_t n,
                            float* out_total_ms, cudaStream_t stream) {
  cudaStream_t s = stream ? stream : (cudaStream_t)0;
  cudaEvent_t e1, e2;
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventCreate(&e2));
  CUDA_CHECK(cudaEventRecord(e1, s));
  CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys, n * sizeof(KeyType), cudaMemcpyHostToDevice, s));
  hash_map_lookup_kernel<<<(n + 255) / 256, 256, 0, s>>>(table->d_device_table, d_keys, d_vals, n);
  CUDA_CHECK(cudaMemcpyAsync(h_results, d_vals, n * sizeof(ValueType), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaEventRecord(e2, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  CUDA_CHECK(cudaEventElapsedTime(out_total_ms, e1, e2));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaEventDestroy(e2));
}

/* True Zero-Copy: kernel-only time; data already in pinned/mapped memory (pre-filled).
   No alloc, no std::memcpy. GPU writes directly into the mapped buffer; PCIe cache
   coherency allows the CPU to read results without explicit migration. */
void measure_zerocopy_total(HashTable* table, KeyType* d_keys_mapped, ValueType* d_vals_mapped,
                            size_t n, float* out_total_ms, cudaStream_t stream) {
  cudaStream_t s = stream ? stream : (cudaStream_t)0;
  cudaEvent_t e1, e2;
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventCreate(&e2));
  CUDA_CHECK(cudaEventRecord(e1, s));
  hash_map_lookup_kernel<<<(n + 255) / 256, 256, 0, s>>>(table->d_device_table, d_keys_mapped, d_vals_mapped, n);
  CUDA_CHECK(cudaEventRecord(e2, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  CUDA_CHECK(cudaEventElapsedTime(out_total_ms, e1, e2));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaEventDestroy(e2));
}

} // namespace

void heuristic_warm_up(HashTable* table, HeuristicState* state, cudaStream_t stream) {
  const size_t probe_n = kWarmUpProbeN;
  size_t free_vram = 0, total_vram = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
  /* Out-of-core: batch (keys+results) must fit in VRAM; use ~50% of free to leave room for table etc. */
  state->max_lookups_fit_vram = (free_vram / 2) / (sizeof(KeyType) + sizeof(ValueType));

  /* Pre-allocate buffers (separate from measurement). */
  const unsigned int pin_flags = cudaHostAllocMapped | cudaHostAllocPortable;
  void* h_pinned_keys = nullptr;
  void* h_pinned_vals = nullptr;
  KeyType* d_keys_mapped = nullptr;
  ValueType* d_vals_mapped = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_pinned_keys, probe_n * sizeof(KeyType), pin_flags));
  CUDA_CHECK(cudaHostAlloc(&h_pinned_vals, probe_n * sizeof(ValueType), pin_flags));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_keys_mapped, h_pinned_keys, 0));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_vals_mapped, h_pinned_vals, 0));

  KeyType* d_keys_std = nullptr;
  ValueType* d_vals_std = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys_std, probe_n * sizeof(KeyType)));
  CUDA_CHECK(cudaMalloc(&d_vals_std, probe_n * sizeof(ValueType)));

  /* Fill probe keys (assume data already in pinned memory for zero-copy path). */
  for (size_t i = 0; i < probe_n; ++i)
    static_cast<KeyType*>(h_pinned_keys)[i] = static_cast<KeyType>(i);

  /* Measure total end-to-end: Standard_Total = H2D + Kernel + D2H. */
  float standard_total_ms = 0.f;
  measure_standard_total(table, static_cast<KeyType const*>(h_pinned_keys),
                         static_cast<ValueType*>(h_pinned_vals),
                         d_keys_std, d_vals_std, probe_n, &standard_total_ms, stream);

  /* Measure ZeroCopy_Total = Kernel only (data already in pinned memory). */
  float zerocopy_total_ms = 0.f;
  measure_zerocopy_total(table, d_keys_mapped, d_vals_mapped, probe_n, &zerocopy_total_ms, stream);

  /* Free pre-allocated buffers. */
  CUDA_CHECK(cudaFreeHost(h_pinned_keys));
  CUDA_CHECK(cudaFreeHost(h_pinned_vals));
  CUDA_CHECK(cudaFree(d_keys_std));
  CUDA_CHECK(cudaFree(d_vals_std));

  /* Safety margin: only switch to Zero-Copy if ZeroCopy_Total < Standard_Total * 0.8. */
  if (zerocopy_total_ms < standard_total_ms * kSafetyMargin) {
    state->crossover_n = probe_n;  /* use Zero-Copy for n >= probe_n */
  } else {
    state->crossover_n = SIZE_MAX;  /* prefer Standard unless out-of-core */
  }
  state->warmed_up = true;
  std::fprintf(stderr, "[heuristic] warm-up: Standard_Total=%.2f ms, ZeroCopy_Total=%.2f ms (margin 0.8) -> crossover_n=%zu, max_lookups_fit_vram=%zu\n",
               standard_total_ms, zerocopy_total_ms, state->crossover_n, state->max_lookups_fit_vram);
}

void hash_map_lookup_batch_heuristic(HashTable* table,
                                     KeyType* h_keys,
                                     ValueType* h_results,
                                     size_t n,
                                     HeuristicState* state,
                                     cudaStream_t stream) {
  LookupPath path = heuristic_choose_path(n, state);
  bool sparsity_driven = (state->table_size_bytes >= kSparsityTableSizeThreshold &&
                           n <= kSparsityLookupCap);
  if (path == LookupPath::ZeroCopy) {
    hash_map_lookup_batch_zero_copy(table, h_keys, h_results, n, stream);
    if (sparsity_driven)
      std::printf("Sparsity detected: Zero-Copy chosen to bypass 2GB table migration tax.\n");
    else
      std::printf("Heuristic chose Zero-Copy for %zu lookups to minimize end-to-end latency.\n", n);
  } else {
    hash_map_lookup_batch_standard_copy(table, h_keys, h_results, n, stream);
    std::printf("Heuristic chose Standard Copy for %zu lookups to minimize end-to-end latency.\n", n);
  }
}

} // namespace gpu_hashmap
