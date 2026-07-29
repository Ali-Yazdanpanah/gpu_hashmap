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

/* Probe sizes span the region where Zero-Copy historically wins (small N) and
 * where the paths are close (256K+). The rule polarity is "ZC below N", so we
 * need the low end of the sweep, not only a single large probe. */
constexpr size_t kWarmUpProbeSizes[] = {
    16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024};
constexpr int kWarmUpProbeCount =
    static_cast<int>(sizeof(kWarmUpProbeSizes) / sizeof(kWarmUpProbeSizes[0]));
constexpr float kSafetyMargin = 0.8f;  /* ZC must be <= 0.8 * Standard to count */

float median3(float const* v) {
  if (v[0] < v[1]) {
    if (v[1] < v[2]) return v[1];
    return (v[0] < v[2]) ? v[2] : v[0];
  }
  if (v[0] < v[2]) return v[0];
  return (v[1] < v[2]) ? v[2] : v[1];
}

/* Collect keys that are actually stored, by walking the chains through the mapped
 * host view of the table. Falls back to dense integers when the table is empty. */
void fill_probe_keys_from_table(HashTable const* table, KeyType* out, size_t n) {
  size_t filled = 0;
  if (table->h_bucket_heads && table->h_nodes) {
    unsigned long long const* heads =
        static_cast<unsigned long long const*>(table->h_bucket_heads);
    Node const* nodes = static_cast<Node const*>(table->h_nodes);
    for (size_t b = 0; b < table->device.num_buckets && filled < n; ++b) {
      unsigned long long slot = heads[b];
      /* Bound the walk by capacity so a corrupt chain cannot spin forever. */
      size_t steps = 0;
      while (slot != kInvalidSlot && filled < n && steps++ < table->device.capacity) {
        out[filled++] = nodes[slot].key;
        slot = nodes[slot].next;
      }
    }
  }
  if (filled == 0) {
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<KeyType>(i);
    return;
  }
  /* Repeat what we found to cover the whole probe batch. */
  for (size_t i = filled; i < n; ++i) out[i] = out[i % filled];
}

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
  size_t free_vram = 0, total_vram = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
  state->max_lookups_fit_vram = (free_vram / 2) / (sizeof(KeyType) + sizeof(ValueType));

  const size_t max_n = kWarmUpProbeSizes[kWarmUpProbeCount - 1];
  const unsigned int pin_flags = cudaHostAllocMapped | cudaHostAllocPortable;
  void* h_pinned_keys = nullptr;
  void* h_pinned_vals = nullptr;
  KeyType* d_keys_mapped = nullptr;
  ValueType* d_vals_mapped = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_pinned_keys, max_n * sizeof(KeyType), pin_flags));
  CUDA_CHECK(cudaHostAlloc(&h_pinned_vals, max_n * sizeof(ValueType), pin_flags));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_keys_mapped, h_pinned_keys, 0));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_vals_mapped, h_pinned_vals, 0));

  KeyType* d_keys_std = nullptr;
  ValueType* d_vals_std = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys_std, max_n * sizeof(KeyType)));
  CUDA_CHECK(cudaMalloc(&d_vals_std, max_n * sizeof(ValueType)));

  fill_probe_keys_from_table(table, static_cast<KeyType*>(h_pinned_keys), max_n);

  /* First-touch warm so mapped pages are not paid inside the timed region. */
  float scratch_ms = 0.f;
  measure_standard_total(table, static_cast<KeyType const*>(h_pinned_keys),
                         static_cast<ValueType*>(h_pinned_vals), d_keys_std, d_vals_std,
                         max_n, &scratch_ms, stream);
  measure_zerocopy_total(table, d_keys_mapped, d_vals_mapped, max_n, &scratch_ms, stream);

  const int probe_reps = 3;
  size_t best_below = 0;
  bool any = false;
  std::fprintf(stderr,
               "[heuristic] warm-up (margin %.2f, median of %d): size  std_ms  zc_ms  ratio  keep?\n",
               kSafetyMargin, probe_reps);

  for (int i = 0; i < kWarmUpProbeCount; ++i) {
    const size_t n = kWarmUpProbeSizes[i];
    if (n > state->max_lookups_fit_vram) break;
    float std_samples[3], zc_samples[3];
    for (int r = 0; r < probe_reps; ++r) {
      measure_standard_total(table, static_cast<KeyType const*>(h_pinned_keys),
                             static_cast<ValueType*>(h_pinned_vals), d_keys_std, d_vals_std, n,
                             &std_samples[r], stream);
      measure_zerocopy_total(table, d_keys_mapped, d_vals_mapped, n, &zc_samples[r], stream);
    }
    const float std_ms = median3(std_samples);
    const float zc_ms = median3(zc_samples);
    const float ratio = zc_ms / (std_ms + 1e-9f);
    const bool keep = (zc_ms < std_ms * kSafetyMargin);
    std::fprintf(stderr, "  %7zu  %7.3f  %7.3f  %5.2f  %s\n", n, std_ms, zc_ms, ratio,
                 keep ? "yes" : "no");
    if (keep) {
      best_below = n;
      any = true;
    }
  }

  CUDA_CHECK(cudaFreeHost(h_pinned_keys));
  CUDA_CHECK(cudaFreeHost(h_pinned_vals));
  CUDA_CHECK(cudaFree(d_keys_std));
  CUDA_CHECK(cudaFree(d_vals_std));

  /* Polarity: Zero-Copy for n <= best_below. If nothing met the margin, leave
   * decided=false so callers see Undecided instead of a silent SIZE_MAX. */
  state->zerocopy_below_n = any ? best_below : 0;
  state->crossover_n = any ? best_below : SIZE_MAX; /* legacy field */
  state->decided = any;
  state->warmed_up = true;
  if (any)
    std::fprintf(stderr,
                 "[heuristic] decided: Zero-Copy when n <= %zu; else Standard "
                 "(max_lookups_fit_vram=%zu)\n",
                 state->zerocopy_below_n, state->max_lookups_fit_vram);
  else
    std::fprintf(stderr,
                 "[heuristic] UNDECIDED: no probe size met margin %.2f; batch-size "
                 "rule will not pick Zero-Copy (max_lookups_fit_vram=%zu)\n",
                 kSafetyMargin, state->max_lookups_fit_vram);
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
      std::printf("Heuristic chose Zero-Copy for %zu lookups (n <= %zu).\n", n,
                  state->zerocopy_below_n);
  } else if (path == LookupPath::Undecided) {
    /* Execute Standard but do not pretend calibration picked it. */
    hash_map_lookup_batch_standard_copy(table, h_keys, h_results, n, stream);
    std::printf("Heuristic UNDECIDED for %zu lookups; falling back to Standard Copy.\n", n);
  } else {
    hash_map_lookup_batch_standard_copy(table, h_keys, h_results, n, stream);
    std::printf("Heuristic chose Standard Copy for %zu lookups (n > %zu).\n", n,
                state->zerocopy_below_n);
  }
}

} // namespace gpu_hashmap
