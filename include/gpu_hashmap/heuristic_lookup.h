/**
 * @file heuristic_lookup.h
 * @brief Heuristic: choose Standard Copy vs Zero-Copy lookup by measured batch cost.
 *
 * Measured on this PCIe Gen3 laptop: zero-copy is fastest at *small* batches and the
 * advantage shrinks as N grows. The old rule ("prefer Zero-Copy when n >= crossover_n")
 * had the polarity wrong and almost never fired. The rule here is:
 *
 *   use Zero-Copy when n_lookups <= zerocopy_below_n
 *
 * calibrated across several batch sizes with a median of k reps and a safety margin.
 * If no size meets the margin, the state is explicitly Undecided rather than silently
 * defaulting to SIZE_MAX.
 */

#ifndef GPU_HASHMAP_HEURISTIC_LOOKUP_H
#define GPU_HASHMAP_HEURISTIC_LOOKUP_H

#include "gpu_hashmap/hash_map_api.h"
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>

namespace gpu_hashmap {

enum class LookupPath { StandardCopy, ZeroCopy, Undecided };

/**
 * State for the heuristic: batch-size rule, out-of-core fallback, sparsity, warm-up.
 *
 * `zerocopy_below_n`: use Zero-Copy when n_lookups <= this (inclusive). 0 means the
 * batch-size rule never picks Zero-Copy (still subject to sparsity / out-of-core).
 * `decided`: false when calibration found no size meeting the margin.
 */
struct HeuristicState {
  size_t zerocopy_below_n;      ///< use Zero-Copy when n <= this; 0 if none
  size_t crossover_n;           ///< deprecated alias kept for older call sites (= zerocopy_below_n or SIZE_MAX)
  size_t max_lookups_fit_vram;  ///< if n > this, force Zero-Copy (out-of-core)
  size_t table_size_bytes;      ///< sparsity: prefer Zero-Copy to avoid table migration tax
  bool warmed_up;
  bool decided;                 ///< false => batch-size rule is Undecided
};

/** Table size above which we consider "massive table" for sparsity-driven crossover. */
constexpr size_t kSparsityTableSizeThreshold = 1500ULL * 1024 * 1024;
/** Lookup count below which we consider "sparse" when table is massive. */
constexpr size_t kSparsityLookupCap = 100 * 1024;

/** Legacy default; warm-up overwrites it. Kept so un-warmed state is defined. */
constexpr size_t kHeuristicDefaultCrossoverN = 256 * 1024;

inline void heuristic_init(HeuristicState* state) {
  state->zerocopy_below_n = 0;
  state->crossover_n = kHeuristicDefaultCrossoverN;
  state->max_lookups_fit_vram = SIZE_MAX;
  state->table_size_bytes = 0;
  state->warmed_up = false;
  state->decided = false;
}

inline void heuristic_set_table_size(HeuristicState* state, size_t table_bytes) {
  state->table_size_bytes = table_bytes;
}

/**
 * Warm-up: measure Standard and Zero-Copy at several batch sizes (median of 3 after a
 * warm pass each), using keys drawn from the table. Sets zerocopy_below_n to the
 * largest N where Zero-Copy is at least (1 - margin) faster; if no such N, decided=false.
 */
void heuristic_warm_up(HashTable* table, HeuristicState* state,
                       cudaStream_t stream = nullptr);

/**
 * Choose path for n_lookups. Priority: out-of-core -> sparsity -> batch-size rule.
 * When the batch-size rule is undecided, returns Undecided (caller should treat as
 * StandardCopy for execution and report the uncertainty).
 */
inline LookupPath heuristic_choose_path(size_t n_lookups, HeuristicState const* state) {
  if (n_lookups > state->max_lookups_fit_vram)
    return LookupPath::ZeroCopy;
  if (state->table_size_bytes >= kSparsityTableSizeThreshold &&
      n_lookups <= kSparsityLookupCap)
    return LookupPath::ZeroCopy;
  if (!state->decided)
    return LookupPath::Undecided;
  return (n_lookups <= state->zerocopy_below_n) ? LookupPath::ZeroCopy
                                                : LookupPath::StandardCopy;
}

/**
 * Run lookup using the heuristic. Undecided falls through to Standard Copy and logs so.
 * For Zero-Copy, h_keys/h_results must be pinned (cudaHostAlloc Mapped).
 */
void hash_map_lookup_batch_heuristic(HashTable* table,
                                     KeyType* h_keys,
                                     ValueType* h_results,
                                     size_t n,
                                     HeuristicState* state,
                                     cudaStream_t stream = nullptr);

} // namespace gpu_hashmap

#endif // GPU_HASHMAP_HEURISTIC_LOOKUP_H
