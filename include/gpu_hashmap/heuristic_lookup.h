/**
 * @file heuristic_lookup.h
 * @brief Heuristic optimizer: choose Standard Copy vs Zero-Copy lookup by N.
 *
 * Uses benchmark data (1.29 ms savings at 256k lookups) for default crossover.
 * Warm-up measures current PCIe/copy cost and adjusts the crossover dynamically.
 */

#ifndef GPU_HASHMAP_HEURISTIC_LOOKUP_H
#define GPU_HASHMAP_HEURISTIC_LOOKUP_H

#include "gpu_hashmap/hash_map_api.h"
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>

namespace gpu_hashmap {

enum class LookupPath { StandardCopy, ZeroCopy };

/**
 * State for the heuristic: crossover threshold, out-of-core fallback, sparsity, and warm-up flag.
 */
struct HeuristicState {
  size_t crossover_n;           ///< use Zero-Copy when n_lookups >= this (set by warm-up with safety margin)
  size_t max_lookups_fit_vram;  ///< if n > this, force Zero-Copy (out-of-core fallback); set in warm-up
  size_t table_size_bytes;      ///< if set and large, sparsity mode: prefer Zero-Copy to avoid table migration tax
  bool warmed_up;               ///< true after heuristic_warm_up()
};

/** Table size above which we consider "massive table" for sparsity-driven crossover (e.g. 2GB). */
constexpr size_t kSparsityTableSizeThreshold = 1500ULL * 1024 * 1024;
/** Lookup count below which we consider "sparse" when table is massive (e.g. N=10,000). */
constexpr size_t kSparsityLookupCap = 100 * 1024;

/** Default crossover from benchmark: 1.29 ms savings at 256k lookups. */
constexpr size_t kHeuristicDefaultCrossoverN = 256 * 1024;

/**
 * Initialize state with default crossover (256k) and no VRAM limit until warm-up.
 */
inline void heuristic_init(HeuristicState* state) {
  state->crossover_n = kHeuristicDefaultCrossoverN;
  state->max_lookups_fit_vram = SIZE_MAX;  /* set by warm_up from cudaMemGetInfo */
  state->table_size_bytes = 0;             /* set by caller for sparsity-driven crossover */
  state->warmed_up = false;
}

/**
 * Set table size (bytes) for sparsity-driven crossover. When table is massive and
 * n_lookups is small, the heuristic will prefer Zero-Copy to avoid the full table
 * migration tax (Standard path would pay Time(Full Table Copy) before sparse lookups).
 */
inline void heuristic_set_table_size(HeuristicState* state, size_t table_bytes) {
  state->table_size_bytes = table_bytes;
}

/**
 * Warm-up: measure Standard Copy and Zero-Copy at 256k lookups (probe keys
 * generated internally), then set crossover_n from current PCIe/copy cost.
 */
void heuristic_warm_up(HashTable* table, HeuristicState* state,
                       cudaStream_t stream = nullptr);

/**
 * Choose path for n_lookups: sparsity (massive table + sparse lookups) and out-of-core
 * force Zero-Copy; else use crossover_n. Sparsity: when table_size_bytes is large and
 * n_lookups is small, Zero-Copy avoids Time(Full Table Copy) > Time(Sparse PCIe Stalls).
 */
inline LookupPath heuristic_choose_path(size_t n_lookups, HeuristicState const* state) {
  if (n_lookups > state->max_lookups_fit_vram)
    return LookupPath::ZeroCopy;  /* out-of-core fallback: batch does not fit in VRAM */
  /* Sparsity-driven: massive table + sparse lookups => avoid full table migration tax */
  if (state->table_size_bytes >= kSparsityTableSizeThreshold &&
      n_lookups <= kSparsityLookupCap)
    return LookupPath::ZeroCopy;
  return (n_lookups >= state->crossover_n) ? LookupPath::ZeroCopy : LookupPath::StandardCopy;
}

/**
 * Run lookup batch using the heuristic (Standard or Zero-Copy) and log the choice.
 * For Zero-Copy path: h_keys and h_results must be pinned (cudaHostAlloc Mapped);
 * results are written in place (true zero-copy, no memcpy). For Standard path,
 * any host pointers are valid (explicit H2D/D2H).
 */
void hash_map_lookup_batch_heuristic(HashTable* table,
                                     KeyType* h_keys,
                                     ValueType* h_results,
                                     size_t n,
                                     HeuristicState* state,
                                     cudaStream_t stream = nullptr);

} // namespace gpu_hashmap

#endif // GPU_HASHMAP_HEURISTIC_LOOKUP_H
