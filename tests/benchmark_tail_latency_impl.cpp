/**
 * @file benchmark_tail_latency_impl.cpp
 * @brief Host-only implementation: std::sort, std::unordered_map, chrono.
 *        Built with C++ compiler so nvcc never sees std_function.h.
 */

#include "benchmark_tail_latency_impl.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <unordered_map>

namespace tail_latency_impl {

static double percentile(std::vector<double>& sorted, double p) {
  if (sorted.empty()) return 0.0;
  double idx = p * (sorted.size() - 1) / 100.0;
  int i = static_cast<int>(idx);
  if (i >= static_cast<int>(sorted.size()) - 1) return sorted.back();
  double frac = idx - i;
  return sorted[i] * (1.0 - frac) + sorted[i + 1] * frac;
}

/* The looked-up values must reach an observable location, or the optimiser deletes
 * the whole probe loop as dead code and every CPU batch times at ~0.00003 ms. */
volatile uint64_t cpu_lookup_sink = 0;

void run_cpu_tail_latency(
    const std::vector<uint64_t>& keys,
    const std::vector<uint64_t>& values,
    const std::vector<uint64_t>& lookup_keys,
    size_t batch_size,
    int n_batches,
    std::vector<double>* out_batch_ms) {
  std::unordered_map<uint64_t, uint64_t> map;
  map.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i)
    map[keys[i]] = values[i];

  out_batch_ms->clear();
  out_batch_ms->reserve(static_cast<size_t>(n_batches));
  const size_t n_lookup_total = lookup_keys.size();
  for (int b = 0; b < n_batches; ++b) {
    size_t offset = (static_cast<size_t>(b) * batch_size) % (n_lookup_total - batch_size + 1);
    uint64_t sum = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < batch_size; ++i) {
      auto it = map.find(lookup_keys[offset + i]);
      sum += (it != map.end()) ? it->second : 0xFFFFFFFFFFFFFFFFull;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    cpu_lookup_sink = sum;
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out_batch_ms->push_back(ms);
  }
}

void sort_and_report_percentiles(
    std::vector<double>* batch_ms,
    double* p50, double* p90, double* p99) {
  if (!batch_ms || !p50 || !p90 || !p99) return;
  std::sort(batch_ms->begin(), batch_ms->end());
  *p50 = percentile(*batch_ms, 50.0);
  *p90 = percentile(*batch_ms, 90.0);
  *p99 = percentile(*batch_ms, 99.0);
}

void print_histogram(const std::vector<double>& batch_ms, int nbins) {
  if (batch_ms.empty()) return;
  double lo = *std::min_element(batch_ms.begin(), batch_ms.end());
  double hi = *std::max_element(batch_ms.begin(), batch_ms.end());
  if (hi <= lo) hi = lo + 0.001;
  std::vector<int> counts(static_cast<size_t>(nbins), 0);
  for (double ms : batch_ms) {
    int bin = static_cast<int>((ms - lo) / (hi - lo) * nbins);
    if (bin >= nbins) bin = nbins - 1;
    counts[static_cast<size_t>(bin)]++;
  }
  int max_count = *std::max_element(counts.begin(), counts.end());
  if (max_count == 0) max_count = 1;
  std::printf("  Histogram (batch latency ms):\n");
  for (int i = 0; i < nbins; ++i) {
    double bin_lo = lo + (hi - lo) * i / nbins;
    double bin_hi = lo + (hi - lo) * (i + 1) / nbins;
    int bar_len = (counts[static_cast<size_t>(i)] * 40 + max_count - 1) / max_count;
    std::printf("    [%7.3f, %7.3f) %4d |", bin_lo, bin_hi, counts[static_cast<size_t>(i)]);
    for (int j = 0; j < bar_len; ++j) std::printf("#");
    std::printf("\n");
  }
}

} // namespace tail_latency_impl
