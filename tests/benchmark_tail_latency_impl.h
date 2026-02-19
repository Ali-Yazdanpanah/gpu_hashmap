/**
 * @file benchmark_tail_latency_impl.h
 * @brief Host-only helpers for tail latency benchmark (no CUDA).
 *        Compiled with C++ compiler to avoid nvcc/stdlib parameter-pack issues.
 */

#ifndef BENCHMARK_TAIL_LATENCY_IMPL_H
#define BENCHMARK_TAIL_LATENCY_IMPL_H

#include <vector>
#include <cstddef>
#include <cstdint>

namespace tail_latency_impl {

/** Run CPU lookup batches (std::unordered_map), fill batch latencies in ms. */
void run_cpu_tail_latency(
    const std::vector<uint64_t>& keys,
    const std::vector<uint64_t>& values,
    const std::vector<uint64_t>& lookup_keys,
    size_t batch_size,
    int n_batches,
    std::vector<double>* out_batch_ms);

/** Sort batch_ms and compute P50, P90, P99 (in-place sort). */
void sort_and_report_percentiles(
    std::vector<double>* batch_ms,
    double* p50, double* p90, double* p99);

/** Print text histogram of batch latencies. */
void print_histogram(const std::vector<double>& batch_ms, int nbins);

} // namespace tail_latency_impl

#endif // BENCHMARK_TAIL_LATENCY_IMPL_H
