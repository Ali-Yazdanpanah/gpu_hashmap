/**
 * @file metrics.hpp
 * @brief Memory architecture metrics: effective bandwidth, load efficiency, occupancy.
 */

#ifndef GPU_HASHMAP_ANALYSIS_METRICS_HPP
#define GPU_HASHMAP_ANALYSIS_METRICS_HPP

#include <cstddef>
#include <cuda_runtime.h>

namespace gpu_hashmap {
namespace analysis {

/** Effective memory bandwidth (GB/s): (Bytes_read + Bytes_written) / Time_s. */
inline double effective_bandwidth_gbps(size_t bytes_read, size_t bytes_written, double time_sec) {
  if (time_sec <= 0) return 0;
  return (bytes_read + bytes_written) / (time_sec * 1e9);
}

/** Get peak theoretical memory bandwidth (GB/s) for current device. */
double get_peak_memory_bandwidth_gbps();

/** Get device name string. */
void get_device_name(char* buf, size_t buf_size);

/**
 * Global memory load efficiency: requested bytes / actual bytes moved.
 * Estimated from coalescing (1.0 = fully coalesced). Use Nsight Compute for exact.
 */
double estimate_load_efficiency_coalesced(size_t num_accesses, size_t bytes_per_access, int warp_size);

} // namespace analysis
} // namespace gpu_hashmap

#endif // GPU_HASHMAP_ANALYSIS_METRICS_HPP
