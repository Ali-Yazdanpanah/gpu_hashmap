/**
 * @file metrics.cu
 * @brief Memory bandwidth, peak BW, occupancy, load efficiency.
 */

#include "gpu_hashmap/analysis/metrics.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace gpu_hashmap {
namespace analysis {

double get_peak_memory_bandwidth_gbps() {
  int device = 0;
  (void)cudaGetDevice(&device);
  int memClockKHz = 0, busWidth = 0;
  cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, device);
  cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, device);
  /* Peak GB/s = 2 * clock * (bus_width/8) (DDR). */
  return 2.0 * (memClockKHz * 1e3) * (busWidth / 8) / 1e9;
}

void get_device_name(char* buf, size_t buf_size) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  strncpy(buf, prop.name, buf_size - 1);
  buf[buf_size - 1] = '\0';
}

double estimate_load_efficiency_coalesced(size_t num_accesses, size_t bytes_per_access, int warp_size) {
  (void)num_accesses;
  (void)bytes_per_access;
  (void)warp_size;
  /* Placeholder: coalesced 64-byte (cache line) aligned loads ≈ 100%. */
  return 1.0;
}

} // namespace analysis
} // namespace gpu_hashmap
