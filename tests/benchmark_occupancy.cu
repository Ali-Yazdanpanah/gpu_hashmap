/**
 * @file benchmark_occupancy.cu
 * @brief Occupancy vs. throughput: sweep thread block sizes (32 to 1024) and
 *        report max active blocks per SM and achieved lookup throughput.
 *        Data for chart: hardware occupancy vs achieved throughput.
 */

#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/lookup_kernel.cuh"
#include "gpu_hashmap/hash_buckets.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
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

namespace {

constexpr size_t N_INSERT = 512 * 1024;
constexpr size_t N_LOOKUP = 1024 * 1024;
const int BLOCK_SIZES[] = { 32, 64, 128, 256, 512, 1024 };
constexpr int N_BLOCK_SIZES = sizeof(BLOCK_SIZES) / sizeof(BLOCK_SIZES[0]);

} // namespace

int main() {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  const size_t num_buckets = 1 << 18;
  const size_t capacity = 2 * 1024 * 1024;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int multiprocessors = prop.multiProcessorCount;
  int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;

  std::printf("================================================================================\n");
  std::printf("  Occupancy vs. Throughput (thread block size sweep)\n");
  std::printf("  Device: %s (SM %d.%d, %d SMs)\n",
              prop.name, prop.major, prop.minor, multiprocessors);
  std::printf("  Lookups: %zu per run\n", N_LOOKUP);
  std::printf("================================================================================\n\n");

  std::vector<gpu_hashmap::KeyType> h_keys(N_INSERT);
  std::vector<gpu_hashmap::ValueType> h_values(N_INSERT);
  std::vector<gpu_hashmap::KeyType> h_lookup_keys(N_LOOKUP);
  for (size_t i = 0; i < N_INSERT; ++i) {
    h_keys[i] = i * 3 + 1;
    h_values[i] = i + 1000;
  }
  for (size_t i = 0; i < N_LOOKUP; ++i)
    h_lookup_keys[i] = h_keys[i % N_INSERT];

  gpu_hashmap::HashTable table = {};
  gpu_hashmap::hash_map_create(&table, num_buckets, capacity);
  gpu_hashmap::hash_map_upload_from_host(&table, h_keys.data(), h_values.data(), N_INSERT);
  CUDA_CHECK(cudaDeviceSynchronize());

  gpu_hashmap::KeyType* d_keys = nullptr;
  gpu_hashmap::ValueType* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, N_LOOKUP * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_out, N_LOOKUP * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_lookup_keys.data(),
                        N_LOOKUP * sizeof(gpu_hashmap::KeyType), cudaMemcpyHostToDevice));

  std::printf("  %-8s  %-12s  %-10s  %-14s\n", "BlockSz", "Blocks/SM", "Occupancy", "Throughput");
  std::printf("  %-8s  %-12s  %-10s  %-14s\n", "------", "--------", "---------", "----------");

  for (int i = 0; i < N_BLOCK_SIZES; ++i) {
    int block_size = BLOCK_SIZES[i];
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, (const void*)&gpu_hashmap::hash_map_lookup_kernel, block_size, 0));
    double occupancy = (block_size * max_blocks) / (double)max_threads_per_sm;
    if (occupancy > 1.0) occupancy = 1.0;

    int num_blocks = (static_cast<int>(N_LOOKUP) + block_size - 1) / block_size;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    gpu_hashmap::hash_map_lookup_kernel<<<num_blocks, block_size>>>(
        table.d_device_table, d_keys, d_out, N_LOOKUP);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double throughput_mops = (N_LOOKUP / 1e6) / (ms / 1000.0);
    std::printf("  %-8d  %-12d  %-10.2f  %-14.2f Mops/s\n",
                block_size, max_blocks, occupancy, throughput_mops);
  }

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_out));
  gpu_hashmap::hash_map_destroy(&table);

  std::printf("\n================================================================================\n");
  std::printf("  Use this data to plot Occupancy vs. Throughput (block size 32--1024).\n");
  std::printf("================================================================================\n");
  return 0;
}
