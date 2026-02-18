/**
 * @file benchmark_zerocopy.cu
 * @brief Zero-copy (mapped memory) lookup benchmark: kernel-only vs copy+kernel.
 *
 * - Device setup: cudaSetDeviceFlags(cudaDeviceMapHost) at start.
 * - Hash table uses cudaHostAlloc(Mapped|Portable); lookup keys/results can be
 *   either device buffers (with cudaMemcpy) or mapped host (zero-copy).
 * - Reports: Kernel execution time only vs Copy + Kernel total.
 * - PhD: Notes on verifying PCIe traffic (bus utilization, Gen3 vs Gen4).
 */

#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/insert_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

__global__ void lookup_kernel(gpu_hashmap::HashTableDevice const* table,
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

} // namespace

int main() {
  /* 1. Device setup: allow mapping host memory (must be before other CUDA calls). */
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

  const size_t num_buckets = 1 << 18;
  const size_t capacity = 2 * 1024 * 1024;
  const size_t n_insert = 512 * 1024;
  const size_t m_lookup = 256 * 1024;

  std::vector<gpu_hashmap::KeyType> h_keys(n_insert), h_lookup_keys(m_lookup);
  std::vector<gpu_hashmap::ValueType> h_values(n_insert);
  for (size_t i = 0; i < n_insert; ++i) {
    h_keys[i] = i * 3 + 1;
    h_values[i] = i + 1000;
  }
  for (size_t i = 0; i < m_lookup; ++i)
    h_lookup_keys[i] = h_keys[i % n_insert];

  gpu_hashmap::HashTable table = {};
  hash_map_create(&table, num_buckets, capacity);
  /* Populate table via zero-copy upload (writes to mapped host memory; no cudaMemcpy). */
  hash_map_upload_from_host(&table, h_keys.data(), h_values.data(), n_insert);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t e1, e2;
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventCreate(&e2));

  std::printf("================================================================================\n");
  std::printf("  Zero-Copy (Mapped Memory) Lookup Benchmark\n");
  std::printf("  Table: %zu buckets, %zu entries; Lookups: %zu\n", num_buckets, n_insert, m_lookup);
  std::printf("================================================================================\n\n");

  /* ---------- Old path: Copy + Kernel ---------- */
  gpu_hashmap::KeyType* d_keys = nullptr;
  gpu_hashmap::ValueType* d_results = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, m_lookup * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_results, m_lookup * sizeof(gpu_hashmap::ValueType)));

  float copy_h2d_ms = 0, kernel_ms = 0, copy_d2h_ms = 0;
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaMemcpy(d_keys, h_lookup_keys.data(), m_lookup * sizeof(gpu_hashmap::KeyType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(e2));
  CUDA_CHECK(cudaEventSynchronize(e2));
  CUDA_CHECK(cudaEventElapsedTime(&copy_h2d_ms, e1, e2));

  CUDA_CHECK(cudaEventRecord(e1));
  lookup_kernel<<<(m_lookup + 255) / 256, 256>>>(table.d_device_table, d_keys, d_results, m_lookup);
  CUDA_CHECK(cudaEventRecord(e2));
  CUDA_CHECK(cudaEventSynchronize(e2));
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, e1, e2));

  CUDA_CHECK(cudaEventRecord(e1));
  std::vector<gpu_hashmap::ValueType> h_results_old(m_lookup);
  CUDA_CHECK(cudaMemcpy(h_results_old.data(), d_results, m_lookup * sizeof(gpu_hashmap::ValueType),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(e2));
  CUDA_CHECK(cudaEventSynchronize(e2));
  CUDA_CHECK(cudaEventElapsedTime(&copy_d2h_ms, e1, e2));

  float total_copy_kernel_ms = copy_h2d_ms + kernel_ms + copy_d2h_ms;
  std::printf("  [Copy + Kernel] path:\n");
  std::printf("    Copy H2D:     %.3f ms\n", copy_h2d_ms);
  std::printf("    Kernel only:  %.3f ms\n", kernel_ms);
  std::printf("    Copy D2H:     %.3f ms\n", copy_d2h_ms);
  std::printf("    Total:        %.3f ms\n\n", total_copy_kernel_ms);

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_results));

  /* ---------- Zero-copy path: no cudaMemcpy for lookup ---------- */
  void* h_lookup_buf = nullptr;
  void* h_result_buf = nullptr;
  gpu_hashmap::KeyType* d_lookup_mapped = nullptr;
  gpu_hashmap::ValueType* d_result_mapped = nullptr;
  const unsigned int map_flags = cudaHostAllocMapped | cudaHostAllocPortable;
  CUDA_CHECK(cudaHostAlloc(&h_lookup_buf, m_lookup * sizeof(gpu_hashmap::KeyType), map_flags));
  CUDA_CHECK(cudaHostAlloc(&h_result_buf, m_lookup * sizeof(gpu_hashmap::ValueType), map_flags));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_lookup_mapped, h_lookup_buf, 0));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_result_mapped, h_result_buf, 0));

  std::memcpy(h_lookup_buf, h_lookup_keys.data(), m_lookup * sizeof(gpu_hashmap::KeyType));
  CUDA_CHECK(cudaDeviceSynchronize());

  float kernel_only_ms = 0;
  CUDA_CHECK(cudaEventRecord(e1));
  lookup_kernel<<<(m_lookup + 255) / 256, 256>>>(table.d_device_table,
                                                 static_cast<gpu_hashmap::KeyType const*>(d_lookup_mapped),
                                                 static_cast<gpu_hashmap::ValueType*>(d_result_mapped),
                                                 m_lookup);
  CUDA_CHECK(cudaEventRecord(e2));
  CUDA_CHECK(cudaEventSynchronize(e2));
  CUDA_CHECK(cudaEventElapsedTime(&kernel_only_ms, e1, e2));

  std::printf("  [Zero-Copy] path (no cudaMemcpy for lookup keys/results):\n");
  std::printf("    Kernel only:  %.3f ms (GPU reads keys, writes results over PCIe)\n\n", kernel_only_ms);

  std::printf("  Comparison:\n");
  std::printf("    Kernel execution time:     %.3f ms (same workload)\n", kernel_ms);
  std::printf("    Copy + Kernel total:       %.3f ms\n", total_copy_kernel_ms);
  std::printf("    Zero-copy kernel-only:     %.3f ms\n", kernel_only_ms);
  if (total_copy_kernel_ms > 0)
    std::printf("    Copy overhead (total - kernel): %.3f ms (%.1f%% of total)\n\n",
                total_copy_kernel_ms - kernel_ms,
                100.0 * (total_copy_kernel_ms - kernel_ms) / total_copy_kernel_ms);

  /* Sanity: compare first few results (zero-copy result buffer is host-visible). */
  gpu_hashmap::ValueType* h_res = static_cast<gpu_hashmap::ValueType*>(h_result_buf);
  size_t mismatches = 0;
  for (size_t i = 0; i < m_lookup && i < 1000; ++i) {
    if (h_res[i] != h_results_old[i]) ++mismatches;
  }
  if (mismatches)
    std::printf("  WARNING: zero-copy results differ from copy path (%zu of first 1000)\n", mismatches);
  else
    std::printf("  Validation: zero-copy results match copy path (first 1000).\n");

  CUDA_CHECK(cudaFreeHost(h_lookup_buf));
  CUDA_CHECK(cudaFreeHost(h_result_buf));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaEventDestroy(e2));
  hash_map_destroy(&table);

  /* ---------- PhD: PCIe verification ---------- */
  std::printf("\n--------------------------------------------------------------------------------\n");
  std::printf("  PhD requirement: Verifying GPU reads over PCIe (zero-copy)\n");
  std::printf("--------------------------------------------------------------------------------\n");
  std::printf("  With zero-copy, the kernel accesses host-mapped memory over the PCIe bus.\n");
  std::printf("  To confirm and measure:\n");
  std::printf("  1. Run 'nvidia-smi -q' and check 'PCIe Link Width' / 'Link Speed' (Gen3 vs Gen4).\n");
  std::printf("  2. Run 'nvidia-smi dmon -s u' during this benchmark to see bus utilization.\n");
  std::printf("  3. Compare kernel-only times at different PCIe link speeds (e.g. BIOS setting).\n");
  std::printf("  4. NVML: nvmlDeviceGetPcieThroughput() gives RX/TX bytes (if available).\n");
  std::printf("================================================================================\n");

  return 0;
}
