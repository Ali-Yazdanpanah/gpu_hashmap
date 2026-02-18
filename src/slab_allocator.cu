/**
 * @file slab_allocator.cu
 * @brief Slab init/destroy (host) and device alloc/free (atomic free list).
 */

#include "gpu_hashmap/slab_allocator.cuh"
#include "gpu_hashmap/hash_buckets.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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

__global__ void init_free_list_kernel(SlotIndex* free_list_next,
                                      unsigned long long* free_list_head,
                                      size_t capacity) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= capacity) return;
  free_list_next[i] = (i + 1 < capacity) ? static_cast<SlotIndex>(i + 1) : kInvalidSlot;
  if (i == 0)
    *free_list_head = 0;
}

} // namespace

void slab_init(SlabAllocator* alloc, size_t num_slots, cudaStream_t stream) {
  if (num_slots == 0 || num_slots > kMaxSlots) {
    std::fprintf(stderr, "slab_init: invalid num_slots %zu\n", num_slots);
    std::abort();
  }
  size_t bytes_next = num_slots * sizeof(SlotIndex);
  size_t bytes_head = sizeof(unsigned long long);
  size_t total = bytes_next + bytes_head;
  void* d_pool = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pool, total));
  alloc->d_pool = d_pool;
  alloc->capacity = num_slots;
  alloc->owns_memory = true;

  SlotIndex* d_next = static_cast<SlotIndex*>(d_pool);
  unsigned long long* d_head = reinterpret_cast<unsigned long long*>(
      static_cast<char*>(d_pool) + bytes_next);

  alloc->device.free_list_next = d_next;
  alloc->device.free_list_head = d_head;
  alloc->device.capacity = num_slots;
  alloc->device.slot_size_bytes = sizeof(Node);

  const int block = 256;
  init_free_list_kernel<<<(num_slots + block - 1) / block, block, 0, stream>>>(
      d_next, d_head, num_slots);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void slab_destroy(SlabAllocator* alloc) {
  if (!alloc) return;
  if (alloc->owns_memory && alloc->d_pool)
    CUDA_CHECK(cudaFree(alloc->d_pool));
  alloc->d_pool = nullptr;
  alloc->capacity = 0;
  alloc->device.free_list_next = nullptr;
  alloc->device.free_list_head = nullptr;
}

} // namespace gpu_hashmap
