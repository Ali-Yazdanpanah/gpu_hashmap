/**
 * @file probe_depth.cu
 * @brief Lookup with probe depth implementation.
 */

#include "gpu_hashmap/analysis/probe_depth.cuh"

namespace gpu_hashmap {
namespace analysis {

__global__ void lookup_with_probe_depth(HashTableDevice const* table,
                                        KeyType const* keys,
                                        ValueType* values,
                                        unsigned int* probe_depths,
                                        size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  KeyType key = keys[i];
  size_t b = hash_key(key, table->num_buckets);
  unsigned long long head = table->bucket_heads[b];
  unsigned int depth = 0;
  while (head != kInvalidSlot) {
    Node const* node = &table->nodes[head];
    if (node->key == key) {
      values[i] = node->value;
      probe_depths[i] = depth + 1;
      return;
    }
    head = node->next;
    ++depth;
  }
  values[i] = 0xFFFFFFFFFFFFFFFFull;
  probe_depths[i] = depth;
}

} // namespace analysis
} // namespace gpu_hashmap
