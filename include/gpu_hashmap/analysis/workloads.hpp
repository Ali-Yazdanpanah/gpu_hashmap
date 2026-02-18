/**
 * @file workloads.hpp
 * @brief Workload generators for stress testing: Zipfian (hot-key), uniform.
 */

#ifndef GPU_HASHMAP_ANALYSIS_WORKLOADS_HPP
#define GPU_HASHMAP_ANALYSIS_WORKLOADS_HPP

#include "../types.cuh"
#include <cmath>
#include <cstddef>
#include <vector>

namespace gpu_hashmap {
namespace analysis {

/**
 * Zipfian distribution generator for hot-key contention.
 * P(rank i) ∝ 1 / (i + 1)^alpha  for i in [0, N).
 * alpha = 0 ~ uniform; alpha > 1 = skewed (hot keys).
 */
class ZipfianGenerator {
 public:
  ZipfianGenerator(size_t num_keys, double alpha, uint64_t seed = 12345);
  /** Generate n samples (key indices 0..num_keys-1). */
  void generate(std::vector<KeyType>& out_keys, size_t n);
  /** Get key at rank (rank 0 = hottest). */
  KeyType key_at_rank(size_t rank) const { return static_cast<KeyType>(rank % num_keys_); }

 private:
  size_t num_keys_;
  double alpha_;
  uint64_t seed_;
  std::vector<double> cdf_;  /* CDF for inverse transform */
};

/** Fill keys/values for load factor sweep: n = capacity * load_factor. */
void fill_uniform_keys_values(std::vector<KeyType>& keys, std::vector<ValueType>& values,
                              size_t n, uint64_t key_seed = 0);

} // namespace analysis
} // namespace gpu_hashmap

#endif // GPU_HASHMAP_ANALYSIS_WORKLOADS_HPP
