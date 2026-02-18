/**
 * @file workloads.cpp
 * @brief Zipfian and uniform workload generators (host code, C++ only to avoid nvcc/std::function issues).
 */

#include "gpu_hashmap/analysis/workloads.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace gpu_hashmap {
namespace analysis {

ZipfianGenerator::ZipfianGenerator(size_t num_keys, double alpha, uint64_t seed)
    : num_keys_(num_keys), alpha_(alpha), seed_(seed) {
  cdf_.resize(num_keys + 1);
  cdf_[0] = 0;
  for (size_t i = 1; i <= num_keys; ++i)
    cdf_[i] = cdf_[i - 1] + 1.0 / std::pow(static_cast<double>(i), alpha_);
  double norm = cdf_[num_keys];
  for (size_t i = 0; i <= num_keys; ++i)
    cdf_[i] /= norm;
}

void ZipfianGenerator::generate(std::vector<KeyType>& out_keys, size_t n) {
  out_keys.resize(n);
  std::mt19937 rng(static_cast<unsigned>(seed_));
  std::uniform_real_distribution<double> u(0, 1);
  for (size_t i = 0; i < n; ++i) {
    double x = u(rng);
    size_t r = static_cast<size_t>(std::lower_bound(cdf_.begin(), cdf_.end(), x) - cdf_.begin());
    if (r >= num_keys_) r = num_keys_ - 1;
    out_keys[i] = key_at_rank(r);
  }
}

void fill_uniform_keys_values(std::vector<KeyType>& keys, std::vector<ValueType>& values,
                              size_t n, uint64_t key_seed) {
  keys.resize(n);
  values.resize(n);
  std::mt19937_64 rng(key_seed);
  for (size_t i = 0; i < n; ++i) {
    keys[i] = rng();
    values[i] = rng();
  }
}

} // namespace analysis
} // namespace gpu_hashmap
