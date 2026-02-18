/**
 * @file validation.hpp
 * @brief Gold-standard CPU verification: bit-perfect comparison with std::unordered_map.
 */

#ifndef GPU_HASHMAP_ANALYSIS_VALIDATION_HPP
#define GPU_HASHMAP_ANALYSIS_VALIDATION_HPP

#include "../hash_map_api.h"
#include <cstddef>
#include <vector>

namespace gpu_hashmap {
namespace analysis {

/**
 * Gold standard validation: insert then lookup on GPU, same on CPU (unordered_map),
 * compare every lookup result. Returns true iff all results match bit-perfectly.
 * Reports first mismatch index and expected/got.
 */
bool validate_against_cpu_gold(
    const std::vector<KeyType>& h_keys,
    const std::vector<ValueType>& h_values,
    const std::vector<KeyType>& h_lookup_keys,
    size_t num_buckets, size_t capacity,
    size_t* out_first_mismatch_index = nullptr,
    ValueType* out_expected = nullptr,
    ValueType* out_got = nullptr);

} // namespace analysis
} // namespace gpu_hashmap

#endif // GPU_HASHMAP_ANALYSIS_VALIDATION_HPP
