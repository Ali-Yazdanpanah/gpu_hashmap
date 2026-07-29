/**
 * @file test_slab_overflow.cu
 * @brief Pin the slab table's drop-on-overflow behaviour.
 *
 * test_slab covers the no-overflow regime and requires dropped == 0, so nothing
 * currently exercises the path a README limitation is built on: a bucket holds
 * exactly kSlabSize entries with no probe and no overflow chain, so a key arriving
 * at a full bucket is discarded.
 *
 * Drops are a deterministic function of (key set, bucket count, hash), so the count
 * can be asserted against a constant rather than merely "some drops happened". That
 * turns any change to the hash function, bucket indexing, or slot-claiming logic
 * into a test failure instead of a silent shift in how much data is lost.
 */

#include "gpu_hashmap/hash_slab.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
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

/* Fixed seed and fixed sizes: 8192 random keys into 1024 buckets of 8 slots is
 * exactly 100% of nominal capacity, so uneven bucket occupancy guarantees that some
 * buckets overflow while others stay underfull. */
constexpr unsigned long long kSeed = 20260729ull;
constexpr size_t kNumBuckets = 1024;
constexpr size_t kNumKeys = 8192;

/* Expected drop count for the constants above. Update only alongside a deliberate
 * change to hashing, bucket count, or slot claiming -- and say so in the commit. */
constexpr unsigned long long kExpectedDrops = 1173ull;

} // namespace

int main() {
  gpu_hashmap::SlabHashTable table = {};
  gpu_hashmap::slab_hash_create(&table, kNumBuckets);

  std::vector<gpu_hashmap::KeyType> h_keys(kNumKeys);
  std::vector<gpu_hashmap::ValueType> h_values(kNumKeys);
  std::mt19937_64 rng(kSeed);
  for (size_t i = 0; i < kNumKeys; ++i) {
    gpu_hashmap::KeyType k = rng();
    /* kEmptyKey marks a free slot, so it can never be stored as a real key. */
    if (k == gpu_hashmap::kEmptyKey) k = 1;
    h_keys[i] = k;
    h_values[i] = i + 1000;
  }

  gpu_hashmap::KeyType* d_keys = nullptr;
  gpu_hashmap::ValueType* d_values = nullptr;
  gpu_hashmap::ValueType* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, kNumKeys * sizeof(gpu_hashmap::KeyType)));
  CUDA_CHECK(cudaMalloc(&d_values, kNumKeys * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMalloc(&d_out, kNumKeys * sizeof(gpu_hashmap::ValueType)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), kNumKeys * sizeof(gpu_hashmap::KeyType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), kNumKeys * sizeof(gpu_hashmap::ValueType),
                        cudaMemcpyHostToDevice));

  gpu_hashmap::slab_hash_insert_batch(table.d_device_table, d_keys, d_values, kNumKeys);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  const unsigned long long dropped = gpu_hashmap::slab_hash_insert_failure_count(&table);

  gpu_hashmap::slab_lookup_kernel<<<(kNumKeys + 255) / 256, 256>>>(
      table.d_device_table, d_keys, d_out, kNumKeys);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  std::vector<gpu_hashmap::ValueType> h_out(kNumKeys);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, kNumKeys * sizeof(gpu_hashmap::ValueType),
                        cudaMemcpyDeviceToHost));

  size_t found = 0, wrong_value = 0;
  for (size_t i = 0; i < kNumKeys; ++i) {
    if (h_out[i] == 0xFFFFFFFFFFFFFFFFull) continue;
    ++found;
    if (h_out[i] != h_values[i]) ++wrong_value;
  }

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_out));
  gpu_hashmap::slab_hash_destroy(&table);

  std::printf("  %zu random keys (seed %llu) into %zu buckets x %d slots\n",
              kNumKeys, kSeed, kNumBuckets, gpu_hashmap::kSlabSize);
  std::printf("  dropped: %llu (expected %llu, %.3f%% of inserts)\n",
              dropped, kExpectedDrops, 100.0 * dropped / kNumKeys);
  std::printf("  found on lookup: %zu (expected %zu)\n", found, kNumKeys - dropped);

  bool ok = true;
  if (dropped != kExpectedDrops) {
    std::fprintf(stderr,
                 "FAIL: drop count changed: got %llu, expected %llu. Hashing, bucket\n"
                 "      count, or slot claiming changed how much data is silently lost.\n",
                 dropped, kExpectedDrops);
    ok = false;
  }
  /* Every key that was not dropped must be retrievable, and with its own value.
   * This is what makes the drop count meaningful rather than just a counter. */
  if (found != kNumKeys - dropped) {
    std::fprintf(stderr,
                 "FAIL: %zu keys found but %zu were not dropped -- the failure counter\n"
                 "      and the table's contents disagree.\n",
                 found, kNumKeys - dropped);
    ok = false;
  }
  if (wrong_value != 0) {
    std::fprintf(stderr, "FAIL: %zu stored keys returned the wrong value\n", wrong_value);
    ok = false;
  }
  if (dropped == 0) {
    std::fprintf(stderr,
                 "FAIL: no drops occurred, so this test is not exercising the overflow\n"
                 "      path it exists to pin. Raise kNumKeys or lower kNumBuckets.\n");
    ok = false;
  }

  std::printf(ok ? "PASS (slab overflow)\n" : "FAIL (slab overflow)\n");
  return ok ? 0 : 1;
}
