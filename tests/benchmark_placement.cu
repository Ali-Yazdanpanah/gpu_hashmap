/**
 * @file benchmark_placement.cu
 * @brief Separate "which scheme" from "which side of the bus the table is on".
 *
 * The headline comparison in this project used to be confounded. The chained table was
 * always allocated in mapped host memory and the slab table always in device memory, so
 * "slab is 124x the CPU, chained is 0.38x" could equally have meant that chaining is a
 * poor scheme or that reaching a table over PCIe is ruinous. Those are different claims
 * with different consequences and the old measurement could not tell them apart.
 *
 * This runs both schemes at both placements -- four cells -- against the same key set,
 * so the two effects can be read off separately:
 *
 *   placement effect (same scheme, mapped-host / device)
 *   scheme effect    (same placement, chained / slab)
 *
 * Every cell reports the median of kReps runs plus the 5th and 95th percentile, because
 * on this laptop GPU a single run of the fast cells is mostly thermal and scheduler noise.
 */

#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/hash_slab.cuh"
#include "gpu_hashmap/insert_kernel.cuh"
#include <cuda_runtime.h>
#include <chrono>
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

using gpu_hashmap::KeyType;
using gpu_hashmap::TablePlacement;
using gpu_hashmap::ValueType;

__global__ void lookup_kernel_chained(gpu_hashmap::HashTableDevice const* table,
                                      KeyType const* keys, ValueType* values, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  KeyType key = keys[i];
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

int compare_double(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x > y) ? 1 : (x < y) ? -1 : 0;
}

struct Stat {
  double median, p05, p95;
};

Stat summarize(std::vector<double> v) {
  Stat s = {0.0, 0.0, 0.0};
  if (v.empty()) return s;
  qsort(v.data(), v.size(), sizeof(double), compare_double);
  const size_t len = v.size();
  s.median = (len % 2 != 0) ? v[len / 2] : 0.5 * (v[len / 2 - 1] + v[len / 2]);
  s.p05 = v[(size_t)(0.05 * (len - 1))];
  s.p95 = v[(size_t)(0.95 * (len - 1) + 0.999)];
  return s;
}

/* One measurement of a chained table at the requested placement. The table is created
 * and destroyed inside the rep so that allocation state cannot carry between reps, but
 * only the insert and lookup kernels are timed. */
void run_chained(TablePlacement placement, std::vector<KeyType> const& h_keys,
                 std::vector<ValueType> const& h_values,
                 std::vector<KeyType> const& h_lookup, size_t num_buckets, size_t capacity,
                 double* out_insert_ms, double* out_lookup_ms,
                 unsigned long long* out_dropped) {
  const size_t n = h_keys.size(), m = h_lookup.size();
  gpu_hashmap::HashTable table = {};
  gpu_hashmap::hash_map_create(&table, num_buckets, capacity, nullptr, placement);

  KeyType* d_keys = nullptr;
  ValueType* d_values = nullptr;
  KeyType* d_lookup = nullptr;
  ValueType* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(KeyType)));
  CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(ValueType)));
  CUDA_CHECK(cudaMalloc(&d_lookup, m * sizeof(KeyType)));
  CUDA_CHECK(cudaMalloc(&d_out, m * sizeof(ValueType)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(KeyType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), n * sizeof(ValueType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lookup, h_lookup.data(), m * sizeof(KeyType), cudaMemcpyHostToDevice));

  gpu_hashmap::hash_map_reset_insert_failure_count(&table);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  gpu_hashmap::hash_map_insert_batch(&table, d_keys, d_values, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaGetLastError());

  const int block = 256;
  lookup_kernel_chained<<<(m + block - 1) / block, block>>>(table.d_device_table, d_lookup,
                                                            d_out, m);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t2 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaGetLastError());

  *out_insert_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  *out_lookup_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  *out_dropped = gpu_hashmap::hash_map_insert_failure_count(&table);

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_lookup));
  CUDA_CHECK(cudaFree(d_out));
  gpu_hashmap::hash_map_destroy(&table);
}

void run_slab(TablePlacement placement, std::vector<KeyType> const& h_keys,
              std::vector<ValueType> const& h_values, std::vector<KeyType> const& h_lookup,
              size_t num_buckets, double* out_insert_ms, double* out_lookup_ms,
              unsigned long long* out_dropped) {
  const size_t n = h_keys.size(), m = h_lookup.size();
  gpu_hashmap::SlabHashTable table = {};
  gpu_hashmap::slab_hash_create(&table, num_buckets, nullptr, placement);

  KeyType* d_keys = nullptr;
  ValueType* d_values = nullptr;
  KeyType* d_lookup = nullptr;
  ValueType* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(KeyType)));
  CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(ValueType)));
  CUDA_CHECK(cudaMalloc(&d_lookup, m * sizeof(KeyType)));
  CUDA_CHECK(cudaMalloc(&d_out, m * sizeof(ValueType)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(KeyType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), n * sizeof(ValueType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lookup, h_lookup.data(), m * sizeof(KeyType), cudaMemcpyHostToDevice));

  gpu_hashmap::slab_hash_reset_insert_failure_count(&table);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  gpu_hashmap::slab_hash_insert_batch(table.d_device_table, d_keys, d_values, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaGetLastError());

  const int block = 256;
  gpu_hashmap::slab_lookup_kernel<<<(m + block - 1) / block, block>>>(table.d_device_table,
                                                                     d_lookup, d_out, m);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t2 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaGetLastError());

  *out_insert_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  *out_lookup_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  *out_dropped = gpu_hashmap::slab_hash_insert_failure_count(&table);

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_lookup));
  CUDA_CHECK(cudaFree(d_out));
  gpu_hashmap::slab_hash_destroy(&table);
}

struct Cell {
  const char* scheme;
  const char* placement;
  Stat insert, lookup, total;
  unsigned long long dropped;
  double dropped_pct;
};

/* Attribution for the chained insert cost. Every chained insert takes a node from the
 * slab allocator, and slab_alloc_device pops the free list by CAS on a single
 * free_list_head, so all inserts serialize on one address no matter which memory that
 * address is in. This kernel does nothing but the allocation, to show how much of the
 * chained insert time is the allocator rather than the hashing or the placement. */
__global__ void alloc_only_kernel(gpu_hashmap::SlabDevice const* slab, unsigned int* sink,
                                  size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  gpu_hashmap::SlotIndex s = gpu_hashmap::slab_alloc_device(slab);
  /* Keep the result observable so the allocation cannot be optimized away. */
  if (s == gpu_hashmap::kInvalidSlot) atomicAdd(sink, 1u);
}

double time_alloc_only(size_t n, size_t capacity) {
  gpu_hashmap::SlabAllocator alloc;
  gpu_hashmap::slab_init(&alloc, capacity);
  gpu_hashmap::SlabDevice* d_slab = nullptr;
  unsigned int* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_slab, sizeof(gpu_hashmap::SlabDevice)));
  CUDA_CHECK(cudaMemcpy(d_slab, &alloc.device, sizeof(gpu_hashmap::SlabDevice),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(unsigned int)));

  const int block = 256;
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  alloc_only_kernel<<<(n + block - 1) / block, block>>>(d_slab, d_sink, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_slab));
  gpu_hashmap::slab_destroy(&alloc);
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

int main() {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

  const size_t num_buckets = 1 << 18;              /* 256K buckets, both schemes */
  const size_t capacity = 2 * 1024 * 1024;         /* chained slab capacity */
  const size_t n = 1024 * 1024;                    /* 1M inserts */
  const size_t m = 512 * 1024;                     /* 512K lookups */
  const int kReps = 5;

  /* Random keys, fixed seed. Arithmetic key sequences interact with the multiplicative
   * hash to produce an unrealistically even bucket distribution -- the error that an
   * earlier version of this project made and that README section 6 discloses. */
  std::vector<KeyType> h_keys(n);
  std::vector<ValueType> h_values(n);
  std::vector<KeyType> h_lookup(m);
  std::mt19937_64 rng(0xC0FFEEull);
  for (size_t i = 0; i < n; ++i) {
    /* Take the raw draw. Forcing a bit (e.g. `rng() | 1`) is not harmless here: the
     * hash multiplies by an odd constant and indexes with the low bits, so pinning the
     * low bit of the key pins the low bit of the bucket index and half the buckets
     * become unreachable -- which showed up as a 14% slab drop rate instead of 0.8%. */
    KeyType k = rng();
    if (k == gpu_hashmap::kEmptyKey) k = 1;  /* reserved as the slab's empty marker */
    h_keys[i] = k;
    h_values[i] = i + 1;
  }
  for (size_t i = 0; i < m; ++i) h_lookup[i] = h_keys[i % n];

  std::printf("=== Placement x scheme: %zu inserts, %zu lookups, %zu buckets, median of %d ===\n",
              n, m, num_buckets, kReps);
  std::printf("Chained capacity %zu nodes; slab capacity %zu entries (%d per bucket)\n\n",
              capacity, num_buckets * gpu_hashmap::kSlabSize, gpu_hashmap::kSlabSize);

  Cell cells[4];
  int ci = 0;
  const TablePlacement placements[2] = {TablePlacement::kMappedHost, TablePlacement::kDevice};
  const char* placement_names[2] = {"mapped-host", "device"};

  for (int p = 0; p < 2; ++p) {
    for (int scheme = 0; scheme < 2; ++scheme) {
      std::vector<double> ins(kReps), lup(kReps), tot(kReps);
      unsigned long long dropped = 0;
      for (int r = 0; r < kReps; ++r) {
        if (scheme == 0)
          run_chained(placements[p], h_keys, h_values, h_lookup, num_buckets, capacity,
                      &ins[r], &lup[r], &dropped);
        else
          run_slab(placements[p], h_keys, h_values, h_lookup, num_buckets, &ins[r], &lup[r],
                   &dropped);
        tot[r] = ins[r] + lup[r];
      }
      Cell& c = cells[ci++];
      c.scheme = (scheme == 0) ? "chained" : "slab";
      c.placement = placement_names[p];
      c.insert = summarize(ins);
      c.lookup = summarize(lup);
      c.total = summarize(tot);
      c.dropped = dropped;
      c.dropped_pct = 100.0 * (double)dropped / (double)n;
      std::printf("  %-8s %-12s insert %9.2f ms  lookup %9.2f ms  total %9.2f ms"
                  "  [p05 %9.2f p95 %9.2f]  dropped %llu (%.3f%%)\n",
                  c.scheme, c.placement, c.insert.median, c.lookup.median, c.total.median,
                  c.total.p05, c.total.p95, c.dropped, c.dropped_pct);
      std::fflush(stdout);
    }
  }

  /* Machine-readable block for scripts/plot_benchmarks.py. */
  std::printf("\nPLACEMENT MATRIX (n_runs=%d)\n", kReps);
  std::printf("%-10s %-12s %10s %10s %10s %10s %10s %10s\n", "scheme", "placement",
              "insert_ms", "lookup_ms", "total_ms", "total_p05", "total_p95", "dropped%");
  for (int i = 0; i < 4; ++i) {
    const Cell& c = cells[i];
    std::printf("%-10s %-12s %10.2f %10.2f %10.2f %10.2f %10.2f %10.3f\n", c.scheme,
                c.placement, c.insert.median, c.lookup.median, c.total.median, c.total.p05,
                c.total.p95, c.dropped_pct);
  }

  /* Decomposition. Cell order is (host,chained) (host,slab) (device,chained) (device,slab). */
  const Cell& host_chained = cells[0];
  const Cell& host_slab = cells[1];
  const Cell& dev_chained = cells[2];
  const Cell& dev_slab = cells[3];

  const double placement_chained = host_chained.total.median / dev_chained.total.median;
  const double placement_slab = host_slab.total.median / dev_slab.total.median;
  const double scheme_device = dev_chained.total.median / dev_slab.total.median;
  const double scheme_host = host_chained.total.median / host_slab.total.median;

  std::printf("\nDECOMPOSITION\n");
  std::printf("  Placement effect (mapped-host / device):  chained %6.1fx   slab %6.1fx\n",
              placement_chained, placement_slab);
  std::printf("  Scheme effect    (chained / slab):        device  %6.1fx   host %6.1fx\n",
              scheme_device, scheme_host);
  std::printf("  Dominant factor: %s\n",
              (placement_chained > scheme_device) ? "placement (PCIe residency)"
                                                  : "scheme (chaining vs slab buckets)");

  /* Where the chained insert time actually goes. */
  std::vector<double> alloc_samples(kReps);
  for (int r = 0; r < kReps; ++r) alloc_samples[r] = time_alloc_only(n, capacity);
  const Stat alloc_only = summarize(alloc_samples);
  std::printf("\nATTRIBUTION\n");
  std::printf("  Free-list pop alone, %zu allocs, all threads at once: %9.2f ms (%.2f us/alloc)\n",
              n, alloc_only.median, 1000.0 * alloc_only.median / (double)n);
  std::printf("  Chained insert, device placement:                     %9.2f ms\n",
              dev_chained.insert.median);
  /* The isolated figure is larger than the full insert because every thread contends
   * for free_list_head simultaneously, whereas in the real kernel hashing and the
   * bucket-head CAS stagger arrivals. So it bounds the cost of that one contended
   * address rather than giving its share -- but the bound is already several times the
   * whole slab-table insert, and the free list is in device memory in every
   * configuration, which is why chained insert barely responds to placement. */
  std::printf("  (upper bound, not a share: the isolated kernel maximises contention)\n");
  std::printf("  Slab-table insert for the same %zu keys, device:      %9.2f ms\n", n,
              dev_slab.insert.median);
  return 0;
}
