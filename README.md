# GPU Concurrent Hash Map

A **high-performance CUDA key-value store** designed for massive-scale lookups with **hierarchical parallelism** and **interconnect-aware data movement**. This document provides an abstract, architectural description, analytical performance models, experimental evaluation with **systems research rigor**, and future directions.

---

## 1. Abstract & Introduction

### Summary

This project implements a GPU-accelerated hash table that addresses two central challenges in high-throughput key-value storage:

1. **Hierarchical parallelism**: Combining *block-level* batch allocation (reducing global slab contention by a factor of the block size), *warp-level* aggregation (reducing global atomics from *N* to *N*/32 via shuffle intrinsics), and *slab-bucketed* layouts (128-byte buckets with one atomic per warp) to maximize occupancy and minimize contention.
2. **Interconnect-aware data movement**: Offering both an **explicit-copy** path (H2D → kernel → D2H) and a **true zero-copy** path (mapped host memory over PCIe), with a calibrated **heuristic** and **out-of-core** fallback when the batch exceeds VRAM.

### The Problem

GPU hash tables face:

- **Massive atomic contention**: Thousands of threads competing on bucket heads and slab free lists serialize progress and limit throughput.
- **PCIe bottlenecks**: Transferring keys and results between host and device can dominate end-to-end latency; naive copy-in/copy-out can negate GPU speedups for moderate batch sizes.

This codebase attacks both: **warp-cooperative inserts** and **slab-hashing** cut global atomics; **zero-copy lookup** and a **crossover heuristic** (with safety margin α = 0.8) choose between standard and in-place mapping based on measured PCIe/copy cost. When the batch does not fit in VRAM, the system **pivots to zero-copy** (system RAM via mapped memory) as an out-of-core fallback.

---

## 2. Architectural Implementation

### 2.1 Slab-Hashing Logic (128-byte buckets)

The **slab hash table** (`gpu_hashmap_slab`) uses fixed-size buckets of *K* = 8 key-value pairs (16 bytes per pair → **128 bytes per bucket**, two cache lines). Each bucket *b* has:

- **Keys/values**: `keys[b * K + slot]`, `values[b * K + slot]` for `slot ∈ [0, K)`.
- **Occupancy mask**: `used_mask[b]` is a bitmask of width *K*; bit *i* set ⇒ slot *i* is occupied.

**Bit-masking selection** for inserts:

1. **Hash**: *b* = `hash_key(key) mod num_buckets`.
2. **Empty slots**: `empty = ¬used_mask[b] ∧ (2^K - 1)` (low *K* bits).
3. **Claim**: The warp cooperatively selects the first `need_count` zero bits via `__ffs(empty)` and builds a **claim mask**.
4. **Atomic**: One `atomicCAS(used_mask[b], old, old ∨ claim)` per warp (not per thread) to reserve slots.

So slot selection is *O*(1) in warp size and uses a single global atomic per warp per bucket.

### 2.2 Warp-Cooperative Inserts

Two strategies reduce global atomic operations:

**Chained table (warp aggregation):**

- Threads in a warp that hash to the **same bucket** are identified with `__ballot_sync`.
- A **leader** lane performs `atomicCAS` on the bucket head; the new head (slot index) is broadcast with `__shfl_sync` so followers can chain their nodes (each node's `next` points to the previous head).
- **Effect**: Global atomic operations drop from *N* (one per insert) to at most *N*/32 (one per distinct bucket per warp), with *O*(1) shuffle/ballot cost per warp.

**Slab table:**

- Same warp ballot to find lanes in the same bucket; **one** `atomicCAS` on `used_mask[b]` claims multiple slots for the warp.
- Slots are assigned to lanes via a small rank/select over the claim mask.

Both rely on ***O*(1) shuffle intrinsics** (`__shfl_sync`, `__ballot_sync`) to avoid extra global memory traffic.

### 2.3 Memory Consistency (lock-free)

The chained insert uses **lock-free** updates:

1. **Allocate** a node from the slab (atomic pop from free list).
2. **Fill** the node (key, value, `next = old_head`) in local memory.
3. **Publish**: `__threadfence()` to ensure the node's fields are visible to other threads, then `atomicCAS(bucket_heads[b], old_head, slot)`.

`__threadfence()` guarantees that writes to the node are ordered before the CAS that makes the node visible. Combined with the atomic CAS, this gives a lock-free linked-list push. **Lock-free reads**: lookups traverse the chain by reading `bucket_heads[b]` and then `nodes[].key` / `nodes[].value` / `nodes[].next` without acquiring locks.

---

## 3. Analytical Performance Model

### 3.1 Effective Bandwidth

The **effective memory bandwidth** (GB/s) for a kernel that reads and writes global (or mapped) memory is:

**BW_eff** = (Bytes_read + Bytes_written) / T_execution

Here *T_execution* is the kernel execution time in seconds. For lookup-only, Bytes_read includes keys and hash-table traffic; Bytes_written is the result buffer. The **PCIe roofline** (see below) uses the same idea for the zero-copy path, where bytes are moved over the interconnect.

### 3.2 Heuristic Crossover Model (Standard vs Zero-Copy)

Let *S* = batch size in bytes (keys + results), *B_pcie* = PCIe bandwidth, *B_vram* = device memory bandwidth, and *A* = access factor (ratio of actual memory traffic to *S* for the kernel). Then:

**Standard path** (explicit copy + kernel on device buffers):

*T_std* = *L_overhead* + *S*/*B_pcie* + (*S* · *A*)/*B_vram*

**Zero-copy path** (mapped host memory; kernel reads/writes over PCIe):

*T_zc* = (*S* · *A*) / (*B_pcie* · η)

The **heuristic** measures *T_std* and *T_zc* at a probe size (e.g. 256K lookups) and chooses zero-copy only when it is **strictly better** than the standard path after a safety margin.

### 3.3 Safety Margin (α = 0.8)

To account for **interconnect jitter** and measurement noise, we switch to zero-copy only if:

*T_zc* < α · *T_std*  (α = 0.8)

So zero-copy must be at least **20% faster** than the standard path at the probe size before we set the crossover threshold to prefer it.

### 3.4 Sparsity Victory: Massive Table vs. Sparse Lookup (Fig 1)

A **Massive Table vs. Sparse Lookup** experiment demonstrates the **Sparsity-Driven Crossover** and the value of the **Interconnect-Aware Heuristic**.

**Scenario**: A ~2 GB hash table (or 80% of free VRAM) resident in pinned host memory, with only **N = 10,000** random lookups.

- **Standard path**: The entire table is copied to VRAM (H2D), then 10K keys H2D, the lookup kernel runs, and 10K results D2H. The one-time **Migration Tax** *T_mig* = *O*(TableSize) dominates.
- **Zero-Copy path**: No table transfer; keys and results live in mapped host memory. The GPU performs the 10K lookups by fetching only the **buckets touched** over PCIe.

**Sparsity Victory — ~1,600× latency reduction**: In measured runs, the Standard path incurs roughly **325 ms** end-to-end (dominated by the 2 GB table migration), while the Zero-Copy path completes the same 10,000 lookups in about **0.2 ms** by bypassing the table transfer. This orders-of-magnitude gap (0.2 ms vs 325 ms) illustrates why the heuristic chooses Zero-Copy for sparse workloads on massive tables.

The **Interconnect-Aware Heuristic** identifies when the **Migration Tax** *O*(TableSize) exceeds the cost of remote PCIe access for the sparse batch. That is a critical optimization for **distributed hash tables**: avoiding full-table migration when only a small fraction of entries are touched.

**Decision boundary**: Optimal Path = arg min(*T_std* + *T_mig*, *T_zc*). When the table is large and the lookup batch is small, *T_std* + *T_mig* ≫ *T_zc*, so Zero-Copy is optimal. The figure below shows the **blue star** (Standard path, full table copy + 10K lookup, ~325 ms) and the **orange diamond** (Zero-Copy path, ~0.2 ms), plus the crossover and safety margin for dense batch sweep.

![Interconnect: Standard vs Zero-Copy (Sparsity-Driven Crossover)](scripts/figures/fig1_interconnect_crossover.png)  
**Figure 1: End-to-end latency vs batch size. Blue star = Standard path with 2 GB table migration + 10K lookup (~325 ms); orange diamond = Zero-Copy path (~0.2 ms). Sparsity Victory: ~1,600× reduction. Crossover and safety margin (α=0.8) for dense batch sweep.**

---

## 4. Experimental Evaluation (Report Around Figures)

All results were obtained on an **NVIDIA RTX 3060 (Laptop)** with **Compute Capability 8.6 (Ampere)**. The evaluation is structured **around the nine figures** below. The CPU baseline (`std::unordered_map`) **times out after 600 seconds** on the Massive Table and high-contention Zipfian stress tests; single-threaded CPU suffers from **O(N) cache-line invalidation** and **memory latency bottlenecks** that the GPU architecture bypasses via warp-cooperative coordination and slab-hashing.

**Figures at a glance**

| Figure | Content |
|--------|---------|
| **Fig 1** | Interconnect crossover: Sparsity Victory (~1,600×), Standard vs Zero-Copy |
| **Fig 2** | Zipfian stress: warp-aggregation vs standard insert (α = 0.5–2.0) |
| **Fig 3** | Heterogeneous speedup: throughput (CPU, GPU Chained, Warp-agg, Slab, Hybrid) |
| **Fig 4** | Load factor: throughput and probe depth |
| **Fig 5** | PCIe roofline: achieved vs theoretical bandwidth |
| **Fig 6** | Tail latency: P50/P90/P99 (CPU, GPU Chained, GPU Slab) |
| **Fig 7** | Occupancy vs throughput (block size 32–1024) |
| **Fig 8** | Speedup (× vs CPU) by implementation |
| **Fig 9** | Insert and Lookup time (ms) by approach (log scale) |

---

### Figure 1 — Interconnect Crossover (Sparsity Victory)

Sparsity-driven crossover: massive table (~2 GB), sparse N = 10K lookups. **Standard path** (~325 ms) pays the full table migration tax; **Zero-Copy path** (~0.2 ms) bypasses it → **~1,600× latency reduction**. The Interconnect-Aware Heuristic chooses Zero-Copy when Migration Tax O(TableSize) exceeds remote PCIe access cost.

![Figure 1: Interconnect crossover — Standard vs Zero-Copy (Sparsity Victory)](scripts/figures/fig1_interconnect_crossover.png)  
**Figure 1: End-to-end latency vs batch size. Blue star = Standard (2 GB migration + 10K lookup, ~325 ms); orange diamond = Zero-Copy (~0.2 ms). Crossover and safety margin α=0.8.**

---

### Figure 2 — Zipfian Stress (Warp-Aggregation)

Zipfian key distribution P(rank i) ∝ 1/(i+1)^α; higher α = hotter keys. Comparison of standard vs warp-aggregated insert time across α = 0.5–2.0.

![Figure 2: Warp-aggregation vs Zipfian skew](scripts/figures/fig2_warp_aggregation_zipfian.png)  
**Figure 2: Standard vs warp-aggregated insert time for Zipfian α = 0.5–2.0.**

---

### Figure 3 — Heterogeneous Speedup and Comparative Table

Throughput (ops/sec) for all implementations. Comparative summary on RTX 3060 (Laptop, CC 8.6):

| Approach | Result | Speedup vs CPU |
|----------|--------|----------------|
| **CPU Baseline** | Timeout / Non-Scalable | — |
| **Naive GPU** | Slower than CPU | **0.26×** |
| **GPU Chained** | — | ~4–5× |
| **GPU Slab-Hash** | 1.60 ms total | **66.67×**, **1.10M ops/sec** peak |

![Figure 3: Heterogeneous speedup — CPU vs GPU variants vs Hybrid](scripts/figures/fig3_heterogeneous_speedup.png)  
**Figure 3: Throughput (ops/sec) for CPU, GPU Chained, GPU Warp-agg, GPU Slab, Hybrid.**

---

### Figure 4 — Load Factor and Probe Depth

Throughput and average probe depth vs load factor (10%–99%).

![Figure 4: Load factor — throughput and probe depth](scripts/figures/fig4_load_factor_throughput_probe_depth.png)  
**Figure 4: Throughput and average probe depth vs load factor.**

---

### Figure 5 — PCIe Roofline (Hardware Bottleneck)

Achieved lookup bandwidth ~**1.8 GB/s** vs theoretical **32 GB/s** (Gen4×16). Gap due to **small, non-coalesced PCIe transactions** (per-bucket, per-key traffic).

![Figure 5: PCIe roofline — achieved vs theoretical bandwidth](scripts/figures/fig5_pcie_roofline.png)  
**Figure 5: Achieved effective bandwidth (Insert/Lookup) vs Gen3/Gen4 peak.**

---

### Figure 6 — Tail Latency (QoS)

**Chained**: ~3× gap between P50 and P99 (warp divergence, collision depth). **Slab-Hash** compresses the tail → better **Quality of Service (QoS)**. All in µs for comparison.

![Figure 6: Tail latency — CPU per-find vs GPU Chained vs GPU Slab](scripts/figures/fig6_tail_latency_p99.png)  
**Figure 6: P99 tail latency — P50/P90/P99 (µs) for CPU, GPU Chained, GPU Slab.**

---

### Figure 7 — Occupancy vs Throughput

Block size sweep (32–1024): max active blocks per SM (occupancy) and achieved lookup throughput (Mops/s).

![Figure 7: Occupancy vs throughput (block size 32–1024)](scripts/figures/fig7_occupancy_throughput.png)  
**Figure 7: Occupancy and achieved throughput (Mops/s) vs block size.**

---

### Figure 8 — Speedup vs CPU

Speedup (× vs CPU) per implementation. CPU = 1.0×; GPU Slab ~46–66× (log scale when >10×).

![Figure 8: Speedup vs CPU](scripts/figures/fig8_speedup_vs_cpu.png)  
**Figure 8: Speedup (× vs CPU) by implementation.**

---

### Figure 9 — Insert and Lookup Time by Approach

Insert (ms) and Lookup (ms) for every approach on a **log scale** so both slow (e.g. GPU Chained ~380 ms) and fast (e.g. GPU Slab ~1 ms) are visible.

![Figure 9: Insert and Lookup time (ms) by approach](scripts/figures/fig9_timings_by_approach.png)  
**Figure 9: Insert and Lookup time (ms) by approach (log scale).**

---

### Out-of-Core Scaling

When the lookup batch exceeds a fraction of free VRAM, the heuristic forces the **zero-copy path** (pinned host memory over PCIe), scaling to batch sizes larger than VRAM.

---

## 5. Conclusion & Future Work

- **66× speedup** (GPU Slab-Hash vs CPU baseline) is the result of **warp-cooperative coordination** and **slab-hashing** reducing global memory contention from *O*(N) to *O*(N/32). The CPU baseline is **non-scalable** (timeout) under Massive Table and high-contention Zipfian stress due to *O*(N) cache-line invalidation and memory latency; the GPU architecture bypasses these bottlenecks.
- **Sparsity Victory** (Fig 1): **~1,600× latency reduction** (0.2 ms vs 325 ms) when the Interconnect-Aware Heuristic chooses Zero-Copy for sparse lookups on a massive table — a critical optimization for distributed hash tables.
- **Hardware bottlenecks** (Fig 5 & 6): Achieved 1.8 GB/s lookup is below 32 GB/s peak (small, non-coalesced PCIe transactions); Slab-Hash compresses the P99 tail vs Chained (~3× P50–P99 gap), improving QoS.

**Future work**:

- **Multi-GPU**: Replicate or partition the table across GPUs; use **NVLink** for fast GPU–GPU transfer when available.
- **Distributed clusters**: **GPUDirect RDMA** and NCCL-style collectives for cross-node, GPU-aware communication.
- **Python/NumPy integration**: CuPy or PyBind11 bindings for data pipelines and distributed DB backends.

---

## Build, Layout & API (reference)

### Build and test

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build .
./test_hashmap
./test_slab
./benchmark_performance
./benchmark_hybrid
./benchmark_vs_cpu
./benchmark_zerocopy
./benchmark_heuristic
./benchmark_tail_latency
./benchmark_occupancy
./performance_validation_suite
```

Use `-DCMAKE_CUDA_ARCHITECTURES=86` for **RTX 3060 (Compute Capability 8.6)**; use `80` for Ampere datacenter (e.g. A100).

### Generate figures (Fig 1–9)

From the repo root, run benchmarks and plot (or plot from existing JSON):

```bash
python scripts/plot_benchmarks.py --run-benchmarks --build-dir build --out scripts/figures --save-json scripts/benchmark_data.json
# Or plot only from saved data:
python scripts/plot_benchmarks.py --data scripts/benchmark_data.json --out scripts/figures
```

### Directory layout

```
gpu_hashmap/
├── CMakeLists.txt
├── include/gpu_hashmap/
│   ├── types.cuh, slab_allocator.cuh, hash_buckets.cuh
│   ├── insert_kernel.cuh, hash_slab.cuh
│   ├── hash_map_api.h, lookup_kernel.cuh, heuristic_lookup.h
│   └── analysis/ (workloads, metrics, validation, probe_depth)
├── src/
│   ├── slab_allocator.cu, insert_kernel.cu, hash_map_api.cu
│   ├── lookup_api.cu, heuristic_lookup.cu
│   ├── slab_hash.cu
│   └── analysis/
├── scripts/
│   ├── plot_benchmarks.py
│   └── figures/  (fig1–fig9 PNGs)
└── tests/
    ├── test_insert.cu, test_slab.cu
    ├── benchmark_*.cu, performance_validation_suite.cu
    └── ...
```

### API (host)

```cpp
#include "gpu_hashmap/hash_map_api.h"
#include "gpu_hashmap/heuristic_lookup.h"

gpu_hashmap::HashTable table = {};
hash_map_create(&table, num_buckets, capacity);

hash_map_insert_batch(&table, d_keys, d_values, n);
// Or: warp-aggregated
hash_map_insert_batch_warp_aggregated(&table, d_keys, d_values, n);
// Or: build on host, upload once (hybrid)
hash_map_upload_from_host(&table, h_keys, h_values, n);

hash_map_lookup_batch_standard_copy(&table, h_keys, h_results, n);
hash_map_lookup_batch_zero_copy(&table, h_pinned_keys, h_pinned_results, n);

gpu_hashmap::HeuristicState state = {};
heuristic_init(&state);
heuristic_warm_up(&table, &state);
hash_map_lookup_batch_heuristic(&table, h_pinned_keys, h_pinned_results, n, &state);

hash_map_destroy(&table);
```

---

## License

Use as needed for academic or portfolio projects.
