# =============================================================================
# GPU Concurrent Hash Map - High-performance, billion-scale lookups
# =============================================================================
# Chained hashing + slab allocation; atomic updates, lock-free reads.
# Python wrapper (CuPy/PyBind11) planned for distributed DB backend.
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(gpu_hashmap
  LANGUAGES CXX CUDA
  VERSION 0.1.0
  DESCRIPTION "Concurrent GPU hash map with slab allocation"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures")
endif()

find_package(CUDAToolkit REQUIRED)

enable_testing()

# -----------------------------------------------------------------------------
# Library: hash map core (slab allocator + buckets + kernels)
# -----------------------------------------------------------------------------
add_library(gpu_hashmap STATIC
  src/slab_allocator.cu
  src/insert_kernel.cu
  src/hash_map_api.cu
  src/lookup_api.cu
  src/heuristic_lookup.cu
)
# Slab-based hash table (bucketed linear probing, one atomic per warp)
add_library(gpu_hashmap_slab STATIC
  src/slab_hash.cu
)
target_include_directories(gpu_hashmap_slab PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gpu_hashmap_slab PUBLIC CUDA::cudart)

# -----------------------------------------------------------------------------
# Analysis library: workloads, metrics, validation, probe depth
# -----------------------------------------------------------------------------
add_library(gpu_hashmap_analysis STATIC
  src/analysis/workloads.cpp
  src/analysis/metrics.cu
  src/analysis/validation.cu
  src/analysis/probe_depth.cu
)
target_include_directories(gpu_hashmap_analysis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gpu_hashmap_analysis PUBLIC CUDA::cudart)
target_include_directories(gpu_hashmap PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(gpu_hashmap PUBLIC cxx_std_17)
target_link_libraries(gpu_hashmap PUBLIC CUDA::cudart)

# -----------------------------------------------------------------------------
# Test: insert and lookup
# -----------------------------------------------------------------------------
add_executable(test_hashmap tests/test_insert.cu)
target_include_directories(test_hashmap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_hashmap PRIVATE gpu_hashmap)

# -----------------------------------------------------------------------------
# Performance benchmark: GPU (parallel) vs CPU (single-threaded)
# -----------------------------------------------------------------------------
add_executable(benchmark_performance tests/benchmark_performance.cu)
target_include_directories(benchmark_performance PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(benchmark_performance PRIVATE gpu_hashmap)

# -----------------------------------------------------------------------------
# Hybrid benchmark: CPU insert + upload once, GPU lookup only; vs full CPU
# -----------------------------------------------------------------------------
add_executable(benchmark_hybrid tests/benchmark_hybrid.cu)
target_include_directories(benchmark_hybrid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(benchmark_hybrid PRIVATE gpu_hashmap)

add_executable(test_slab tests/test_slab.cu)
target_include_directories(test_slab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_slab PRIVATE gpu_hashmap_slab)

# Correctness tests run under CTest (`ctest --output-on-failure`). The benchmarks
# are deliberately not registered: they are timing runs, not pass/fail checks.
add_test(NAME test_hashmap COMMAND test_hashmap)
add_test(NAME test_slab COMMAND test_slab)

# Zero-copy (mapped memory) lookup benchmark: kernel-only vs copy+kernel
add_executable(benchmark_zerocopy tests/benchmark_zerocopy.cu)
target_include_directories(benchmark_zerocopy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(benchmark_zerocopy PRIVATE gpu_hashmap)

# Heuristic optimizer benchmark: warm-up, path sweep, correctness
add_executable(benchmark_heuristic tests/benchmark_heuristic.cu)
target_include_directories(benchmark_heuristic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(benchmark_heuristic PRIVATE gpu_hashmap)

# Benchmark: all GPU variants vs CPU
add_executable(benchmark_vs_cpu tests/benchmark_vs_cpu.cu)
target_include_directories(benchmark_vs_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(benchmark_vs_cpu PRIVATE gpu_hashmap gpu_hashmap_slab)

# P99 tail latency: percentile breakdown (P50, P90, P99) and histogram of batch latencies
add_executable(benchmark_tail_latency
  tests/benchmark_tail_latency.cu
  tests/benchmark_tail_latency_impl.cpp
)
target_include_directories(benchmark_tail_latency PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
target_link_libraries(benchmark_tail_latency PRIVATE gpu_hashmap gpu_hashmap_slab)

# Occupancy vs throughput: sweep block sizes 32--1024 for chart data
add_executable(benchmark_occupancy tests/benchmark_occupancy.cu)
target_include_directories(benchmark_occupancy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(benchmark_occupancy PRIVATE gpu_hashmap)

# -----------------------------------------------------------------------------
# Performance analysis & validation suite (academic metrics)
# -----------------------------------------------------------------------------
add_executable(performance_validation_suite tests/performance_validation_suite.cu)
target_include_directories(performance_validation_suite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(performance_validation_suite PRIVATE gpu_hashmap gpu_hashmap_analysis)
