# Benchmark plotting scripts

Generate academic-grade figures for the GPU hash map portfolio.

## Setup

```bash
pip install -r requirements-plot.txt
```

## Data

**Option A — Run benchmarks from the script (recommended)**

From the repo root (or `scripts/`), after building in `build/`:

```bash
cd build && cmake .. && make -j
cd ../scripts
python plot_benchmarks.py --run-benchmarks --build-dir ../build --out figures --save-json benchmark_data.json
```

This runs `benchmark_heuristic`, `benchmark_vs_cpu`, `performance_validation_suite`, and optionally `benchmark_zerocopy`, parses their stdout/stderr, merges the data, and generates the figures. Use `--save-json` to write the merged data for later runs without re-running benchmarks.

**Option B — Manual runs and JSON**

1. Run the CUDA benchmarks and note the outputs:
   - `./benchmark_heuristic` → interconnect sweep (warm-up line on stderr; crossover N)
   - `./performance_validation_suite` → Zipfian (α), load factor, probe depth, roofline
   - `./benchmark_vs_cpu` → CPU vs Chained vs Slab (insert/lookup ms)

2. Edit `benchmark_data.json` and fill in the numbers (see `"note"` fields per section).

## Generate figures (JSON only)

```bash
python plot_benchmarks.py --data benchmark_data.json --out ../figures
```

Output (default `figures/` if `--out` omitted):

| File | Description |
|------|-------------|
| `fig1_interconnect_crossover.png` | Standard vs Zero-Copy latency; crossover point and safety margin (α=0.8) |
| `fig2_warp_aggregation_zipfian.png` | Standard vs warp-aggregated insert time for Zipfian α = 0.5–2.0 |
| `fig3_heterogeneous_speedup.png` | Throughput (ops/sec): CPU baseline, GPU Chained, GPU Slab |
| `fig4_load_factor_throughput_probe_depth.png` | Dual-axis: throughput and avg probe depth vs load factor (10%–99%) |
| `fig5_pcie_roofline.png` | Achieved effective bandwidth (Insert/Lookup) vs Gen3/Gen4 peak lines |

Figures use IEEE/academic style (serif fonts, grid, 300 DPI) for README and papers.
