# Benchmark plotting scripts

Generate the figures used in the top-level README from measured benchmark output.

## Setup

```bash
pip install -r requirements-plot.txt
```

## Measuring and plotting

Build first, then run the benchmarks and plot in one step:

```bash
python scripts/plot_benchmarks.py --run-benchmarks \
    --build-dir build \
    --out scripts/figures \
    --save-json scripts/benchmark_data.json
```

On Windows with a multi-config generator the executables land in a config
subdirectory, so pass that instead:

```powershell
python scripts\plot_benchmarks.py --run-benchmarks `
    --build-dir build\Release `
    --out scripts\figures `
    --save-json scripts\benchmark_data.json
```

To re-plot without re-measuring:

```bash
python scripts/plot_benchmarks.py --data scripts/benchmark_data.json --out scripts/figures
```

## Which benchmark feeds which figure

| Figure | Source benchmark |
|--------|------------------|
| `fig1_interconnect_crossover.png` | `benchmark_heuristic` |
| `fig2_warp_aggregation_zipfian.png` | `performance_validation_suite` |
| `fig3_heterogeneous_speedup.png` | `benchmark_vs_cpu` |
| `fig4_load_factor_throughput_probe_depth.png` | `performance_validation_suite` |
| `fig5_pcie_roofline.png` | `performance_validation_suite` |
| `fig6_tail_latency_p99.png` | `benchmark_tail_latency` (all three series, same batch size) |
| `fig7_occupancy_throughput.png` | `benchmark_occupancy` |
| `fig8_speedup_vs_cpu.png` | `benchmark_vs_cpu` |
| `fig9_timings_by_approach.png` | `benchmark_vs_cpu` |

## No synthetic data

Every number the script plots is parsed from benchmark stdout/stderr. The script
does not extrapolate, interpolate, or substitute placeholder values.

If a benchmark fails to run, or its output does not parse, the script prints the
reason to stderr and **skips the affected figure** rather than plotting invented
numbers. A missing figure therefore means "not measured", and any figure present
in `figures/` corresponds to data present in `benchmark_data.json`.

When re-plotting from JSON, a missing `--data` file is a hard error rather than a
fallback to sample data.

Figures use a serif/grid style at 300 DPI.

## Measurement notes

- `benchmark_vs_cpu` reports the **median of 5 runs** per approach. Its slab totals are
  near 1 ms, so single-shot timings were dominated by scheduler noise.
- `benchmark_heuristic` reports the median of 9 timed reps per batch size, and measures
  *both* paths at every size rather than only the one the heuristic picks.
- Several figures use log axes because the measured spans cover two to three orders of
  magnitude; the axis labels say so.
- Absolute timings on a laptop GPU vary with thermal state — see section 5 of the
  top-level README. Ratios are far more stable than absolute milliseconds.
