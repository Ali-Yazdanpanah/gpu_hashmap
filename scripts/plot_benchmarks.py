#!/usr/bin/env python3
"""
Generate academic-grade figures for the CUDA Key-Value Store PhD portfolio.
Reads benchmark_data.json, or runs benchmarks and parses stdout, then produces
IEEE-style high-DPI PNGs.

Usage:
  pip install matplotlib seaborn numpy
  # From JSON only:
  python plot_benchmarks.py [--data benchmark_data.json] [--out figures]
  # Run benchmarks then plot (from repo root or scripts/):
  python plot_benchmarks.py --run-benchmarks [--build-dir ../build] [--out figures]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns  # optional; style still works without it
except ImportError:
    pass


# -----------------------------------------------------------------------------
# IEEE / Academic style
# -----------------------------------------------------------------------------
DPI = 300
FIGW = 6.0
FIGH = 4.0
SERIF = "serif"
FONTSIZE = 11
plt.rcParams.update({
    "font.family": SERIF,
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": FONTSIZE,
    "axes.labelsize": FONTSIZE + 1,
    "axes.titlesize": FONTSIZE + 2,
    "xtick.labelsize": FONTSIZE - 1,
    "ytick.labelsize": FONTSIZE - 1,
    "legend.fontsize": FONTSIZE - 1,
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_data(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Run benchmarks and parse stdout
# -----------------------------------------------------------------------------
def run_benchmark(build_dir: Path, exe: str, timeout_s: int = 300) -> str:
    path = build_dir / exe
    if not path.is_file():
        path_exe = build_dir / (exe + ".exe")
        path = path_exe if path_exe.is_file() else path
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark not found: {build_dir / exe} (run cmake --build first)")
    try:
        result = subprocess.run(
            [str(path)],
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"{exe} timed out after {timeout_s}s")
    if result.returncode != 0:
        sys.stderr.write(result.stderr or "")
        result.check_returncode()
    # Merge stderr so parsers can find lines printed to stderr (e.g. heuristic warm-up)
    return (result.stdout or "") + "\n" + (result.stderr or "")


def parse_heuristic(stdout: str) -> dict:
    """Parse benchmark_heuristic: warm-up, per-N sweep of BOTH paths, accuracy, sparsity.

    Every batch size in the sweep is measured on both paths, so nothing here is
    extrapolated. If the sweep table cannot be parsed, the caller is told (by the
    absence of the keys) rather than being handed synthetic numbers.
    """
    data = {"safety_margin_alpha": 0.8}

    m = re.search(r"Zero-Copy when n <= (\d+)", stdout)
    if m:
        data["zerocopy_below_n"] = int(m.group(1))
        data["crossover_n"] = int(m.group(1))  # fig1 vertical line = ZC threshold
        data["decided"] = True
    elif re.search(r"UNDECIDED", stdout):
        data["zerocopy_below_n"] = 0
        data["crossover_n"] = (1 << 64) - 1
        data["decided"] = False
    else:
        m_legacy = re.search(r"crossover_n=(\d+)", stdout)
        if m_legacy:
            data["crossover_n"] = int(m_legacy.group(1))

    # Sweep (human table): N Std ZC Faster Chosen Correct ZC/Std
    rows = re.findall(
        r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+(Standard|Zero-Copy)\s+"
        r"(Standard|Zero-Copy|Undecided)\s+(yes|no)\s+([\d.]+)\s*$",
        stdout,
        re.MULTILINE,
    )
    if not rows:
        # Older 4-column format (pre accuracy columns)
        rows_old = re.findall(
            r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+(Standard|Zero-Copy)\s+([\d.]+)\s*$",
            stdout,
            re.MULTILINE,
        )
        if rows_old:
            rows_old.sort(key=lambda r: int(r[0]))
            data["batch_sizes_n"] = [int(r[0]) for r in rows_old]
            data["batch_sizes_k"] = [int(r[0]) / 1024.0 for r in rows_old]
            data["standard_path_ms"] = [float(r[1]) for r in rows_old]
            data["zerocopy_path_ms"] = [float(r[2]) for r in rows_old]
            data["chosen_path"] = [r[3] for r in rows_old]
    else:
        rows.sort(key=lambda r: int(r[0]))
        data["batch_sizes_n"] = [int(r[0]) for r in rows]
        data["batch_sizes_k"] = [int(r[0]) / 1024.0 for r in rows]
        data["standard_path_ms"] = [float(r[1]) for r in rows]
        data["zerocopy_path_ms"] = [float(r[2]) for r in rows]
        data["chosen_path"] = [r[4] for r in rows]

    # Machine-readable accuracy block
    acc = re.search(r"HEURISTIC ACCURACY:\s+(\d+)\s*/\s*(\d+)", stdout)
    if acc:
        data_acc = {
            "correct": int(acc.group(1)),
            "total": int(acc.group(2)),
            "rows": [],
        }
        for mrow in re.finditer(
            r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+(Standard|Zero-Copy)\s+"
            r"(Standard|Zero-Copy|Undecided)\s+(yes|no)\s*$",
            stdout,
            re.MULTILINE,
        ):
            data_acc["rows"].append(
                {
                    "n": int(mrow.group(1)),
                    "standard_ms": float(mrow.group(2)),
                    "zerocopy_ms": float(mrow.group(3)),
                    "faster": mrow.group(4),
                    "chosen": mrow.group(5),
                    "correct": mrow.group(6) == "yes",
                }
            )
        data["heuristic_accuracy"] = data_acc

    # Sparsity block: live-data migration, naive full-capacity migration, zero-copy
    std_live = re.search(r"Standard path \(live-data copy \+ lookup\):\s+([\d.]+)\s+ms", stdout)
    naive_full = re.search(r"Naive full-capacity migration only:\s+([\d.]+)\s+ms", stdout)
    zc_nomig = re.search(r"Zero-Copy path \(no table migration\):\s+([\d.]+)\s+ms", stdout)
    if std_live and zc_nomig:
        data["sparsity_n_k"] = 10  # 10K lookups
        data["sparsity_standard_ms"] = float(std_live.group(1))
        data["sparsity_zerocopy_ms"] = float(zc_nomig.group(1))
    if naive_full:
        data["sparsity_naive_full_migration_ms"] = float(naive_full.group(1))
    return data


def parse_placement(stdout: str) -> dict:
    """Parse benchmark_placement machine-readable matrix."""
    rows = []
    for m in re.finditer(
        r"^(chained|slab)\s+(mapped-host|device)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
        stdout,
        re.MULTILINE,
    ):
        rows.append(
            {
                "scheme": m.group(1),
                "placement": m.group(2),
                "insert_ms": float(m.group(3)),
                "lookup_ms": float(m.group(4)),
                "total_ms": float(m.group(5)),
                "total_p05": float(m.group(6)),
                "total_p95": float(m.group(7)),
                "dropped_pct": float(m.group(8)),
            }
        )
    n_runs = re.search(r"PLACEMENT MATRIX \(n_runs=(\d+)\)", stdout)
    dominant = re.search(r"Dominant factor:\s+(.+)$", stdout, re.MULTILINE)
    out: dict = {"rows": rows}
    if n_runs:
        out["n_runs"] = int(n_runs.group(1))
    if dominant:
        out["dominant"] = dominant.group(1).strip()
    return out


def parse_hit_rate(stdout: str):
    rows = re.findall(
        r"^(\d+(?:\.\d+)?)\s+([\d.]+)\s+([\d.]+)\s*$",
        stdout.split("HIT RATE SWEEP")[-1] if "HIT RATE SWEEP" in stdout else "",
        re.MULTILINE,
    )
    # Filter to plausible hit-rate percentages (0..100)
    parsed = []
    for pct, ms, pd in rows:
        p = float(pct)
        if 0.0 <= p <= 100.0:
            parsed.append((p, float(ms), float(pd)))
    if not parsed:
        return None
    parsed.sort(key=lambda r: r[0])
    return {
        "hit_rate_pct": [p for p, _, _ in parsed],
        "lookup_ms": [ms for _, ms, _ in parsed],
        "avg_probe_depth": [pd for _, _, pd in parsed],
    }


def parse_vs_cpu(stdout: str) -> dict:
    """Parse benchmark_vs_cpu: all 5 rows (CPU, GPU chained, warp-agg, slab, Hybrid). Insert(ms), Lookup(ms), Total(ms), vs CPU.
    Also parses CPU per-find latency (µs): P50, P90, P99 for Figure 6."""
    # Short display label per benchmark row label, so the plot legend cannot
    # silently desync from what the benchmark actually printed.
    display_names = {
        "cpu (unordered_map)": "CPU (unordered_map)",
        "gpu chained": "GPU Chained",
        "gpu warp-aggregated": "GPU Warp-agg",
        "gpu slab (8/bucket)": "GPU Slab",
        "hybrid (cpu+gpu lup)": "Hybrid (CPU+GPU lup)",
    }
    data = {
        "approaches": [],
        "insert_ms": [],
        "lookup_ms": [],
        "total_ms": [],
        "speedup_x": [],
        "total_ops": 1536000,
        "cpu_per_find_us": None,
        "dropped_inserts": {},
        "dropped_pct": {},
        "inserts_requested": None,
    }
    pattern = re.compile(
        r"^\s*(CPU \(unordered_map\)|GPU chained|GPU warp-aggregated|GPU slab \(8/bucket\)|Hybrid \(CPU\+GPU lup\))\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x",
        re.IGNORECASE,
    )
    for line in stdout.splitlines():
        m = pattern.match(line)
        if m:
            label = m.group(1).strip().lower()
            data["approaches"].append(display_names.get(label, m.group(1).strip()))
            data["insert_ms"].append(float(m.group(2)))
            data["lookup_ms"].append(float(m.group(3)))
            data["total_ms"].append(float(m.group(4)))
            data["speedup_x"].append(float(m.group(5)))
    # CPU per-find latency (µs): "  CPU per-find latency (µs):  P50=    1.23   P90=    2.34   P99=    4.56"
    cpu_lat = re.search(
        r"CPU per-find latency \(µs\):\s+P50=\s+([\d.]+)\s+P90=\s+([\d.]+)\s+P99=\s+([\d.]+)",
        stdout,
    )
    if cpu_lat:
        data["cpu_per_find_us"] = {
            "p50_us": float(cpu_lat.group(1)),
            "p90_us": float(cpu_lat.group(2)),
            "p99_us": float(cpu_lat.group(3)),
        }
    # Dropped inserts are a headline limitation, so they belong in the data rather
    # than only in the benchmark's stdout:
    #   "  Inserts not stored (of 1048576 requested):"
    #   "    GPU slab (8/bucket)         8792  (0.838%)"
    req = re.search(r"Inserts not stored \(of (\d+) requested\)", stdout)
    if req:
        data["inserts_requested"] = int(req.group(1))
    for m in re.finditer(
        r"^\s{4,}(CPU \(unordered_map\)|GPU chained|GPU warp-aggregated|GPU slab \(8/bucket\))\s+"
        r"(\d+)\s+\(([\d.]+)%\)",
        stdout,
        re.MULTILINE | re.IGNORECASE,
    ):
        label = display_names.get(m.group(1).strip().lower(), m.group(1).strip())
        data["dropped_inserts"][label] = int(m.group(2))
        data["dropped_pct"][label] = float(m.group(3))
    return data


def parse_validation_suite(stdout: str) -> dict:
    """Parse performance_validation_suite: Zipfian, load factor, probe depth, roofline."""
    result = {
        "zipfian": None,
        "load_factor": None,
        "probe_depth_success": None,
        "roofline": None,
    }
    # 2a Zipfian: "  0.5   45.20   38.10   12.30" (alpha, InsStd, InsWarp, Lookup)
    zipf_alphas, zipf_ins, zipf_warp, zipf_lup = [], [], [], []
    for line in stdout.splitlines():
        m = re.match(r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line.strip())
        if m:
            try:
                a = float(m.group(1))
                if 0.4 <= a <= 2.1:
                    zipf_alphas.append(a)
                    zipf_ins.append(float(m.group(2)))
                    zipf_warp.append(float(m.group(3)))
                    zipf_lup.append(float(m.group(4)))
            except ValueError:
                pass
    if zipf_alphas:
        result["zipfian"] = {
            "alphas": zipf_alphas,
            "standard_insert_ms": zipf_ins,
            "warp_agg_insert_ms": zipf_warp,
            "lookup_ms": zipf_lup,
        }

    # 2b Load factor: "  10%  8.10  2.10  185000 ops/s  1.350"
    lf_pct, lf_ins, lf_lup, lf_throughput, lf_probe = [], [], [], [], []
    for line in stdout.splitlines():
        m = re.match(
            r"^\s*(\d+)\s*%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+ops/s\s+([\d.]+)\s*$",
            line.strip(),
        )
        if m:
            lf_pct.append(int(m.group(1)))
            lf_ins.append(float(m.group(2)))
            lf_lup.append(float(m.group(3)))
            lf_throughput.append(float(m.group(4)))
            lf_probe.append(float(m.group(5)))
    if lf_pct:
        result["load_factor"] = {
            "load_factor_pct": lf_pct,
            "insert_ms": lf_ins,
            "lookup_ms": lf_lup,
            "throughput_ops_per_sec": lf_throughput,
            "avg_probe_depth": lf_probe,
        }

    # 2c Probe depth: "  Successful find — avg probe depth: 1.234 (count ...)"
    m = re.search(r"Successful find — avg probe depth: ([\d.]+)", stdout)
    if m:
        result["probe_depth_success"] = float(m.group(1))
    m = re.search(r"Unsuccessful find — avg probe depth: ([\d.]+) \(count (\d+)\)", stdout)
    if m and int(m.group(2)) > 0:
        result["probe_depth_unsuccessful"] = float(m.group(1))

    # Section 4: "  Insert (effective)             4.20       16.00      26.2%"
    ins_bw = re.search(r"Insert \(effective\)\s+([\d.]+)\s+[\d.]+\s+", stdout)
    lup_bw = re.search(r"Lookup \(effective\)\s+([\d.]+)\s+[\d.]+\s+", stdout)
    if ins_bw and lup_bw:
        result["roofline"] = {
            "operations": ["Insert", "Lookup"],
            "achieved_gbps": [float(ins_bw.group(1)), float(lup_bw.group(1))],
            "peak_gen3_gbps": 16,
            "peak_gen4_gbps": 32,
        }
    return result


def parse_tail_latency(stdout: str) -> dict:
    """Parse benchmark_tail_latency: GPU Chained, GPU Slab, and CPU P50, P90, P99 (batch latency in ms)."""
    data = {"gpu": {}, "gpu_slab": {}, "cpu": {}}
    # P50 (median)  X.XXXX ms  and  P90  X.XXXX ms  and  P99  X.XXXX ms
    def extract_percentiles(block):
        out = {}
        m = re.search(r"P50 \(median\)\s+([\d.]+)\s+ms", block)
        if m:
            out["p50_ms"] = float(m.group(1))
        m = re.search(r"P90\s+([\d.]+)\s+ms", block)
        if m:
            out["p90_ms"] = float(m.group(1))
        m = re.search(r"P99\s+([\d.]+)\s+ms", block)
        if m:
            out["p99_ms"] = float(m.group(1))
        return out
    idx_gpu = stdout.find("GPU (chained lookup")
    idx_slab = stdout.find("GPU Slab (batch latency")
    idx_cpu = stdout.find("CPU (std::unordered_map")
    if idx_gpu >= 0:
        data["gpu"] = extract_percentiles(stdout[idx_gpu:])
    if idx_slab >= 0:
        data["gpu_slab"] = extract_percentiles(stdout[idx_slab:])
    if idx_cpu >= 0:
        data["cpu"] = extract_percentiles(stdout[idx_cpu:])
    return data


def parse_occupancy(stdout: str) -> dict:
    """Parse benchmark_occupancy: block size, blocks/SM, occupancy, throughput Mops/s."""
    data = {"block_sizes": [], "occupancy": [], "throughput_mops": []}
    # Lines like "  32        12          0.25      45.32 Mops/s"
    for m in re.finditer(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+Mops/s", stdout, re.MULTILINE):
        data["block_sizes"].append(int(m.group(1)))
        data["occupancy"].append(float(m.group(3)))
        data["throughput_mops"].append(float(m.group(4)))
    return data


def run_and_parse_all(build_dir: Path) -> dict:
    """Run all benchmarks, parse stdout, return merged data dict for plotting.

    Every value returned comes from a benchmark run. When a benchmark fails or its
    output cannot be parsed, the corresponding key is left absent and a warning is
    printed; the affected figure is then skipped. Nothing is filled in with
    placeholder or extrapolated numbers.
    """
    build_dir = Path(build_dir).resolve()
    data: dict = {}
    failures: list = []

    # benchmark_heuristic -> fig1
    try:
        out = run_benchmark(build_dir, "benchmark_heuristic")
        parsed = parse_heuristic(out)
        if "standard_path_ms" in parsed:
            interconnect = {k: v for k, v in parsed.items() if k != "heuristic_accuracy"}
            data["interconnect"] = interconnect
            if "heuristic_accuracy" in parsed:
                data["heuristic_accuracy"] = parsed["heuristic_accuracy"]
            print("Parsed benchmark_heuristic")
        else:
            failures.append("benchmark_heuristic: ran, but the sweep table did not parse")
    except Exception as e:
        failures.append(f"benchmark_heuristic: {e}")

    # benchmark_placement -> placement_matrix (scheme x residency)
    try:
        out = run_benchmark(build_dir, "benchmark_placement", timeout_s=900)
        pm = parse_placement(out)
        if pm.get("rows"):
            data["placement_matrix"] = pm
            print("Parsed benchmark_placement")
        else:
            failures.append("benchmark_placement: ran, but the matrix did not parse")
    except Exception as e:
        failures.append(f"benchmark_placement: {e}")

    # benchmark_vs_cpu -> fig3, fig8, fig9 (+ CPU per-find latency for fig6)
    cpu_per_find_us_from_vs_cpu = None
    try:
        out = run_benchmark(build_dir, "benchmark_vs_cpu")
        vs_cpu = parse_vs_cpu(out)
        if vs_cpu.get("insert_ms"):
            data["heterogeneous"] = {k: v for k, v in vs_cpu.items() if k != "cpu_per_find_us"}
            print("Parsed benchmark_vs_cpu")
        else:
            failures.append("benchmark_vs_cpu: ran, but the comparison table did not parse")
        if vs_cpu.get("cpu_per_find_us"):
            cpu_per_find_us_from_vs_cpu = vs_cpu["cpu_per_find_us"]
    except Exception as e:
        failures.append(f"benchmark_vs_cpu: {e}")

    # performance_validation_suite -> fig2, fig4, fig5 (slow)
    try:
        out = run_benchmark(build_dir, "performance_validation_suite", timeout_s=1800)
        vs = parse_validation_suite(out)
        if vs["zipfian"] and vs["zipfian"].get("warp_agg_insert_ms"):
            data["warp_aggregation"] = {
                "alphas": vs["zipfian"]["alphas"],
                "standard_insert_ms": vs["zipfian"]["standard_insert_ms"],
                "warp_agg_insert_ms": vs["zipfian"]["warp_agg_insert_ms"],
            }
        else:
            failures.append("performance_validation_suite: Zipfian table did not parse")
        if vs["load_factor"] and vs["load_factor"].get("avg_probe_depth"):
            data["load_factor"] = vs["load_factor"]
        else:
            failures.append("performance_validation_suite: load-factor table did not parse")
        if vs["roofline"]:
            data["roofline"] = vs["roofline"]
        else:
            failures.append("performance_validation_suite: roofline section did not parse")
        for k in ("probe_depth_success", "probe_depth_unsuccessful"):
            if vs.get(k) is not None:
                data[k] = vs[k]
        hr = parse_hit_rate(out)
        if hr:
            data["hit_rate"] = hr
        print("Parsed performance_validation_suite")
    except Exception as e:
        failures.append(f"performance_validation_suite: {e}")

    # benchmark_tail_latency -> fig6
    try:
        out = run_benchmark(build_dir, "benchmark_tail_latency")
        tl = parse_tail_latency(out)
        if cpu_per_find_us_from_vs_cpu:
            tl["cpu_per_find_us"] = cpu_per_find_us_from_vs_cpu
        data["tail_latency"] = tl
        print("Parsed benchmark_tail_latency")
    except Exception as e:
        failures.append(f"benchmark_tail_latency: {e}")

    # benchmark_occupancy -> fig7
    try:
        out = run_benchmark(build_dir, "benchmark_occupancy")
        occ = parse_occupancy(out)
        if occ.get("block_sizes"):
            data["occupancy"] = occ
            print("Parsed benchmark_occupancy")
        else:
            failures.append("benchmark_occupancy: ran, but no throughput rows parsed")
    except Exception as e:
        failures.append(f"benchmark_occupancy: {e}")

    if failures:
        print("\n!! Some data is missing; the affected figures will be SKIPPED, not faked:",
              file=sys.stderr)
        for f in failures:
            print(f"   - {f}", file=sys.stderr)
        print("", file=sys.stderr)

    return data


# -----------------------------------------------------------------------------
# 1. Interconnect Latency vs. Batch Size (Crossover + Safety Margin)
# -----------------------------------------------------------------------------
def plot_interconnect_crossover(data: dict, outdir: Path) -> None:
    d = data.get("interconnect") or {}
    if not d.get("batch_sizes_k") or not d.get("standard_path_ms"):
        print("Skipping fig1 (no measured interconnect sweep data)")
        return
    n_k = np.array(d["batch_sizes_k"], dtype=float)
    t_std = np.array(d["standard_path_ms"])
    t_zc = np.array(d["zerocopy_path_ms"])
    # crossover_n: from benchmark = actual N (e.g. 262144); from JSON/sample may be in K (e.g. 256)
    crossover_n_raw = d.get("crossover_n", 0)
    size_max = 1 << 64  # heuristic uses SIZE_MAX when no crossover
    if crossover_n_raw >= size_max - 1 or crossover_n_raw <= 0:
        crossover_n_k = None
    elif crossover_n_raw < 1024:
        crossover_n_k = float(crossover_n_raw)  # already in thousands (e.g. 256)
    else:
        crossover_n_k = crossover_n_raw / 1024.0  # convert actual N to thousands

    alpha = d.get("safety_margin_alpha", 0.8)

    # Full-page size so the chart is not condensed (x-axis has room for 10K–512K)
    fig, ax = plt.subplots(figsize=(12.0, 7.0))

    # Safety margin first (behind lines) so it is not masked
    t_thresh = alpha * t_std
    ax.fill_between(n_k, t_zc, t_thresh, where=(t_zc <= t_thresh), alpha=0.28, color="C1", zorder=0,
                    label=f"Safety margin (α={alpha})")

    ax.plot(n_k, t_std, "o-", color="C0", label="Standard path (Copy + Kernel)", linewidth=2, markersize=8, zorder=2)
    ax.plot(n_k, t_zc, "s-", color="C1", label="Zero-Copy path (Mapped Memory)", linewidth=2, markersize=8, zorder=2)

    # Sparsity-driven crossover: offset in x so both markers visible; use distinct markers so they don't mask each other
    sparsity_n = d.get("sparsity_n_k")
    sparsity_std = d.get("sparsity_standard_ms")
    sparsity_zc = d.get("sparsity_zerocopy_ms")
    if sparsity_n is not None and sparsity_std is not None and sparsity_zc is not None:
        x_std, x_zc = sparsity_n * 0.93, sparsity_n * 1.07
        ax.plot(x_std, sparsity_std, "*", color="C0", markersize=16, markeredgecolor="black", markeredgewidth=1.2,
                zorder=5, label="Standard (2GB table copy + 10K lookup)")
        ax.plot(x_zc, sparsity_zc, "D", color="C1", markersize=10, markeredgecolor="black", markeredgewidth=1.2,
                zorder=5, label="Zero-Copy (no table migration, 10K lookup)")
        ax.annotate("Sparsity:\nZero-Copy bypasses\ntable migration tax",
                    xy=(x_zc, sparsity_zc), xytext=(sparsity_n * 1.8, (sparsity_std + sparsity_zc) / 2),
                    fontsize=8, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1))
        if sparsity_zc < sparsity_std * 0.1:  # call out the low point so it's visible
            ax.annotate(f"  {sparsity_zc:.1f} ms", xy=(x_zc, sparsity_zc), fontsize=8, va="bottom", ha="left")

    # Only draw crossover line when it's within the plotted batch-size range (crossover_n is in actual N, so convert to K)
    x_min, x_max = float(np.min(n_k)), float(np.max(n_k))
    if sparsity_n is not None:
        x_min = min(x_min, float(sparsity_n) * 0.93)
    # The two curves do not intersect on this machine, so this line is where the
    # heuristic switches paths, not a point where the faster path changes.
    if crossover_n_k is not None and x_min <= crossover_n_k <= x_max:
        ax.axvline(x=crossover_n_k, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Zero-Copy when N ≤ {int(round(crossover_n_k))}K (calibrated)")

    ax.set_xlabel("Batch size (thousands of lookups, log scale)")
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title("Interconnect: Standard vs Zero-Copy Path")
    # Batch sizes double each step, so a linear axis crushes every small-N point into
    # the left edge and the tick labels overlap.
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(x_min * 0.85, x_max * 1.18)
    xticks = list(n_k)
    xtick_labels = [f"{int(x)}K" for x in n_k]
    if sparsity_n is not None and sparsity_n not in xticks:
        xticks.append(float(sparsity_n))
        xticks.sort()
        xtick_labels = [f"{int(x)}K" if x != sparsity_n else "10K\n(sparse)" for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.minorticks_off()
    # Legend outside chart (right) so it never hides data
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(1.02, 1), frameon=True)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(outdir / "fig1_interconnect_crossover.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_interconnect_crossover.png")


# -----------------------------------------------------------------------------
# 2. Warp-Aggregation Efficiency (Zipfian skew)
# -----------------------------------------------------------------------------
def plot_warp_aggregation(data: dict, outdir: Path) -> None:
    d = data.get("warp_aggregation") or {}
    if not d.get("alphas") or not d.get("warp_agg_insert_ms"):
        print("Skipping fig2 (no measured warp-aggregation data)")
        return
    alphas = np.array(d["alphas"])
    std_ms = np.array(d["standard_insert_ms"])
    warp_ms = np.array(d["warp_agg_insert_ms"])
    # Throughput as 1/time (higher is better); or show time (lower is better)
    x = np.arange(len(alphas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    bars1 = ax.bar(x - width / 2, std_ms, width, label="Standard insert_kernel", color="C0", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, warp_ms, width, label="Warp-aggregated insert", color="C1", edgecolor="black", linewidth=0.5)

    # Insert time spans three orders of magnitude across these skews, so a linear
    # axis flattens the low-skew bars into the baseline.
    ax.set_yscale("log")
    ax.set_xlabel("Zipfian skew α")
    ax.set_ylabel("Insert time (ms, log scale)")
    ax.set_title("Warp-aggregation efficiency under hot-key contention")
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in alphas])
    for xi, s, w in zip(x, std_ms, warp_ms):
        if w > 0:
            ax.annotate(f"{s / w:.1f}x", xy=(xi, max(s, w)), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=9)
    ax.legend(loc="upper left")
    fig.savefig(outdir / "fig2_warp_aggregation_zipfian.png")
    plt.close(fig)
    print("Saved fig2_warp_aggregation_zipfian.png")


# -----------------------------------------------------------------------------
# 3. Heterogeneous Speedup (CPU vs Chained vs Slab)
# -----------------------------------------------------------------------------
def plot_heterogeneous_speedup(data: dict, outdir: Path) -> None:
    d = data.get("heterogeneous") or {}
    approaches = d.get("approaches") or []
    if not approaches:
        print("Skipping fig3 (no heterogeneous data)")
        return
    insert_ms = np.array(d["insert_ms"])
    lookup_ms = np.array(d["lookup_ms"])
    total_ops = d["total_ops"]
    total_ms = insert_ms + lookup_ms
    # total_ms is in milliseconds, so seconds = total_ms / 1000.
    ops_per_sec = total_ops / (total_ms / 1000.0)

    n = len(approaches)
    x = np.arange(n)
    width = 0.5

    fig, ax = plt.subplots(figsize=(max(FIGW, n * 1.2), FIGH))
    colors = ["C2", "C0", "C3", "C1", "C4"][:n]
    bars = ax.bar(x, ops_per_sec, width, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Throughput (ops/sec)")
    ax.set_xlabel("Implementation")
    ax.set_title("Heterogeneous speedup: CPU vs GPU variants vs Hybrid")
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, rotation=20, ha="right")
    for i, (bar, v) in enumerate(zip(bars, ops_per_sec)):
        if v >= 1e9:
            label = f"{v/1e9:.2f}G"
        elif v >= 1e6:
            label = f"{v/1e6:.2f}M"
        elif v >= 1e3:
            label = f"{v/1e3:.1f}K"
        else:
            label = f"{v:.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ops_per_sec) * 0.02,
                label, ha="center", va="bottom", fontsize=9)
    fig.savefig(outdir / "fig3_heterogeneous_speedup.png")
    plt.close(fig)
    print("Saved fig3_heterogeneous_speedup.png")


# -----------------------------------------------------------------------------
# 3b. Speedup (vs CPU)
# -----------------------------------------------------------------------------
def plot_timings_and_speedup(data: dict, outdir: Path) -> None:
    d = data.get("heterogeneous") or {}
    approaches = d.get("approaches") or []
    insert_ms = np.array(d.get("insert_ms") or [])
    lookup_ms = np.array(d.get("lookup_ms") or [])
    total_ms = np.array(d.get("total_ms") or [])
    speedup_x = np.array(d.get("speedup_x") or [])
    if len(approaches) == 0 or len(insert_ms) != len(approaches):
        print("Skipping timings/speedup chart (no heterogeneous data)")
        return
    n = len(approaches)
    # Support old JSON without total_ms / speedup_x: derive from insert_ms + lookup_ms
    if len(total_ms) != n:
        total_ms = insert_ms + lookup_ms
    if len(speedup_x) != n:
        cpu_total = float(total_ms[0]) if n > 0 else 1.0
        speedup_x = np.array([cpu_total / t if t > 0 else 0.0 for t in total_ms])
    x = np.arange(n)

    fig, ax2 = plt.subplots(1, 1, figsize=(max(FIGW, n * 1.2), FIGH))

    # Speedup (vs CPU)
    colors = ["C2", "C0", "C3", "C1", "C4"][:n]
    bars = ax2.bar(x, speedup_x, width=0.5, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="CPU baseline (1.0×)")
    ax2.set_ylabel("Speedup (× vs CPU)")
    ax2.set_xlabel("Implementation")
    ax2.set_title("Speedup vs CPU (unordered_map)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(approaches, rotation=20, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if speedup_x.max() > 10:
        ax2.set_yscale("log")
        ax2.set_ylim(max(0.05, speedup_x.min() * 0.5), speedup_x.max() * 1.5)
    for bar, v in zip(bars, speedup_x):
        h = bar.get_height()
        if v >= 1:
            y_pos = h * 1.08
            va = "bottom"
        else:
            y_pos = h * 0.92
            va = "top"
        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{v:.2f}×", ha="center", va=va, fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "fig8_speedup_vs_cpu.png", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print("Saved fig8_speedup_vs_cpu.png")


# -----------------------------------------------------------------------------
# 9. Insert and Lookup time by approach (log scale)
# -----------------------------------------------------------------------------
def plot_timings_by_approach(data: dict, outdir: Path) -> None:
    d = data.get("heterogeneous") or {}
    approaches = d.get("approaches") or []
    insert_ms = np.array(d.get("insert_ms") or [], dtype=float)
    lookup_ms = np.array(d.get("lookup_ms") or [], dtype=float)
    if not approaches or len(insert_ms) != len(approaches) or len(lookup_ms) != len(approaches):
        print("Skipping fig9 (no heterogeneous timing data)")
        return
    n = len(approaches)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(FIGW, n * 1.3), FIGH))
    ax.bar(x - width / 2, insert_ms, width, label="Insert (ms)",
           color="C0", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, lookup_ms, width, label="Lookup (ms)",
           color="C1", edgecolor="black", linewidth=0.5)
    # Log scale so sub-millisecond and hundreds-of-milliseconds bars are both readable.
    ax.set_yscale("log")
    ax.set_ylabel("Time (ms, log scale)")
    ax.set_xlabel("Implementation")
    ax.set_title("Insert and lookup time by approach")
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, rotation=20, ha="right")
    ax.legend()
    for xi, v in zip(x - width / 2, insert_ms):
        if v > 0:
            ax.text(xi, v * 1.1, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + width / 2, lookup_ms):
        if v > 0:
            ax.text(xi, v * 1.1, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig9_timings_by_approach.png", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print("Saved fig9_timings_by_approach.png")


# -----------------------------------------------------------------------------
# 4. VRAM Occupancy vs Throughput (dual-axis with Probe Depth)
# -----------------------------------------------------------------------------
def plot_load_factor_throughput(data: dict, outdir: Path) -> None:
    d = data.get("load_factor") or {}
    if not d.get("load_factor_pct") or not d.get("avg_probe_depth"):
        print("Skipping fig4 (no measured load-factor / probe-depth data)")
        return
    lf = np.array(d["load_factor_pct"])
    throughput = np.array(d["throughput_ops_per_sec"])
    probe_depth = np.array(d["avg_probe_depth"])

    fig, ax1 = plt.subplots(figsize=(FIGW, FIGH))
    color1, color2 = "C0", "C1"
    ax1.set_xlabel("Load factor (%)")
    ax1.set_ylabel("Throughput (ops/sec)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.plot(lf, throughput, "o-", color=color1, linewidth=2, markersize=8)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Avg. probe depth", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.plot(lf, probe_depth, "s--", color=color2, linewidth=2, markersize=8)

    ax1.set_title("Chained table: throughput and probe depth vs load factor")
    ax1.set_xticks(lf)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_load_factor_throughput_probe_depth.png")
    plt.close(fig)
    print("Saved fig4_load_factor_throughput_probe_depth.png")


# -----------------------------------------------------------------------------
# 5. PCIe Roofline Model
# -----------------------------------------------------------------------------
def plot_roofline(data: dict, outdir: Path) -> None:
    d = data.get("roofline") or {}
    if not d.get("operations") or not d.get("achieved_gbps"):
        print("Skipping fig5 (no measured roofline data)")
        return
    ops = d["operations"]
    achieved = np.array(d["achieved_gbps"])
    peak_gen3 = d["peak_gen3_gbps"]
    peak_gen4 = d["peak_gen4_gbps"]

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    x = np.arange(len(ops))
    width = 0.35

    # Theoretical peaks (horizontal lines)
    ax.axhline(y=peak_gen3, color="gray", linestyle=":", linewidth=1.5, label=f"PCIe Gen3×16 ({peak_gen3} GB/s)")
    ax.axhline(y=peak_gen4, color="gray", linestyle="--", linewidth=1, label=f"PCIe Gen4×16 ({peak_gen4} GB/s)")

    # Achieved bandwidth (scatter/bar)
    colors = ["C0", "C1"]
    for i, (op, gbps) in enumerate(zip(ops, achieved)):
        ax.bar(i, gbps, width=0.4, color=colors[i], edgecolor="black", linewidth=0.5)
        ax.text(i, gbps * 1.15, f"{gbps:.2f} GB/s\n({100.0 * gbps / peak_gen3:.1f}% of Gen3)",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Effective bandwidth (GB/s, log scale)")
    ax.set_xlabel("Operation")
    ax.set_title("PCIe roofline: Achieved vs theoretical peak bandwidth")
    ax.set_xticks(x)
    ax.set_xticklabels(ops)
    # Achieved bandwidth is ~100x below the interconnect ceiling, so on a linear axis
    # the measured bars are not visible at all.
    ax.set_yscale("log")
    ax.set_ylim(min(achieved) * 0.3, peak_gen4 * 3.0)
    ax.legend(loc="upper right")
    fig.savefig(outdir / "fig5_pcie_roofline.png")
    plt.close(fig)
    print("Saved fig5_pcie_roofline.png")


# -----------------------------------------------------------------------------
# 6. P99 Tail Latency: CPU (per-find µs) vs GPU Chained vs GPU Slab (batch → µs)
# -----------------------------------------------------------------------------
def plot_tail_latency(data: dict, outdir: Path) -> None:
    tl = data.get("tail_latency") or {}
    cpu = tl.get("cpu") or {}
    gpu = tl.get("gpu") or {}
    gpu_slab = tl.get("gpu_slab") or {}
    # All three are batch latencies over the same batch size, so they compare directly
    # once converted to µs. Plotting the CPU's per-find latency here instead made the
    # CPU look ~3 orders of magnitude faster than it is.
    def ms_to_us(ms):
        return (float(ms) * 1000.0) if ms is not None else None
    cpu_vals = [ms_to_us(cpu.get("p50_ms")), ms_to_us(cpu.get("p90_ms")), ms_to_us(cpu.get("p99_ms"))]
    gpu_vals = [ms_to_us(gpu.get("p50_ms")), ms_to_us(gpu.get("p90_ms")), ms_to_us(gpu.get("p99_ms"))]
    slab_vals = [ms_to_us(gpu_slab.get("p50_ms")), ms_to_us(gpu_slab.get("p90_ms")), ms_to_us(gpu_slab.get("p99_ms"))]
    if not any(cpu_vals) and not any(gpu_vals) and not any(slab_vals):
        print("Skipping fig6 (no tail_latency data)")
        return
    percentiles = ["P50", "P90", "P99"]
    x = np.arange(len(percentiles))
    width = 0.26
    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    cpu_vals = [v if v is not None else 0 for v in cpu_vals]
    gpu_vals = [v if v is not None else 0 for v in gpu_vals]
    slab_vals = [v if v is not None else 0 for v in slab_vals]
    if any(cpu_vals):
        ax.bar(x - width, cpu_vals, width, label="CPU (std::unordered_map)", color="C0")
    if any(gpu_vals):
        ax.bar(x, gpu_vals, width, label="GPU (Chained)", color="C1")
    if any(slab_vals):
        ax.bar(x + width, slab_vals, width, label="GPU (Slab)", color="C2")
    ax.set_ylabel("Batch latency (µs)")
    ax.set_xlabel("Percentile")
    ax.set_title("Tail latency per 4096-lookup batch (P50 / P90 / P99)")
    ax.set_xticks(x)
    ax.set_xticklabels(percentiles)
    ax.legend()
    ax.set_ylim(0, max(cpu_vals + gpu_vals + slab_vals) * 1.15 if (cpu_vals + gpu_vals + slab_vals) else 1)
    fig.savefig(outdir / "fig6_tail_latency_p99.png", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print("Saved fig6_tail_latency_p99.png")


# -----------------------------------------------------------------------------
# 7. Occupancy vs. Throughput (block size sweep)
# -----------------------------------------------------------------------------
def plot_occupancy_throughput(data: dict, outdir: Path) -> None:
    occ = data.get("occupancy") or {}
    block_sizes = occ.get("block_sizes") or []
    throughput_mops = occ.get("throughput_mops") or []
    occupancy = occ.get("occupancy") or []
    if not block_sizes or not throughput_mops:
        print("Skipping fig7 (no occupancy data)")
        return
    fig, ax1 = plt.subplots(figsize=(FIGW, FIGH))
    ax1.set_xlabel("Block size (threads)")
    ax1.set_xticks(block_sizes)
    ax1.set_xticklabels(block_sizes)
    color1 = "C0"
    ax1.plot(block_sizes, throughput_mops, "o-", color=color1, label="Throughput (Mops/s)")
    ax1.set_ylabel("Throughput (Mops/s)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, max(throughput_mops) * 1.15 if throughput_mops else 1)
    if occupancy:
        ax2 = ax1.twinx()
        color2 = "C1"
        ax2.plot(block_sizes, occupancy, "s--", color=color2, label="Occupancy")
        ax2.set_ylabel("Occupancy", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, 1.05)
    ax1.legend(loc="upper left")
    ax1.set_title("Occupancy vs. throughput (block size 32–1024)")
    fig.tight_layout()
    fig.savefig(outdir / "fig7_occupancy_throughput.png", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print("Saved fig7_occupancy_throughput.png")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark figures for GPU hash map portfolio.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to benchmark_data.json (default: same dir as script; ignored if --run-benchmarks)")
    parser.add_argument("--out", type=str, default="figures",
                        help="Output directory for PNGs (default: figures)")
    parser.add_argument("--run-benchmarks", action="store_true",
                        help="Run benchmarks from build dir, parse stdout, then plot")
    parser.add_argument("--build-dir", type=str, default=None,
                        help="Build directory (e.g. ../build); default: ../build when run from scripts/")
    parser.add_argument("--save-json", type=str, default=None,
                        help="After --run-benchmarks, save merged data to this path")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    outdir = Path(args.out)
    ensure_outdir(outdir)

    if args.run_benchmarks:
        build_dir = Path(args.build_dir) if args.build_dir else (script_dir / ".." / "build")
        build_dir = build_dir.resolve()
        data = run_and_parse_all(build_dir)
        if args.save_json:
            with open(args.save_json, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved merged data to {args.save_json}")
    else:
        data_path = Path(args.data) if args.data else (script_dir / "benchmark_data.json")
        if not data_path.is_file():
            print(f"Data file not found: {data_path}", file=sys.stderr)
            print("Run with --run-benchmarks to measure, or pass --data <file>.", file=sys.stderr)
            sys.exit(1)
        data = load_data(str(data_path))

    plot_interconnect_crossover(data, outdir)
    plot_warp_aggregation(data, outdir)
    plot_heterogeneous_speedup(data, outdir)
    plot_timings_and_speedup(data, outdir)
    plot_timings_by_approach(data, outdir)
    plot_load_factor_throughput(data, outdir)
    plot_roofline(data, outdir)
    plot_tail_latency(data, outdir)
    plot_occupancy_throughput(data, outdir)
    print(f"Figures written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
