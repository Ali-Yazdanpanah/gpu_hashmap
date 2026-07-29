#!/usr/bin/env python3
"""Check that every measured number in README.md agrees with benchmark_data.json.

The README is the project's honesty claim, so it must not be possible to edit a
figure by hand and leave it disagreeing with the data it came from. Two kinds of
claim are enforced:

  Scalars   A marker names one or more keys and the value each should have:
                | Slab vs CPU | **124.51x** | <!-- generated: slab_vs_cpu=124.51 -->
            The script recomputes each key from the JSON, checks it against the
            marker, and checks that the formatted value actually appears in the
            visible text of that line. Both directions are covered: the marker
            cannot drift from the data, and the prose cannot drift from the marker.

  Tables    A fenced block is regenerated from the JSON and compared verbatim:
                <!-- generated:table:vs_cpu -->
                ...markdown table...
                <!-- /generated:table -->

Usage:
    python scripts/check_results.py                  # verify, non-zero on mismatch
    python scripts/check_results.py --write          # rewrite markers and tables
    python scripts/check_results.py --list           # show every key and its value

Re-running the benchmarks changes the data, so the intended workflow is to
regenerate benchmark_data.json, run with --write, and review the resulting diff.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "scripts" / "benchmark_data.json"
DEFAULT_README = REPO / "README.md"

SCALAR_MARKER = re.compile(r"<!--\s*generated:\s*(?P<body>[^>]*?)\s*-->")
TABLE_OPEN = re.compile(r"<!--\s*generated:table:(?P<name>[A-Za-z0-9_]+)\s*-->")
TABLE_CLOSE = re.compile(r"<!--\s*/generated:table\s*-->")


class DataError(Exception):
    pass


def _approach_index(data: dict, name: str) -> int:
    approaches = data["heterogeneous"]["approaches"]
    try:
        return approaches.index(name)
    except ValueError as exc:
        raise DataError(f"approach {name!r} not in {approaches}") from exc


def _het(data: dict, field: str, approach: str) -> float:
    return data["heterogeneous"][field][_approach_index(data, approach)]


def _placement(data: dict, scheme: str, placement: str, field: str) -> float:
    """Look up the Phase 3 placement matrix, which may be absent in older data."""
    matrix = data.get("placement_matrix")
    if not matrix:
        raise DataError("placement_matrix missing (run benchmark_placement)")
    for row in matrix["rows"]:
        if row["scheme"] == scheme and row["placement"] == placement:
            return row[field]
    raise DataError(f"no placement row for {scheme}/{placement}")


# key -> (callable(data) -> float, format spec)
SCALARS: dict[str, tuple, ] = {
    # --- Section 1 headline table ---
    "slab_vs_cpu": (lambda d: _het(d, "speedup_x", "GPU Slab"), "{:.2f}"),
    "slab_drop_pct": (lambda d: d["heterogeneous"]["dropped_pct"]["GPU Slab"], "{:.3f}"),
    "slab_drop_count": (lambda d: d["heterogeneous"]["dropped_inserts"]["GPU Slab"], "{:d}"),
    "inserts_requested": (lambda d: d["heterogeneous"]["inserts_requested"], "{:d}"),
    "chained_vs_cpu": (lambda d: _het(d, "speedup_x", "GPU Chained"), "{:.2f}"),
    "warpagg_vs_cpu": (lambda d: _het(d, "speedup_x", "GPU Warp-agg"), "{:.2f}"),
    "hybrid_vs_cpu": (lambda d: _het(d, "speedup_x", "Hybrid (CPU+GPU lup)"), "{:.2f}"),
    "zipf_alpha_lo": (lambda d: d["warp_aggregation"]["alphas"][0], "{:.1f}"),
    "zipf_alpha_hi": (lambda d: d["warp_aggregation"]["alphas"][-1], "{:.1f}"),
    "warpagg_speedup_lo": (
        lambda d: d["warp_aggregation"]["standard_insert_ms"][0]
        / d["warp_aggregation"]["warp_agg_insert_ms"][0],
        "{:.2f}",
    ),
    "warpagg_speedup_hi": (
        lambda d: d["warp_aggregation"]["standard_insert_ms"][-1]
        / d["warp_aggregation"]["warp_agg_insert_ms"][-1],
        "{:.2f}",
    ),
    "sparsity_zerocopy_ms": (lambda d: d["interconnect"]["sparsity_zerocopy_ms"], "{:.3f}"),
    "sparsity_live_ms": (lambda d: d["interconnect"]["sparsity_standard_ms"], "{:.3f}"),
    "sparsity_naive_ms": (
        lambda d: d["interconnect"]["sparsity_naive_full_migration_ms"],
        "{:.3f}",
    ),
    "sparsity_speedup_vs_naive": (
        lambda d: d["interconnect"]["sparsity_naive_full_migration_ms"]
        / d["interconnect"]["sparsity_zerocopy_ms"],
        "{:.0f}",
    ),
    "sparsity_speedup_vs_live": (
        lambda d: d["interconnect"]["sparsity_standard_ms"]
        / d["interconnect"]["sparsity_zerocopy_ms"],
        "{:.0f}",
    ),
    # --- Bandwidth / roofline ---
    "insert_gbps": (lambda d: d["roofline"]["achieved_gbps"][0], "{:.2f}"),
    "lookup_gbps": (lambda d: d["roofline"]["achieved_gbps"][1], "{:.2f}"),
    "peak_gen3_gbps": (lambda d: d["roofline"]["peak_gen3_gbps"], "{:d}"),
    "insert_pct_gen3": (
        lambda d: 100.0 * d["roofline"]["achieved_gbps"][0] / d["roofline"]["peak_gen3_gbps"],
        "{:.1f}",
    ),
    "lookup_pct_gen3": (
        lambda d: 100.0 * d["roofline"]["achieved_gbps"][1] / d["roofline"]["peak_gen3_gbps"],
        "{:.1f}",
    ),
    # --- Probe depth ---
    "probe_depth_success": (lambda d: d["probe_depth_success"], "{:.3f}"),
    "probe_depth_unsuccessful": (lambda d: d["probe_depth_unsuccessful"], "{:.3f}"),
    "probe_depth_lf_lo": (lambda d: d["load_factor"]["avg_probe_depth"][0], "{:.3f}"),
    "probe_depth_lf_hi": (lambda d: d["load_factor"]["avg_probe_depth"][-1], "{:.3f}"),
    "throughput_decline_pct": (
        lambda d: 100.0
        * (
            1.0
            - d["load_factor"]["throughput_ops_per_sec"][-1]
            / d["load_factor"]["throughput_ops_per_sec"][0]
        ),
        "{:.0f}",
    ),
    # --- Interconnect sweep summary ---
    "sweep_sizes": (lambda d: len(d["interconnect"]["batch_sizes_n"]), "{:d}"),
    "zc_faster_count": (
        lambda d: sum(
            1
            for s, z in zip(
                d["interconnect"]["standard_path_ms"], d["interconnect"]["zerocopy_path_ms"]
            )
            if z <= s
        ),
        "{:d}",
    ),
    # --- Tail latency ---
    "tail_cpu_p50_us": (lambda d: d["tail_latency"]["cpu"]["p50_ms"] * 1000.0, "{:.1f}"),
    "tail_chained_p50_us": (lambda d: d["tail_latency"]["gpu"]["p50_ms"] * 1000.0, "{:.1f}"),
    "tail_slab_p50_us": (lambda d: d["tail_latency"]["gpu_slab"]["p50_ms"] * 1000.0, "{:.1f}"),
    # --- Phase 3: placement vs scheme ---
    "chained_host_total_ms": (
        lambda d: _placement(d, "chained", "mapped-host", "total_ms"),
        "{:.2f}",
    ),
    "chained_device_total_ms": (
        lambda d: _placement(d, "chained", "device", "total_ms"),
        "{:.2f}",
    ),
    "slab_device_total_ms": (lambda d: _placement(d, "slab", "device", "total_ms"), "{:.2f}"),
    "placement_speedup_chained": (
        lambda d: _placement(d, "chained", "mapped-host", "total_ms")
        / _placement(d, "chained", "device", "total_ms"),
        "{:.1f}",
    ),
    "algorithm_speedup_device": (
        lambda d: _placement(d, "chained", "device", "total_ms")
        / _placement(d, "slab", "device", "total_ms"),
        "{:.1f}",
    ),
    # --- Phase 4: heuristic accuracy ---
    "heuristic_correct": (lambda d: d["heuristic_accuracy"]["correct"], "{:d}"),
    "heuristic_total": (lambda d: d["heuristic_accuracy"]["total"], "{:d}"),
    "heuristic_accuracy_pct": (
        lambda d: 100.0 * d["heuristic_accuracy"]["correct"] / d["heuristic_accuracy"]["total"],
        "{:.0f}",
    ),
    # --- Phase 5: hit-rate sweep ---
    "hitrate_lookup_ms_0": (lambda d: d["hit_rate"]["lookup_ms"][0], "{:.2f}"),
    "hitrate_lookup_ms_100": (lambda d: d["hit_rate"]["lookup_ms"][-1], "{:.2f}"),
    "hitrate_miss_over_hit": (
        lambda d: d["hit_rate"]["lookup_ms"][0] / d["hit_rate"]["lookup_ms"][-1],
        "{:.2f}",
    ),
}


def fmt_scalar(key: str, data: dict) -> str:
    fn, spec = SCALARS[key]
    value = fn(data)
    if spec.endswith("d}"):
        value = int(round(float(value)))
    return spec.format(value)


# -----------------------------------------------------------------------------
# Table renderers. Each returns the markdown body (no surrounding markers).
# -----------------------------------------------------------------------------
def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _k_label(n: int) -> str:
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)


def table_interconnect_sweep(d: dict) -> list[str]:
    ic = d["interconnect"]
    out = [
        _row(["Batch (lookups)", "Standard (ms)", "Zero-copy (ms)", "ZC / Std"]),
        _row(["---"] * 4),
    ]
    for n, s, z in zip(ic["batch_sizes_n"], ic["standard_path_ms"], ic["zerocopy_path_ms"]):
        out.append(_row([_k_label(n), f"{s:.3f}", f"{z:.3f}", f"{z / s:.2f}"]))
    return out


def table_sparsity_paths(d: dict) -> list[str]:
    ic = d["interconnect"]
    return [
        _row(["Path", "Time"]),
        _row(["---", "---"]),
        _row(
            [
                "Naive full-capacity migration + lookup",
                f"**{ic['sparsity_naive_full_migration_ms']:.3f} ms**",
            ]
        ),
        _row(
            [
                "Standard, copying only live data + lookup",
                f"**{ic['sparsity_standard_ms']:.3f} ms**",
            ]
        ),
        _row(["Zero-copy, no table migration", f"**{ic['sparsity_zerocopy_ms']:.3f} ms**"]),
    ]


def table_zipfian(d: dict) -> list[str]:
    w = d["warp_aggregation"]
    out = [
        _row(["Zipfian α", "Standard insert (ms)", "Warp-aggregated (ms)", "Speedup"]),
        _row(["---"] * 4),
    ]
    for a, s, wa in zip(w["alphas"], w["standard_insert_ms"], w["warp_agg_insert_ms"]):
        out.append(_row([f"{a:.1f}", f"{s:.2f}", f"{wa:.2f}", f"{s / wa:.2f}x"]))
    return out


def table_vs_cpu(d: dict) -> list[str]:
    h = d["heterogeneous"]
    out = [
        _row(["Approach", "Insert (ms)", "Lookup (ms)", "Total (ms)", "vs CPU", "Inserts dropped"]),
        _row(["---"] * 6),
    ]
    for i, name in enumerate(h["approaches"]):
        dropped = h.get("dropped_pct", {}).get(name)
        drop_cell = "n/a" if dropped is None else (f"{dropped:.3f}%" if dropped else "0")
        speed = h["speedup_x"][i]
        speed_cell = "1.00x" if abs(speed - 1.0) < 1e-9 else f"**{speed:.2f}x**"
        out.append(
            _row(
                [
                    name,
                    f"{h['insert_ms'][i]:.2f}",
                    f"{h['lookup_ms'][i]:.2f}",
                    f"{h['total_ms'][i]:.2f}",
                    speed_cell,
                    drop_cell,
                ]
            )
        )
    return out


def table_load_factor(d: dict) -> list[str]:
    lf = d["load_factor"]
    out = [
        _row(["Load factor", "Insert (ms)", "Lookup (ms)", "Throughput (ops/s)", "Avg. probe depth"]),
        _row(["---"] * 5),
    ]
    for pct, ins, lup, thr, pd in zip(
        lf["load_factor_pct"],
        lf["insert_ms"],
        lf["lookup_ms"],
        lf["throughput_ops_per_sec"],
        lf["avg_probe_depth"],
    ):
        out.append(_row([f"{pct}%", f"{ins:.2f}", f"{lup:.2f}", f"{int(thr):,d}", f"{pd:.3f}"]))
    return out


def table_roofline(d: dict) -> list[str]:
    r = d["roofline"]
    gen3 = r["peak_gen3_gbps"]
    out = [
        _row(["Operation", "Achieved", f"vs PCIe Gen3x16 ({gen3} GB/s)", "vs VRAM peak (336 GB/s)"]),
        _row(["---"] * 4),
    ]
    for op, gbps in zip(r["operations"], r["achieved_gbps"]):
        out.append(
            _row(
                [
                    op,
                    f"{gbps:.2f} GB/s",
                    f"{100.0 * gbps / gen3:.1f}%",
                    f"{100.0 * gbps / 336.0:.2f}%",
                ]
            )
        )
    return out


def table_tail_latency(d: dict) -> list[str]:
    tl = d["tail_latency"]
    series = [
        ("CPU (`std::unordered_map`)", "cpu"),
        ("GPU chained", "gpu"),
        ("GPU slab", "gpu_slab"),
    ]
    out = [_row(["", "P50", "P90", "P99", "P99/P50"]), _row(["---"] * 5)]
    for label, key in series:
        s = tl[key]
        p50, p90, p99 = (s["p50_ms"] * 1000.0, s["p90_ms"] * 1000.0, s["p99_ms"] * 1000.0)
        out.append(
            _row(
                [
                    label,
                    f"{p50:.1f} µs",
                    f"{p90:.1f} µs",
                    f"{p99:.1f} µs",
                    f"{p99 / p50:.1f}x",
                ]
            )
        )
    return out


def table_occupancy(d: dict) -> list[str]:
    o = d["occupancy"]
    out = [_row(["Block size", "Occupancy", "Throughput (Mops/s)"]), _row(["---"] * 3)]
    for b, occ, thr in zip(o["block_sizes"], o["occupancy"], o["throughput_mops"]):
        out.append(_row([str(b), f"{occ:.2f}", f"{thr:.2f}"]))
    return out


def table_placement_matrix(d: dict) -> list[str]:
    m = d.get("placement_matrix")
    if not m:
        raise DataError("placement_matrix missing (run benchmark_placement)")
    out = [
        _row(["Scheme", "Table placement", "Insert (ms)", "Lookup (ms)", "Total (ms)", "Dropped"]),
        _row(["---"] * 6),
    ]
    for r in m["rows"]:
        out.append(
            _row(
                [
                    r["scheme"],
                    r["placement"],
                    f"{r['insert_ms']:.2f}",
                    f"{r['lookup_ms']:.2f}",
                    f"**{r['total_ms']:.2f}**",
                    f"{r['dropped_pct']:.3f}%" if r["dropped_pct"] else "0",
                ]
            )
        )
    return out


def table_heuristic_accuracy(d: dict) -> list[str]:
    h = d.get("heuristic_accuracy")
    if not h:
        raise DataError("heuristic_accuracy missing (run benchmark_heuristic)")
    out = [
        _row(["Batch (lookups)", "Standard (ms)", "Zero-copy (ms)", "Faster path", "Heuristic chose", "Correct"]),
        _row(["---"] * 6),
    ]
    for r in h["rows"]:
        out.append(
            _row(
                [
                    _k_label(r["n"]),
                    f"{r['standard_ms']:.3f}",
                    f"{r['zerocopy_ms']:.3f}",
                    r["faster"],
                    r["chosen"],
                    "yes" if r["correct"] else "**no**",
                ]
            )
        )
    return out


def table_hit_rate(d: dict) -> list[str]:
    h = d.get("hit_rate")
    if not h:
        raise DataError("hit_rate missing (run performance_validation_suite)")
    out = [
        _row(["Hit rate", "Lookup (ms)", "Avg. probe depth", "vs 100% hits"]),
        _row(["---"] * 4),
    ]
    base = h["lookup_ms"][-1]
    for pct, ms, pd in zip(h["hit_rate_pct"], h["lookup_ms"], h["avg_probe_depth"]):
        out.append(_row([f"{pct}%", f"{ms:.2f}", f"{pd:.3f}", f"{ms / base:.2f}x"]))
    return out


TABLES = {
    "interconnect_sweep": table_interconnect_sweep,
    "sparsity_paths": table_sparsity_paths,
    "zipfian": table_zipfian,
    "vs_cpu": table_vs_cpu,
    "load_factor": table_load_factor,
    "roofline": table_roofline,
    "tail_latency": table_tail_latency,
    "occupancy": table_occupancy,
    "placement_matrix": table_placement_matrix,
    "heuristic_accuracy": table_heuristic_accuracy,
    "hit_rate": table_hit_rate,
}


# -----------------------------------------------------------------------------
# README parsing / verification
# -----------------------------------------------------------------------------
def parse_marker_body(body: str) -> list[tuple[str, str]]:
    """'a=1, b=2' -> [('a','1'), ('b','2')]. Table markers return []."""
    if body.startswith("table:") or body.startswith("/generated"):
        return []
    pairs = []
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"marker entry {part!r} is not key=value")
        k, v = part.split("=", 1)
        pairs.append((k.strip(), v.strip()))
    return pairs


def value_visible(text: str, formatted: str) -> bool:
    """True when `formatted` appears in `text` as a standalone number."""
    return re.search(rf"(?<![\d.]){re.escape(formatted)}(?![\d])", text) is not None


def check_readme(readme: str, data: dict) -> tuple[list[str], str]:
    """Return (errors, rewritten_readme)."""
    errors: list[str] = []
    lines = readme.splitlines()
    out_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        m_open = TABLE_OPEN.search(line)
        if m_open:
            name = m_open.group("name")
            # Collect the existing block up to the closing marker.
            j = i + 1
            existing: list[str] = []
            while j < len(lines) and not TABLE_CLOSE.search(lines[j]):
                existing.append(lines[j])
                j += 1
            if j >= len(lines):
                errors.append(f"table {name}: opening marker has no closing marker")
                out_lines.append(line)
                i += 1
                continue
            if name not in TABLES:
                errors.append(f"table {name}: no renderer registered")
                rendered = existing
            else:
                try:
                    rendered = TABLES[name](data)
                except DataError as exc:
                    errors.append(f"table {name}: {exc}")
                    rendered = existing
            stripped_existing = [ln for ln in existing if ln.strip()]
            if stripped_existing != rendered:
                errors.append(
                    f"table {name}: README does not match data\n"
                    f"    expected:\n      " + "\n      ".join(rendered) + "\n"
                    f"    found:\n      " + "\n      ".join(stripped_existing)
                )
            out_lines.append(line)
            out_lines.extend(rendered)
            out_lines.append(lines[j])
            i = j + 1
            continue

        if SCALAR_MARKER.search(line):
            # A line may carry several independent markers. Validate each against the
            # visible prose (markers stripped), and rewrite each marker's claimed value.
            visible = SCALAR_MARKER.sub("", line)
            new_line = line
            # Walk matches right-to-left so string indices stay valid while rewriting.
            matches = list(SCALAR_MARKER.finditer(line))
            for m_scalar in reversed(matches):
                body = m_scalar.group("body")
                try:
                    pairs = parse_marker_body(body)
                except ValueError as exc:
                    errors.append(f"{exc} (line {i + 1})")
                    continue
                new_pairs = []
                for key, claimed in pairs:
                    if key not in SCALARS:
                        errors.append(f"line {i + 1}: unknown key {key!r}")
                        new_pairs.append((key, claimed))
                        continue
                    try:
                        actual = fmt_scalar(key, data)
                    except (DataError, KeyError, IndexError, TypeError, ZeroDivisionError) as exc:
                        errors.append(f"line {i + 1}: {key} could not be computed: {exc}")
                        new_pairs.append((key, claimed))
                        continue
                    if claimed != actual:
                        errors.append(
                            f"line {i + 1}: {key} marker says {claimed}, data says {actual}"
                        )
                    if not value_visible(visible, actual):
                        errors.append(
                            f"line {i + 1}: {key}={actual} does not appear in the text of that line"
                        )
                    new_pairs.append((key, actual))
                if new_pairs:
                    rebuilt = (
                        "<!-- generated: "
                        + ", ".join(f"{k}={v}" for k, v in new_pairs)
                        + " -->"
                    )
                    new_line = new_line[: m_scalar.start()] + rebuilt + new_line[m_scalar.end() :]
            out_lines.append(new_line)
            i += 1
            continue

        out_lines.append(line)
        i += 1

    return errors, "\n".join(out_lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--readme", type=Path, default=DEFAULT_README)
    ap.add_argument("--write", action="store_true", help="rewrite markers and tables in place")
    ap.add_argument("--list", action="store_true", help="print every key and its computed value")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"FAIL: {args.data} not found. Run the benchmarks first.", file=sys.stderr)
        return 2
    data = json.loads(args.data.read_text(encoding="utf-8"))

    if args.list:
        for key in sorted(SCALARS):
            try:
                print(f"{key:32s} {fmt_scalar(key, data)}")
            except Exception as exc:  # noqa: BLE001 - reporting, not handling
                print(f"{key:32s} <unavailable: {exc}>")
        return 0

    readme = args.readme.read_text(encoding="utf-8")
    errors, rewritten = check_readme(readme, data)

    if args.write:
        args.readme.write_text(rewritten, encoding="utf-8")
        print(f"Wrote {args.readme} from {args.data}")
        if errors:
            print(f"  ({len(errors)} value(s) updated or still failing; re-run to verify)")
        return 0

    if errors:
        print(f"FAIL: README.md disagrees with {args.data.name} ({len(errors)} problem(s)):\n",
              file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print(
            "\nIf the data is newer than the prose, run:\n"
            "  python scripts/check_results.py --write\n"
            "and review the diff.",
            file=sys.stderr,
        )
        return 1

    n_tables = len(TABLE_OPEN.findall(readme))
    n_scalars = sum(
        len(parse_marker_body(m.group("body"))) for m in SCALAR_MARKER.finditer(readme)
    )
    print(f"OK: {n_scalars} scalar(s) and {n_tables} table(s) in README.md match {args.data.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
