#!/usr/bin/env python3
"""Generate benchmark charts from results/benchmark_results.csv.

Usage: python visualization/plot_results.py [--csv PATH] [--out DIR]
Charts 1, 2 and 4 use, for each vector size, the block size with the lowest kernel time.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd

REQUIRED = ["gpu_name", "vector_size", "block_size", "cpu_time_ms", "gpu_kernel_ms",
            "gpu_total_ms", "speedup", "bandwidth_gbps", "max_error", "validation"]


def load(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}. Run the benchmark on a CUDA machine first.")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing columns: {missing}")
    if df.empty:
        sys.exit("CSV has no data rows. Run the benchmark first.")
    return df


def best_per_size(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("vector_size")["gpu_kernel_ms"].idxmin()
    return df.loc[idx].sort_values("vector_size").reset_index(drop=True)


def chart_time(best, gpu, out):
    x = range(len(best))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w for i in x], best["cpu_time_ms"], w, label="CPU")
    ax.bar(list(x), best["gpu_total_ms"], w, label="GPU total (with transfers)")
    ax.bar([i + w for i in x], best["gpu_kernel_ms"], w, label="GPU kernel only")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{int(s):,}" for s in best["vector_size"]])
    ax.set_yscale("log")
    ax.set_xlabel("Vector size (elements)")
    ax.set_ylabel("Time (ms, log scale)")
    ax.set_title(f"CPU vs GPU execution time - {gpu}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "cpu_vs_gpu_time.png", dpi=150)
    plt.close(fig)


def chart_speedup(best, gpu, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(best["vector_size"], best["speedup"], marker="o")
    ax.axhline(1.0, color="gray", linestyle="--", label="break-even (1x)")
    ax.set_xscale("log")
    ax.set_xlabel("Vector size (elements)")
    ax.set_ylabel("Speedup (CPU time / GPU total time)")
    ax.set_title(f"End-to-end speedup - {gpu}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "speedup_by_size.png", dpi=150)
    plt.close(fig)


def chart_blocks(df, gpu, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for size, g in df.groupby("vector_size"):
        g = g.sort_values("block_size")
        ax.plot(g["block_size"], g["gpu_kernel_ms"], marker="o", label=f"{int(size):,}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Block size (threads per block)")
    ax.set_ylabel("GPU kernel time (ms, log scale)")
    ax.set_title(f"Kernel time vs block size - {gpu}")
    ax.legend(title="Vector size")
    fig.tight_layout()
    fig.savefig(out / "block_size_performance.png", dpi=150)
    plt.close(fig)


def chart_bandwidth(best, gpu, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([f"{int(s):,}" for s in best["vector_size"]], best["bandwidth_gbps"])
    ax.set_xlabel("Vector size (elements)")
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title(f"Estimated kernel memory bandwidth - {gpu}")
    fig.tight_layout()
    fig.savefig(out / "memory_bandwidth.png", dpi=150)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=here.parent / "results" / "benchmark_results.csv")
    ap.add_argument("--out", type=Path, default=here)
    args = ap.parse_args()

    df = load(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)
    gpu = df["gpu_name"].iloc[0]
    best = best_per_size(df)

    chart_time(best, gpu, args.out)
    chart_speedup(best, gpu, args.out)
    chart_blocks(df, gpu, args.out)
    chart_bandwidth(best, gpu, args.out)
    print(f"Saved 4 charts to {args.out}")


if __name__ == "__main__":
    main()
