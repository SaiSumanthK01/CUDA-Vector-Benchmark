#!/usr/bin/env python3
"""Build results/RESULTS.md (table + charts + plain-language findings) from the CSV."""
import sys
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent.parent
csv = root / "results" / "benchmark_results.csv"
if not csv.exists():
    sys.exit("Run the benchmark first.")
df = pd.read_csv(csv)
if df.empty:
    sys.exit("CSV is empty. Run the benchmark first.")

gpu = df["gpu_name"].iloc[0]
best = df.loc[df.groupby("vector_size")["gpu_kernel_ms"].idxmin()].sort_values("vector_size")

lines = [f"# Benchmark Results\n", f"**GPU:** {gpu}  ", f"**Validation:** {'ALL PASS' if (df['validation'] == 'PASS').all() else 'FAILURES PRESENT'}\n",
         "## Best block size per vector size\n",
         "| Vector size | Best block | CPU (ms) | GPU kernel (ms) | GPU total (ms) | Speedup | Bandwidth (GB/s) |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
for _, r in best.iterrows():
    lines.append(f"| {int(r.vector_size):,} | {int(r.block_size)} | {r.cpu_time_ms:.4f} | {r.gpu_kernel_ms:.4f} | "
                 f"{r.gpu_total_ms:.4f} | {r.speedup:.2f}x | {r.bandwidth_gbps:.1f} |")

lines += ["\n## Findings (computed from the data)\n"]
for _, r in best.iterrows():
    kernel_x = r.cpu_time_ms / r.gpu_kernel_ms
    verdict = "GPU faster end-to-end" if r.speedup > 1 else "CPU faster end-to-end (transfer cost dominates)"
    lines.append(f"- **{int(r.vector_size):,} elements:** kernel alone is {kernel_x:.1f}x vs CPU; "
                 f"with transfers it is {r.speedup:.2f}x. {verdict}.")

lines += ["\n## Charts\n"]
for title, f in [("CPU vs GPU time", "cpu_vs_gpu_time.png"), ("Speedup by size", "speedup_by_size.png"),
                 ("Block size performance", "block_size_performance.png"), ("Memory bandwidth", "memory_bandwidth.png")]:
    lines.append(f"### {title}\n![{title}](../visualization/{f})\n")

(root / "results" / "RESULTS.md").write_text("\n".join(lines))
print("Wrote results/RESULTS.md")
