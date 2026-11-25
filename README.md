# CUDA Vector Benchmark

A CPU vs GPU benchmark for vector addition written in C++ and CUDA. It sweeps vector sizes and thread-block sizes, times the CPU loop, the CUDA kernel and the full GPU round trip, validates the GPU output against the CPU, writes a CSV and plots the results.

See **[results/RESULTS.md](results/RESULTS.md)** for the measured results and charts.

## Problem statement

GPUs are widely described as faster than CPUs for data-parallel work, but the real gain depends on problem size, memory traffic and host-device transfer cost. Vector addition is a minimal, memory-bound workload that makes those effects easy to see and measure.

## Objectives

- Implement a correct CUDA kernel with explicit thread indexing and a sequential CPU baseline.
- Separate **kernel time** from **end-to-end time** (including PCIe transfers).
- Study the effect of block size (32 to 1024) and vector size (10^4 to 10^7).
- Validate every result and fail loudly on mismatch or CUDA errors.
- Produce reproducible, machine-readable output (CSV) and charts.

## Technologies

C++17, CUDA C++ (nvcc), CMake, `std::chrono`, CUDA events, Python 3 with pandas and matplotlib, Git/GitHub.

## Project structure

```
cuda-vector-benchmark/
├── src/
│   ├── cpu_vector_add.cpp     # sequential baseline + chrono timing
│   ├── gpu_vector_add.cu      # kernel, RAII memory, CUDA-event timing, error checks
│   └── main.cu                # CLI, sweep, validation, console + CSV output
├── include/benchmark.h
├── results/benchmark_results.csv   # header only until you run the benchmark
├── visualization/plot_results.py
├── docs/project_architecture.md
├── tests/validation.md
├── CMakeLists.txt  requirements.txt  .gitignore  LICENSE  README.md
```

See [docs/project_architecture.md](docs/project_architecture.md) for the design.

## Requirements

- NVIDIA GPU with a recent driver, CUDA Toolkit 11.0+ (`nvcc --version`), CMake 3.18+, a C++17 host compiler.
- Python 3.8+ for plotting.

No local GPU? Use a free Google Colab GPU runtime (`Runtime > Change runtime type > T4 GPU`), `git clone` the repo, and run the same commands below.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

The build defaults to Release. On CMake older than 3.24, set your GPU architecture, e.g. `-DCMAKE_CUDA_ARCHITECTURES=75` (T4) or `86` (RTX 30-series).

## Run (one command)

```bash
bash run_all.sh
```

This builds, benchmarks, plots, and writes `results/RESULTS.md`. Or step by step:

```bash
./build/cuda_vector_benchmark                       # full sweep -> results/benchmark_results.csv
./build/cuda_vector_benchmark --sizes 1000000 --blocks 128,256 --iters 50
./build/cuda_vector_benchmark --help

pip install -r requirements.txt
python visualization/plot_results.py                # writes 4 PNGs to visualization/
```

Options: `--sizes`, `--blocks` (max 1024), `--iters`, `--output`.

## Methodology

- Inputs: random floats in [-1, 1], fixed seed, identical for CPU and GPU.
- One untimed warm-up, then the **median** of N timed iterations (default 20).
- CPU: `std::chrono::steady_clock` around the loop.
- GPU: CUDA events. `gpu_kernel_ms` covers the launch only. `gpu_total_ms` covers H2D copies + kernel + D2H copy. `cudaMalloc`/`cudaFree` are excluded.
- Memory is pageable (not pinned) host memory; pinned memory would speed up transfers.

## Metrics

| Metric | Definition |
| --- | --- |
| `speedup` | `cpu_time_ms / gpu_total_ms` (end-to-end) |
| `bandwidth_gbps` | `3 * N * 4 bytes / kernel time`, i.e. two reads and one write |
| `max_error` | max absolute difference between CPU and GPU output |
| `validation` | PASS if `max_error <= 1e-6` |

CSV columns: `gpu_name, vector_size, block_size, cpu_time_ms, gpu_kernel_ms, gpu_total_ms, speedup, bandwidth_gbps, max_error, validation`.

## Console output format

```
========================================
       CUDA VECTOR BENCHMARK
========================================

GPU DEVICE: <detected name>

VECTOR SIZE: <n>
BLOCK SIZE: <b>

CPU EXECUTION TIME: <ms>
GPU KERNEL TIME: <ms>
GPU TOTAL TIME: <ms> (incl. transfers)

SPEEDUP (CPU / GPU total): <x>
MEMORY BANDWIDTH (kernel): <GB/s>

MAXIMUM ERROR: <value>
VALIDATION: PASS
```

## Testing

See [tests/validation.md](tests/validation.md). Every run self-validates; the exit code is non-zero on failure.

## Results

See [results/RESULTS.md](results/RESULTS.md) (table, findings and charts) and `results/benchmark_results.csv` (raw data).

## Limitations and future work

- Single operation (vector add) that is memory-bound; not representative of compute-bound kernels.
- Pageable host memory; try pinned memory, `cudaMemcpyAsync` and streams to overlap copies and compute.
- CPU baseline is single-threaded; compare against OpenMP and `-O3 -march=native`.
- Add unified memory, vectorised loads (`float4`), grid-stride loops, and Nsight Compute profiling.
- Compare against cuBLAS/Thrust and add WebGPU/GLSL implementations.
- Add CI (build-only on GitHub Actions) and a regression threshold on bandwidth.

## License

MIT
