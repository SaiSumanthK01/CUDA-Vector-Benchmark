# Architecture

```
main.cu  ──►  parse args ─► for each vector size:
                              generate inputs (fixed seed)
                              cpu_vector_add_timed()      [cpu_vector_add.cpp, chrono]
                              for each block size:
                                gpu_vector_add()          [gpu_vector_add.cu, CUDA events]
                                max_abs_error(cpu, gpu) ─► PASS / FAIL
                                print + store row
                           ─► write CSV
```

## Modules

| File | Responsibility |
| --- | --- |
| `include/benchmark.h` | Shared structs and function declarations. No CUDA headers, so `.cpp` files stay host-only. |
| `src/cpu_vector_add.cpp` | Sequential baseline and `std::chrono::steady_clock` timing (median of N runs). |
| `src/gpu_vector_add.cu` | Kernel, RAII device buffers/events, `CUDA_CHECK` error handling, CUDA-event timing. |
| `src/main.cu` | CLI parsing, sweep over sizes/blocks, validation, console output, CSV writer. |
| `visualization/plot_results.py` | Reads the CSV and writes four PNG charts. |

## Design decisions

- **Kernel:** one thread per element, `i = blockIdx.x * blockDim.x + threadIdx.x`, with a bounds check for the final partial block. Grid size is `ceil(n / block)`.
- **Two GPU timings:** `gpu_kernel_ms` wraps only the launch; `gpu_total_ms` wraps H2D copies + kernel + D2H copy. `cudaMalloc`/`cudaFree` are excluded and done once per configuration (documented, so the numbers are not misread).
- **Warm-up + median:** one untimed run to absorb context creation and clock ramp-up, then the median of N timed runs (default 20).
- **Speedup** = `cpu_time_ms / gpu_total_ms`. It is deliberately end-to-end; for a memory-bound op like vector add, PCIe transfer time usually dominates.
- **Bandwidth** = `3 * N * 4 bytes / kernel_time` (read a, read b, write c).
- **Error handling:** every CUDA call goes through `CUDA_CHECK`, which throws; RAII wrappers free device memory and destroy events during unwinding. `main` catches and returns a non-zero exit code.
- **Validation:** max absolute difference between CPU and GPU outputs against a 1e-6 tolerance. A single IEEE float add is correctly rounded on both, so 0 is expected.
