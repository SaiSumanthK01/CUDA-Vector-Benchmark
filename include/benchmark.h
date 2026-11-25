// benchmark.h - shared declarations for the CUDA vector benchmark.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

// Timing returned by the GPU path (all values are medians over N iterations, in ms).
struct GpuTiming {
    double kernel_ms = 0.0;  // kernel only (CUDA events around the launch)
    double total_ms  = 0.0;  // H2D copy + kernel + D2H copy (CUDA events)
};

// One row of the CSV report.
struct BenchResult {
    std::string gpu_name;
    size_t vector_size = 0;
    int block_size = 0;
    double cpu_ms = 0.0;
    double gpu_kernel_ms = 0.0;
    double gpu_total_ms = 0.0;
    double speedup = 0.0;         // cpu_ms / gpu_total_ms (end-to-end)
    double bandwidth_gbps = 0.0;  // 3 * N * sizeof(float) / kernel time
    double max_error = 0.0;
    bool passed = false;
};

// ---- CPU (src/cpu_vector_add.cpp) ----
// Plain sequential loop: c[i] = a[i] + b[i].
void cpu_vector_add(const float* a, const float* b, float* c, size_t n);
// Runs the loop `iters` times (after one warm-up) and returns the median time in ms.
double cpu_vector_add_timed(const float* a, const float* b, float* c, size_t n, int iters);

// ---- GPU (src/gpu_vector_add.cu) ----
// Runs vector addition on the GPU. Throws std::runtime_error on any CUDA failure
// (device memory is released automatically via RAII before the exception propagates).
GpuTiming gpu_vector_add(const float* a, const float* b, float* c,
                         size_t n, int block_size, int iters);
// Name of the CUDA device in use. Throws if no device is available.
std::string gpu_device_name(int device = 0);
