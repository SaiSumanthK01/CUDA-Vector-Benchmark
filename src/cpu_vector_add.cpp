// cpu_vector_add.cpp - sequential CPU baseline.
#include "benchmark.h"

#include <algorithm>
#include <chrono>
#include <vector>

void cpu_vector_add(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

double cpu_vector_add_timed(const float* a, const float* b, float* c, size_t n, int iters) {
    using clock = std::chrono::steady_clock;

    cpu_vector_add(a, b, c, n);  // warm-up: touches pages and fills caches

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clock::now();
        cpu_vector_add(a, b, c, n);
        auto t1 = clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];  // median is robust to outliers
}
