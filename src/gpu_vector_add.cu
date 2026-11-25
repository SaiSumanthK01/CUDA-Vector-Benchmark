// gpu_vector_add.cu - CUDA kernel, device memory management and GPU timing.
#include "benchmark.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

// Turn any CUDA runtime error into a C++ exception with file/line context.
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::ostringstream os_;                                               \
            os_ << "CUDA error at " << __FILE__ << ":" << __LINE__ << " ("        \
                << #call << "): " << cudaGetErrorString(err_);                    \
            throw std::runtime_error(os_.str());                                  \
        }                                                                         \
    } while (0)

namespace {

// RAII wrapper: device memory is freed even when an exception is thrown.
struct DeviceBuffer {
    float* ptr = nullptr;
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// RAII wrapper for CUDA events.
struct Event {
    cudaEvent_t e = nullptr;
    Event() { CUDA_CHECK(cudaEventCreate(&e)); }
    ~Event() { if (e) cudaEventDestroy(e); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

// One thread computes one output element.
// Global index = blockIdx.x * blockDim.x + threadIdx.x.
__global__ void vector_add_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {  // the last block may have threads past the end of the array
        c[i] = a[i] + b[i];
    }
}

double median(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

std::string gpu_device_name(int device) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) throw std::runtime_error("No CUDA-capable device found");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.name;
}

GpuTiming gpu_vector_add(const float* a, const float* b, float* c,
                         size_t n, int block_size, int iters) {
    if (n == 0 || block_size <= 0 || block_size > 1024 || iters <= 0)
        throw std::invalid_argument("gpu_vector_add: invalid arguments");

    const size_t bytes = n * sizeof(float);
    const size_t grid = (n + block_size - 1) / block_size;  // round up
    if (grid > 2147483647ULL) throw std::invalid_argument("grid too large");

    // Allocation is done once and is NOT part of the timed region.
    DeviceBuffer d_a(n), d_b(n), d_c(n);
    Event t_start, k_start, k_stop, t_stop;

    auto run_once = [&]() {
        CUDA_CHECK(cudaEventRecord(t_start.e));
        CUDA_CHECK(cudaMemcpy(d_a.ptr, a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.ptr, b, bytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(k_start.e));
        vector_add_kernel<<<static_cast<unsigned>(grid), block_size>>>(d_a.ptr, d_b.ptr, d_c.ptr, n);
        CUDA_CHECK(cudaGetLastError());  // catches launch-configuration errors
        CUDA_CHECK(cudaEventRecord(k_stop.e));

        CUDA_CHECK(cudaMemcpy(c, d_c.ptr, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(t_stop.e));
        CUDA_CHECK(cudaEventSynchronize(t_stop.e));  // also surfaces async kernel errors
    };

    run_once();  // warm-up (context creation, clocks, caches)

    std::vector<float> kernel_samples, total_samples;
    for (int i = 0; i < iters; ++i) {
        run_once();
        float k = 0.f, t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&k, k_start.e, k_stop.e));
        CUDA_CHECK(cudaEventElapsedTime(&t, t_start.e, t_stop.e));
        kernel_samples.push_back(k);
        total_samples.push_back(t);
    }

    GpuTiming out;
    out.kernel_ms = median(kernel_samples);
    out.total_ms = median(total_samples);
    return out;
}
