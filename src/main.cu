// main.cu - benchmark driver: parses args, runs CPU/GPU, validates, prints, writes CSV.
#include "benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kTolerance = 1e-6;  // float add is exactly rounded, so expect 0

struct Options {
    std::vector<size_t> sizes{10000, 100000, 1000000, 10000000};
    std::vector<int> blocks{32, 64, 128, 256, 512, 1024};
    int iters = 20;
    std::string output = "results/benchmark_results.csv";
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --sizes  N1,N2,...   vector sizes        (default 10000,100000,1000000,10000000)\n"
              << "  --blocks B1,B2,...   threads per block   (default 32,64,128,256,512,1024; max 1024)\n"
              << "  --iters  K           timed iterations    (default 20)\n"
              << "  --output FILE        CSV output path     (default results/benchmark_results.csv)\n"
              << "  --help               show this message\n";
}

template <class T>
std::vector<T> parse_list(const std::string& s, const char* what) {
    std::vector<T> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t pos = 0;
        long long v = 0;
        try { v = std::stoll(tok, &pos); } catch (...) { pos = 0; }
        if (pos == 0 || pos != tok.size() || v <= 0)
            throw std::invalid_argument(std::string("invalid value for ") + what + ": '" + tok + "'");
        out.push_back(static_cast<T>(v));
    }
    if (out.empty()) throw std::invalid_argument(std::string("empty list for ") + what);
    return out;
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--sizes") o.sizes = parse_list<size_t>(next(), "--sizes");
        else if (a == "--blocks") {
            o.blocks = parse_list<int>(next(), "--blocks");
            for (int b : o.blocks)
                if (b > 1024) throw std::invalid_argument("block size must be <= 1024");
        }
        else if (a == "--iters") o.iters = parse_list<int>(next(), "--iters")[0];
        else if (a == "--output") o.output = next();
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); std::exit(0); }
        else throw std::invalid_argument("unknown option: " + a);
    }
    return o;
}

double max_abs_error(const std::vector<float>& x, const std::vector<float>& y) {
    double m = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::fabs(static_cast<double>(x[i]) - static_cast<double>(y[i])));
    return m;
}

void print_result(const BenchResult& r) {
    std::cout << "\nVECTOR SIZE: " << r.vector_size
              << "\nBLOCK SIZE: " << r.block_size << "\n\n"
              << std::fixed << std::setprecision(4)
              << "CPU EXECUTION TIME: " << r.cpu_ms << " ms\n"
              << "GPU KERNEL TIME: " << r.gpu_kernel_ms << " ms\n"
              << "GPU TOTAL TIME: " << r.gpu_total_ms << " ms (incl. transfers)\n\n"
              << std::setprecision(2)
              << "SPEEDUP (CPU / GPU total): " << r.speedup << "x\n"
              << "MEMORY BANDWIDTH (kernel): " << r.bandwidth_gbps << " GB/s\n\n"
              << std::scientific << std::setprecision(3)
              << "MAXIMUM ERROR: " << r.max_error << "\n"
              << "VALIDATION: " << (r.passed ? "PASS" : "FAIL") << "\n"
              << "----------------------------------------\n";
}

void write_csv(const std::string& path, const std::vector<BenchResult>& rows) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open output file: " + path);
    f << "gpu_name,vector_size,block_size,cpu_time_ms,gpu_kernel_ms,gpu_total_ms,"
         "speedup,bandwidth_gbps,max_error,validation\n";
    f << std::setprecision(9);
    for (const auto& r : rows) {
        f << '"' << r.gpu_name << '"' << ',' << r.vector_size << ',' << r.block_size << ','
          << r.cpu_ms << ',' << r.gpu_kernel_ms << ',' << r.gpu_total_ms << ','
          << r.speedup << ',' << r.bandwidth_gbps << ',' << r.max_error << ','
          << (r.passed ? "PASS" : "FAIL") << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        const std::string gpu = gpu_device_name();

        std::cout << "========================================\n"
                  << "       CUDA VECTOR BENCHMARK\n"
                  << "========================================\n\n"
                  << "GPU DEVICE: " << gpu << "\n";

        std::vector<BenchResult> rows;
        bool all_passed = true;
        std::mt19937 rng(12345);  // fixed seed -> reproducible inputs
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (size_t n : opt.sizes) {
            std::vector<float> a(n), b(n), c_cpu(n), c_gpu(n);
            for (size_t i = 0; i < n; ++i) { a[i] = dist(rng); b[i] = dist(rng); }

            const double cpu_ms = cpu_vector_add_timed(a.data(), b.data(), c_cpu.data(), n, opt.iters);

            for (int block : opt.blocks) {
                std::fill(c_gpu.begin(), c_gpu.end(), 0.0f);  // make sure stale data can't pass
                GpuTiming t = gpu_vector_add(a.data(), b.data(), c_gpu.data(), n, block, opt.iters);

                BenchResult r;
                r.gpu_name = gpu;
                r.vector_size = n;
                r.block_size = block;
                r.cpu_ms = cpu_ms;
                r.gpu_kernel_ms = t.kernel_ms;
                r.gpu_total_ms = t.total_ms;
                r.speedup = cpu_ms / t.total_ms;
                // Kernel reads a and b and writes c: 3 * N * 4 bytes of traffic.
                r.bandwidth_gbps = (3.0 * n * sizeof(float)) / (t.kernel_ms * 1e-3) / 1e9;
                r.max_error = max_abs_error(c_cpu, c_gpu);
                r.passed = r.max_error <= kTolerance;
                all_passed = all_passed && r.passed;

                print_result(r);
                rows.push_back(r);
            }
        }

        write_csv(opt.output, rows);
        std::cout << "\nResults written to " << opt.output << "\n"
                  << "========================================\n"
                  << "BENCHMARK COMPLETED (" << (all_passed ? "ALL PASS" : "FAILURES DETECTED") << ")\n"
                  << "========================================\n";
        return all_passed ? 0 : 2;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Input error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
