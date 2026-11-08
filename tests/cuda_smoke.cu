#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) do { \
    const cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::cerr << #call << " failed at line " << __LINE__ << ": " \
                  << cudaGetErrorString(status) << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::cout << "CUDA device: " << properties.name << ", compute capability "
              << properties.major << '.' << properties.minor << '\n';

    // An incomplete last block exercises the kernel's bounds guard.
    constexpr int n = 100003;
    constexpr int threads = 256;
    constexpr int blocks = (n + threads - 1) / threads;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> a(n), b(n), out(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 97) * 0.25f;
        b[i] = -static_cast<float>(i % 31) * 0.5f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    // NaN sentinel makes any unwritten output fail the comparison.
    CUDA_CHECK(cudaMemset(d_out, 0xff, bytes));

    vector_add<<<blocks, threads>>>(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    for (int i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        // All inputs and sums here are exactly representable in float32.
        if (!std::isfinite(out[i]) || out[i] != expected) {
            std::cerr << "CUDA check: FAIL at index " << i << ", got " << out[i]
                      << ", expected " << expected << '\n';
            return 1;
        }
    }
    std::cout << "Custom CUDA compile and execution: PASS (" << n
              << " elements, " << blocks << " blocks, " << threads
              << " threads/block)\n";
}
