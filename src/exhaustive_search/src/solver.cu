#include "solver.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

__device__ float compute_energy(const float* J, const int* S, int N) {
    float energy = 0.0f;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            energy += J[i * N + j] * S[i] * S[j];
    return -energy;
}

__global__ void exhaustive_kernel(const float* J, int N, float* min_energy, int* best_config) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = 1ULL << N;
    if (idx >= total) return;

    __shared__ float shared_best_energy;
    __shared__ int shared_best_config[64];  // N <= 64

    int S[64];  // 当前配置
    for (int i = 0; i < N; ++i)
        S[i] = ((idx >> i) & 1) ? 1 : -1;

    float E = compute_energy(J, S, N);

    if (threadIdx.x == 0) {
        shared_best_energy = FLT_MAX;
    }
    __syncthreads();

    atomicMin((int*)&shared_best_energy, __float_as_int(E));
    __syncthreads();

    if (E == shared_best_energy) {
        for (int i = 0; i < N; ++i)
            shared_best_config[i] = S[i];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        *min_energy = shared_best_energy;
        for (int i = 0; i < N; ++i)
            best_config[i] = shared_best_config[i];
    }
}

void solve_exhaustive_gpu(const float* J, int N, std::vector<int>& best_S, float& best_energy) {
    unsigned long long total = 1ULL << N;
    if (total > (1ULL << 20)) {
        throw std::runtime_error("N too large for brute-force");
    }

    float* d_J;
    float* d_min_energy;
    int* d_best_config;

    cudaMalloc(&d_J, sizeof(float) * N * N);
    cudaMalloc(&d_min_energy, sizeof(float));
    cudaMalloc(&d_best_config, sizeof(int) * N);

    cudaMemcpy(d_J, J, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    exhaustive_kernel<<<blocks, threads>>>(d_J, N, d_min_energy, d_best_config);
    cudaDeviceSynchronize();

    float min_E;
    std::vector<int> S(N);
    cudaMemcpy(&min_E, d_min_energy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(S.data(), d_best_config, sizeof(int) * N, cudaMemcpyDeviceToHost);

    best_S = std::move(S);
    best_energy = min_E;

    cudaFree(d_J);
    cudaFree(d_min_energy);
    cudaFree(d_best_config);
}
