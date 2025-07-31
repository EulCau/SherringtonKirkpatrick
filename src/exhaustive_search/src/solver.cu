#include "solver.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdexcept>

__device__ float compute_energy(const float* J, int N, unsigned int k) {
    float energy = 0.0f;
    for (int i = 0; i < N; ++i) {
        int Si = ((k >> i) & 1) ? 1 : -1;
        for (int j = i + 1; j < N; ++j) {
            int Sj = ((k >> j) & 1) ? 1 : -1;
            energy -= J[i * N + j] * Si * Sj;
        }
    }
    return energy;
}

__global__ void compute_energies_and_min(
    const float* __restrict__ J,
    int N,
    float* __restrict__ energies)
{
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (1U << (N - 1))) return;

    float energy = compute_energy(J, N, k);
    energies[k] = energy;
}

void launch_energy_kernel(const float* h_J, int N, float*& h_energies) {
    size_t total_k = 1ULL << (N - 1);

    float* d_J;
    float* d_energies;

    cudaMalloc(&d_J, sizeof(float) * N * N);
    cudaMalloc(&d_energies, sizeof(float) * total_k);

    cudaMemcpy(d_J, h_J, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (total_k + threadsPerBlock - 1) / threadsPerBlock;

    compute_energies_and_min<<<blocks, threadsPerBlock>>>(d_J, N, d_energies);
    cudaDeviceSynchronize();

    h_energies = new float[total_k];

    cudaMemcpy(h_energies, d_energies, sizeof(float) * total_k, cudaMemcpyDeviceToHost);

    cudaFree(d_J);
    cudaFree(d_energies);
}
