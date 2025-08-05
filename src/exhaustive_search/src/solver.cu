#include "solver.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

__device__ float compute_energy(const float* J, int N, unsigned int k)
{
    float energy = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        int Si = ((k >> i) & 1) ? 1 : -1;
        for (int j = i + 1; j < N; ++j)
        {
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

__global__ void reduce_min_energy(const float* energies, size_t total_k, float* min_energy, unsigned int* min_index) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float my_min = 1e20f;
    unsigned int my_idx = 0;

    if (i < total_k) {
        float val1 = energies[i];
        float val2 = (i + blockDim.x < total_k) ? energies[i + blockDim.x] : 1e20f;
        if (val1 < val2) {
            my_min = val1;
            my_idx = i;
        } else {
            my_min = val2;
            my_idx = i + blockDim.x;
        }
    }

    __shared__ unsigned int sidx[1024];
    sdata[tid] = my_min;
    sidx[tid] = my_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] < sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        min_energy[blockIdx.x] = sdata[0];
        min_index[blockIdx.x] = sidx[0];
    }
}

__global__ void final_reduce_min(const float* block_min_energy, const unsigned int* block_min_index, int num_blocks, unsigned int* global_min_index) {
    float min_val = 1e20f;
    unsigned int min_idx = 0;
    for (int i = 0; i < num_blocks; ++i) {
        if (block_min_energy[i] < min_val) {
            min_val = block_min_energy[i];
            min_idx = block_min_index[i];
        }
    }
    *global_min_index = min_idx;
}

void launch_energy_kernel(const float* h_J, int N, unsigned int* h_min_index)
{
    size_t total_k = 1ULL << (N - 1);

    float* d_J;
    float* d_energies;
    unsigned int* d_min_index;

    float* d_block_min_energy;
    unsigned int* d_block_min_index;

    cudaMalloc(&d_J, sizeof(float) * N * N);
    cudaMalloc(&d_energies, sizeof(float) * total_k);
    cudaMalloc(&d_min_index, sizeof(unsigned int));

    cudaMemcpy(d_J, h_J, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (total_k + threadsPerBlock - 1) / threadsPerBlock;  // 每个block处理2 * threadsPerBlock个元素

    compute_energies_and_min<<<blocks, threadsPerBlock>>>(d_J, N, d_energies);
    cudaDeviceSynchronize();

    blocks = (total_k + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    cudaMalloc(&d_block_min_energy, sizeof(float) * blocks);
    cudaMalloc(&d_block_min_index, sizeof(unsigned int) * blocks);
    reduce_min_energy<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_energies, total_k, d_block_min_energy, d_block_min_index);
    cudaDeviceSynchronize();

    final_reduce_min<<<1, 1>>>(d_block_min_energy, d_block_min_index, blocks, d_min_index);
    cudaDeviceSynchronize();

    cudaMemcpy(h_min_index, d_min_index, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_J);
    cudaFree(d_energies);
    cudaFree(d_min_index);
    cudaFree(d_block_min_energy);
    cudaFree(d_block_min_index);
}
