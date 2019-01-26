//
// Created by Pujun Lun on 2019-01-25.
//

#include "matmul.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

static const uint TILE_WIDTH = 8;

__global__ void sgemmKernel(const float* A, const float* B, float* C,
                            const uint m, const uint n, const uint k) {
    uint row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    uint col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0;
    for (uint i = 0; i < k; ++i)
        sum += A[row * k + i] * B[i * n + col];

    C[row * n + col] = sum;
}

void sgemm(const uint m, const uint n, const uint k,
           const float* A, const float* B, float* C) {
    uint aSize = m * k * sizeof(float);
    uint bSize = k * n * sizeof(float);
    uint cSize = m * n * sizeof(float);

    float *aData, *bData, *cData;
    cudaMalloc(&aData, aSize);
    cudaMemcpy(aData, A, aSize, cudaMemcpyHostToDevice);
    cudaMalloc(&bData, bSize);
    cudaMemcpy(bData, B, bSize, cudaMemcpyHostToDevice);
    cudaMalloc(&cData, cSize);

    dim3 dimBlock {TILE_WIDTH, TILE_WIDTH};
    dim3 dimGrid {n / TILE_WIDTH, m / TILE_WIDTH};
    sgemmKernel <<<dimGrid, dimBlock>>>(aData, bData, cData, m, n, k);
    cudaMemcpy(C, cData, cSize, cudaMemcpyDeviceToHost);

    cudaFree(aData);
    cudaFree(bData);
    cudaFree(cData);
}
