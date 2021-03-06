//
// Created by Pujun Lun on 2019-02-02.
//

#include "matmul_impl.h"

#include "macro.h"

namespace Math {
    static const uint TILE_WIDTH = 32;

    MatrixMultiplier::MatrixMultiplier(const Matrix &a, const Matrix &b, const Matrix &c,
            const Matrix *ref)
            : mC{c}, pRef{ref} {
        if (Matrix::verbose) {
            cout << "Extracting matrices data" << endl;
            cout << "A: " << a << endl;
            cout << "B: " << b << endl;
            cout << "C: " << c << endl;
        }

        m = a.rows() == c.rows() ?
            (uint)a.rows() : DEBUG_INFO("First dimension not match")
        n = b.cols() == c.cols() ?
            (uint)b.cols() : DEBUG_INFO("Second dimension not match")
        k = a.cols() == b.rows() ?
            (uint)a.cols() : DEBUG_INFO("Third dimension not match")

        sA = m * k * sizeof(float);
        sB = k * n * sizeof(float);
        sC = m * n * sizeof(float);

        cudaMalloc(&pA, sA);
        cudaMemcpy(pA, a.data(), sA, cudaMemcpyHostToDevice);
        cudaMalloc(&pB, sB);
        cudaMemcpy(pB, b.data(), sB, cudaMemcpyHostToDevice);
        cudaMalloc(&pC, sC);
    }

    bool MatrixMultiplier::predicate() const {
        if (!pRef)
            DEBUG_INFO("No reference matrix")

        return mC == *pRef;
    }

    void MatrixMultiplier::transfer() const {
        cudaMemcpy(mC.data(), pC, sC, cudaMemcpyDeviceToHost);
    }

    void MatrixMultiplier::clear() const {
        mC.clear();
        cudaMemset(pC, 0, sC);
    }

    MatrixMultiplier::~MatrixMultiplier() {
        if (Matrix::verbose)
            cout << "De-allocating matrices on device" << endl;

        cudaFree(pA);
        cudaFree(pB);
        cudaFree(pC);
    }

    /* Naive algorithm - 32 registers */
    __global__ void matMulKernelA(const float *A, const float *B, float *C,
                                  const uint m, const uint n, const uint k) {
        uint row = blockIdx.y * TILE_WIDTH + threadIdx.y;
        uint col = blockIdx.x * TILE_WIDTH + threadIdx.x;

        float sum = 0.0f;
        for (uint i = 0; i < k; ++i)
            sum += A[row * k + i] * B[i * n + col];

        C[row * n + col] = sum;
    }

    /* Reuse data with shared memory - 29 registers */
    __global__ void matMulKernelB(const float *A, const float *B, float *C,
                                  const uint m, const uint n, const uint k) {
        __shared__ float aD[TILE_WIDTH][TILE_WIDTH];
        __shared__ float bD[TILE_WIDTH][TILE_WIDTH];

        uint tx = threadIdx.x, ty = threadIdx.y;
        uint row = blockIdx.y * TILE_WIDTH + ty;
        uint col = blockIdx.x * TILE_WIDTH + tx;

        float sum = 0.0f;
        for (int i = 0; i < k / TILE_WIDTH; ++i) {
            aD[ty][tx] = A[row * k + (i * TILE_WIDTH + tx)];
            bD[ty][tx] = B[(i * TILE_WIDTH + ty) * n + col];
            __syncthreads();

            for (uint j = 0; j < TILE_WIDTH; ++j)
                sum += aD[ty][j] * bD[j][tx];
            __syncthreads();
        }
        C[row * n + col] = sum;
    }

    /* Prefetch data (store in shared memory) - 26 registers */
    __global__ void matMulKernelC(const float *A, const float *B, float *C,
                                  const uint m, const uint n, const uint k) {
        __shared__ float aD1[TILE_WIDTH][TILE_WIDTH];
        __shared__ float bD1[TILE_WIDTH][TILE_WIDTH];
        __shared__ float aD2[TILE_WIDTH][TILE_WIDTH];
        __shared__ float bD2[TILE_WIDTH][TILE_WIDTH];

        uint tx = threadIdx.x, ty = threadIdx.y;
        uint row = blockIdx.y * TILE_WIDTH + ty;
        uint col = blockIdx.x * TILE_WIDTH + tx;

        bool useMat1 = true;
        aD1[ty][tx] = A[row * k + tx];
        bD1[ty][tx] = B[ty * n + col];

        float sum = 0.0f;
        for (int i = 1, itr = k / TILE_WIDTH; i <= itr; ++i, useMat1 = !useMat1) {
            __syncthreads();

            if (useMat1) {
                // prefetch data for next iteration
                if (i != itr) {
                    aD2[ty][tx] = A[row * k + (i * TILE_WIDTH + tx)];
                    bD2[ty][tx] = B[(i * TILE_WIDTH + ty) * n + col];
                }

                // consume data fetched in last iteration
                for (uint j = 0; j < TILE_WIDTH; ++j)
                    sum += aD1[ty][j] * bD1[j][tx];
            } else {
                // prefetch data for next iteration
                if (i != itr) {
                    aD1[ty][tx] = A[row * k + (i * TILE_WIDTH + tx)];
                    bD1[ty][tx] = B[(i * TILE_WIDTH + ty) * n + col];
                }

                // consume data fetched in last iteration
                for (uint j = 0; j < TILE_WIDTH; ++j)
                    sum += aD2[ty][j] * bD2[j][tx];
            }
        }
        C[row * n + col] = sum;
    }

    /* Prefetch data (store in registers) - 29 registers */
    __global__ void matMulKernelD(const float *A, const float *B, float *C,
                                  const uint m, const uint n, const uint k) {
        __shared__ float aD[TILE_WIDTH][TILE_WIDTH];
        __shared__ float bD[TILE_WIDTH][TILE_WIDTH];

        uint tx = threadIdx.x, ty = threadIdx.y;
        uint row = blockIdx.y * TILE_WIDTH + ty;
        uint col = blockIdx.x * TILE_WIDTH + tx;

        float nextElemA = A[row * k + tx];
        float nextElemB = B[ty * n + col];

        float sum = 0.0f;
        for (int i = 1, itr = k / TILE_WIDTH; i <= itr; ++i) {
            // store data fetched in last iteration
            aD[ty][tx] = nextElemA;
            bD[ty][tx] = nextElemB;
            __syncthreads();

            // prefetch data for next iteration
            if (i != itr) {
                nextElemA = A[row * k + (i * TILE_WIDTH + tx)];
                nextElemB = B[(i * TILE_WIDTH + ty) * n + col];
            }

            // consume data fetched in last iteration
            for (uint j = 0; j < TILE_WIDTH; ++j)
                sum += aD[ty][j] * bD[j][tx];
            __syncthreads();
        }
        C[row * n + col] = sum;
    }

    /* Unroll for loop - 30 registers */
    __global__ void matMulKernelE(const float *A, const float *B, float *C,
                                  const uint m, const uint n, const uint k) {
        __shared__ float aD[TILE_WIDTH][TILE_WIDTH];
        __shared__ float bD[TILE_WIDTH][TILE_WIDTH];

        uint tx = threadIdx.x, ty = threadIdx.y;
        uint row = blockIdx.y * TILE_WIDTH + ty;
        uint col = blockIdx.x * TILE_WIDTH + tx;

        float nextElemA = A[row * k + tx];
        float nextElemB = B[ty * n + col];

        float sum = 0.0f;
        for (int i = 1, itr = k / TILE_WIDTH; i <= itr; ++i) {
            // store data fetched in last iteration
            aD[ty][tx] = nextElemA;
            bD[ty][tx] = nextElemB;
            __syncthreads();

            // prefetch data for next iteration
            if (i != itr) {
                nextElemA = A[row * k + (i * TILE_WIDTH + tx)];
                nextElemB = B[(i * TILE_WIDTH + ty) * n + col];
            }

            // consume data fetched in last iteration
            // for i in range(32): print(f"aD[ty][{i}] * bD[{i}][tx] +");
            sum += aD[ty][ 0] * bD[ 0][tx] +
                   aD[ty][ 1] * bD[ 1][tx] +
                   aD[ty][ 2] * bD[ 2][tx] +
                   aD[ty][ 3] * bD[ 3][tx] +
                   aD[ty][ 4] * bD[ 4][tx] +
                   aD[ty][ 5] * bD[ 5][tx] +
                   aD[ty][ 6] * bD[ 6][tx] +
                   aD[ty][ 7] * bD[ 7][tx] +
                   aD[ty][ 8] * bD[ 8][tx] +
                   aD[ty][ 9] * bD[ 9][tx] +
                   aD[ty][10] * bD[10][tx] +
                   aD[ty][11] * bD[11][tx] +
                   aD[ty][12] * bD[12][tx] +
                   aD[ty][13] * bD[13][tx] +
                   aD[ty][14] * bD[14][tx] +
                   aD[ty][15] * bD[15][tx] +
                   aD[ty][16] * bD[16][tx] +
                   aD[ty][17] * bD[17][tx] +
                   aD[ty][18] * bD[18][tx] +
                   aD[ty][19] * bD[19][tx] +
                   aD[ty][20] * bD[20][tx] +
                   aD[ty][21] * bD[21][tx] +
                   aD[ty][22] * bD[22][tx] +
                   aD[ty][23] * bD[23][tx] +
                   aD[ty][24] * bD[24][tx] +
                   aD[ty][25] * bD[25][tx] +
                   aD[ty][26] * bD[26][tx] +
                   aD[ty][27] * bD[27][tx] +
                   aD[ty][28] * bD[28][tx] +
                   aD[ty][29] * bD[29][tx] +
                   aD[ty][30] * bD[30][tx] +
                   aD[ty][31] * bD[31][tx];
            __syncthreads();
        }
        C[row * n + col] = sum;
    }

    using Kernel = void (const float*, const float*, float*,
                         const uint, const uint, const uint);

    void matMul(const MatrixMultiplier &multiplier, Kernel kernel) {
        if (multiplier.m % TILE_WIDTH != 0 || multiplier.n % TILE_WIDTH != 0)
            DEBUG_INFO("Dimension not supported")

        dim3 dimBlock{TILE_WIDTH, TILE_WIDTH};
        dim3 dimGrid{multiplier.n / TILE_WIDTH, multiplier.m / TILE_WIDTH};
        kernel<<<dimGrid, dimBlock>>>(
                multiplier.pA, multiplier.pB, multiplier.pC,
                multiplier.m, multiplier.n, multiplier.k);
    }

    void matMulA(const MatrixMultiplier &multiplier) {
        matMul(multiplier, matMulKernelA);
    }

    void matMulB(const MatrixMultiplier &multiplier) {
        matMul(multiplier, matMulKernelB);
    }

    void matMulC(const MatrixMultiplier &multiplier) {
        matMul(multiplier, matMulKernelC);
    }

    void matMulD(const MatrixMultiplier &multiplier) {
        matMul(multiplier, matMulKernelD);
    }

    void matMulE(const MatrixMultiplier &multiplier) {
        matMul(multiplier, matMulKernelE);
    }
}
