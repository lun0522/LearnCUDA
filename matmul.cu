//
// Created by Pujun Lun on 2019-01-25.
//

#include "matmul.h"

#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>

#include "macro.h"

namespace Math {
    static const uint REPEAT_TIMES = 10;
    static const uint TILE_WIDTH = 32;

    __global__ void matMulKernelA(const float* A, const float* B, float* C,
                                  const uint m, const uint n, const uint k) {
        uint row = blockIdx.y * TILE_WIDTH + threadIdx.y;
        uint col = blockIdx.x * TILE_WIDTH + threadIdx.x;

        float sum = 0;
        for (uint i = 0; i < k; ++i)
            sum += A[row * k + i] * B[i * n + col];

        C[row * n + col] = sum;
    }

    void matMulA(const float* A, const float* B, float* C,
                 const uint m, const uint n, const uint k) {
        dim3 dimBlock {TILE_WIDTH, TILE_WIDTH};
        dim3 dimGrid {n / TILE_WIDTH, m / TILE_WIDTH};
        matMulKernelA<<<dimGrid, dimBlock>>>(A, B, C, m, n, k);
    }

    template <typename Transfer, typename Predicate,
              typename Cleanup, typename Compute>
    void verifyKernel(bool run, string name,
                      Transfer transfer, Predicate predicate,
                      Cleanup cleanup, Compute compute) {
        if (run) {
            compute();
            transfer();
            if (!predicate())
                DEBUG_INFO("Wrong result by kernel " + name)
            cleanup();
        }
    }

    template <typename Compute>
    void repeatWithTimer(bool run, string name, Compute compute) {
        if (run) {
            using chrono::steady_clock;
            steady_clock::time_point begin = steady_clock::now();

            for (uint i = 0; i < REPEAT_TIMES; ++i) compute();
            cudaDeviceSynchronize();

            steady_clock::time_point end = steady_clock::now();
            auto time = chrono::duration_cast<chrono::microseconds>(end - begin).count();
            cout << name << ": " << time / 1000.0 / REPEAT_TIMES << "ms" << std::endl;
        }
    }

    void testMatMul(const Matrix& a, const Matrix& b, Matrix& c, MatMulAlgo algo) {
        if (Matrix::verbose)
            cout << "Multiplying " << a << " and " << b << endl;

        uint m = a.rows() == c.rows() ?
                 (uint)a.rows() : DEBUG_INFO("First dimension not match")
        uint n = b.cols() == c.cols() ?
                 (uint)b.cols() : DEBUG_INFO("Second dimension not match")
        uint k = a.cols() == b.rows() ?
                 (uint)a.cols() : DEBUG_INFO("Third dimension not match")

        if (m % TILE_WIDTH != 0 || n % TILE_WIDTH != 0)
            DEBUG_INFO("Dimension not supported")

        uint aSize = m * k * sizeof(float);
        uint bSize = k * n * sizeof(float);
        uint cSize = m * n * sizeof(float);

        float *aData, *bData, *cData;
        cudaMalloc(&aData, aSize);
        cudaMemcpy(aData, a.data(), aSize, cudaMemcpyHostToDevice);
        cudaMalloc(&bData, bSize);
        cudaMemcpy(bData, b.data(), bSize, cudaMemcpyHostToDevice);
        cudaMalloc(&cData, cSize);

        /* verify algorithms */
        if (Matrix::verbose)
            cout << endl << "Verifying correctness of algorithms" << endl;

        Matrix ref = eigenMatMul(a, b);
        auto trans = [&]() { cudaMemcpy(c.data(), cData, cSize, cudaMemcpyDeviceToHost); };
        auto pred  = [&]() -> bool { return c == ref; };
        auto clean = [&]() { c.clear(); cudaMemset(cData, 0, cSize); };

        verifyKernel(algo & MatMulAlgoA, "A", trans, pred, clean,
                     [&]() { matMulA(aData, bData, cData, m, n, k); });

        /* record elapsed time */
        if (Matrix::verbose)
            cout << endl << "Reporting elapsed time of algorithms" << endl;

        repeatWithTimer(algo & MatMulAlgoA, "A",
                        [&]() { matMulA(aData, bData, cData, m, n, k); });

        cudaFree(aData);
        cudaFree(bData);
        cudaFree(cData);

        if (Matrix::verbose) cout << endl;
    }

    Matrix eigenMatMul(const Matrix& a, const Matrix& b) {
        if (Matrix::verbose)
            cout << "Verifying matrix multiply result" << endl;

        if (a.cols() != b.rows())
            DEBUG_INFO("Dimension not match")

        return Matrix {MatrixXf(a) * MatrixXf(b)};
    }
}
