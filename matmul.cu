//
// Created by Pujun Lun on 2019-01-25.
//

#include "matmul.h"

#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>

#include "macro.h"

namespace Math {
    struct MatrixMultiplier;
    using KernelFunc = void (const MatrixMultiplier&);

    static const uint REPEAT_TIMES = 10;
    static const uint TILE_WIDTH = 32;

    struct MatrixMultiplier {
        uint m, n, k;
        uint sA, sB, sC;
        float *pA, *pB, *pC;
        const Matrix &mC, *pRef;

        MatrixMultiplier(const Matrix& a, const Matrix& b, const Matrix& c, const Matrix* ref)
            : mC {c}, pRef {ref} {
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

        bool predicate() const {
            if (!pRef) DEBUG_INFO("No reference matrix")
            return mC == *pRef;
        }

        void transfer() const {
            cudaMemcpy(mC.data(), pC, sC, cudaMemcpyDeviceToHost);
        }

        void clear() const {
            mC.clear();
            cudaMemset(pC, 0, sC);
        }

        ~MatrixMultiplier() {
            if (Matrix::verbose)
                cout << "De-allocating matrices on device" << endl;

            cudaFree(pA);
            cudaFree(pB);
            cudaFree(pC);
        }
    };

    __global__ void matMulKernelA(const float* A, const float* B, float* C,
                                  const uint m, const uint n, const uint k) {
        uint row = blockIdx.y * TILE_WIDTH + threadIdx.y;
        uint col = blockIdx.x * TILE_WIDTH + threadIdx.x;

        float sum = 0;
        for (uint i = 0; i < k; ++i)
            sum += A[row * k + i] * B[i * n + col];

        C[row * n + col] = sum;
    }

    void matMulA(const MatrixMultiplier& multiplier) {
        if (multiplier.m % TILE_WIDTH != 0 || multiplier.n % TILE_WIDTH != 0)
            DEBUG_INFO("Dimension not supported")

        dim3 dimBlock {TILE_WIDTH, TILE_WIDTH};
        dim3 dimGrid {multiplier.n / TILE_WIDTH, multiplier.m / TILE_WIDTH};
        matMulKernelA<<<dimGrid, dimBlock>>>(multiplier.pA, multiplier.pB, multiplier.pC,
                                             multiplier.m, multiplier.n, multiplier.k);
    }

    void verifyKernel(bool run, string name, KernelFunc func, const MatrixMultiplier& multiplier) {
        if (run) {
            func(multiplier);
            multiplier.transfer();
            if (!multiplier.predicate())
                DEBUG_INFO("Wrong result by kernel " + name)
            multiplier.clear();
        }
    }

    void repeatWithTimer(bool run, string name, KernelFunc func, const MatrixMultiplier& multiplier) {
        if (run) {
            using chrono::steady_clock;
            steady_clock::time_point begin = steady_clock::now();

            for (uint i = 0; i < REPEAT_TIMES; ++i) func(multiplier);
            cudaDeviceSynchronize();

            steady_clock::time_point end = steady_clock::now();
            auto time = chrono::duration_cast<chrono::microseconds>(end - begin).count();
            cout << name << ": " << time / 1000.0 / REPEAT_TIMES << "ms" << std::endl;
        }
    }

    void testMatMul(const Matrix& a, const Matrix& b, MatMulAlgo algo) {
        if (Matrix::verbose)
            cout << "Multiplying " << a << " and " << b << endl;

        /* verify algorithms */
        if (Matrix::verbose)
            cout << endl << "Verifying correctness of algorithms" << endl;

        Matrix ref = blasMatMul(a, b);
        Matrix c {a.rows(), b.cols(), Matrix::Mode::undefined};
        MatrixMultiplier multiplier {a, b, c, &ref};

        verifyKernel(algo & MatMulAlgoA, "A", matMulA, multiplier);

        /* record elapsed time */
        if (Matrix::verbose)
            cout << endl << "Reporting elapsed time of algorithms" << endl;

        repeatWithTimer(algo & MatMulAlgoA, "A", matMulA, multiplier);
    }

    Matrix blasMatMul(const Matrix& a, const Matrix& b) {
        if (Matrix::verbose)
            cout << "Matrix multiply via cuBLAS" << endl;

        Matrix c {a.rows(), b.cols(), Matrix::Mode::undefined};
        MatrixMultiplier multiplier {a, b, c, nullptr};

        int lda = (int)multiplier.m, ldb = (int)multiplier.k, ldc = (int)multiplier.m;
        float alpha = 1.0f, beta = 0.0f;

        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, multiplier.m, multiplier.n, multiplier.k,
                    &alpha, multiplier.pA, lda, multiplier.pB, ldb, &beta, multiplier.pC, ldc);
        cublasDestroy(handle);

        multiplier.transfer();
        return c;
    }
}
