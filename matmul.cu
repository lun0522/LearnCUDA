//
// Created by Pujun Lun on 2019-01-25.
//

#include "matmul.h"

#include <chrono>
#include <cublas_v2.h>
#include <string>

#include "macro.h"
#include "matmul_impl.h"

namespace Math {
    using KernelFunc = void (const MatrixMultiplier&);

    static const uint REPEAT_TIMES = 10;

    void verifyKernel(uint run, const string &name, KernelFunc func,
                      const MatrixMultiplier &multiplier) {
        if (run) {
            func(multiplier);
            multiplier.transfer();
            if (!multiplier.predicate())
                DEBUG_INFO("Wrong result by kernel " + name)
            multiplier.clear();
        }
    }

    void repeatWithTimer(uint run, const string &name, KernelFunc func,
                         const MatrixMultiplier &multiplier) {
        if (run) {
            using namespace chrono;
            auto begin = steady_clock::now();

            for (uint i = 0; i < REPEAT_TIMES; ++i)
                func(multiplier);
            cudaDeviceSynchronize();

            auto end = steady_clock::now();
            auto time = duration_cast<microseconds>(end - begin);
            cout << name << ": ";
            cout << time.count() / 1000.0 / REPEAT_TIMES << "ms" << endl;
        }
    }

    void testMatMul(const Matrix &a, const Matrix &b, uint algo) {
        if (Matrix::verbose)
            cout << "Multiplying " << a << " and " << b << endl;

        /* verify algorithms */
        if (Matrix::verbose)
            cout << endl << "Verifying correctness of algorithms" << endl;

        Matrix ref = blasMatMul(a, b);
        Matrix c{a.rows(), b.cols(), Matrix::Mode::undefined};
        MatrixMultiplier multiplier{a, b, c, &ref};

        verifyKernel(algo & MatMulAlgoA, "A", matMulA, multiplier);
        verifyKernel(algo & MatMulAlgoB, "B", matMulB, multiplier);
        verifyKernel(algo & MatMulAlgoC, "C", matMulC, multiplier);
        verifyKernel(algo & MatMulAlgoD, "D", matMulD, multiplier);
        verifyKernel(algo & MatMulAlgoE, "E", matMulE, multiplier);

        /* record elapsed time */
        if (Matrix::verbose)
            cout << endl << "Reporting elapsed time of algorithms" << endl;

        repeatWithTimer(algo & MatMulAlgoA, "A", matMulA, multiplier);
        repeatWithTimer(algo & MatMulAlgoB, "B", matMulB, multiplier);
        repeatWithTimer(algo & MatMulAlgoC, "C", matMulC, multiplier);
        repeatWithTimer(algo & MatMulAlgoD, "D", matMulD, multiplier);
        repeatWithTimer(algo & MatMulAlgoE, "E", matMulE, multiplier);

        if (Matrix::verbose)
            cout << endl;
    }

    Matrix blasMatMul(const Matrix &a, const Matrix &b) {
        if (Matrix::verbose)
            cout << "Matrix multiply via cuBLAS" << endl;

        Matrix c{a.rows(), b.cols(), Matrix::Mode::undefined};
        MatrixMultiplier multiplier{a, b, c, nullptr};

        /* Note that cuBLAS expects matrices stored in column-major */
        int lda = multiplier.n, ldb = multiplier.k, ldc = multiplier.n;
        float alpha = 1.0f, beta = 0.0f;

        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    multiplier.n, multiplier.m, multiplier.k,
                    &alpha, multiplier.pB, lda, multiplier.pA, ldb,
                    &beta, multiplier.pC, ldc);
        cublasDestroy(handle);

        multiplier.transfer();
        return c;
    }
} // namespace Math
