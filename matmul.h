//
// Created by Pujun Lun on 2019-01-25.
//

#ifndef LEARNCUDA_MATMUL_H
#define LEARNCUDA_MATMUL_H

#include "matrix.h"

namespace Math {
    enum MatMulAlgo {
        MatMulAlgoA = 1 << 0,
        MatMulAlgoB = 1 << 1,
        MatMulAlgoC = 1 << 2,
        MatMulAlgoD = 1 << 3,
        MatMulAlgoE = 1 << 4,
    };

    void testMatMul(const Matrix &a, const Matrix &b, uint algo);
    Matrix blasMatMul(const Matrix &a, const Matrix &b);
} // namespace Math

#endif // LEARNCUDA_MATMUL_H
