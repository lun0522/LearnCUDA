//
// Created by Pujun Lun on 2019-01-25.
//

#ifndef LEARNCUDA_MATMUL_H
#define LEARNCUDA_MATMUL_H

#include "matrix.h"

namespace Math {
    enum MatMulAlgo {
        MatMulAlgoA = 1 << 0,
    };
    void testMatMul(const Matrix& a, const Matrix& b, Matrix& c, MatMulAlgo algo);
    Matrix eigenMatMul(const Matrix& a, const Matrix& b);
}

#endif //LEARNCUDA_MATMUL_H
