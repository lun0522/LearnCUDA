//
// Created by Pujun Lun on 2019-01-25.
//

#ifndef LEARNCUDA_MATMUL_H
#define LEARNCUDA_MATMUL_H

#import "matrix.h"

namespace Math {
    void matMul(const Matrix& m, const Matrix& n, Matrix& p);
    bool verifyMatMul(const Matrix& m, const Matrix& n, const Matrix& p);
}

#endif //LEARNCUDA_MATMUL_H
