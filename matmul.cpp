//
// Created by Pujun Lun on 2019-01-25.
//

#include "matmul.h"

#include "macro.h"

namespace Math {
    void matMul(const Matrix& m, const Matrix& n, Matrix& p) {
        if (m.getDims().width != n.getDims().height)
            DEBUG_INFO("Wrong dimension for matmul")
    }

    bool verifyMatMul(const Matrix& m, const Matrix& n, const Matrix& p) {
        return p == Matrix {MatrixXf(m) * MatrixXf(n)};
    }
}
