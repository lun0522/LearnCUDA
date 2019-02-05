//
// Created by Pujun Lun on 2019-02-02.
//

#ifndef LEARNCUDA_MATMUL_IMPL_H
#define LEARNCUDA_MATMUL_IMPL_H

#include "matrix.h"

namespace Math {
    struct MatrixMultiplier {
        uint m, n, k;
        uint sA, sB, sC;
        float *pA, *pB, *pC;
        const Matrix &mC, *pRef;

        MatrixMultiplier(const Matrix &a, const Matrix &b, const Matrix &c,
                         const Matrix *ref);
        bool predicate() const;
        void transfer() const;
        void clear() const;
        ~MatrixMultiplier();
    };

    void matMulA(const MatrixMultiplier &multiplier);
    void matMulB(const MatrixMultiplier &multiplier);
    void matMulC(const MatrixMultiplier &multiplier);
    void matMulD(const MatrixMultiplier &multiplier);
    void matMulE(const MatrixMultiplier &multiplier);
}

#endif //LEARNCUDA_MATMUL_IMPL_H
