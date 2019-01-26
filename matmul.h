//
// Created by Pujun Lun on 2019-01-25.
//

#ifndef LEARNCUDA_MATMUL_H
#define LEARNCUDA_MATMUL_H

void sgemm(const uint m, const uint n, const uint k,
           const float *A, const float *B, float *C);

#endif //LEARNCUDA_MATMUL_H
