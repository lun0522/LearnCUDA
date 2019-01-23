//
// Created by Pujun Lun on 2019-01-22.
//

#ifndef LEARNCUDA_MACRO_H
#define LEARNCUDA_MACRO_H

// https://stackoverflow.com/a/42288429
#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__

#include <device_functions.h>
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_cmath.h>
#endif // __JETBRAINS_IDE__

#endif //LEARNCUDA_MACRO_H
