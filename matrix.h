//
// Created by Pujun Lun on 2019-01-21.
//

#ifndef LEARNCUDA_MATRIX_H
#define LEARNCUDA_MATRIX_H

#include <string>

class Matrix {
    using UInt = unsigned int;
    UInt width, height;
    UInt* data;
public:
    Matrix(UInt width, UInt height);
    Matrix(const std::string& path);
    void dump(const std::string& path);
    ~Matrix();
};

#endif //LEARNCUDA_MATRIX_H
