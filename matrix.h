//
// Created by Pujun Lun on 2019-01-21.
//

#ifndef LEARNCUDA_MATRIX_H
#define LEARNCUDA_MATRIX_H

#include <string>

class Matrix {
public:
    using UInt = unsigned int;
    struct Dims {
        UInt width;
        UInt height;
    };

    explicit Matrix(UInt width, UInt height);
    explicit Matrix(const std::string& path);
    ~Matrix();

    Dims getDims() const;
    const UInt* getData() const;
    void print() const;
    void dump(const std::string& path) const;
    static void setVerbose(bool v) { verbose = v; }

private:
    static bool verbose;
    Dims dims;
    UInt* data;
};

#endif //LEARNCUDA_MATRIX_H
