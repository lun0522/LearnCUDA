//
// Created by Pujun Lun on 2019-01-21.
//

#ifndef LEARNCUDA_MATRIX_H
#define LEARNCUDA_MATRIX_H

#include <iostream>
#include <string>

namespace Math {
    using namespace std;
    using UInt = unsigned int;

    class Matrix {
    public:
        struct Dims {
            UInt width;
            UInt height;
        };

        explicit Matrix(UInt width, UInt height);
        explicit Matrix(const string& path);
        ~Matrix();

        Dims getDims() const;
        const UInt* getData() const;
        void print() const;
        void dump(const string& path) const;

    private:
        Dims dims;
        UInt* data;
    };

    static bool verbose = true;
    static void setVerbose(bool v) { verbose = v; }
    ostream& operator<<(ostream& os, const Matrix& matrix);
}

#endif //LEARNCUDA_MATRIX_H
