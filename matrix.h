//
// Created by Pujun Lun on 2019-01-21.
//

#ifndef LEARNCUDA_MATRIX_H
#define LEARNCUDA_MATRIX_H

#include <Dense>
#include <iostream>
#include <string>

namespace Math {
    using namespace std;
    using Eigen::MatrixXi;

    class Matrix {
    public:
        struct Dims {
            uint width;
            uint height;
        };

        enum class Mode {
            random,
            unit,
            zero,
        };

        explicit Matrix(uint width, uint height, Mode mode = Mode::random);
        explicit Matrix(const MatrixXi& other);
        explicit Matrix(const string& path);
        Matrix(const Matrix& other);
        Matrix(Matrix&& other) noexcept;
        ~Matrix();

        static void setVerbose(bool v) { verbose = v; }
        Dims getDims() const { return dims; }
        const uint* getData() const { return data; }

        uint& operator()(uint row, uint col) const { return data[row * dims.width + col]; }
        friend ostream& operator<<(ostream& os, const Matrix& matrix);
        bool operator==(const Matrix& other) const;
        Matrix& operator=(const Matrix& other);
        Matrix& operator=(Matrix&& other) noexcept;
        operator MatrixXi() const;

        void print() const;
        void dump(const string& path) const;

    private:
        static bool verbose;
        Dims dims;
        uint* data;
    };
    
    inline bool verifyMatMul(const Matrix& m, const Matrix& n, const Matrix& p) {
        return p == Matrix {MatrixXi(m) * MatrixXi(n)};
    }
}

#endif //LEARNCUDA_MATRIX_H
