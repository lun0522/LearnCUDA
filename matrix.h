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
    using Eigen::MatrixXf;

    class Matrix {
    public:
        using val_t = float;
        struct Dims {
            size_t width;
            size_t height;
        };
        enum class Mode {
            randFloat,
            randInt,
            undefined,
            unit,
            zero,
        };

        explicit Matrix(size_t width, size_t height, Mode mode);
        explicit Matrix(const MatrixXf& other);
        explicit Matrix(const string& path);
        Matrix(const Matrix& other);
        Matrix(Matrix&& other) noexcept;
        ~Matrix();

        static void setVerbose(bool v) { verbose = v; }
        Dims getDims() const { return dims; }
        val_t* getData() const { return data; }

        val_t& operator()(size_t row, size_t col) const {
            return data[row * dims.width + col];
        }
        friend ostream& operator<<(ostream& os, const Matrix& matrix);
        bool operator==(const Matrix& other) const;
        Matrix& operator=(const Matrix& other);
        Matrix& operator=(Matrix&& other) noexcept;
        operator MatrixXf() const;

        void print() const;
        void dump(const string& path) const;

    private:
        static bool verbose;
        Dims dims;
        val_t* data;
    };
}

#endif //LEARNCUDA_MATRIX_H
