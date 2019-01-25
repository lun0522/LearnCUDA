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
        using dim_t = unsigned int;

        struct Dims {
            dim_t width;
            dim_t height;
        };
        enum class Mode {
            randFloat,
            randInt,
            undefined,
            unit,
            zero,
        };

        explicit Matrix(dim_t width, dim_t height, Mode mode);
        explicit Matrix(const MatrixXf& other);
        explicit Matrix(const string& path);
        Matrix(const Matrix& other);
        Matrix(Matrix&& other) noexcept;
        ~Matrix();

        static void setVerbose(bool v) { verbose = v; }
        Dims getDims() const { return dims; }
        const val_t* getData() const { return data; }

        val_t& operator()(dim_t row, dim_t col) const {
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
    
    inline bool verifyMatMul(const Matrix& m, const Matrix& n, const Matrix& p) {
        return p == Matrix {MatrixXf(m) * MatrixXf(n)};
    }
}

#endif //LEARNCUDA_MATRIX_H
