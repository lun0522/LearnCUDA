//
// Created by Pujun Lun on 2019-01-21.
//

#ifndef LEARNCUDA_MATRIX_H
#define LEARNCUDA_MATRIX_H

#include <iostream>
#include <string>

namespace Math {
    using namespace std;

    class Matrix {
    public:
        using val_t = float;
        static bool verbose;

        enum class Mode {
            randFloat,
            randInt,
            undefined,
            unit,
            zero,
        };

        explicit Matrix(size_t rows, size_t cols, Mode mode);
        explicit Matrix(const string& path);
        Matrix(const Matrix& other);
        Matrix(Matrix&& other) noexcept;
        ~Matrix();

        size_t rows() const { return num_row; }
        size_t cols() const { return num_col; }
        size_t size() const { return num_elem; }
        val_t* data() const { return raw_data; }

        val_t& operator()(size_t row, size_t col) const {
            return data()[row * cols() + col];
        }
        bool operator==(const Matrix& other) const;
        bool operator!=(const Matrix& other) const { return !(*this == other); }
        Matrix& operator=(const Matrix& other);
        Matrix& operator=(Matrix&& other) noexcept;
        friend ostream& operator<<(ostream& os, const Matrix& matrix);

        void clear() const;
        void print() const;
        void dump(const string& path) const;

    private:
        size_t num_row, num_col, num_elem;
        val_t* raw_data;
    };
}

#endif //LEARNCUDA_MATRIX_H
