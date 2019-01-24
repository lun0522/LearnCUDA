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
        explicit Matrix(const string& path);
        Matrix(const Matrix& other);
        Matrix(Matrix&& other) noexcept;
        ~Matrix();

        static void setVerbose(bool v) { verbose = v; }
        Dims getDims() const { return dims; }
        const uint* getData() const { return data; }

        friend ostream& operator<<(ostream& os, const Matrix& matrix);
        bool operator==(const Matrix& other) const;
        Matrix& operator=(const Matrix& other);
        Matrix& operator=(Matrix&& other) noexcept;

        void clear();
        void print() const;
        void dump(const string& path) const;

    private:
        static bool verbose;
        Dims dims;
        uint* data;
    };
}

#endif //LEARNCUDA_MATRIX_H
