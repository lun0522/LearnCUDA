//
// Created by Pujun Lun on 2019-01-21.
//

#include "matrix.h"

#include <fstream>
#include <random>

#include "macro.h"

namespace Math {
    bool Matrix::verbose{false};

    Matrix::Matrix(size_t rows, size_t cols, Mode mode)
        : num_row {rows}, num_col {cols}, num_elem {rows * cols} {
        raw_data = (val_t *)malloc(size() * sizeof(val_t));
        switch (mode) {
            case Mode::randFloat: {
                if (verbose)
                    cout << "Generating " << *this << " (random float)" << endl;

                mt19937 randGen{random_device{}()};
                uniform_real_distribution<> dist {0.0, 1.0};
                for (size_t i = 0; i < size(); ++i)
                    data()[i] = (val_t)dist(randGen);
                break;
            }
            case Mode::randInt: {
                if (verbose)
                    cout << "Generating " << *this << " (random int)" << endl;

                mt19937 randGen{random_device{}()};
                uniform_int_distribution<> dist {0, 9};
                for (size_t i = 0; i < size(); ++i)
                    data()[i] = (val_t)dist(randGen);
                break;
            }
            case Mode::undefined: {
                if (verbose)
                    cout << "Generating " << *this << " (undefined)" << endl;
                // leave uninitialized
                break;
            }
            case Mode::unit: {
                if (rows != cols)
                    DEBUG_INFO("Cannot create non-square unit matrix")

                if (verbose)
                    cout << "Generating " << *this << " (unit)" << endl;

                memset(data(), 0, size() * sizeof(val_t));
                for (size_t r = 0; r < rows; ++r)
                    data()[r * cols + r] = 1;
                break;
            }
            case Mode::zero: {
                if (verbose)
                    cout << "Generating " << *this << " (zero)" << endl;

                clear();
                break;
            }
        }
    }

    Matrix::Matrix(const MatrixXf& other)
        : num_row {(size_t)other.rows()}, num_col {(size_t)other.cols()} {
        num_elem = rows() * cols();
        raw_data = (val_t *)malloc(size() * sizeof(val_t));
        for (size_t r = 0; r < rows(); ++r)
            for (size_t c = 0; c < cols(); ++c)
                (*this)(r, c) = other(r, c);
    }

    Matrix::Matrix(const string &path) {
        if (verbose)
            cout << "Reading matrix from file " << path << endl;

        ifstream ifs {path, ios::in | ios::binary};
        if (!ifs.is_open())
            DEBUG_INFO("Failed to open file " + path)

        size_t buf[2];
        ifs.read((char *)buf, 2 * sizeof(size_t));
        num_row = buf[0];
        num_col = buf[1];
        num_elem = rows() * cols();
        if (verbose)
            cout << "Found " << *this << endl;

        raw_data = (val_t *)malloc(size() * sizeof(val_t));
        ifs.read((char *)data(), size() * sizeof(val_t));
        ifs.close();
    }

    Matrix::Matrix(const Matrix& other)
        : num_row {other.rows()}, num_col {other.cols()}, num_elem {other.size()} {
        if (verbose)
            cout << "Copying " << other << endl;

        raw_data = (val_t *)malloc(size() * sizeof(val_t));
        memcpy(data(), other.data(), size() * sizeof(val_t));
    }

    Matrix::Matrix(Matrix&& other) noexcept
        : num_row {other.rows()}, num_col {other.cols()}, num_elem {other.size()} {
        if (verbose)
            cout << "Moving " << other << endl;

        raw_data = other.data();
        other.raw_data = nullptr;
    }

    Matrix::~Matrix() {
        if (verbose)
            cout << "Destruct " << *this << endl;

        delete data();
    }

    bool Matrix::operator==(const Matrix& other) const {
        if (rows() != other.rows() || cols() != other.cols())
            DEBUG_INFO("Comparing matrices of different dimensions")

        if (verbose)
            cout << "Comparing " << *this << endl;

        for (size_t r = 0; r < rows(); ++r) {
            for (size_t c = 0; c < cols(); ++c) {
                if (abs((*this)(r, c) - other(r, c)) > 1e-3) {
                    cout << "Matrices are different at (" << r << ", " << c << ")" << endl;
                    cout << "Left matrix:" << endl;
                    print();
                    cout << "Right matrix: " << endl;
                    other.print();
                    return false;
                }
            }
        }

        if (verbose)
            cout << "Matrices are identical" << endl;
        return true;
    }

    Matrix& Matrix::operator=(const Matrix& other) {
        if (rows() != other.rows() || cols() != other.cols())
            DEBUG_INFO("Copying matrices of different dimensions")

        if (verbose)
            cout << "Assigning " << *this << endl;
        memcpy(data(), other.data(), size() * sizeof(val_t));
        return *this;
    }

    Matrix& Matrix::operator=(Matrix&& other) noexcept {
        if (verbose)
            cout << "Moving " << other << " to " << *this << endl;

        num_row = other.rows();
        num_col = other.cols();
        num_elem = other.size();
        swap(raw_data, other.raw_data);
        return *this;
    }

    Matrix::operator MatrixXf() const {
        MatrixXf ret {rows(), cols()};
        for (size_t r = 0; r < rows(); ++r)
            for (size_t c = 0; c < cols(); ++c)
                ret(r, c) = (*this)(r, c);
        return ret;
    }

    ostream& operator<<(ostream& os, const Matrix& matrix) {
        os << matrix.rows() << "x" << matrix.cols() << " matrix";
        return os;
    }

    void Matrix::clear() const {
        memset(data(), 0, size() * sizeof(val_t));
    }

    void Matrix::print() const {
        cout << "Printing " << *this << endl;
        for (size_t r = 0; r < rows(); ++r) {
            for (size_t c = 0; c < cols(); ++c) {
                cout.width(4);
                cout << right << (uint)data()[r * cols() + c];
            }
            cout << endl;
        }
    }

    void Matrix::dump(const string& path) const {
        if (verbose)
            cout << "Writing matrix to file " << path << endl;

        ofstream ofs {path, ios::out | ios::binary};
        if (!ofs.is_open())
            DEBUG_INFO("Failed to open file " + path)

        size_t buf[] {rows(), cols()};
        ofs.write((char *)buf, 2 * sizeof(size_t));
        ofs.write((char *)data(), size() * sizeof(val_t));
        ofs.close();

        if (verbose)
            cout << "Matrix written to file " << path << endl;
    }
}
