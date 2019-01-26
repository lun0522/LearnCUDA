//
// Created by Pujun Lun on 2019-01-21.
//

#include "matrix.h"

#include <fstream>
#include <random>

#include "macro.h"
#include "matmul.h"

namespace Math {
    bool Matrix::verbose {false};

    Matrix::Matrix(size_t width, size_t height, Mode mode)
        : dims {width, height} {
        size_t size = getSize();
        data = (val_t *)malloc(size * sizeof(val_t));
        switch (mode) {
            case Mode::randFloat: {
                if (verbose)
                    cout << "Generating " << *this << " (random float)" << endl;

                mt19937 randGen{random_device{}()};
                uniform_real_distribution<> dist {0.0, 1.0};
                for (size_t i = 0; i < size; ++i)
                    data[i] = (val_t)dist(randGen);
                break;
            }
            case Mode::randInt: {
                if (verbose)
                    cout << "Generating " << *this << " (random int)" << endl;

                mt19937 randGen{random_device{}()};
                uniform_int_distribution<> dist {0, 9};
                for (size_t i = 0; i < size; ++i)
                    data[i] = (val_t)dist(randGen);
                break;
            }
            case Mode::undefined: {
                if (verbose)
                    cout << "Generating " << *this << " (undefined)" << endl;
                // leave uninitialized
                break;
            }
            case Mode::unit: {
                if (width != height)
                    DEBUG_INFO("Cannot create non-square unit matrix")

                if (verbose)
                    cout << "Generating " << *this << " (unit)" << endl;

                memset(data, 0, size * sizeof(val_t));
                for (size_t i = 0; i < height; ++i)
                    data[i * width + i] = 1;
                break;
            }
            case Mode::zero: {
                if (verbose)
                    cout << "Generating " << *this << " (zero)" << endl;

                memset(data, 0, size * sizeof(val_t));
                break;
            }
        }
    }

    Matrix::Matrix(const MatrixXf& other)
        : dims {(size_t)other.cols(), (size_t)other.rows()} {
        data = (val_t *)malloc(getSize() * sizeof(val_t));
        for (size_t i = 0; i < dims.height; ++i)
            for (size_t j = 0; j < dims.width; ++j)
                (*this)(i, j) = other(i, j);
    }

    Matrix::Matrix(const string &path) {
        if (verbose)
            cout << "Reading matrix from file " << path << endl;

        ifstream ifs {path, ios::in | ios::binary};
        if (!ifs.is_open())
            DEBUG_INFO("Failed to open file " + path)

        size_t buf[2];
        ifs.read((char *)buf, 2 * sizeof(size_t));
        dims = {buf[0], buf[1]};
        if (verbose)
            cout << "Found " << *this << endl;

        size_t size = getSize();
        data = (val_t *)malloc(size * sizeof(val_t));
        ifs.read((char *)data, size * sizeof(val_t));
        ifs.close();
    }

    Matrix::Matrix(const Matrix& other)
        : dims {other.dims} {
        if (verbose)
            cout << "Copying " << other << endl;

        size_t size = getSize();
        data = (val_t *)malloc(size * sizeof(val_t));
        memcpy(data, other.getData(), size * sizeof(val_t));
    }

    Matrix::Matrix(Matrix&& other) noexcept {
        if (verbose)
            cout << "Moving " << other << endl;

        dims = other.dims;
        data = other.data;
        other.data = nullptr;
    }

    Matrix::~Matrix() {
        if (verbose)
            cout << "Destruct " << *this << endl;

        delete data;
    }

    ostream& operator<<(ostream& os, const Matrix& matrix) {
        Matrix::Dims dims = matrix.getDims();
        os << dims.width << "x" << dims.height << " matrix";
        return os;
    }

    bool Matrix::operator==(const Matrix& other) const {
        if (dims.width != other.dims.width || dims.height != other.dims.height)
            DEBUG_INFO("Comparing matrices of different dimensions")

        if (verbose)
            cout << "Comparing " << *this << endl;

        size_t size = getSize();
        for (size_t i = 0; i < size; ++i) {
            if (abs(data[i] - other.getData()[i]) > 1e-3) {
                cout << "Matrices are different" << endl;
                cout << "Left matrix:" << endl;
                print();
                cout << "Right matrix: " << endl;
                other.print();
                return false;
            }
        }

        if (verbose)
            cout << "Matrices are identical" << endl;
        return true;
    }

    Matrix& Matrix::operator=(const Matrix& other) {
        if (dims.width != other.dims.width || dims.height != other.dims.height)
            DEBUG_INFO("Copying matrices of different dimensions")

        if (verbose)
            cout << "Assigning " << *this << endl;
        memcpy(data, other.getData(), dims.width * dims.height * sizeof(val_t));
        return *this;
    }

    Matrix& Matrix::operator=(Matrix&& other) noexcept {
        if (verbose)
            cout << "Moving " << other << " to " << *this << endl;

        swap(dims, other.dims);
        swap(data, other.data);
        return *this;
    }

    Matrix::operator MatrixXf() const {
        MatrixXf ret {dims.height, dims.width};
        for (size_t i = 0; i < dims.height; ++i)
            for (size_t j = 0; j < dims.width; ++j)
                ret(i, j) = (*this)(i, j);
        return ret;
    }

    void Matrix::print() const {
        cout << "Printing " << *this << endl;
        for (size_t i = 0; i < dims.height; ++i) {
            for (size_t j = 0; j < dims.width; ++j) {
                cout.width(4);
                cout << right << (uint)data[i * dims.width + j];
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

        size_t buf[] {dims.width, dims.height};
        ofs.write((char *)buf, 2 * sizeof(size_t));
        ofs.write((char *)data, dims.width * dims.height * sizeof(val_t));
        ofs.close();

        if (verbose)
            cout << "Matrix written to file " << path << endl;
    }

    void matMul(const Matrix& a, const Matrix& b, Matrix& c) {
        if (Matrix::verbose)
            cout << "Multiplying " << a << " and " << b << endl;

        uint m = a.getDims().height == c.getDims().height ?
                 (uint)a.getDims().height : DEBUG_INFO("First dimension not match")
        uint n = b.getDims().width == c.getDims().width ?
                 (uint)b.getDims().width : DEBUG_INFO("Second dimension not match")
        uint k = a.getDims().width == b.getDims().height ?
                 (uint)a.getDims().width : DEBUG_INFO("Third dimension not match")

        if (m % 8 != 0 || n % 8 != 0)
            DEBUG_INFO("Dimension not supported")

        sgemm(m, n, k, a.getData(), b.getData(), c.getData());
    }

    bool verifyMatMul(const Matrix& a, const Matrix& b, const Matrix& c) {
        return Matrix {MatrixXf(a) * MatrixXf(b)} == c;
    }
}
