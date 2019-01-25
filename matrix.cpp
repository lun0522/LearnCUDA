//
// Created by Pujun Lun on 2019-01-21.
//

#include "matrix.h"

#include <fstream>
#include <random>

#include "macro.h"

namespace Math {
    bool Matrix::verbose {false};

    Matrix::Matrix(dim_t width, dim_t height, Mode mode)
        : dims {width, height} {
        dim_t size = width * height;
        data = (val_t *)malloc(size * sizeof(val_t));
        switch (mode) {
            case Mode::randFloat: {
                if (verbose)
                    cout << "Generating " << *this << " (random float)" << endl;

                mt19937 randGen{random_device{}()};
                uniform_real_distribution<> dist {0.0, 1.0};
                for (dim_t i = 0; i < size; ++i)
                    data[i] = (val_t)dist(randGen);
                break;
            }
            case Mode::randInt: {
                if (verbose)
                    cout << "Generating " << *this << " (random int)" << endl;

                mt19937 randGen{random_device{}()};
                uniform_int_distribution<> dist {0, 9};
                for (dim_t i = 0; i < size; ++i)
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
                    throw runtime_error("Cannot create non-square unit matrix");

                if (verbose)
                    cout << "Generating " << *this << " (unit)" << endl;

                memset(data, 0, size * sizeof(val_t));
                for (dim_t i = 0; i < height; ++i)
                    data[i * width + i] = (val_t)1.0;
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
        : dims {(dim_t)other.cols(), (dim_t)other.rows()} {
        dim_t size = dims.width * dims.height;
        data = (val_t *)malloc(size * sizeof(val_t));
        for (dim_t i = 0; i < dims.height; ++i)
            for (dim_t j = 0; j < dims.width; ++j)
                (*this)(i, j) = other(i, j);
    }

    Matrix::Matrix(const string &path) {
        if (verbose)
            cout << "Reading matrix from file " << path << endl;

        ifstream ifs {path, ios::in | ios::binary};
        if (!ifs.is_open())
            throw runtime_error(DEBUG_INFO("Failed to open file " + path));

        dim_t buf[2];
        ifs.read((char *)buf, 2 * sizeof(dim_t));
        dims = {buf[0], buf[1]};
        if (verbose)
            cout << "Found " << *this << endl;

        dim_t size = dims.width * dims.height;
        data = (val_t *)malloc(size * sizeof(val_t));
        ifs.read((char *)data, size * sizeof(val_t));
        ifs.close();
    }

    Matrix::Matrix(const Matrix& other)
        : dims {other.dims} {
        if (verbose)
            cout << "Copying " << other << endl;

        dim_t size = dims.width * dims.height;
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
            throw runtime_error(DEBUG_INFO("Comparing matrices of different dimensions"));

        if (verbose)
            cout << "Comparing " << *this << endl;

        dim_t size = dims.width * dims.height;
        for (dim_t i = 0; i < size; ++i) {
            if (data[i] != other.getData()[i]) {
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
            throw runtime_error(DEBUG_INFO("Copying matrices of different dimensions"));

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
        for (dim_t i = 0; i < dims.height; ++i)
            for (dim_t j = 0; j < dims.width; ++j)
                ret(i, j) = (*this)(i, j);
        return ret;
    }

    void Matrix::print() const {
        cout << "Printing " << *this << endl;
        for (dim_t i = 0; i < dims.height; ++i) {
            for (dim_t j = 0; j < dims.width; ++j) {
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
            throw runtime_error(DEBUG_INFO("Failed to open file " + path));

        dim_t buf[] {dims.width, dims.height};
        ofs.write((char *)buf, 2 * sizeof(val_t));
        ofs.write((char *)data, dims.width * dims.height * sizeof(val_t));
        ofs.close();

        if (verbose)
            cout << "Matrix written to file " << path << endl;
    }
}
