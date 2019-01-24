//
// Created by Pujun Lun on 2019-01-21.
//

#include "matrix.h"

#include <fstream>
#include <random>

#include "macro.h"

namespace Math {
    bool Matrix::verbose {false};

    Matrix::Matrix(uint width, uint height, Mode mode):
        dims {width, height} {
        uint size = width * height;
        data = (uint *)malloc(size * sizeof(uint));
        switch (mode) {
            case Mode::random: {
                if (verbose)
                    cout << "Generating " << *this << " (random)" << endl;

                mt19937 randGen;
                randGen.seed(random_device{}());
                uniform_int_distribution<mt19937::result_type> rand10 {0, 9};
                for (uint i = 0; i < size; ++i)
                    data[i] = rand10(randGen);
                break;
            }
            case Mode::unit: {
                if (width != height)
                    throw runtime_error("Cannot create non-square unit matrix");

                if (verbose)
                    cout << "Generating " << *this << " (unit)" << endl;

                memset(data, 0, size * sizeof(uint));
                for (uint i = 0; i < height; ++i)
                    data[i * width + i] = 1;
                break;
            }
            case Mode::zero: {
                if (verbose)
                    cout << "Generating " << *this << " (zero)" << endl;

                memset(data, 0, size * sizeof(uint));
                break;
            }
        }
    }

    Matrix::Matrix(const string &path) {
        if (verbose)
            cout << "Reading matrix from file " << path << endl;

        ifstream ifs {path, ios::in | ios::binary};
        if (!ifs.is_open())
            throw runtime_error(DEBUG_INFO("Failed to open file " + path));

        uint buf[2];
        ifs.read((char *)buf, 2 * sizeof(uint));
        dims = {buf[0], buf[1]};
        if (verbose)
            cout << "Found " << *this << endl;

        uint size = dims.width * dims.height;
        data = (uint *)malloc(size * sizeof(uint));
        ifs.read((char *)data, size * sizeof(uint));
        ifs.close();
    }

    Matrix::Matrix(const Matrix& other): dims {other.dims} {
        if (verbose)
            cout << "Copying " << other << endl;

        uint size = dims.width * dims.height;
        data = (uint *)malloc(size * sizeof(uint));
        copy(other.data, other.data + size, data);
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

        uint size = dims.width * dims.height;
        for (int i = 0; i < size; ++i) {
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
        copy(other.data, other.data + dims.width * dims.height, data);
        return *this;
    }

    Matrix& Matrix::operator=(Matrix&& other) noexcept {
        if (verbose)
            cout << "Moving " << other << " to " << *this << endl;

        swap(dims, other.dims);
        swap(data, other.data);
        return *this;
    }

    void Matrix::clear() {
        memset(data, 0, dims.width * dims.height * sizeof(uint));
    }

    void Matrix::print() const {
        streamsize ssize = to_string(dims.width * dims.height - 1).length() + 2;
        cout << "Printing " << *this << endl;
        for (uint i = 0; i < dims.height; ++i) {
            for (uint j = 0; j < dims.width; ++j) {
                cout.width(ssize);
                cout << right << data[i * dims.width + j];
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

        uint buf[] = {dims.width, dims.height};
        ofs.write((char *)buf, 2 * sizeof(uint));
        ofs.write((char *)data, dims.width * dims.height * sizeof(uint));
        ofs.close();

        if (verbose)
            cout << "Matrix written to file " << path << endl;
    }
}
