//
// Created by Pujun Lun on 2019-01-21.
//

#include "matrix.h"

#include <fstream>

#define DEBUG_INFO(x)   string{"Fatal error! "} + __func__ + "() in file " + \
__FILE__ + " on line " + to_string(__LINE__) + ": " + x

namespace Math {
    Matrix::Matrix(UInt width, UInt height): dims {width, height} {
        if (verbose)
            cout << "Generating " << *this << endl;

        data = (UInt *)malloc(width * height * sizeof(UInt));
        for (UInt i = 0; i < height; ++i)
            for (UInt j = 0; j < width; ++j)
                data[i * width + j] = i + j + 1;

        if (verbose) print();
    }

    Matrix::Matrix(const string &path) {
        if (verbose)
            cout << "Reading matrix from file " << path << endl;

        ifstream ifs {path, ios::in | ios::binary};
        if (!ifs.is_open())
            throw runtime_error(DEBUG_INFO("Failed to open file " + path));

        UInt buf[2];
        ifs.read((char *)buf, 2 * sizeof(UInt));
        dims = {buf[0], buf[1]};
        if (verbose)
            cout << "Found " << *this << " matrix" << endl;

        UInt size = dims.width * dims.height;
        data = (UInt *)malloc(size * sizeof(UInt));
        ifs.read((char *)data, size * sizeof(UInt));
        ifs.close();

        if (verbose) print();
    }

    Matrix::~Matrix() {
        if (verbose)
            cout << "Destruct " << *this << endl;
        free(data);
    }

    Matrix::Dims Matrix::getDims() const {
        return dims;
    }

    const UInt* Matrix::getData() const {
        return data;
    }

    ostream& operator<<(ostream& os, const Matrix& matrix) {
        Matrix::Dims dims = matrix.getDims();
        os << dims.width << "x" << dims.height << " matrix";
        return os;
    }

    void Matrix::print() const {
        streamsize ssize = to_string(dims.width * dims.height - 1).length() + 2;
        cout << "Printing " << *this << endl;
        for (UInt i = 0; i < dims.height; ++i) {
            for (UInt j = 0; j < dims.width; ++j) {
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

        UInt buf[] = {dims.width, dims.height};
        ofs.write((char *)buf, 2 * sizeof(UInt));
        ofs.write((char *)data, dims.width * dims.height * sizeof(UInt));
        ofs.close();

        if (verbose)
            cout << "Matrix written to file " << path << endl;
    }
}
