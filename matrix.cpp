//
// Created by Pujun Lun on 2019-01-21.
//

#include "matrix.h"

#include <fstream>
#include <iostream>

using namespace std;

bool Matrix::verbose = true;

Matrix::Matrix(UInt width, Matrix::UInt height): dims {width, height} {
    if (verbose)
        cout << "Generating " << width << "x" << height << " matrix" << endl;

    data = (UInt *)malloc(width * height * sizeof(UInt));
    for (UInt i = 0; i < height; ++i)
        for (UInt j = 0; j < width; ++j)
            data[i * width + j] = i + j + 1;
    if (verbose) print();
}

Matrix::Matrix(const string &path) {
    if (verbose)
        cout << "Reading matrix from file " << path << endl;

    ifstream fs {path, ios::in | ios::binary};
    if (!fs.is_open())
        throw runtime_error("Failed to open file " + path);

    UInt buf[2];
    fs.read((char *)buf, 2 * sizeof(UInt));
    dims = {buf[0], buf[1]};
    if (verbose)
        cout << "Found " << dims.width << "x" << dims.height << " matrix" << endl;

    UInt size = dims.width * dims.height;
    data = (UInt *)malloc(size * sizeof(UInt));
    fs.read((char *)data, size * sizeof(UInt));
    fs.close();
    if (verbose) print();
}

Matrix::~Matrix() {
    if (verbose)
        cout << "Destruct " << dims.width << "x" << dims.height << " matrix" << endl;
    free(data);
}

Matrix::Dims Matrix::getDims() const {
    return dims;
}

const Matrix::UInt* Matrix::getData() const {
    return data;
}

void Matrix::print() const {
    streamsize ssize = to_string(dims.width * dims.height - 1).length() + 2;
    for (UInt i = 0; i < dims.height; ++i) {
        for (UInt j = 0; j < dims.width; ++j) {
            cout.width(ssize);
            cout << right << data[i * dims.width + j];
        }
        cout << endl;
    }
}

void Matrix::dump(const std::string &path) const {
    if (verbose)
        cout << "Writing matrix to file " << path << endl;

    ofstream fs {path, ios::out | ios::binary};
    if (!fs.is_open())
        throw runtime_error("Failed to open file " + path);

    UInt buf[] = {dims.width, dims.height};
    fs.write((char *)buf, 2 * sizeof(UInt));
    fs.write((char *)data, dims.width * dims.height * sizeof(UInt));
    fs.close();

    if (verbose)
        cout << "Matrix written to file " << path << endl;
}
