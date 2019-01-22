//
// Created by Pujun Lun on 2019-01-21.
//

#include "util.h"

#include <fstream>
#include <iostream>

namespace Util {

    using namespace std;

    void PrintMatrix(UInt* matrix,
                     UInt width,
                     UInt height) {
        streamsize ssize = to_string(width * height - 1).length() + 2;
        for (UInt i = 0; i < height; ++i) {
            for (UInt j = 0; j < width; ++j) {
                cout.width(ssize);
                cout << right << matrix[i * width + j];
            }
            cout << endl;
        }
    }

    void GenMatrixFile(const string& filename,
                       UInt width,
                       UInt height,
                       bool verbose) {
        if (verbose)
            cout << "Generating a " << width << "x" << height << " matrix" << endl;

        ofstream fs {filename, ios::out | ios::binary};
        if (!fs.is_open())
            throw runtime_error("Failed to open file " + filename);

        UInt size = width * height;
        UInt* matrix = (UInt *)malloc(size * sizeof(UInt));
        for (UInt i = 0; i < height; ++i)
            for (UInt j = 0; j < width; ++j)
                matrix[i * width + j] = i + j + 1;
        PrintMatrix(matrix, width, height);

        fs.write((char *)matrix, size * sizeof(UInt));
        fs.close();
        free(matrix);
        if (verbose)
            cout << "Matrix written to file " << filename << endl;
    }

    UInt* ReadMatrixFile(const string& filename,
                         UInt width,
                         UInt height,
                         bool verbose) {
        if (verbose)
            cout << "Reading matrix from file " << filename << endl;

        ifstream fs {filename, ios::in | ios::binary};
        if (!fs.is_open())
            throw runtime_error("Failed to open file " + filename);

        UInt size = width * height;
        UInt* matrix = (UInt *)malloc(size * sizeof(UInt));
        fs.read((char *)matrix, size * sizeof(UInt));
        fs.close();
        if (verbose) PrintMatrix(matrix, width, height);

        return matrix;
    }

}
