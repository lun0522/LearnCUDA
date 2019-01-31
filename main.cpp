#include <assert.h>
#include <iostream>

#include "macro.h"
#include "matmul.h"
#include "matrix.h"

using namespace std;
using namespace Math;

void testGenMatrix() {
    BEGIN_TEST
    Matrix m{16, 16, Matrix::Mode::randInt};
    m.print();
    Matrix n{16, 16, Matrix::Mode::unit};
    n.print();
    Matrix p{16, 16, Matrix::Mode::zero};
    p.print();
}

void testReadWrite() {
    BEGIN_TEST
    const string path = "mat";
    Matrix m{8, 16, Matrix::Mode::randInt};
    m.print();
    m.dump(path);
    Matrix n{path};
    n.print();

    assert(m == n);
}

void testCmp() {
    BEGIN_TEST
    Matrix m{16, 16, Matrix::Mode::randInt};
    Matrix q{15, 15, Matrix::Mode::undefined};
    try {
        if (m == q) {}
        assert(0);
    } catch (const exception &e) {
        cout << e.what() << " (Expected behavior)" << endl;
    }

    Matrix n{16, 16, Matrix::Mode::unit};
    Matrix p = blasMatMul(m, n);
    assert(m == p);
}

void testCopy() {
    BEGIN_TEST
    Matrix m{16, 16, Matrix::Mode::randInt};
    Matrix n{m};
    m.print();
    n.print();
    cout << "m: " << m.data() << endl;
    cout << "n: " << n.data() << endl;

    assert(m == n);
    assert(m.data() != n.data());
}

void testAssign() {
    BEGIN_TEST
    Matrix m{16, 16, Matrix::Mode::randInt};
    Matrix n{16, 16, Matrix::Mode::undefined};
    n = m;
    m.print();
    n.print();

    assert(m == n);
}

void testMove() {
    BEGIN_TEST
    Matrix m{16, 16, Matrix::Mode::randInt};
    Matrix n{15, 15, Matrix::Mode::randInt};
    n = move(m);
    n.print();

    assert(n.rows() == 16);
}

void testMatMul() {
    const size_t dim = 1024;
    Matrix m{dim, dim, Matrix::Mode::randFloat};
    Matrix n{dim, dim, Matrix::Mode::randFloat};
    testMatMul(m, n, MatMulAlgoA | MatMulAlgoB);
}

void testAll() {
    Matrix::verbose = true;
    for (auto func : {testGenMatrix, testReadWrite, testCmp, testCopy,
                      testAssign, testMove, testMatMul}) {
        func();
        cout << endl;
    }
    cout << "All tests passed" << endl << endl;
}

int main() {
    try {
        testAll();
    } catch (const exception &e) {
        cout << e.what() << endl;
        exit(-1);
    }

    return 0;
}