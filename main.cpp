#include <Dense>
#include <iostream>

#include "macro.h"
#include "matmul.h"
#include "matrix.h"

using namespace std;
using namespace Math;

void testGenMatrix() {
    BEGIN_TEST
    Matrix m {16, 16, Matrix::Mode::randInt};
    m.print();
    Matrix n {16, 16, Matrix::Mode::unit};
    n.print();
    Matrix p {16, 16, Matrix::Mode::zero};
    p.print();
}

void testReadWrite() {
    BEGIN_TEST
    const string path = "mat";
    Matrix m {16, 16, Matrix::Mode::randInt};
    m.print();
    m.dump(path);
    Matrix n {path};
    n.print();

    assert(m == n);
}

void testCmp() {
    BEGIN_TEST
    Matrix m {16, 16, Matrix::Mode::randFloat};
    Matrix q {15, 15, Matrix::Mode::undefined};
    try {
        if (m == q) {}
        assert(0);
    } catch (const exception& e) {
        cout << e.what() << " (Expected behavior)" << endl;
    }

    Matrix n {16, 16, Matrix::Mode::unit};
    assert(verifyMatMul(m, n, m));
}

void testCopy() {
    BEGIN_TEST
    Matrix m {16, 16, Matrix::Mode::randInt};
    Matrix n {m};
    m.print();
    n.print();
    cout << "m: " << m.getData() << endl;
    cout << "n: " << n.getData() << endl;

    assert(m == n);
    assert(m.getData() != n.getData());
}

void testAssign() {
    BEGIN_TEST
    Matrix m {16, 16, Matrix::Mode::randInt};
    Matrix n {16, 16, Matrix::Mode::undefined};
    n = m;
    m.print();
    n.print();

    assert(m == n);
}

void testMove() {
    BEGIN_TEST
    Matrix m {16, 16, Matrix::Mode::randInt};
    Matrix n {15, 15, Matrix::Mode::randInt};
    n = move(m);
    m.print();
    n.print();

    assert(m.getDims().width  == 15);
    assert(n.getDims().height == 16);
}

void testAll() {
    Matrix::setVerbose(true);
    for (auto func : {testGenMatrix, testReadWrite, testCmp,
                      testCopy, testAssign, testMove}) {
        func();
        cout << endl;
    }
}

int main() {
    try {
        testAll();
    } catch (const exception& e) {
        cout << e.what() << endl;
        exit(-1);
    }

    return 0;
}