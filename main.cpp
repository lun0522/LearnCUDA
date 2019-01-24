#include <iostream>

#include "macro.h"
#include "matrix.h"

using namespace std;
using namespace Math;

void testGenMatrix() {
    BEGIN_TEST
    Matrix m {16, 16, Matrix::Mode::random};
    m.print();
    Matrix n {16, 16, Matrix::Mode::unit};
    n.print();
    Matrix p {16, 16, Matrix::Mode::zero};
    p.print();
}

void testReadWrite() {
    BEGIN_TEST
    const string path = "mat";
    Matrix m {16, 16};
    m.print();
    m.dump(path);
    Matrix n {path};
    n.print();

    assert(m == n);
}

void testCmp() {
    BEGIN_TEST
    Matrix m {16, 16};
    Matrix n {15, 15};
    try {
        if (m == n) {}
        assert(1);
    } catch (const exception& e) {
        cout << e.what() << " (Expected behavior)" << endl;
    }
}

void testCopy() {
    BEGIN_TEST
    Matrix m {16, 16};
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
    Matrix m {16, 16, Matrix::Mode::random};
    Matrix n {16, 16, Matrix::Mode::zero};
    n = m;
    m.print();
    n.print();

    assert(m == n);
}

void testMove() {
    BEGIN_TEST
    Matrix m {16, 16};
    Matrix n {15, 15};
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
//        testAll();
        Matrix::setVerbose(true);
        testMove();
    } catch (const exception& e) {
        cout << e.what() << endl;
        exit(-1);
    }

    return 0;
}