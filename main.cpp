#include <iostream>

#include "matrix.h"

using namespace std;
using namespace Math;

void testReadWrite() {
    const string path = "mat";
    Matrix m {16, 16};
    m.print();
    m.dump(path);
    Matrix n {path};
    n.print();

    assert(m == n);
}

void testCmp() {
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
    Matrix m {16, 16};
    Matrix n {16, 16};
    m.print();
    n.clear();
    n.print();
    n = m;
    n.print();

    assert(m == n);
}

void testMove() {
    Matrix m {16, 16};
    Matrix n {15, 15};
    n = move(m);
    m.print();
    n.print();

    assert(m.getDims().width  == 15);
    assert(n.getDims().height == 16);
}

int main() {
    try {
        Matrix::setVerbose(true);
        testReadWrite();
        testCmp();
        testCopy();
        testAssign();
        testMove();
    } catch (const exception& e) {
        cout << e.what() << endl;
        exit(-1);
    }

    return 0;
}