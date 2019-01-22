#include <iostream>

#include "matrix.h"

using namespace std;
using namespace Math;

int main() {
    try {
        const UInt width = 16, height = 16;
        const string path = "mat";

        setVerbose(true);
        Matrix m {width, height};
        m.dump(path);
        Matrix n {path};
    } catch (const exception& e) {
        cout << e.what() << endl;
        exit(-1);
    }

    return 0;
}