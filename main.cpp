#include <iostream>

#include "util.h"

using namespace std;
using namespace Util;

int main() {
    try {
        const UInt width = 16, height = 16;
        const string filename = "matrix";
        const bool verbose = true;
        GenMatrixFile(filename, width, height, verbose);
        ReadMatrixFile(filename, width, height, verbose);
    } catch (const exception& e) {
        cout << e.what() << endl;
        exit(-1);
    }

    return 0;
}