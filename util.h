//
// Created by Pujun Lun on 2019-01-21.
//

#ifndef LEARNCUDA_UTIL_H
#define LEARNCUDA_UTIL_H

#include <string>

namespace Util {

    using UInt = unsigned int;

    void PrintMatrix(UInt* matrix,
                     UInt width,
                     UInt height);

    void GenMatrixFile(const std::string& filename,
                       UInt width,
                       UInt height,
                       bool verbose);

    UInt* ReadMatrixFile(const std::string& filename,
                         UInt width,
                         UInt height,
                         bool verbose);

}

#endif //LEARNCUDA_UTIL_H
