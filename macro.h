//
// Created by Pujun Lun on 2019-01-22.
//

#ifndef LEARNCUDA_MACRO_H
#define LEARNCUDA_MACRO_H

#include <iostream>
#include <string>

#define BEGIN_TEST          std::cout << "Running " << __func__ << std::endl;
#define DEBUG_INFO(info)    throw std::runtime_error(std::string{"Fatal error!\n"} + \
    __func__ + "() in file " + \
    __FILE__ + " on line " + std::to_string(__LINE__) + \
    ": \n\t" + info);

#endif //LEARNCUDA_MACRO_H
