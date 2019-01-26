//
// Created by Pujun Lun on 2019-01-22.
//

#ifndef LEARNCUDA_MACRO_H
#define LEARNCUDA_MACRO_H

#define BEGIN_TEST          cout << __func__ << endl;
#define DEBUG_INFO(info)    throw runtime_error(string{"Fatal error!\n"} + \
    __func__ + "() in file " + \
    __FILE__ + " on line " + to_string(__LINE__) + \
    ": \n\t" + info);

#endif //LEARNCUDA_MACRO_H
