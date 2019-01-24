cmake_minimum_required(VERSION 3.13)
project(LearnCUDA)

set(CMAKE_CXX_STANDARD 11)

if (APPLE)
    set(EIGEN_PATH /Users/lun/Desktop/Code/libs/Eigen)
endif()

include_directories(${EIGEN_PATH})

add_executable(LearnCUDA main.cpp matrix.cpp matrix.h macro.h)