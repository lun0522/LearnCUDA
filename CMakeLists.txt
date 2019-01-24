cmake_minimum_required(VERSION 3.10)
project(LearnCUDA)

set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (APPLE)
    set(EIGEN_PATH /Users/lun/Desktop/Code/libs/eigen)
else()
    set(EIGEN_PATH /home/lpj0522/eigen)
endif()

include_directories(${EIGEN_PATH})

add_executable(LearnCUDA main.cpp matrix.cpp matrix.h macro.h)