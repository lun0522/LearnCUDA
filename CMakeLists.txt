cmake_minimum_required(VERSION 3.13)
project(LearnCUDA)

set(CMAKE_CXX_STANDARD 11)

add_executable(LearnCUDA main.cpp matrix.cpp matrix.h)