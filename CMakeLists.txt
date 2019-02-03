cmake_minimum_required(VERSION 3.10)
project(LearnCUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_37,code=sm_37 -std=c++11")

if (APPLE)
    set(CUDA_INCLUDE /Developer/NVIDIA/CUDA-10.0/include/)
else()
    set(CUDA_PATH /usr/local/cuda-10.0)
    set(CUDA_INCLUDE ${CUDA_PATH}/include)
endif()

include_directories(${CUDA_INCLUDE})

cuda_add_executable(LearnCUDA main.cpp matrix.cpp matrix.h macro.h matmul.cu matmul.h matmul_impl.cu matmul_impl.h)
target_link_libraries(LearnCUDA -L${CUDA_PATH}/lib64 cublas)