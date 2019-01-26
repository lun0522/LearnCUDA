#!/usr/bin/env bash

export CC=clang
export CXX=clang++

cd $(dirname $0)
BUILD_DIR="build"
if [[ ! -d ${BUILD_DIR} ]]; then
    mkdir ${BUILD_DIR}
fi
cd ${BUILD_DIR}

# load cudart dylib
if [[ $OSTYPE == darwin* ]]; then
    export DYLD_LIBRARY_PATH=/usr/local/cuda/lib
fi

cmake ..
cmake --build .
cd "bin"
./LearnCUDA
