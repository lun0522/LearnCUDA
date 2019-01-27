#!/usr/bin/env bash

cd $(dirname $0)
BUILD_DIR="build"
if [[ ! -d ${BUILD_DIR} ]]; then
    mkdir ${BUILD_DIR}
fi
cd ${BUILD_DIR}

cmake ..
cmake --build .

if [[ $1 == "e" ]]; then
    cd "bin"
    ./LearnCUDA
fi