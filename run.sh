BASE_DIR=$(dirname $0)
cd $BASE_DIR

BUILD_DIR="build"
if [ ! -d $BUILD_DIR ]; then
    mkdir $BUILD_DIR
fi
cd $BUILD_DIR

cmake ..
make
cd "bin"
./LearnCUDA
