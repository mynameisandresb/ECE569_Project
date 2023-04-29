#! /bin/bash

# Put open in path
export PATH=/contrib/ece569/group1/.local/include:${PATH}
export LD_LIBRARY_PATH=/contrib/ece569/group1/.local/lib
export LIBRARY_PATH=/contrib/ece569/group1/.local/lib
export CPLUS_INCLUDE_PATH=/contrib/ece569/group1/.local/include
export PKG_CONFIG_PATH=/contrib/ece569/group1/.local/lib/pkgconfig
export OpenCV_DIR=/contrib/ece569/group1/.local/share/OpenCV/

rm -r build
mkdir build
cd build
module load cuda11/11.0
CC=gcc cmake3 ..
make clean
make