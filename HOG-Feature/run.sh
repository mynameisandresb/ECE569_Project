#! /bin/bash

export PATH=/contrib/ece569/group1/.local/include:${PATH}
export LD_LIBRARY_PATH=/contrib/ece569/group1/.local/lib
export LIBRARY_PATH=/contrib/ece569/group1/.local/lib
export CPLUS_INCLUDE_PATH=/contrib/ece569/group1/.local/include
export PKG_CONFIG_PATH=/contrib/ece569/group1/.local/lib/pkgconfig
export OpenCV_DIR=/contrib/ece569/group1/.local/share/OpenCV/

cd bin 

if [ -z "$1" ]
  then
    ./HOG 0 0 0 0 0
else
    ./HOG 2 2 3 1 1
fi

