#! /bin/bash

if [ -z "$1" ]
  then
    FAST=0
else
    FAST=1
fi

export PATH=/contrib/ece569/group1/.local/include:${PATH}
export LD_LIBRARY_PATH=/contrib/ece569/group1/.local/lib
export LIBRARY_PATH=/contrib/ece569/group1/.local/lib
export CPLUS_INCLUDE_PATH=/contrib/ece569/group1/.local/include
export PKG_CONFIG_PATH=/contrib/ece569/group1/.local/lib/pkgconfig
export OpenCV_DIR=/contrib/ece569/group1/.local/share/OpenCV/

./bin/BGS $FAST