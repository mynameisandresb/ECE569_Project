#! /bin/bash

# Disable and enable fast mode
if [ -z "$1" ]
  then
    FAST=0
else
    FAST="$1"
fi

# Disable and enable display mode
if [ -z "$2" ]
  then
    DISPLAY_MODE=0
else
    DISPLAY_MODE="$2"
fi

# Put open in path
export PATH=/contrib/ece569/group1/.local/include:${PATH}
export LD_LIBRARY_PATH=/contrib/ece569/group1/.local/lib
export LIBRARY_PATH=/contrib/ece569/group1/.local/lib
export CPLUS_INCLUDE_PATH=/contrib/ece569/group1/.local/include
export PKG_CONFIG_PATH=/contrib/ece569/group1/.local/lib/pkgconfig
export OpenCV_DIR=/contrib/ece569/group1/.local/share/OpenCV/

./bin/BGS $FAST $DISPLAY_MODE