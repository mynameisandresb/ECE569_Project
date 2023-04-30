#! /bin/bash

# Put open in path
export PATH=/contrib/ece569/group1/.local/include:${PATH}
export LD_LIBRARY_PATH=/contrib/ece569/group1/.local/lib
export LIBRARY_PATH=/contrib/ece569/group1/.local/lib
export CPLUS_INCLUDE_PATH=/contrib/ece569/group1/.local/include
export PKG_CONFIG_PATH=/contrib/ece569/group1/.local/lib/pkgconfig
export OpenCV_DIR=/contrib/ece569/group1/.local/share/OpenCV/


# Disable and enable fast mode
if [ -z "$1" ]
  then
    ./bin/BGS 0 1
else
    ./bin/BGS 1 1
fi