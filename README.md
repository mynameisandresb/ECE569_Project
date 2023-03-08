Source code from https://github.com/Breakend/MotionDetection

A parallelized motion detection implementation of <a href="http://ieeexplore.ieee.org/document/6595847/">Yi et al.’s dual-mode SGM background subtraction model</a>. 

If you use their implementation please cite:

```
@article{henderson2017analysis,
  title={An Analysis of Parallelized Motion Masking Using Dual-Mode Single Gaussian Models},
  author={Henderson, Peter and Vertescher, Matthew},
  journal={arXiv preprint arXiv:1702.05156},
  year={2017}
}
```

The CUDA implementation of our code is in the cuda folder. Both the serial and parallelized tbb versions of our code reside in the same folder: serial_and_tbb. They can both be seen in the main.cpp and DualSGM.cpp files.


Install instructions

# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/3.3.0.zip

unzip opencv.zip

# Create build directory
mkdir -p build && cd build

# Configure
cmake  ../opencv-3.3.0

# Build
cmake --build .

cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local .

make install -j64

#.bashrc file edits, add exports to bashrc
vim $HOME/.bashrc

export PATH=$HOME/.local/include:${PATH}
export LD_LIBRARY_PATH=$HOME/.local/lib
export LIBRARY_PATH=$HOME/.local/lib
export CPLUS_INCLUDE_PATH=$HOME/.local/include
export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig

# source bashrc
source $HOME/.bashrc

# download source code
[https://github.com/zpiernater/ECE569_Project](https://github.com/zpiernater/ECE569_Project.git)

# change to cloned code and relevant part
cd NAMEOFCLONEDCODE
cd serial_and_tbb

# get dataset
http://jacarini.dinf.usherbrooke.ca/static/dataset/dataset2014.zip

# change dir path on line 12 serial_main.cpp to your data

# then do make inside the in directory
make

# run code
./main 1 0 0
