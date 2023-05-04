Source code for motion detection https://github.com/Breakend/MotionDetection

Source code for HOG https://github.com/cchinmai19/GPU-Parallel-Programming

HOG-Feature PDF From Source Code:

https://github.com/cchinmai19/GPU-Parallel-Programming/blob/master/HOG-Feature/Hog%20Feature%20Extractor_GPU.pdf

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


# README.md for running respective code bases is located in ./cuda and ./HOG-Feature

# Make sure terminal is in either the cuda directory or the HOG-Feature directory to properly execute their README

# Below code is if you want to run optimized versions only, follow these steps exactly (must be on hpc if you want it to just work)

# Follow README in each directory however for more detailed instructions
```
# Execute cuda kernels
cd cuda
./build.sh
./run.sh 1

cd ..

# Execute hog kernels
cd HOG-Feature
./build.sh
./run.sh 1
```


# Install instructions if running outside hpc node


# Download and unpack sources
```
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/3.3.0.zip

unzip opencv.zip
```

# Create build directory
```
mkdir -p build && cd build
```

# Configure
```
cmake  ../opencv-3.3.0
```

# Build
```
cmake --build .

cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local .

make install -j64
```

# Add exports to bashrc
```
vim $HOME/.bashrc
```
```
export PATH=$HOME/.local/include:${PATH}
export LD_LIBRARY_PATH=$HOME/.local/lib
export LIBRARY_PATH=$HOME/.local/lib
export CPLUS_INCLUDE_PATH=$HOME/.local/include
export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig
```

# Source bashrc
```
source $HOME/.bashrc
```

# Download source code
[https://github.com/zpiernater/ECE569_Project](https://github.com/zpiernater/ECE569_Project.git)

# Change directory to relevant cloned code
```
cd ECE569_Project
cd serial_and_tbb
```

# get dataset
http://jacarini.dinf.usherbrooke.ca/static/dataset/dataset2014.zip

# change dir path on line 12 main.cpp to your data
```
...CODE SNIPPET FROM serial_and_tbb/main.cpp...

// Specify path to image set 
// (ex: badminton, boulevard, sofa, traffic)
const char* PATH = "../Videos/sofa/input/";
```

# Make inside serial_and_tbb
```
make
```

# run code
```
./main 1 0 0
```