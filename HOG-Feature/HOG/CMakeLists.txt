find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# collect source files
#SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c11" )
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -cc /usr/bin/clang )

file( GLOB  hdr *.hpp *.h )
SET (HOG_files main.cu HogSupport.cu)

CUDA_ADD_EXECUTABLE(HOG ${HOG_files} ${hdr})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(HOG ${OpenCV_LIBS})

