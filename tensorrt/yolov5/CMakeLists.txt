cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# 10.23.4.176-vtxdemo容器
# tensorrt + cudnn
# include_directories(
#     /usr/include/x86_64-linux-gnu/
#     /workspace/install_package/nvidia-pkg/TensorRT-22.06/include
# )
# link_directories(
#     /usr/lib/x86_64-linux-gnu/
#     /workspace/install_package/nvidia-pkg/TensorRT-22.06/include
# )
# find_package(OpenCV)
# include_directories(${OpenCV_INCLUDE_DIRS})

# 10.23.4.176-trtdev容器
# tensorrt + cudnn
include_directories(
    /usr/include/x86_64-linux-gnu/
    /workspace/install_package/nvidia-pkg/TensorRT-8.4.0.6/include
    /workspace/project/install_package/opencv3410/include
)
link_directories(
    /usr/lib/x86_64-linux-gnu/
    /workspace/install_package/nvidia-pkg/TensorRT-22.06/build/out
    /workspace/project/install_package/opencv3410/lib
    /workspace/project/install_package/opencv3410/jasper
)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
cuda_add_library(myplugins SHARED yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)


cuda_add_executable(yolov5 calibrator.cpp yolov5.cpp preprocess.cu)

target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 ${OpenCV_LIBS} 
    opencv_dnn
    opencv_imgproc
    opencv_core 
    opencv_highgui 
    opencv_imgcodecs
)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


