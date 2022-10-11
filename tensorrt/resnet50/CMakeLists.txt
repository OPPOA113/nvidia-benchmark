cmake_minimum_required(VERSION 2.6)

project(resnet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)

add_definitions(-DAPI_EXPORTS)

find_package(CUDA REQUIRED)

if(WIN32)
    enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/include/opencv4)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda-11.3
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt 
# include_directories(/usr/include/x86_64-linux-gnu/ /home/ermengz/sw_source/TensorRT-7.2.1.6/include)
# link_directories(/usr/lib/x86_64-linux-gnu/ /home/ermengz/sw_source/TensorRT-7.2.1.6/lib) 
include_directories(
    /workspace/install_package/nvidia-pkg/TensorRT-8.2.5.1/include
    /workspace/install_package/nvidia-pkg/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/include
)
link_directories(
    /workspace/install_package/nvidia-pkg/TensorRT-8.2.5.1/lib
    /workspace/install_package/nvidia-pkg/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/lib
)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

#add preprocess cuda library
cuda_add_library(preprocess SHARED ${PROJECT_SOURCE_DIR}/preprocess.cu)
target_link_libraries(preprocess nvinfer cudart ${OpenCV_LIBS})

add_executable(classifier ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/resnet50.cpp ${PROJECT_SOURCE_DIR}/data_reader.cpp)
target_link_libraries(classifier nvinfer preprocess)
target_link_libraries(classifier cudart ${OpenCV_LIBS} cuda)

add_definitions(-O2 -pthread)

