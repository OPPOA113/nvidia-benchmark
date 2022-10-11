cmake_minimum_required(VERSION 2.6)

project(yolov3)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)

# Debug/Release
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/include/opencv4)

# cuda
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(yololayer nvinfer cudart ${OpenCV_LIBS})

#add preprocess cuda library
cuda_add_library(preprocess SHARED ${PROJECT_SOURCE_DIR}/preprocess.cu)
target_link_libraries(preprocess nvinfer cudart ${OpenCV_LIBS})

add_executable(yolov3 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/yolov3.cpp ${PROJECT_SOURCE_DIR}/data_reader.cpp ${PROJECT_SOURCE_DIR}/eval_model.cpp)
target_link_libraries(yolov3 nvinfer)
target_link_libraries(yolov3 cudart cuda)
target_link_libraries(yolov3 yololayer)
target_link_libraries(yolov3 ${OpenCV_LIBS})
target_link_libraries(yolov3 nvinfer_plugin)
target_link_libraries(yolov3 preprocess)

add_definitions(-O2 -pthread)

