cmake_minimum_required(VERSION 3.17)
project(darknet_cpp)
set(CMAKE_CXX_STANDARD 17)

set(DARKNET_DIR CACHE STRING "darknet installation location")


if(NOT DARKNET_DIR STREQUAL "")
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    add_library(cuda INTERFACE)
    target_include_directories(cuda INTERFACE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(cuda INTERFACE ${CUDA_LIBRARIES})
    set(CMAKE_CUDA_FLAGS "--expt-extended-lambda")

    add_library(darknet INTERFACE)
    target_include_directories(darknet INTERFACE ${DARKNET_DIR}/include)
    target_link_libraries(darknet
        INTERFACE
        ${DARKNET_DIR}/libdarknet.so
        cuda
        )
endif()

find_package(fmt CONFIG REQUIRED)
find_package(OpenCV
    REQUIRED
    opencv_core
    opencv_highgui
    opencv_videoio
    opencv_imgproc
    )

add_executable(live live.cpp)
target_link_libraries(live
    PRIVATE
    darknet
    opencv_videoio
    opencv_highgui
    opencv_imgproc
    fmt::fmt
    )
target_compile_options(live PRIVATE -Wall -Wextra -pedantic)
