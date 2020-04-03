cmake_minimum_required(VERSION 3.17)
project(darknet_cpp)
set(CMAKE_CXX_STANDARD 17)

set(DARKNET_DIR CACHE STRING "darknet installation location")

if(DARKNET_DIR STREQUAL "")
    message(FATAL_ERROR "DARKNET_DIR not defined")
endif()

find_package(range-v3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(OpenCV
    REQUIRED
    opencv_core
    opencv_highgui
    opencv_videoio
    opencv_imgproc
    )
find_package(CUDA REQUIRED)
enable_language(CUDA)
add_library(cuda INTERFACE)
target_include_directories(cuda INTERFACE ${CUDA_INCLUDE_DIRS})
target_link_libraries(cuda INTERFACE ${CUDA_LIBRARIES})

add_library(darknet_c INTERFACE)
target_include_directories(darknet_c INTERFACE ${DARKNET_DIR}/include)
target_link_libraries(darknet_c
    INTERFACE
    ${DARKNET_DIR}/libdarknet.so
    cuda
    )


set(public_hdr
    include/darknet/detector.hpp
    )
set(private_src
    src/detector.cpp
    )
set(dependencies
    darknet_c
    opencv_core
    opencv_imgproc
    range-v3)

add_library(darknet_cpp ${private_src} ${public_hdr})
target_link_libraries(darknet_cpp PUBLIC ${dependencies})
target_include_directories(darknet_cpp PUBLIC include)
target_compile_options(darknet_cpp PRIVATE -Wall -Wextra -pedantic)

install(TARGETS darknet_cpp darknet_c
    EXPORT darknet_cpp
    )


add_executable(live examples/live.cpp)
target_link_libraries(live
    PRIVATE
    darknet_cpp
    opencv_videoio
    opencv_highgui
    opencv_imgproc
    fmt::fmt
    range-v3
    )
target_compile_options(live PRIVATE -Wall -Wextra -pedantic)
