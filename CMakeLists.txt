cmake_minimum_required(VERSION 3.17)
project(darknet)
set(CMAKE_CXX_STANDARD 17)

set(DARKNET_DIR CACHE STRING "darknet installation location")

if(DARKNET_DIR STREQUAL "")
    message(FATAL_ERROR "DARKNET_DIR not defined")
endif()

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
    )

add_library(darknet ${private_src})
target_link_libraries(darknet PUBLIC ${dependencies})
target_include_directories(darknet PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )
set_target_properties(darknet PROPERTIES PUBLIC_HEADER ${public_hdr})
set_property(TARGET darknet PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(darknet PRIVATE -Wall -Wextra -pedantic)

set(main_lib_dest lib/darknet)
install(TARGETS darknet darknet_c cuda
    EXPORT darknet
    ARCHIVE DESTINATION ${main_lib_dest}
    PUBLIC_HEADER DESTINATION include/darknet
    )
install(EXPORT darknet DESTINATION ${main_lib_dest})
install(FILES darknet-config.cmake DESTINATION ${main_lib_dest})


add_executable(live examples/live.cpp)
target_link_libraries(live
    PRIVATE
    darknet
    opencv_videoio
    opencv_highgui
    opencv_imgproc
    fmt::fmt
    )
target_compile_options(live PRIVATE -Wall -Wextra -pedantic)
