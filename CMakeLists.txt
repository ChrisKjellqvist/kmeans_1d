cmake_minimum_required(VERSION 3.26)
project(kmeans_hacking)

set(CMAKE_CXX_STANDARD 20)

# if apple, then use these options
if(APPLE)
    add_compile_options(-Xclang -fnative-half-type)
endif ()

add_executable(kmeans_hacking main.cpp util.cpp util.h constants.h kmeans.cpp kmeans.h)

add_library(kmeans_hacking_lib SHARED util.cpp util.h constants.h kmeans.cpp kmeans.h)