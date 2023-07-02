cmake_minimum_required(VERSION 3.26)
project(kmeans_hacking)

set(CMAKE_CXX_STANDARD 20)

find_package(Torch REQUIRED)
add_executable(pytorch_test pytorch_test.cpp util.cpp util.h constants.h kmeans.cpp kmeans.h)
target_link_libraries(pytorch_test "${TORCH_LIBRARIES}")
set_property(TARGET pytorch_test PROPERTY CXX_STANDARD 14)
add_executable(kmeans_hacking main.cpp util.cpp util.h constants.h kmeans.cpp kmeans.h)

