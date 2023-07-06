cmake_minimum_required(VERSION 3.0)
project(kmeans_hacking)

set(CMAKE_CXX_STANDARD 20)

if (APPLE)
  add_compile_options(-Xclang -fnative-half-type)
endif()

add_subdirectory(ckmeans)

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
add_executable(kmeans_hacking test/basic_test.cpp src/util.cpp src/util.h src/constants.h src/kmeans.cpp src/kmeans.h)
add_executable(pytorch_test test/pytorch_test.cpp src/util.cpp src/util.h src/constants.h src/kmeans.cpp src/kmeans.h)
add_executable(degenerate_test test/degenerate_local_placement_test.cpp src/util.cpp src/util.h src/constants.h src/kmeans.cpp src/kmeans.h)
target_link_libraries(pytorch_test "${TORCH_LIBRARIES}")
target_link_libraries(pytorch_test CKMeans)
target_include_directories(pytorch_test PRIVATE "ckmeans/src")
#target_include_directories(pytorch_test PRIVATE "ckmeans/src")
set_property(TARGET pytorch_test PROPERTY CXX_STANDARD 14)

#add_library(binding SHARED src/binding.cpp src/util.cpp src/util.h src/constants.h src/kmeans.cpp src/kmeans.h)
#target_link_libraries(binding PRIVATE ${TORCH_LIBRARIES} pybind11::module)
#set_target_properties(binding PROPERTIES PREFIX "")

