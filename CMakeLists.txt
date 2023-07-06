cmake_minimum_required(VERSION 3.0)
project(kmeans_hacking)

set(CMAKE_CXX_STANDARD 20)

if (APPLE)
  add_compile_options(-Xclang -fnative-half-type)
endif()

include(FindPkgConfig)
pkg_search_module(GMP REQUIRED gmp)


# Main CKKMeans library
add_library(CKKMeans SHARED src/kmeans.cpp src/util.cpp src/util.h src/constants.h)
target_include_directories(CKKMeans PUBLIC src ${GMP_INCLUDE_DIRS} )
message(STATUS "GMP include dirs: ${GMP_INCLUDE_DIRS}")
message(STATUS "GMP libraries: ${GMP_LIBRARIES}")
message(STATUS "env: $ENV{LD_LIBRARY_PATH}")
target_link_libraries(CKKMeans PUBLIC ${GMP_LIBRARIES})
target_link_directories(CKKMeans PUBLIC ${GMP_LIBRARY_DIRS})

add_subdirectory(ckmeans)

# Wrapper for external c-kmeans library
add_library(ckmeans_wrapper SHARED src/ckmeans_wrapper.cpp)
target_include_directories(ckmeans_wrapper PUBLIC src)
target_link_libraries(ckmeans_wrapper PUBLIC CKMeans)

find_package(Torch REQUIRED)
add_executable(pytorch_test test/pytorch_test.cpp)
target_link_libraries(pytorch_test "${TORCH_LIBRARIES}" ckmeans_wrapper CKKMeans)
set_property(TARGET pytorch_test PROPERTY CXX_STANDARD 17)

add_executable(kmeans_hacking test/basic_test.cpp )
target_link_libraries(kmeans_hacking CKKMeans)
add_executable(degenerate_test test/degenerate_local_placement_test.cpp )
target_link_libraries(degenerate_test CKKMeans)
add_executable(tensor_load_test test/tensor_load_test.cpp)
target_link_libraries(tensor_load_test CKKMeans)

find_package(pybind11)
if(pybind11_FOUND)
  add_library(binding SHARED src/binding.cpp)
  target_link_libraries(binding PRIVATE ${TORCH_LIBRARIES} pybind11::module CKKMeans)
  set_target_properties(binding PROPERTIES PREFIX "")
endif()

target_link_libraries(tensor_load_test "${TORCH_LIBRARIES}" CKKMeans)

