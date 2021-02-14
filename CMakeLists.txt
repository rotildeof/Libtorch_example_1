cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(classify_example1 CXX)

list(APPEND CMAKE_PREFIX_PATH ~/Torch/libtorch)
find_package(Torch REQUIRED)

add_executable(classify_ex1 classify_example1.cpp)
add_compile_options(-Wall)
target_link_libraries(classify_ex1 "${TORCH_LIBRARIES}")
set_property(TARGET classify_ex1 PROPERTY CXX_STANDARD 14)

