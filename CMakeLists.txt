cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(mem_cuda LANGUAGES CXX CUDA)

add_subdirectory(vendor EXCLUDE_FROM_ALL)

add_library(${PROJECT_NAME} STATIC
    src/cuda_pattern.cu
    include/mem/cuda_pattern.h)

target_include_directories(${PROJECT_NAME}
    PUBLIC include)

target_link_libraries(${PROJECT_NAME}
    mem)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON)
