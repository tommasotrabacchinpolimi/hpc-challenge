cmake_minimum_required(VERSION 3.20)
project(matrix_vector_multiplication CUDA CXX)



add_executable(main src/conjugate_gradient.cpp )
add_executable(main_cuda src/main.cu )
set_target_properties(main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

target_compile_options(main_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        >)

