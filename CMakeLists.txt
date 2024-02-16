cmake_minimum_required(VERSION 3.20)
project(matrix_vector_multiplication)

set(CMAKE_CXX_STANDARD 23)

add_executable(main src/conjugate_gradient.cpp )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")