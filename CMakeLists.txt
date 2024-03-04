cmake_minimum_required(VERSION 3.20)
project(matrix_vector_multiplication CUDA CXX)

include_directories(include)
include_directories(src/heterogeneous)
include_directories(include/CL)

add_executable(main src/conjugate_gradient.cpp )
add_executable(main_cuda src/main.cu )
#add_executable(test_fpga src/fpga/test.cpp)
add_executable(templated_cuda src/templated_cuda.cu)
add_executable(test_nccl src/nccl_test.cu)
add_executable(fpga_main src/fpga/main_fpga.cpp)
add_executable(multi_fpga src/fpga/multi_fpga.cpp)
add_executable(multi_node src/fpga/multi_node.cpp)
add_executable(main_node src/heterogeneous/main.cpp src/heterogeneous/FPGAMatrixVectorMultiplier.cpp src/heterogeneous/utils.cpp)
add_executable(matrix_gen src/matrix_generator.cpp)
add_executable(rhs_gen src/rhs_generator.cpp)
find_package(MPI REQUIRED)
target_link_libraries(main_node MPI::MPI_CXX)
set_target_properties(main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")


#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lOpenCL")


#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcublas")


target_compile_options(main_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -lcublas
        >)

target_compile_options(templated_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -lcublas
        >)

