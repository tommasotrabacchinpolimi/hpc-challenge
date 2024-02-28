//
// Created by tomma on 23/02/2024.
//

#include "CL/opencl.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <cmath>
#define CL_TARGET_OPENCL_VERSION 120

#define MATRIX_VECTOR_KERNEL_NAME gemv
#define REDUCE_ROWS_KERNEL_NAME reduce_rows
#define MATRIX_VECTOR_KERNEL_PATH bin/matrix_vector_kernel.aocx
#define REDUCE_ROWS_KERNEL bin/matrix_vector_kernel.aock


cl_kernel matrix_vector_kernel;
cl_kernel reduce_rows;
cl_kernel cg;


void check_cl(const std::string& msg, cl_int err) {
    if(err != CL_SUCCESS) {
        std::cout << msg << " : " <<err << std::endl;
    }
}

cl_int init_cl(cl_uint device_numbers, cl_command_queue** queues, cl_context* context, cl_device_id** mydev) {
    cl_int err;
    cl_uint num_plat_found;
    cl_platform_id* myp = (cl_platform_id*)malloc(2*sizeof(cl_platform_id));

    err = clGetPlatformIDs(2, myp, &num_plat_found);
    /*for(int i = 0; i < num_plat_found; i++) {
        char name[100];
        clGetPlatformInfo(myp[i],CL_PLATFORM_NAME, 100, name, NULL);
        std::cout << name << std::endl;
    }*/

    std::cout << "found platform " << num_plat_found << std::endl;

    *mydev = (cl_device_id*)malloc(device_numbers * sizeof(cl_device_id));

    cl_uint found_device_n;
    err = clGetDeviceIDs(myp[1], CL_DEVICE_TYPE_ACCELERATOR, device_numbers, *mydev, &found_device_n);
    if(err == CL_DEVICE_NOT_FOUND) {
        std::cout << "no device found" << std::endl;
    }

    if(device_numbers > found_device_n) {
        std::cerr << "not enough devices : " << found_device_n << std::endl;
        free(*mydev);
        exit(1);
    }

    std::cout << "ok" << std::endl;

    *context = clCreateContext(NULL, device_numbers, *mydev, NULL, NULL, &err);

    std::cout << "okk" << std::endl;

    *queues = (cl_command_queue*)malloc(sizeof(cl_command_queue) * device_numbers);
    for(cl_uint i = 0; i < device_numbers; i++) {
        (*queues)[i] = clCreateCommandQueueWithProperties(*context, (*mydev)[i], 0, &err);
        std::cout << "queue " << i << std::endl;
    }
    std::cout << "okkk" << std::endl;

    return err;
}

template<size_t N>
struct clDim {
    int dims[N];
    int dim = N;
};

void generate_matrix(size_t n, double** matrix_out) {
    auto* matrix = new double[n * n];
    for(size_t i = 0; i < n * n; i++) {
        matrix[i] = 0.0;
    }
    for(size_t i = 0; i < n; i++) {
        matrix[i*n + i] = 2.0;
        if(i != n-1) {
            matrix[(i+1)*n + i] = -1;
            matrix[i*n + (i+1)] = -1;
        }
    }
    *matrix_out = matrix;
}

void generate_rhs(size_t n, double value, double** rhs_out) {
    auto* rhs = new double[n];
    for(size_t i = 0; i < n; i++) {
        rhs[i] = value;
    }
    *rhs_out = rhs;
}

template<typename Type>
cl_mem allocateDeviceReadOnly(const double* host_array, cl_int* err, size_t size, cl_context context, cl_command_queue queue) {
    cl_mem ret = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(Type), NULL, NULL, err);
    check_cl("error read-only", *err);
    *err = clEnqueueWriteBuffer(queue, ret, CL_TRUE, 0, size, host_array, 0, NULL, NULL);
    check_cl("error read-only2", *err);

    return ret;
}



template<typename Type>
cl_mem allocateDevice(const double* host_array, cl_int* err, size_t size, cl_context context, cl_command_queue queue) {
    cl_mem ret = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(Type), NULL, NULL, err);
    clEnqueueWriteBuffer(queue, ret, CL_TRUE, 0, size, host_array, 0, NULL, NULL);
    return ret;
}

template<typename Type>
cl_mem allocateDevice(cl_int* err, size_t size, cl_context context) {
    return clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size * sizeof(Type), NULL, NULL, err);
}


template<size_t N>
cl_int launch_kernel(cl_kernel kernel, clDim<N> global_work_size, clDim<N> local_work_size, cl_command_queue queue) {
    return clEnqueueNDRangeKernel(queue, kernel, N, NULL, global_work_size.dims, local_work_size.dims);
}

template<typename Type>
void matrix_vector_multiplication(cl_mem device_A, cl_mem device_p, cl_mem device_Ap, size_t nrows, size_t ncols, cl_command_queue queue, cl_context context) {
    cl_int err;
    int threadsPerRow = 128;
    int rowsPerBlock = 1024;
    int shared_memory = ncols / threadsPerRow;
    clDim<2> local_work_size_mult({1, rowsPerBlock});
    clDim<2> global_work_size_mult({threadsPerRow, static_cast<int>((size_t)((nrows + rowsPerBlock - 1) / rowsPerBlock))});
    cl_mem y_partial = allocateDevice<double>(&err, nrows * threadsPerRow, context);



    clSetKernelArg(matrix_vector_kernel, 0, nrows * ncols * sizeof(Type), device_A);
    clSetKernelArg(matrix_vector_kernel, 1, nrows * sizeof(Type), device_p);
    clSetKernelArg(matrix_vector_kernel, 2, nrows * threadsPerRow * sizeof(Type), y_partial);
    clSetKernelArg(matrix_vector_kernel, 3, ncols * sizeof(Type), device_Ap);
    clSetKernelArg(matrix_vector_kernel, 4, shared_memory * sizeof(Type), NULL);
    clSetKernelArg(matrix_vector_kernel, 5, sizeof(Type), &nrows);
    clSetKernelArg(matrix_vector_kernel, 6, sizeof(Type), &ncols);

    launch_kernel(matrix_vector_kernel, global_work_size_mult, local_work_size_mult, queue);

    clSetKernelArg(reduce_rows, 0, nrows * threadsPerRow * sizeof(Type), y_partial);
    clSetKernelArg(reduce_rows, 1, sizeof(Type), &nrows);
    clSetKernelArg(reduce_rows, 2, sizeof(Type), &threadsPerRow);



    clDim<1> global_work_size_reduce{rowsPerBlock};
    clDim<1> local_work_size_reduce{static_cast<int>((size_t)((nrows + rowsPerBlock - 1) / rowsPerBlock))};

    launch_kernel(reduce_rows, global_work_size_reduce, local_work_size_reduce, queue);

}




void conjugate_gradients(const double * host_A, const double * host_b, double * host_x, size_t size, int max_iters, double rel_error, cl_context context, cl_command_queue queue) {
    cl_int err;
    cl_mem device_A = allocateDevice<double>(host_A, &err, size * size, context, queue);
    check_cl("error A", err);
    std::cout << "allocated A" << std::endl;
    cl_mem device_b = allocateDevice<double>(host_b, &err, size, context, queue);
    check_cl("error b", err);
    std::cout << "allocated b" << std::endl;
    cl_mem device_x = allocateDevice<double>(host_x, &err, size, context, queue);
    check_cl("error x", err);
    std::cout << "allocated x" << std::endl;
    cl_mem device_r = allocateDevice<double>(host_b, &err, size, context, queue);
    check_cl("error r", err);
    std::cout << "allocated r" << std::endl;
    cl_mem device_p = allocateDevice<double>(host_b, &err, size, context, queue);
    check_cl("error p", err);
    std::cout << "allocated p" << std::endl;
    cl_mem device_Ap = allocateDevice<double>(&err, size, context);
    check_cl("error Ap", err);
    std::cout << "allocated Ap" << std::endl;


    double squared_tol = rel_error*rel_error;
    double bb = 0;
    for(int i = 0; i < size; i++) {
        bb += host_b[i]*host_b[i];
    }
    std::cout << "bb = " << bb << std::endl;

    check_cl("set A", clSetKernelArg(cg, 0, sizeof(cl_mem), &device_A));
    check_cl("set b", clSetKernelArg(cg, 1, sizeof(cl_mem), &device_b));
    check_cl("set x", clSetKernelArg(cg, 2, sizeof(cl_mem), &device_x));
    check_cl("set size", clSetKernelArg(cg, 3, sizeof(int), &size));
    check_cl("set r", clSetKernelArg(cg, 4, sizeof(double), &device_r));
    check_cl("set p", clSetKernelArg(cg, 5, sizeof(cl_mem), &device_p));
    check_cl("set Ap", clSetKernelArg(cg, 6, sizeof(cl_mem), &device_Ap));
    check_cl("set niters", clSetKernelArg(cg, 7, sizeof(int), &max_iters));
    check_cl("set tol", clSetKernelArg(cg, 8, sizeof(double), &squared_tol));
    check_cl("set bb", clSetKernelArg(cg, 9, sizeof(double), &bb));

    cl_event wait_finish_kernel;
    check_cl("error with kernel launch", clEnqueueTask(queue, cg, 0, NULL, &wait_finish_kernel));
    std::cout << "size " << size << std::endl;
    check_cl("error with copy",clEnqueueReadBuffer(queue, device_x, CL_TRUE, 0, size * sizeof(double), host_x, 1, &wait_finish_kernel, NULL));
    double* tmp = new double[size];
    memset(tmp, 0, size*sizeof(double));
    for(int i = 0; i < size; i++) {
        tmp[i] = 0.0;
        for(int j = 0; j < size; j++) {
            tmp[i] += host_A[i*size+j]*host_x[j];
        }
    }
    double res_err = 0;
    for(int i = 0; i < size; i++) {

        res_err += (host_b[i] - tmp[i])*(host_b[i] - tmp[i]);
    }
    std::cout << "error: " << res_err << std::endl;

}

void load_program(const std::string& path, cl_program* program, cl_context context, cl_uint num_devices, const cl_device_id* device_list) {
    std::ifstream input_file;
    input_file.open(path);
    size_t length;
    unsigned char* buffer;
    input_file.seekg (0, std::ios::end);
    length = input_file.tellg();
    input_file.seekg (0, std::ios::beg);
    buffer = new unsigned char [length];
    input_file.read (reinterpret_cast<char *>(buffer), length);
    input_file.close();
    const unsigned char** binaries = (const unsigned char**)malloc(sizeof(unsigned char*) * num_devices);
    for(int i = 0; i < num_devices; i++) {
        binaries[i] = buffer;
    }
    cl_int binary_status;
    cl_int errorcode_ret;

    *program = clCreateProgramWithBinary(context, num_devices, device_list, &length, binaries, &binary_status, &errorcode_ret);
    if(errorcode_ret != CL_SUCCESS) {
        std::cout << "error in loading the program" << std::endl;
    }
}

cl_kernel create_kernel(cl_program program, const char* kernel_name, cl_int* errorcode) {
    return clCreateKernel(program, kernel_name, errorcode);
}




int main() {
    cl_context context;
    cl_command_queue* command_queues;
    cl_program program;
    cl_device_id* devices;
    if(init_cl(2, &command_queues, &context, &devices)!=0) {
        std::cout << "error" << std::endl;
    }
    load_program("../src/fpga/CG_kernel_reduced.aocx", &program, context, 1, devices);

    cl_int err;
    cg = create_kernel(program, "conjugate_gradient_kernel", &err);
    if(err == CL_SUCCESS) {
        std::cout << "Success" << std::endl;
    }
    size_t size = 100;
    double* rhs;
    double* matrix;
    double* sol = new double[size];

    for(int i = 0; i < size; i++) {
        sol[i] = 0.0;
    }

    generate_rhs(size, 1.0, &rhs);
    generate_matrix(size, &matrix);
    conjugate_gradients(matrix, rhs, sol, size, 1, 1e-12, context, command_queues[0]);

}

