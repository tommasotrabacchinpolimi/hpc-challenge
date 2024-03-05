//
// Created by tomma on 05/03/2024.
//



#include "utils.h"


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


    *context = clCreateContext(NULL, device_numbers, *mydev, NULL, NULL, &err);


    *queues = (cl_command_queue*)malloc(sizeof(cl_command_queue) * device_numbers);
    for(cl_uint i = 0; i < device_numbers; i++) {
        (*queues)[i] = clCreateCommandQueueWithProperties(*context, (*mydev)[i], 0, &err);
    }

    return err;
}


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


void conjugate_gradient(const double* matrix, const double* rhs, double* x, int size, double tol, int max_iters, cl_context context, cl_command_queue queue, cl_kernel kernel) {

    std::cout << "starting conjugate gradient" << std::endl;
    cl_mem device_matrix;
    cl_mem device_rhs;
    cl_mem sol;
    cl_mem p;
    cl_mem Ap;
    cl_mem r;
    cl_mem iter_number;
    cl_mem achieved_error;
    cl_int err;



    device_matrix = allocateDeviceReadOnly(&err, size * size, context);
    check_cl(err, "error allocating matrix");
    device_rhs = allocateDeviceReadOnly(&err, size, context);
    check_cl(err, "error allocating rhs");
    sol = allocateDevice(&err, size, context);
    check_cl(err, "error allocating the solution");
    p = allocateDevice(&err, size, context);
    check_cl(err, "error allocating p");
    Ap = allocateDevice(&err, size, context);
    check_cl(err, "error allocating Ap");
    r = allocateDevice(&err, size, context);
    check_cl(err, "error allocating r");
    iter_number = allocateDeviceSingleInt(&err, context);
    check_cl(err, "error allocating iter_number");
    achieved_error = allocateDeviceSingleDouble(&err, context);
    check_cl(err, "error allocating achieved error");

    writeToBufferNoBlock(queue, p, 0, size, rhs, 0);
    writeToBufferNoBlock(queue, device_matrix, 0, size * size, matrix, 0);
    writeToBufferNoBlock(queue, device_rhs, 0, size, rhs, 0);
    writeToBufferNoBlock(queue, sol, 0, size, x, 0);

    double squared_tol = tol * tol;
    double bb = dot(rhs, rhs, size);

    clFinish(queue);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_matrix);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_rhs);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &sol);
    clSetKernelArg(kernel, 3, sizeof(int), &size);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &r);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &p);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &Ap);
    clSetKernelArg(kernel, 7, sizeof(int), &max_iters);
    clSetKernelArg(kernel, 8, sizeof(double), &squared_tol);
    clSetKernelArg(kernel, 9, sizeof(double), &bb);
    clSetKernelArg(kernel, 10, sizeof(cl_mem), &iter_number);
    clSetKernelArg(kernel, 11, sizeof(cl_mem), &achieved_error);

    cl_event wait_finish_kernel;
    clEnqueueTask(queue, kernel, 0, NULL, &wait_finish_kernel);
    clEnqueueReadBuffer(queue, sol, CL_TRUE, 0, size * sizeof(double), x, 1, &wait_finish_kernel, NULL);

    double* tmp = new double[size];
    for(int i = 0; i < size; i++) {
        tmp[i] = 0.0;
        for(int j = 0; j < size; j++) {
            tmp[i] += matrix[i*size+j]*x[j];
        }
    }
    double res_err = 0;
    for(int i = 0; i < size; i++) {
        //std::cout << x[i] << " " << rhs[i] << " " << tmp[i] << std::endl;
        res_err += (rhs[i] - tmp[i])*(rhs[i] - tmp[i]);
    }
    std::cout << "error: " << res_err << std::endl;

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
    cl_kernel kernel  = create_kernel(program, "conjugate_gradient_kernel", &err);
    if(err == CL_SUCCESS) {
        std::cout << "Success" << std::endl;
    }
    size_t size = 1000;
    double* rhs;
    double* matrix;
    double* sol = new double[size];

    for(int i = 0; i < size; i++) {
        sol[i] = 0.0;
    }

    generate_rhs(size, 1.0, &rhs);
    generate_matrix(size, &matrix);
    conjugate_gradient(matrix, rhs, sol, size, 1e-12, 500, context, command_queues[0], kernel);

}