//
// Created by tomma on 04/03/2024.
//

#ifndef MATRIX_VECTOR_MULTIPLICATION_FPGAMATRIXVECTORMULTIPLIER_H
#define MATRIX_VECTOR_MULTIPLICATION_FPGAMATRIXVECTORMULTIPLIER_H

#include "utils.h"

class FPGAMatrixVectorMultiplier {
public:
    void init() {
        cl_int err;
        cl_program program;
        init_cl(platform_index, &queues, &context, &devices, &num_device);
        kernels = new cl_kernel[num_device];

        load_program(MATRIX_VECTOR_KERNEL_PATH, &program, context, num_device, devices);
        for(int i = 0; i < num_device; i++) {
            kernels[i] = create_kernel(program, MATRIX_VECTOR_KERNEL_NAME, &err);
        }
    }

    void setup() {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        cl_int err;
        if(rank == 0) {
            std::cout << "device number = " << num_device << std::endl;
        }
        splitted_matrix = new double*[num_device];
        local_offset.resize(num_device);
        local_partial_size.resize(num_device);
        if(rank == 0) {
            std::cout << "check1" << std::endl;
        }
        local_offset[0] = 0;
        for(size_t i = 1; i < num_device; i++) {
            local_offset[i] = local_offset[i-1] + partial_size/num_device;
            local_offset[i] = ( (local_offset[i] * sizeof(double)) + (mem_alignment - ((local_offset[i] * sizeof(double))%mem_alignment)))/sizeof(double);
        }
        if(rank == 0) {
            std::cout << "check2" << std::endl;
            std::cout << "partial size = " << partial_size  << std::endl;

        }
        for(size_t i = 0; i < num_device; i++) {
            if(i != num_device - 1) {
                local_partial_size[i] = local_offset[i+1] - local_offset[i];
            } else {
                local_partial_size[num_device - 1] = partial_size - local_offset[num_device - 1];
            }
            splitted_matrix[i] = new (std::align_val_t(mem_alignment)) double[local_partial_size[i] * size];

        }
        if(rank == 0) {
            std::cout << "check3" << std::endl;
        }
        for(size_t i = 0; i < num_device; i++) {
            for(size_t j = 0; j < size * local_partial_size[i]; j++) {
                splitted_matrix[i][j] = matrix[size * local_offset[i] + j];
            }
        }

        device_A = new cl_mem[num_device];
        device_p = new cl_mem[num_device];
        device_Ap = new cl_mem[num_device];

        if(rank == 0) {
            std::cout << "check4" << std::endl;
        }

        for(int i = 0; i < num_device; i++) {
            if(rank == 0) {
                std::cout << "allocating read only with size = " << local_partial_size[i] * size << std::endl;
            }
            err = 0;
            device_A[i] = allocateDeviceReadOnly(&err, local_partial_size[i] * size, context);
            linkBufferToDevice(queues[i], device_A[i]);
            writeToBuffer(queues[i], device_A[i], 0, local_partial_size[i] * size, splitted_matrix[i], 0);

            device_p[i] = allocateDevice(&err, size, context);
            linkBufferToDevice(queues[i], device_p[i]);

            device_Ap[i] = allocateDevice(&err, local_partial_size[i], context);
            linkBufferToDevice(queues[i], device_Ap[i]);

        }

    }

    void compute(double* p, double* Ap) {
        for (int i = 0; i < num_device; i++) {
            writeToBuffer(queues[i], device_p[i], 0, size, p, 0);
            matrix_vector_multiplication(Ap, local_offset[i], &(device_A[i]), &(device_p[i]), &(device_Ap[i]),
                                         local_partial_size[i], size, &(queues[i]), &(kernels[i]));

        }

    }


    int getDevicesNumber() {
        return num_device;
    }

    void setSize(size_t size_) {
        this->size = size_;
    }

    void setPartialSize(size_t size_) {
        this->partial_size = size_;
    }

    void setMatrix(double* matrix_) {
        this->matrix = matrix_;
    }
private:
    size_t size;
    size_t partial_size;
    cl_uint num_device;
    cl_command_queue* queues;
    cl_context context;
    cl_device_id* devices;
    cl_kernel* kernels;
    int mem_alignment = 64;
    int platform_index = 1;
    double* matrix;

    double** splitted_matrix;
    std::vector<size_t> local_offset;
    std::vector<size_t> local_partial_size;
    cl_mem* device_A;
    cl_mem* device_p;
    cl_mem* device_Ap;
};


#endif //MATRIX_VECTOR_MULTIPLICATION_FPGAMATRIXVECTORMULTIPLIER_H
