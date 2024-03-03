//
// Created by tomma on 03/03/2024.
//

#ifndef MATRIX_VECTOR_MULTIPLICATION_FPGAACCELERATORNODE_H
#define MATRIX_VECTOR_MULTIPLICATION_FPGAACCELERATORNODE_H

#define MATRIX_VECTOR_KERNEL_PATH "../src/fpga/MVV.aocx"
#define MATRIX_VECTOR_KERNEL_NAME "matrix_vector_kernel"
#include "utils.h"
#include <mpi.h>
class FpgaAcceleratorNode {
public:
    void setup() {
        cl_command_queue* queues;
        cl_context context;
        cl_device_id* devices;
        cl_program program;
        init_cl(platform_index, &queues, &context, &devices, &num_device);
        kernels = new cl_kernel[num_device];

        load_program(MATRIX_VECTOR_KERNEL_PATH, &program, context, num_device, devices);
        for(int i = 0; i < num_device; i++) {
            kernels[i] = create_kernel(program, MATRIX_VECTOR_KERNEL_NAME, &err);
        }
    }

    void handshake() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        //std::cout << "rank " << rank << "received size " << size << std::endl;
        size_t max_rows = max_memory / (size * sizeof (double));
        MPI_Gather(&max_rows, 1, MPI_UNSIGNED_LONG, &max_rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Type_contiguous(2, MPI_UNSIGNED_LONG, &matrixDataType);
        MPI_Type_commit(&matrixDataType);
        MPI_Scatter(NULL, 0, matrixDataType, &matrixData, 1, matrixDataType, 0, MPI_COMM_WORLD);
        matrix = new double[size * matrixData.partial_size];
        MPI_Recv(matrix, size * matrixData.partial_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "completed handshake" << std::endl;
    }

    void compute() {
        double* p = new (std::align_val_t(mem_alignment)) double[matrixData.partial_size];
        double* Ap = new (std::align_val_t(mem_alignment)) double[matrixData.partial_size];
        //std::vector<std::vector<double>> splitted_matrix(num_device);
        double** splitted_matrix = new double * [num_device];
        std::vector<size_t> local_offset(num_device);
        std::vector<size_t> local_partial_size(num_device);
        std::cout << "compute started" << std::endl;

        local_offset[0] = 0;
        for(size_t i = 1; i < num_device; i++) {
            local_offset[i] = local_offset[i-1] + matrixData.partial_size/num_device;
            local_offset[i] = ( (local_offset[i] * sizeof(double)) + (mem_alignment - ((local_offset[i] * sizeof(double))%mem_alignment)))/sizeof(double);
        }

        for(size_t i = 0; i < num_device; i++) {
            if(i != num_device - 1) {
                local_partial_size[i] = local_offset[i+1] - local_offset[i];
            } else {
                local_partial_size[num_device - 1] = size - local_offset[num_device - 1];
            }
            splitted_matrix[i] = new double[local_partial_size[i] * size];
            std::cout << "splitted matrix created " << std::endl;

        }
        for(size_t i = 0; i < num_device; i++) {
            for(size_t j = 0; j < size * local_partial_size[i]; j++) {
                splitted_matrix[i][j] = matrix[size * local_offset[i] + j];
            }
        }

        cl_mem* device_A = new cl_mem[num_device];
        cl_mem* device_p = new cl_mem[num_device];
        cl_mem* device_Ap = new cl_mem[num_device];

        std::cout << "starting cycle" << std::endl;

        for(int i = 0; i < num_device; i++) {
            device_A[i] = allocateDeviceReadOnly(&err, local_partial_size[i] * size, context);
            std::cout << "allocating" << std::endl;
            linkBufferToDevice(queues[i], device_A[i]);
            std::cout << "linking" << std::endl;
            writeToBuffer(queues[i], device_A[i], 0, local_partial_size[i] * size, splitted_matrix[i], 0);
            std::cout << "writing" << std::endl;

            std::cout << "device_A" << std::endl;
            device_p[i] = allocateDevice(&err, size, context);
            linkBufferToDevice(queues[i], device_p[i]);
            std::cout << "device_p" << std::endl;

            device_Ap[i] = allocateDevice(&err, local_partial_size[i], context);
            linkBufferToDevice(queues[i], device_Ap[i]);
            std::cout << "device_Ap" << std::endl;

        }

        std::cout << "starting while loop" << std::endl;

        while(true) {
            MPI_Bcast(p, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            for (int i = 0; i < num_device; i++) {
                writeToBuffer(queues[i], device_p[i], 0, size, p, 0);
                matrix_vector_multiplication(Ap, local_offset[i], &(device_A[i]), &(device_p[i]), &(device_Ap[i]),
                                             local_partial_size[i], size, &(queues[i]), &(kernels[i]));
            }
            std::cout << "sending... " << std::endl;
            for(int i = 0; i < matrixData.partial_size; i++) {
                std::cout << Ap[i] << ", ";
            }
            std::cout<<std::endl<<"over"<<std::endl;
            MPI_Gatherv(Ap, matrixData.partial_size, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
private:



    std::string matrix_file_path;
    double* matrix;
    int platform_index = 1;
    cl_int err = 0;
    size_t max_memory = 2e30 * 16;
    size_t size;
    cl_uint num_device;
    MPI_Datatype matrixDataType;
    size_t mem_alignment = 64;
    MatrixData matrixData;
    cl_command_queue* queues;
    cl_context context;
    cl_device_id* devices;
    cl_kernel* kernels;
    int rank;


};


#endif //MATRIX_VECTOR_MULTIPLICATION_FPGAACCELERATORNODE_H
