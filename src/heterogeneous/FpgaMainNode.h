//
// Created by tomma on 03/03/2024.
//

#ifndef MATRIX_VECTOR_MULTIPLICATION_FPGAMAINNODE_H
#define MATRIX_VECTOR_MULTIPLICATION_FPGAMAINNODE_H

#include <vector>
#include <fstream>
#include <mpi.h>
#include <cmath>
#include "utils.h"

class FpgaMainNode {

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

    void handshake() {

        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        max_size.resize(world_size);
        std::ifstream is;
        /*is.open(matrix_file_path, std::ios::binary);
        is.read((char*)&size,sizeof(size_t));
        is.close();
         */
        size = 100;

        MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        MPI_Gather(MPI_IN_PLACE, 1, MPI_UNSIGNED_LONG, &max_size[0], 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        size_t total_capacity = 0;
        for(int i = 0; i < world_size; i++) {
            total_capacity += max_size[i];
        }
        std::vector<double> quota(world_size);
        for(int i = 0; i < world_size; i++) {
            quota[i] = (double)max_size[i]/(double)total_capacity;
        }

        offset.resize(world_size);
        partial_size.resize(world_size);
        offset[0] = 0;
        for(int i = 1; i < world_size; i++) {
            offset[i] = offset[i-1] + std::min((size_t)(size*quota[i-1]), max_size[i-1]);
            partial_size[i-1] = std::min((size_t)(size*quota[i-1]), max_size[i-1]);
        }
        partial_size[world_size - 1] = size - offset[world_size - 1];


        MPI_Datatype matrixDataType;
        MPI_Type_contiguous(2, MPI_UNSIGNED_LONG, &matrixDataType);
        MPI_Type_commit(&matrixDataType);

        std::vector<MatrixData> matrixData(world_size);
        for(int i = 0; i < world_size; i++) {
            matrixData[i] = {(size_t)offset[i], (size_t)partial_size[i]};
        }

        MatrixData myMatrixData;
        MPI_Scatter(&matrixData[0], 1, matrixDataType, &myMatrixData, 1, matrixDataType, 0, MPI_COMM_WORLD);
        read_rhs();
        sol.resize(size);
        read_and_send_matrix();
    }

    void compute_conjugate_gradient() {
        double alpha;
        double beta;
        double rr;
        double rr_new;
        double bb;
        cl_int err;
        std::vector<double> r(size);
        double* p = new (std::align_val_t(mem_alignment))double[size];
        double* Ap = new (std::align_val_t(mem_alignment))double[size];
        r = rhs;
        for(int i = 0; i < size; i++) {
            p[i] = rhs[i];
        }
        //p = rhs;
        bb = dot(rhs,rhs,size);
        rr = bb;
        for(auto& s : sol) {
            s = 0.0;
        }




        double** splitted_matrix = new double * [num_device];
        std::vector<size_t> local_offset(num_device);
        std::vector<size_t> local_partial_size(num_device);

        local_offset[0] = 0;
        for(size_t i = 1; i < num_device; i++) {
            local_offset[i] = local_offset[i-1] + partial_size[0]/num_device;
            local_offset[i] = ( (local_offset[i] * sizeof(double)) + (mem_alignment - ((local_offset[i] * sizeof(double))%mem_alignment)))/sizeof(double);
        }



        for(size_t i = 0; i < num_device; i++) {
            if(i != num_device - 1) {
                local_partial_size[i] = local_offset[i+1] - local_offset[i];
            } else {
                local_partial_size[num_device - 1] = partial_size[0] - local_offset[num_device - 1];
            }
            splitted_matrix[i] = new (std::align_val_t(mem_alignment)) double[local_partial_size[i] * size];

        }
        for(size_t i = 0; i < num_device; i++) {
            for(size_t j = 0; j < size * local_partial_size[i]; j++) {
                splitted_matrix[i][j] = local_matrix[size * local_offset[i] + j];
            }
        }

        cl_mem* device_A = new cl_mem[num_device];
        cl_mem* device_p = new cl_mem[num_device];
        cl_mem* device_Ap = new cl_mem[num_device];


        for(int i = 0; i < num_device; i++) {

            device_A[i] = allocateDeviceReadOnly(&err, local_partial_size[i] * size, context);
            linkBufferToDevice(queues[i], device_A[i]);
            writeToBuffer(queues[i], device_A[i], 0, local_partial_size[i] * size, splitted_matrix[i], 0);

            device_p[i] = allocateDevice(&err, size, context);
            linkBufferToDevice(queues[i], device_p[i]);

            device_Ap[i] = allocateDevice(&err, local_partial_size[i], context);
            linkBufferToDevice(queues[i], device_Ap[i]);

        }


        int iters;
        for(iters = 1; iters <= max_iters; iters++) {
            MPI_Bcast(&p[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, &Ap[0], (&(partial_size[0])),
                        (&(offset[0])), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            for (int i = 0; i < num_device; i++) {
                writeToBuffer(queues[i], device_p[i], 0, size, p, 0);
                matrix_vector_multiplication(Ap, local_offset[i], &(device_A[i]), &(device_p[i]), &(device_Ap[i]),
                                             local_partial_size[i], size, &(queues[i]), &(kernels[i]));

            }

            alpha = rr / dot(p, Ap, size);
            axpby(alpha, p, 1.0, sol, size);
            axpby(-alpha, Ap, 1.0, r, size);
            rr_new = dot(r, r, size);
            beta = rr_new / rr;
            rr = rr_new;
            if(std::sqrt(rr / bb) < tol) { break; }
            axpby(1.0, r, beta, p, size);
        }

        if(iters <= max_iters)
        {
            printf("Converged in %d iterations, relative error is %e\n", iters, std::sqrt(rr / bb));
        }
        else
        {
            printf("Did not converge in %d iterations, relative error is %e\n", iters, std::sqrt(rr / bb));
        }
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
private:

    /*
    void read_rhs() {
        int buff;
        std::ifstream is;
        is.open(rhs_file_path, std::ios::binary);
        is.read((char*)&buff,sizeof(size_t));
        is.read((char*)&rhs[0], size * sizeof(double));
        is.close();
    }
     */
    void read_rhs() {
        rhs.resize(size);
        for(auto& r : rhs) {
            r = 1.0;
        }
    }


    void read_and_send_matrix() {
        std::vector<double> matrix(size * size);
        for(size_t i = 0; i < size * size; i++) {
            matrix[i] = 0.0;
        }
        for(size_t i = 0; i < size; i++) {
            matrix[i*size + i] = 2.0;
            if(i != size-1) {
                matrix[(i+1)*size + i] = -1;
                matrix[i*size + (i+1)] = -1;
            }
        }

        local_matrix = new (std::align_val_t(mem_alignment))double[size * partial_size[0]];
        for(int i = 0; i < size * partial_size[0];i++) {
            local_matrix[i] = matrix[i];
        }
        for(size_t i = 1; i < world_size; i++) {
            MPI_Send(&matrix[0] + offset[i]*size, size * partial_size[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }

    std::string matrix_file_path;
    std::string rhs_file_path;
    size_t size;
    int world_size;
    std::vector<int> offset;
    std::vector<int> partial_size;
    std::vector<size_t> max_size;
    std::vector<double> rhs;
    std::vector<double> sol;
    size_t total_device_number;
    int max_iters;
    double tol;
    double* local_matrix;
    cl_command_queue* queues;
    cl_context context;
    cl_device_id* devices;
    cl_kernel* kernels;
    cl_uint num_device;
    int platform_index;
    int mem_alignment = 64;
};


#endif //MATRIX_VECTOR_MULTIPLICATION_FPGAMAINNODE_H
