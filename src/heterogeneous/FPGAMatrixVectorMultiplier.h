//
// Created by tomma on 04/03/2024.
//

#ifndef MATRIX_VECTOR_MULTIPLICATION_FPGAMATRIXVECTORMULTIPLIER_H
#define MATRIX_VECTOR_MULTIPLICATION_FPGAMATRIXVECTORMULTIPLIER_H
#include "MatrixVectorMultiplier.h"
#include "mpi.h"
#include "utils.h"
class FPGAMatrixVectorMultiplier : public MatrixVectorMultiplier{
public:
    void init() override;

    void setup() override;

    void compute(double* p, double* Ap) override;


    void setSize(size_t size_) override;

    void setPartialSize(size_t size_) override;

    void setMatrix(double* matrix_) override;

    ~FPGAMatrixVectorMultiplier() override;
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
