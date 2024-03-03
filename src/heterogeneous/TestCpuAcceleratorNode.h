//
// Created by tomma on 02/03/2024.
//

#ifndef MATRIX_VECTOR_MULTIPLICATION_TESTCPUACCELERATORNODE_H
#define MATRIX_VECTOR_MULTIPLICATION_TESTCPUACCELERATORNODE_H

#include <mpi.h>
#include "utils.h"

class TestCpuAcceleratorNode {
public:
    void setup() {

    }

    void handshake() {
        MPI_Init(nullptr, nullptr);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&max_size, 1, MPI_INT, &max_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&device_number, 1, MPI_INT, &device_number, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Datatype matrixDataType;
        MPI_Type_contiguous(2, MPI_INT, &matrixDataType);
        MPI_Type_commit(&matrixDataType);
        MatrixData matrixData;
        MPI_Scatter(&matrixData, 1, matrixDataType, &matrixData, 1, matrixDataType, 0, MPI_COMM_WORLD);

    }
private:
    int world_size;
    int size;
    int max_size = 3;
    int device_number = 2;
};


#endif //MATRIX_VECTOR_MULTIPLICATION_TESTCPUACCELERATORNODE_H
