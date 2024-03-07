#ifndef MATRIX_VECTOR_MULTIPLICATION_ACCELERATORNODE_H
#define MATRIX_VECTOR_MULTIPLICATION_ACCELERATORNODE_H
#include "utils.h"
#include "mpi.h"
template<typename Accelerator>
class AcceleratorNode {
public:
    void init() {
        accelerator.init();
    }

    void handshake() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        size_t max_rows = max_memory / (size * sizeof (double));
        MPI_Gather(&max_rows, 1, MPI_UNSIGNED_LONG, &max_rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Type_contiguous(2, MPI_UNSIGNED_LONG, &matrixDataType);
        MPI_Type_commit(&matrixDataType);
        MPI_Scatter(NULL, 0, matrixDataType, &matrixData, 1, matrixDataType, 0, MPI_COMM_WORLD);
        matrix = new double[size * matrixData.partial_size];
        std::cout << "check1" << std::endl;
        #pragma omp parallel for default(none)
        for(int i = 0; i < size * matrixData.partial_size; i++) {
            matrix[i] = 0.0;
        }
        std::cout << "check2" << std::endl;

        MPI_Recv(matrix, size * matrixData.partial_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "check3" << std::endl;

        accelerator.setSize(size);
        accelerator.setPartialSize(matrixData.partial_size);
        accelerator.setMatrix(matrix);
        std::cout << "check4, partial_size = " << matrixData.partial_size << " size =  "<<size << std::endl;

        accelerator.setup();
        std::cout << "setup completed" << std::endl;

    }

    void compute() {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank == 1)
        std::cout << "computing" << std::endl;

        double* p = new (std::align_val_t(mem_alignment))double[size];
        double* Ap = new (std::align_val_t(mem_alignment))double[matrixData.partial_size];
        if(rank == 1)
        std::cout << "starting cycle" << std::endl;

        while(true) {

            std::cout << "new cycle" << std::endl;

            MPI_Bcast(p, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            std::cout << "broadcast completed" << std::endl;

            //accelerator.compute(p, Ap);
            std::cout << "compute completed" << std::endl;

            MPI_Gatherv(Ap, matrixData.partial_size, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            std::cout << "gather completed" << std::endl;

        }
    }

    ~AcceleratorNode() {
        delete[] matrix;
    }

private:
    Accelerator accelerator;

    double* matrix;
    size_t max_memory = 2e30 * 16;
    size_t size;
    MPI_Datatype matrixDataType;
    size_t mem_alignment = 64;
    MatrixData matrixData;
    int rank;
};


#endif //MATRIX_VECTOR_MULTIPLICATION_ACCELERATORNODE_H