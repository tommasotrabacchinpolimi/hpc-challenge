#ifndef MATRIX_VECTOR_MULTIPLICATION_ACCELERATORNODE_H
#define MATRIX_VECTOR_MULTIPLICATION_ACCELERATORNODE_H
#include "utils.h"
#include "mpi.h"
class AcceleratorNode {
public:
    void init() {
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
        #pragma omp parallel for default(none) num_threads(num_threads)
        for(int i = 0; i < size * matrixData.partial_size; i++) {
            matrix[i] = 0.0;
        }
        MPI_Recv(matrix, size * matrixData.partial_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }

    void compute() {

        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm communicator;
        int new_root;
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        for(int i = 1; i < world_size; i++) {

            int ranks[] = {0, i};
            MPI_Group new_group;
            MPI_Group_incl(world_group, 2, ranks, &new_group);
            MPI_Comm new_communicator;
            MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_communicator);
            if(i == rank) {
                communicator = new_communicator;
                MPI_Bcast(&new_root,1, MPI_INTEGER, NULL, MPI_COMM_WORLD);
            }
            else {
                int tmp;
                MPI_Bcast(&tmp, 1, MPI_INTEGER, NULL, MPI_COMM_WORLD);
            }
        }

        double* p = new (std::align_val_t(mem_alignment))double[size];
        double* Ap = new (std::align_val_t(mem_alignment))double[matrixData.partial_size];
#pragma omp parallel for default(none) shared(p) num_threads(num_threads)
        for(int i = 0; i < size;i++) {
            p[i] = 0;
        }
#pragma omp parallel for default(none) shared(Ap) num_threads(num_threads)
        for(int i = 0; i < matrixData.partial_size;i++) {
            Ap[i] = 0;
        }
        int cont = 0;

#pragma omp parallel default(none) shared(communicator, p, Ap, matrixData, cont) num_threads(num_threads)
        {
            while (cont < size) {

#pragma omp single
                {
                    MPI_Request r;

                    MPI_Ibcast(p, size, MPI_DOUBLE, 0, communicator, &r);
                    MPI_Wait(&r, MPI_STATUS_IGNORE);
                }


#pragma omp for simd
                for (size_t i = 0; i < matrixData.partial_size; i += 1) {
                    Ap[i] = 0.0;
#pragma omp simd
                    for (size_t j = 0; j < size; j++) {
                        Ap[i] += matrix[i * size + j] * p[j];
                    }
                }
#pragma omp single
                {
                    MPI_Request r;
                    MPI_Igatherv(Ap, matrixData.partial_size, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0,
                                MPI_COMM_WORLD, &r);
                    MPI_Wait(&r, MPI_STATUS_IGNORE);
                    cont++;

                }


            }
        }

    }

    ~AcceleratorNode() {
        delete[] matrix;
    }

private:

    double* matrix;
    size_t max_memory = 2e30 * 16;
    size_t size;
    MPI_Datatype matrixDataType;
    size_t mem_alignment = 64;
    MatrixData matrixData;
    int rank;
    int num_threads = 100;
};


#endif //MATRIX_VECTOR_MULTIPLICATION_ACCELERATORNODE_H