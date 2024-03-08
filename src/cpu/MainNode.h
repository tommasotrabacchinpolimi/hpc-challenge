//
// Created by tomma on 02/03/2024.
//

#ifndef MATRIX_VECTOR_MULTIPLICATION_MAINNODE_H
#define MATRIX_VECTOR_MULTIPLICATION_MAINNODE_H

#include <string>
#include <fstream>
#include <utility>
#include <vector>
#include <mpi.h>
#include <math.h>
#include "utils.h"
#include <algorithm>
#include <chrono>
#include <omp.h>

class MainNode {
public:

    MainNode(std::string&  matrix_file_path, std::string&  rhs_file_path, std::string& output_file_path, int max_iters, double tol) : matrix_file_path(std::move(matrix_file_path)), rhs_file_path(std::move(rhs_file_path)), output_file_path(output_file_path), max_iters(max_iters), tol(tol) {}

    void init() {

    }
    void handshake() {

        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        max_size.resize(world_size);


        read_rhs();
        MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        max_size[0] = max_memory / (size * sizeof(double));
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

        MPI_Scatter(&matrixData[0], 1, matrixDataType, &myMatrixData, 1, matrixDataType, 0, MPI_COMM_WORLD);


        sol.resize(size);

        read_and_send_matrix();

    }

    void compute_conjugate_gradient1() {
        double alpha;
        double beta;
        double rr;
        double rr_new;
        double bb;
        std::vector<double> r(size);

        double* p = new (std::align_val_t(mem_alignment))double[size];
        double* Ap = new (std::align_val_t(mem_alignment))double[size];

        double* Ap_ = new (std::align_val_t(mem_alignment))double[size];
        //std::cout << "check1" << std::endl;


#pragma omp parallel for default(none) shared(p, Ap, r, Ap_) num_threads(100)
        for(int i = 0; i < size; i++) {
            p[i] = rhs[i];
            Ap[i] = 0.0;
            Ap_[i] = 0.0;
            sol[i] = 0;
            r[i] = rhs[i];
        }

        bb = dot(rhs,rhs,size);
        rr = bb;

        int iters, total_iterations;
        long comm_overhead = 0;
        double dot_result = 0;
#pragma omp parallel default(none) shared(Ap_, comm_overhead, max_iters, size, tol, matrix, p, Ap, sol, r, dot_result, rr_new, total_iterations, partial_size) firstprivate(alpha, beta, rr, bb, iters) num_threads(100)
        {

            for (iters = 1; iters <= max_iters; iters++) {

#pragma omp for
                for(int i = 0; i < myMatrixData.partial_size; i++) {
                    Ap_[i] = Ap[i];
                }

#pragma omp master
                {

                    auto tmp1 = std::chrono::high_resolution_clock::now();
                    MPI_Request request_broadcast;
                    MPI_Request request_gather;
                    MPI_Ibcast(&p[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request_broadcast);
                    MPI_Igatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, &Ap[0], (&(partial_size[0])),
                                (&(offset[0])), MPI_DOUBLE, 0, MPI_COMM_WORLD, &request_gather);
                    MPI_Wait(&request_gather, MPI_STATUS_IGNORE);
                    MPI_Wait(&request_broadcast, MPI_STATUS_IGNORE);
                    auto tmp2 = std::chrono::high_resolution_clock::now();
                    comm_overhead += std::chrono::duration_cast<std::chrono::microseconds>(tmp2 - tmp1).count();
                }
#pragma omp for simd nowait
                for (size_t i = 0; i < myMatrixData.partial_size; i += 1) {
                    Ap_[i] = 0.0;
#pragma omp simd
                    for (size_t j = 0; j < size; j++) {
                        Ap_[i] += matrix[i * size + j] * p[j];
                    }
                }

#pragma omp barrier

#pragma omp for
                for(int i = 0; i < myMatrixData.partial_size; i++) {
                    Ap[i] = Ap_[i];
                }

#pragma omp single
                {
                    dot_result = 0.0;
                    rr_new = 0.0;
                }


#pragma omp for simd reduction(+:dot_result)
                for (size_t i = 0; i < size; i++) {
                    dot_result += p[i] * Ap[i];
                }
                alpha = rr / dot_result;


#pragma omp for simd nowait
                for(size_t i = 0; i < size; i++) {
                    sol[i] = alpha * p[i] + sol[i];
                }


#pragma omp for simd nowait
                for(size_t i = 0; i < size; i++) {
                    r[i] = -alpha * Ap[i] + r[i];
                }


#pragma omp for simd reduction(+:rr_new)
                for (size_t i = 0; i < size; i++) {
                    rr_new += r[i] * r[i];
                }


                beta = rr_new / rr;
                rr = rr_new;
                if (std::sqrt(rr / bb) < tol) {
#pragma omp single
                    {
                        total_iterations = iters;
                    }
                    break; }

#pragma omp for simd
                for(size_t i = 0; i < size; i++) {
                    p[i] =  r[i] + beta * p[i];
                }
            }
        }

        if(iters <= max_iters)
        {
            printf("Converged in %d iterations, relative error is %e\n", total_iterations, std::sqrt(rr_new / bb));
        }
        else
        {
            printf("Did not converge in %d iterations, relative error is %e\n", total_iterations, std::sqrt(rr_new / bb));
        }
        std::cout << "communication overhead " << comm_overhead << std::endl,

        write_matrix_to_file(output_file_path.c_str(), sol.data(), size, 1);

        delete[] Ap;
        delete[] p;

        //MPI_Abort(MPI_COMM_WORLD, 0);
    }

    void compute_conjugate_gradient() {
        double alpha;
        double beta;
        double rr;
        double rr_new;
        double bb;
        std::vector<double> r(size);

        double* p = new (std::align_val_t(mem_alignment))double[size];
        double* Ap = new (std::align_val_t(mem_alignment))double[size];

        double* Ap_ = new (std::align_val_t(mem_alignment))double[size];
        //std::cout << "check1" << std::endl;


#pragma omp parallel for default(none) shared(p, Ap, r, Ap_) num_threads(num_threads)
        for(int i = 0; i < size; i++) {
            p[i] = rhs[i];
            Ap[i] = 0.0;
            Ap_[i] = 0.0;
            sol[i] = 0;
            r[i] = rhs[i];
        }

        bb = dot(rhs,rhs,size);
        rr = bb;

        int iters, total_iterations;
        double dot_result = 0;
        MPI_Request request_gather;

#pragma omp parallel default(none) shared(std::cout, request_gather, Ap_, max_iters, size, tol, matrix, p, Ap, sol, r, dot_result, rr_new, total_iterations, partial_size) firstprivate(alpha, beta, rr, bb, iters) num_threads(num_threads)
        {


            for (iters = 1; iters <= max_iters; iters++) {

#pragma omp for nowait
                for(int i = 0; i < myMatrixData.partial_size; i++) {
                    Ap_[i] = Ap[i];
                }


#pragma omp master
                {
                    //std::cout << "iter  " << iters << std::endl;
                    dot_result = 0.0;
                    rr_new = 0.0;
                    MPI_Request request_broadcast;
                    MPI_Ibcast(&p[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request_broadcast);

                    MPI_Igatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, &Ap[0], (&(partial_size[0])),
                                 (&(offset[0])), MPI_DOUBLE, 0, MPI_COMM_WORLD, &request_gather);
                    //MPI_Wait(&request_gather, MPI_STATUS_IGNORE);
                    //MPI_Wait(&request_broadcast, MPI_STATUS_IGNORE);
                }

#pragma omp for simd nowait
                for (size_t i = 0; i < myMatrixData.partial_size; i += 1) {
                    Ap_[i] = 0.0;
#pragma omp simd
                    for (size_t j = 0; j < size; j++) {
                        Ap_[i] += matrix[i * size + j] * p[j];
                    }
                }



#pragma omp for
                for(int i = 0; i < myMatrixData.partial_size; i++) {
                    Ap[i] = Ap_[i];
                }

/*#pragma omp single
                {
                    dot_result = 0.0;
                    rr_new = 0.0;
                }*/

#pragma omp master
                {
                    MPI_Wait(&request_gather, MPI_STATUS_IGNORE);
                }
#pragma omp barrier

#pragma omp for simd reduction(+:dot_result)
                for (size_t i = 0; i < size; i++) {
                    dot_result += p[i] * Ap[i];
                }
                alpha = rr / dot_result;


#pragma omp for simd nowait
                for(size_t i = 0; i < size; i++) {
                    sol[i] = alpha * p[i] + sol[i];
                }


#pragma omp for simd nowait
                for(size_t i = 0; i < size; i++) {
                    r[i] = -alpha * Ap[i] + r[i];
                }


#pragma omp for simd reduction(+:rr_new)
                for (size_t i = 0; i < size; i++) {
                    rr_new += r[i] * r[i];
                }


                beta = rr_new / rr;
                rr = rr_new;
                if (std::sqrt(rr / bb) < tol) {
#pragma omp single
                    {
                        total_iterations = iters;
                    }
                    break; }

#pragma omp for simd
                for(size_t i = 0; i < size; i++) {
                    p[i] =  r[i] + beta * p[i];
                }
            }
        }

        if(iters <= max_iters)
        {
            printf("Converged in %d iterations, relative error is %e\n", total_iterations, std::sqrt(rr_new / bb));
        }
        else
        {
            printf("Did not converge in %d iterations, relative error is %e\n", total_iterations, std::sqrt(rr_new / bb));
        }

                write_matrix_to_file(output_file_path.c_str(), sol.data(), size, 1);

        delete[] Ap;
        delete[] p;

        //MPI_Abort(MPI_COMM_WORLD, 0);
    }

    ~MainNode() {
        delete[] matrix;
    }
private:


    void read_rhs() {
        std::ifstream is;
        size_t tmp;
        is.open(rhs_file_path, std::ios::binary);
        is.read((char*)&size,sizeof(size_t));
        is.read((char*)&tmp,sizeof(size_t));
        rhs.resize(size);
#pragma omp parallel for default(none) num_threads(num_threads)
        for(int i = 0; i < size; i++) {
            rhs[i] = 0;
        }
        is.read((char*)&rhs[0], size * sizeof(double));
        is.close();
    }



    void check_matrix(double* matrix, int nrows, int offset) {
        std::vector<double> matrix_(size * size);
        for(size_t i = 0; i < size * size; i++) {
            matrix_[i] = 0.0;
        }
        for(size_t i = 0; i < size; i++) {
            matrix_[i * size + i] = 2.0;
            if(i != size-1) {
                matrix_[(i + 1) * size + i] = -1;
                matrix_[i * size + (i + 1)] = -1;
            }
        }
        for(int i = 0; i < size * nrows; i++) {
            if(matrix_[i + offset*size] != matrix[i]) {
                std::cout << "matrices doesnt match " << matrix_[i] << " " << matrix[i] << std::endl;
            }
        }
    }

    void read_and_send_matrix() {
        auto it = std::max_element(partial_size.begin(), partial_size.end());
        size_t msize = *it;
        double* matrix_ = new (std::align_val_t(mem_alignment)) double[msize * size];
        matrix = new (std::align_val_t(mem_alignment)) double[partial_size[0] * size];

#pragma omp parallel for default(none) num_threads(num_threads)
        for(int i = 0; i < partial_size[0] * size; i++) {
            matrix[i] = 0.0;
        }
        std::ifstream is;
        size_t buff;
        is.open(matrix_file_path, std::ios::binary);
        if(!is.is_open()) {
            std::cout << "matrix file doesn't exist" << std::endl;
            exit(1);
        }
        is.read((char*)&buff, sizeof(size_t));
        is.read((char*)&buff, sizeof(size_t));
        is.read((char*)matrix, size * partial_size[0] * sizeof(double));
        //check_matrix(matrix, partial_size[0], 0);

        for(int i = 1; i < world_size; i++) {
            is.read((char*)matrix_, size * partial_size[i] * sizeof(double));
            MPI_Send(matrix_, size * partial_size[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        is.close();
        delete[] matrix_;

    }

    static bool write_matrix_to_file(const char * filename, const double * matrix, size_t num_rows, size_t num_cols)
    {
        FILE * file = fopen(filename, "wb");
        if(file == nullptr)
        {
            fprintf(stderr, "Cannot open output file\n");
            return false;
        }

        fwrite(&num_rows, sizeof(size_t), 1, file);
        fwrite(&num_cols, sizeof(size_t), 1, file);
        fwrite(matrix, sizeof(double), num_rows * num_cols, file);

        fclose(file);

        return true;
    }

    std::string matrix_file_path;
    std::string rhs_file_path;
    std::string output_file_path;
    size_t size;
    int world_size;
    std::vector<int> offset;
    std::vector<int> partial_size;
    std::vector<size_t> max_size;
    std::vector<double> rhs;
    std::vector<double> sol;
    double* matrix;
    size_t total_device_number;
    int max_iters;
    double tol;
    MatrixData myMatrixData;
    int mem_alignment = 64;
    size_t max_memory = 2e30 * 16;

    int num_threads = 300;



};


#endif //MATRIX_VECTOR_MULTIPLICATION_MAINNODE_H
