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


class MainNode {
public:

    MainNode(std::string  matrix_file_path, std::string  rhs_file_path, int max_iters, double tol) : matrix_file_path(std::move(matrix_file_path)), rhs_file_path(std::move(rhs_file_path)), max_iters(max_iters), tol(tol) {}

    void handshake() {

        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        max_size.resize(world_size);
        std::ifstream is;
        /*is.open(matrix_file_path, std::ios::binary);
        is.read((char*)&size,sizeof(size_t));
        is.close();
         */
        size = 10;

        MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        MPI_Gather(MPI_IN_PLACE, 1, MPI_UNSIGNED_LONG, &max_size[0], 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        world_device_number.resize(world_size);
        world_device_number[0] = 0;

        MPI_Gather(MPI_IN_PLACE, 1, MPI_UNSIGNED_LONG, &world_device_number[0], 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        total_device_number = 0;
        for(int i = 1; i < world_size; i++) {
            total_device_number += world_device_number[i];
        }

        offset.resize(world_size);
        partial_size.resize(world_size);
        offset[1] = 0;
        for(int i = 2; i < world_size; i++) {
            offset[i] = offset[i-1] + std::min(size/total_device_number*world_device_number[i], max_size[i]);
            partial_size[i-1] = std::min(size/total_device_number*world_device_number[i], max_size[i]);
        }
        partial_size[world_size - 1] = size - offset[world_size - 1];

        MPI_Datatype matrixDataType;
        MPI_Type_contiguous(2, MPI_UNSIGNED_LONG, &matrixDataType);
        MPI_Type_commit(&matrixDataType);

        std::vector<MatrixData> matrixData(world_size);
        for(int i = 1; i < world_size; i++) {
            matrixData[i] = {offset[i], partial_size[i]};
        }

        MatrixData myMatrixData;//ignored
        MPI_Scatter(&matrixData[0], 1, matrixDataType, &myMatrixData, 1, matrixDataType, 0, MPI_COMM_WORLD);
        //read_rhs();
        sol.resize(size);
        read_and_send_matrix();
    }

    void compute_conjugate_gradient() {
        double alpha;
        double beta;
        double rr;
        double rr_new;
        double bb;
        std::vector<double> r(size);
        std::vector<double> p(size);
        std::vector<double> Ap(size);
        r = rhs;
        bb = dot(rhs,rhs,size);
        rr = bb;

        for(int iters = 0; iters < max_iters; iters++) {
            MPI_Bcast(&p[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, &Ap[0], reinterpret_cast<const int *>(&(partial_size[0])),
                        reinterpret_cast<const int *>(&(offset[0])), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            alpha = rr / dot(p, Ap, size);
            axpby(alpha, p, 1.0, sol, size);
            axpby(-alpha, Ap, 1.0, r, size);
            rr_new = dot(r, r, size);
            beta = rr_new / rr;
            rr = rr_new;
            if(std::sqrt(rr / bb) < tol) { break; }
            axpby(1.0, r, beta, p, size);
        }
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
private:

    void read_rhs() {
        int buff;
        std::ifstream is;
        is.open(rhs_file_path, std::ios::binary);
        is.read((char*)&buff,sizeof(size_t));
        is.read((char*)&rhs[0], size * sizeof(double));
        is.close();
    }

    void read_and_send_matrix() {
        std::vector<double> matrix(size);
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

        for(size_t i = 1; i < world_size; i++) {
            MPI_Send(&matrix[0] + offset[i]*size, size * partial_size[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }

    std::string matrix_file_path;
    std::string rhs_file_path;
    int size;
    int world_size;
    std::vector<size_t> offset;
    std::vector<size_t> partial_size;
    std::vector<size_t> world_device_number;
    std::vector<size_t> max_size;
    std::vector<double> rhs;
    std::vector<double> sol;
    size_t total_device_number;
    int max_iters;
    double tol;
};


#endif //MATRIX_VECTOR_MULTIPLICATION_MAINNODE_H
