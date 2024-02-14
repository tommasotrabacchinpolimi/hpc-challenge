//
// Created by tomma on 14/02/2024.
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

long parallel_time;
long serial_time;


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

void transpose_matrix(size_t n, double* matrix) {
    for(size_t i = 0; i < n; i++) {
        for(size_t j = i + 1; j < n; j++) {
            double swap = matrix[i*n+j];
            matrix[i*n+j] = matrix[j*n+i];
            matrix[j*n+i] = swap;
        }
    }
}

double dot(const double* x, const double* y, size_t n) {
    double result = 0.0;
//#pragma omp parallel for default(none) shared(n,x,y) reduction(+:result)
    for(size_t i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    return result;
}

void axpby(double alpha, const double* x, double beta, double* y, size_t n) {
//#pragma omp parallel for default(none) shared(n,alpha,beta) firstprivate(x,y)
    for(size_t i = 0; i < n; i++) {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols) {

//#pragma omp parallel for default(none) shared(num_rows, num_cols, alpha, beta) firstprivate(A,x,y)
    for(size_t row = 0; row < num_rows; row++) {
        double y_val = 0.0;
//#pragma omp parallel for default(none) shared(num_cols, alpha) firstprivate(A, row, x) reduction(+:y_val)
        for(size_t col = 0; col < num_cols; col++) {
            y_val += alpha * A[row * num_cols + col] * x[col];
        }
        y[row] = y_val + beta * y[row];
    }
}

void conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) {
    double alpha, beta, bb, rr, rr_new;
    auto * r = new double[size];
    auto * p = new double[size];
    auto * Ap = new double[size];
    int num_iters;

    for(size_t i = 0; i < size; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    bb = dot(b, b, size);
    rr = bb;
    auto start = std::chrono::high_resolution_clock::now();
    for(num_iters = 1; num_iters <= max_iters; num_iters++) {
        gemv(1.0, A, p, 0.0, Ap, size, size);
        alpha = rr / dot(p, Ap, size);
        axpby(alpha, p, 1.0, x, size);
        axpby(-alpha, Ap, 1.0, r, size);
        rr_new = dot(r, r, size);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / bb) < rel_error) { break; }
        axpby(1.0, r, beta, p, size);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Serial execution time = " <<
              std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;
    serial_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    delete[] r;
    delete[] p;
    delete[] Ap;
    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
}

void conjugate_gradient_v2(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) {
    double alpha, beta, bb, rr, rr_new;
    auto * r = new double[size];
    auto * p = new double[size];
    auto * Ap = new double[size];
    int num_iters;

    for(size_t i = 0; i < size; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    bb = dot(b, b, size);
    rr = bb;
    double dot_result1 = 0.0;
    rr_new = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) shared(max_iters, size, rel_error, A, p, Ap, x, r, dot_result1, rr_new) firstprivate(alpha, beta, rr, bb, num_iters)
    {
        int th_id = omp_get_thread_num();
        int th_n = omp_get_num_threads();
        for (num_iters = 1; num_iters <= max_iters; num_iters++) {

            /*
            #pragma omp for schedule(static, (int)size/4)
            for(size_t i = 0; i < size;i++) {
                Ap[i] = 0.0;
                for(size_t j = 0; j < size; j++) {
                    Ap[i] += A[j*size + i] * p[j];
                }
            }
             */


            for (size_t i = th_id; i < size; i += th_n) {
                Ap[i] = 0.0;
                for (size_t j = 0; j < size; j++) {
                    Ap[i] += A[j * size + i] * p[j];
                }
            }
#pragma omp barrier


#pragma omp single
            {
                dot_result1 = 0.0;
            }
#pragma omp for schedule(static, (int)size/4) reduction(+:dot_result1)
            for (size_t i = 0; i < size; i++) {
                dot_result1 += p[i] * Ap[i];
            }
            alpha = rr / dot_result1;


            /*
            #pragma omp for schedule(static, (int)size/4)
            for(size_t i = 0; i < size; i++) {
                x[i] = alpha * p[i] + x[i];
            }
             */
            for (size_t i = th_id; i < size; i += th_n) {
                x[i] = alpha * p[i] + x[i];
            }
#pragma omp barrier
            /*
            #pragma omp for schedule(static, (int)size/4)
            for(size_t i = 0; i < size; i++) {
                r[i] = -alpha * Ap[i] + r[i];
            }
             */
            for (size_t i = th_id; i < size; i += th_n) {
                r[i] = -alpha * Ap[i] + r[i];
            }
#pragma omp barrier


#pragma omp single
            {
                rr_new = 0.0;
            }
#pragma omp for schedule(static, (int)size/4) reduction(+:rr_new)
            for (size_t i = 0; i < size; i++) {
                rr_new += r[i] * r[i];
            }
#pragma omp barrier


            beta = rr_new / rr;
            rr = rr_new;
            if (std::sqrt(rr / bb) < rel_error) { break; }
            /*
            #pragma omp for schedule(static, (int)size/4)
            for(size_t i = 0; i < size; i++) {
                p[i] =  r[i] + beta * p[i];
            }
             */
            for (size_t i = th_id; i < size; i += th_n) {
                p[i] = r[i] + beta * p[i];

            }
#pragma omp barrier
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel execution time = " <<
              std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;
    parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    delete[] r;
    delete[] p;
    delete[] Ap;
    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr_new / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr_new / bb));
    }
}

void print_sol_head(size_t n, double* sol) {
    for(size_t i = 0; i < n; i++) {
        std::cout << sol[i] << std::endl;
    }
}


int main() {
    size_t size = 250;
    int max_iters = 2000;
    double tol = 1e-6;
    double* matrix;
    double* rhs;
    generate_matrix(size, &matrix);
    generate_rhs(size, 2.0, &rhs);
    auto* sol1 = new double[size];
    auto* sol2 = new double[size];
    std::cout << "starting conjugate gradient" << std::endl;
    transpose_matrix(size, matrix);
    conjugate_gradient_v2(matrix, rhs, sol1, size, max_iters, tol);
    print_sol_head(5, sol1);
    conjugate_gradient(matrix, rhs, sol2, size, max_iters, tol);
    print_sol_head(5, sol2);

    std::cout << "speedup: " <<(double)serial_time/(double)parallel_time << std::endl;
    return 0;
}

