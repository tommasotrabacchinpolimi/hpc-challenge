#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <bits/stdc++.h>



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



double dot(const double * x, const double * y, size_t size)
{
    double result = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}



void axpby(double alpha, const double * x, double beta, double * y, size_t size)
{
    // y = alpha * x + beta * y

    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}



void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols)
{
    // y = alpha * A * x + beta * y;

    for(size_t r = 0; r < num_rows; r++)
    {
        double y_val = 0.0;
        for(size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[c];
        }
        y[r] = beta * y[r] + y_val;
    }
}



void conjugate_gradients_parallel(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error, int threads_number, long* execution_time) {
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
    double dot_result = 0.0;
    rr_new = 0.0;
    int total_iterations = 0;
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel default(none) shared(max_iters, size, rel_error, A, p, Ap, x, r, dot_result, rr_new, total_iterations) firstprivate(alpha, beta, rr, bb, num_iters) num_threads(4)
    {
        for (num_iters = 1; num_iters <= max_iters; num_iters++) {

            #pragma omp for nowait
            for (size_t i = 0; i < size; i += 1) {
                Ap[i] = 0.0;
                for (size_t j = 0; j < size; j++) {
                    Ap[i] += A[i * size + j] * p[j];
                }
            }



            #pragma omp single
            {
                dot_result = 0.0;
            }

            #pragma omp for reduction(+:dot_result)
            for (size_t i = 0; i < size; i++) {
                dot_result += p[i] * Ap[i];
            }
            alpha = rr / dot_result;


            #pragma omp for nowait
            for(size_t i = 0; i < size; i++) {
                x[i] = alpha * p[i] + x[i];
            }


            #pragma omp for nowait
            for(size_t i = 0; i < size; i++) {
                r[i] = -alpha * Ap[i] + r[i];
            }

            #pragma omp single
            {
                rr_new = 0.0;
            }

            #pragma omp for reduction(+:rr_new)
            for (size_t i = 0; i < size; i++) {
                rr_new += r[i] * r[i];
            }


            beta = rr_new / rr;
            rr = rr_new;
            if (std::sqrt(rr / bb) < rel_error) {
                #pragma omp single
                {
                    total_iterations = num_iters;
                }
                break; }

            #pragma omp for
            for(size_t i = 0; i < size; i++) {
                p[i] =  r[i] + beta * p[i];
            }
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    *execution_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    delete[] r;
    delete[] p;
    delete[] Ap;
    if(total_iterations <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", total_iterations, std::sqrt(rr_new / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", total_iterations, std::sqrt(rr_new / bb));
    }
}



void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error, long* execution_time)
{
    double alpha, beta, bb, rr, rr_new;
    double * r = new double[size];
    double * p = new double[size];
    double * Ap = new double[size];
    int num_iters;

    for(size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    bb = dot(b, b, size);
    rr = bb;
    auto start = std::chrono::high_resolution_clock::now();
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
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
    *execution_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

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





int main(int argc, char ** argv)
{
    printf("Usage: size max_iters rel_error\n");
    printf("\n");

    int size = 5000;
    int max_iters = 1000;
    double rel_error = 1e-9;
    int trials = 1;
    int blank_trials = 0;
    int threads_number = 1;


    if(argc > 1) size = atoi(argv[1]);
    if(argc > 2) max_iters = atoi(argv[2]);
    if(argc > 3) rel_error = atof(argv[3]);
    if(argc > 4) trials = atoi(argv[4]);
    if(argc > 5) blank_trials = atoi(argv[5]);
    if(argc > 6) threads_number = atoi(argv[6]);

    printf("Command line arguments:\n");
    printf("  matrix_size: %d\n", size);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("  trials number:         %d\n", trials);
    printf("  blank trials number:         %d\n", blank_trials);
    printf("  threads number:         %d\n", threads_number);
    printf("\n");

    double* matrix;
    double* rhs;
    generate_matrix(size, &matrix);
    generate_rhs(size, 2.0, &rhs);
    auto* sol = new double[size];
    long serial_execution_time = 0;
    long parallel_execution_time = 0;



    for(int i = 0; i < trials; i++) {
        long tmp;
        conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error, &tmp);
        serial_execution_time += tmp;
        memset(sol, 0, sizeof(double) * size);
    }

    for(int i = 0; i < trials; i++) {
        long tmp;
        conjugate_gradients_parallel(matrix, rhs, sol, size, max_iters, rel_error, threads_number, &tmp);
        parallel_execution_time += tmp;
        memset(sol, 0, sizeof(double) * size);
    }

    std::cout << "Serial average execution time: " << (double)serial_execution_time/trials << std::endl;
    std::cout << "Parallel average execution time: " << (double)parallel_execution_time/trials << std::endl;
    std::cout << "Speedup: " << (double)serial_execution_time/(double)parallel_execution_time << std::endl;
    printf("Finished successfully\n");

    return 0;
}