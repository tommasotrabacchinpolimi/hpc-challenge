//
// Created by tomma on 14/02/2024.
//
#include <iostream>
#include <cuda.h>


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

template<int blockSize>
__device__ void row_column_mult(const double* A, unsigned int row, int size, const double* p, double* Ap) {
    __shared__ double sArr[blockSize];
    __shared__ double partial;
    int iter_n = 0;
    if(threadIdx.x == 0) {
        partial = 0.0;
    }

    for(unsigned int i = threadIdx.x; iter_n*blockSize < size; i+=blockSize) {
        sArr[threadIdx.x] = (i<size)?A[row*size + i]*p[i]:0.0;
        for (unsigned int stride = blockSize/2; stride >= 1;
             stride = stride>>1)
        {

            __syncthreads();
            if (threadIdx.x < stride)
                sArr[threadIdx.x] += sArr[threadIdx.x+stride];
        }
        iter_n++;
        __syncthreads();
        if(threadIdx.x == 0) {
            partial += sArr[0];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        Ap[row] = partial;
    }

}

template<int gridSize, int blockSize>
__global__ void matrix_vector_kernel(double* A, double* p, double* Ap, int size) {
    for(unsigned int i = blockIdx.x; i < size; i+=gridSize) {
        row_column_mult<blockSize>(A,i,size,p,Ap);
    }

}

template<int gridSize, int blockSize>
void matrix_vector_mult(double* A, double* p, double* Ap, int size) {
    matrix_vector_kernel<gridSize, blockSize><<<gridSize, blockSize>>>(A, p, Ap, size);
}


template<int blockSize>
__global__ void sumArray(const double* array, int size, double* result) {
    __shared__ double sArr[blockSize];
    __shared__ double partial;
    int iter_n = 0;
    if(threadIdx.x == 0) {
        partial = 0;
    }
    sArr[threadIdx.x] = 0.0;
    for(unsigned int i = threadIdx.x; iter_n*blockSize < size; i+=blockSize) {
        sArr[threadIdx.x] = (i<size)?array[i]:0.0;
        for (unsigned int stride = blockSize/2; stride >= 1;
             stride = stride>>1)
        {

            __syncthreads();
            if (threadIdx.x < stride)
                sArr[threadIdx.x] += sArr[threadIdx.x+stride];
        }
        iter_n++;
        __syncthreads();
        if(threadIdx.x == 0) {
            partial += sArr[0];
        }
        __syncthreads();

    }
    if(threadIdx.x == 0) {
        *result = partial;
    }
}


template<int gridSize, int blockSize>
__global__ void dot_product_kernel(const double* x, const double* y, double* outArray, int size) {
    __shared__ double sArr[blockSize];
    if(threadIdx.x == 0) {
        outArray[blockIdx.x] = 0.0;
    }
    for(unsigned int i = blockIdx.x; blockSize*i < size; i+=gridSize) {
        sArr[threadIdx.x] = (i*blockSize + threadIdx.x<size)?x[i*blockSize + threadIdx.x]*y[i*blockSize + threadIdx.x]:0.0;
        for (unsigned int stride = blockSize/2; stride >= 1;
             stride = stride>>1)
        {

            __syncthreads();
            if (threadIdx.x < stride)
                sArr[threadIdx.x] += sArr[threadIdx.x+stride];
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            outArray[blockIdx.x] += sArr[0];
        }
        __syncthreads();
    }
}

template<int gridSize, int blockSize>
void dot_product(double* x, double* y, double* outArray, int size, double* result) {
    dot_product_kernel<gridSize, blockSize><<<gridSize, blockSize>>>(x, y, outArray, size);
    sumArray<blockSize><<<1, blockSize>>>(outArray, gridSize, result);
}


template<int gridSize, int blockSize>
void axpby(double )


int main() {
    int size = 2000;
    int* size_cuda;
    int max_iters = 2000;
    int* max_iters_cuda;
    double tol = 1e-6;
    double* tol_cuda;
    double* matrix;
    double* matrix_cuda;
    double* rhs;
    double* rhs_cuda;
    double* r_cuda;
    double* p_cuda;
    double* Ap_cuda;
    generate_matrix(size, &matrix);
    generate_rhs(size, 2.0, &rhs);
    auto* sol = new double[size];
    double* sol_cuda;

    for(int i = 0; i < size; i++) {
        sol[i] = 1.0;
    }



    cudaMalloc(&matrix_cuda, size*size*sizeof(double));
    cudaMalloc(&rhs_cuda, size*sizeof(double));
    cudaMalloc(&sol_cuda, size*sizeof(double));
    cudaMalloc(&max_iters_cuda, sizeof(int));
    cudaMalloc(&size_cuda, sizeof(int));
    cudaMalloc(&tol_cuda, sizeof(double));
    cudaMalloc(&r_cuda, size*sizeof(double));
    cudaMalloc(&p_cuda, size*sizeof(double));
    cudaMalloc(&Ap_cuda, size*sizeof(double));
    cudaMemcpy(matrix_cuda, matrix, size*size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rhs_cuda, rhs, size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(sol_cuda, sol, size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(max_iters_cuda, &max_iters, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(size_cuda, &size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tol_cuda, &tol, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(r_cuda, rhs, size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(p_cuda, rhs, size*sizeof(double), cudaMemcpyHostToDevice);

    std::cout << "starting dot product" << std::endl;
    dot_product<2, 1024>(sol_cuda, sol_cuda, rhs_cuda, size, tol_cuda);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << err << std::endl;
        std::cout << "execution failed" << std::endl;
        exit(1);
    }
    cudaMemcpy(&tol, tol_cuda, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << tol << std::endl;
}
