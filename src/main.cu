
#include <iostream>
#include <cuda.h>
#include <chrono>
#define GRID_SIZE 3
#define BLOCK_SIZE 1024


void check_cuda(const std::string& msg) {
    cudaDeviceSynchronize();
    cudaError_t err;
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cout << "cuda error: " << msg << std::endl;
        std::cout << "description: " << err << std::endl;
    }
}

float dot(const float * x, const float * y, size_t size)
{
    float result = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}



void axpby(float alpha, const float * x, float beta, float * y, size_t size)
{
    // y = alpha * x + beta * y

    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}



void gemv(float alpha, const float * A, const float * x, float beta, float * y, size_t num_rows, size_t num_cols)
{
    // y = alpha * A * x + beta * y;

    for(size_t r = 0; r < num_rows; r++)
    {
        float y_val = 0.0;
        for(size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[c];
        }
        y[r] = beta * y[r] + y_val;
    }
}

void generate_matrix(size_t n, float** matrix_out) {
    auto* matrix = new float[n * n];
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

void generate_rhs(size_t n, float value, float** rhs_out) {
    auto* rhs = new float[n];
    for(size_t i = 0; i < n; i++) {
        rhs[i] = value;
    }
    *rhs_out = rhs;
}



template<int blockSize>
__device__ void row_column_mult(const float* A, unsigned int row, int size, const float* p, float* Ap) {
    __shared__ float sArr[blockSize];
    __shared__ float partial;
    int iter_n = 0;
    if(threadIdx.x == 0) {
        partial = 0.0;
    }

    for(unsigned int i = threadIdx.x; iter_n < size; i+=blockSize) {
        sArr[threadIdx.x] = (i<size)?A[row*size + i]*p[i]:0.0;
        for (unsigned int stride = blockSize/2; stride >= 1;
             stride = stride>>1)
        {

            __syncthreads();
            if (threadIdx.x < stride)
                sArr[threadIdx.x] += sArr[threadIdx.x+stride];
        }
        iter_n += blockSize;
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
__global__ void matrix_vector_kernel(const float* A, float* p, float* Ap, int size) {
    for(unsigned int i = blockIdx.x; i < size; i+=gridSize) {
        row_column_mult<blockSize>(A,i,size,p,Ap);
    }

}

template<int gridSize, int blockSize>
void matrix_vector_mult(const float* A, float* p, float* Ap, int size, cudaStream_t stream) {
    matrix_vector_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(A, p, Ap, size);
}


template<int blockSize>
__global__ void sumArray(const float* array, int size, float* result) {
    __shared__ float sArr[blockSize];
    __shared__ float partial;
    int iter_n = 0;
    if(threadIdx.x == 0) {
        partial = 0;
    }
    sArr[threadIdx.x] = 0.0;
    for(unsigned int i = threadIdx.x; iter_n < size; i+=blockSize) {
        sArr[threadIdx.x] = (i<size)?array[i]:0.0;
        for (unsigned int stride = blockSize/2; stride >= 1;
             stride = stride>>1)
        {

            __syncthreads();
            if (threadIdx.x < stride)
                sArr[threadIdx.x] += sArr[threadIdx.x+stride];
        }
        iter_n += blockSize;
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
__global__ void dot_product_kernel(const float* x, const float* y, float* outArray, int size) {
    __shared__ float sArr[blockSize];
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
void dot_product(const float* x, const float* y, float* outArray, int size, float* result, cudaStream_t stream) {
    dot_product_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(x, y, outArray, size);
    sumArray<blockSize><<<1, blockSize>>>(outArray, gridSize, result);
}

template<int gridSize, int blockSize>
__global__ void axpby_kernel(const float* alpha, const float* x, float* y, int size) {
    int th_id = threadIdx.x + blockIdx.x * blockSize;
    if(th_id < size) {
        y[th_id] = (*alpha) * x[th_id] + y[th_id];
    }
}


template<int gridSize, int blockSize>
__global__ void _minus_axpby_kernel(const float* alpha, const float* x, float* y, int size) {
    int th_id = threadIdx.x + blockIdx.x * blockSize;
    if(th_id < size) {
        y[th_id] = -(*alpha) * x[th_id] + y[th_id];
    }
}

template<int gridSize, int blockSize>
__global__ void xpby_kernel( const float* x, float* y, const float* beta, int size) {
    int th_id = threadIdx.x + blockIdx.x * blockSize;
    if(th_id < size) {
        y[th_id] = (x[th_id] + (*beta) * y[th_id]);
    }
}



__global__ void divide(float* div1, float* div2, float* result) {
    if(threadIdx.x == 0) {
        *result = *div1 / *div2;
    }

}

void matrix_vector(float* matrix, float* vector, float* sol, int size) {
    for(int i = 0; i < size; i++) {
        sol[i] = 0;
        for(int j = 0; j < size; j++) {
            sol[i] += matrix[i*size + j] * vector[j];
        }
    }
}

template<int gridSize, int blockSize>
void axpby(float* alpha, const float * x, float * y, int size, cudaStream_t stream)
{
    axpby_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(alpha, x, y, size);
}

template<int gridSize, int blockSize>
void _minus_axpby(float* alpha, const float * x, float * y, int size, cudaStream_t stream)
{
    _minus_axpby_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(alpha, x, y, size);
}


template<int gridSize, int blockSize>
void xpby(const float * x, float * y, const float* beta, int size, cudaStream_t stream)
{
    xpby_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(x, y, beta, size);
}



void conjugate_gradients_serial(const float * A, const float * b, float * x, size_t size, int max_iters, float rel_error, long* execution_time)
{
    float alpha, beta, bb, rr, rr_new;
    float * r = new float[size];
    float * p = new float[size];
    float * Ap = new float[size];
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

void conjugate_gradients(const float * A, const float * b, float * x, size_t size, int max_iters, float rel_error, long* execution_time) {
    float* r_cuda;
    float* p_cuda;
    float* Ap_cuda;
    float *alpha;
    float *beta;
    float* bb;
    float bb_cpu;
    float* rr;
    float* rr_new;
    float* dot_product_out_array;
    float err;
    cudaMalloc(&r_cuda, size*sizeof(float));
    cudaMalloc(&p_cuda, size*sizeof(float));
    cudaMalloc(&Ap_cuda, size*sizeof(float));
    cudaMalloc(&dot_product_out_array, sizeof(float)*GRID_SIZE);
    cudaMalloc(&alpha, sizeof(float));
    cudaMalloc(&beta, sizeof(float));
    cudaMalloc(&bb, sizeof(float));
    cudaMalloc(&rr, sizeof(float));
    cudaMalloc(&rr_new, sizeof(float));
    cudaMemcpy(r_cuda, b, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p_cuda, b, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(x,0,sizeof(float) * size);
    int niters;
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    dot_product<GRID_SIZE, BLOCK_SIZE>(b, b, dot_product_out_array, (int) size, bb, stream1);
    cudaMemcpy(&bb_cpu, bb, sizeof(float), cudaMemcpyDeviceToHost);
    err = bb_cpu;
    cudaMemcpy(rr, bb, sizeof(float), cudaMemcpyDeviceToDevice);
    auto start = std::chrono::high_resolution_clock::now();
    for(niters = 1; niters < max_iters; niters++) {
        matrix_vector_mult<GRID_SIZE, BLOCK_SIZE>(A, p_cuda, Ap_cuda, (int)size, stream1);
        dot_product<GRID_SIZE, BLOCK_SIZE>(p_cuda, Ap_cuda, dot_product_out_array,(int)size, alpha, stream1);
        divide<<<1,1, 0, stream1>>>(rr,alpha, alpha);
        axpby<GRID_SIZE, BLOCK_SIZE>(alpha, p_cuda, x, (int)size, stream1);
        _minus_axpby<GRID_SIZE, BLOCK_SIZE>(alpha, Ap_cuda, r_cuda, (int) size, stream1);
        dot_product<GRID_SIZE, BLOCK_SIZE>(r_cuda, r_cuda, dot_product_out_array, (int)size, rr_new, stream1);
        divide<<<1, 1, 0, stream1>>>(rr_new, rr, beta);
        cudaMemcpy(rr, rr_new, sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&err, rr, sizeof(float), cudaMemcpyDeviceToHost);
        if(std::sqrt(err / bb_cpu) < rel_error) { break; }
        xpby<GRID_SIZE, BLOCK_SIZE>(r_cuda, p_cuda, beta,  (int)size, stream1);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    if(niters < max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", niters, std::sqrt(err / bb_cpu));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(err / bb_cpu));
    }
    *execution_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    cudaFree(r_cuda);
    cudaFree(p_cuda);
    cudaFree(Ap_cuda);
    cudaFree(dot_product_out_array);
    cudaFree(alpha);
    cudaFree(beta);
    cudaFree(bb);
    cudaFree(rr);
    cudaFree(rr_new);
}




int main(int argc, char ** argv) {

    int size = 500;
    int max_iters = 1000;
    float rel_error = 1e-9;
    int serial_trials = 1;
    int parallel_trials = 1;
    if(argc > 1) size = atoi(argv[1]);
    if(argc > 2) max_iters = atoi(argv[2]);
    if(argc > 3) rel_error = atof(argv[3]);
    if(argc > 4) serial_trials = atoi(argv[4]);
    if(argc > 5) parallel_trials = atoi(argv[5]);

    printf("Command line arguments:\n");
    printf("  matrix_size: %d\n", size);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("  serial trials number:         %d\n", serial_trials);
    printf("  parallel trials number:         %d\n", parallel_trials);
    printf("\n");

    long serial_execution_time = 0;
    long parallel_execution_time = 0;

    int* size_cuda;
    int* max_iters_cuda;
    float* tol_cuda;
    float* matrix;
    float* matrix_cuda;
    float* rhs;
    float* rhs_cuda;
    float* r_cuda;
    float* p_cuda;
    float* Ap_cuda;
    generate_matrix(size, &matrix);
    generate_rhs(size, 2.0, &rhs);
    auto* sol = new float[size];
    float* sol_cuda;

    for(int i = 0; i < size; i++) {
        sol[i] = 1.0;
    }

    cudaMalloc(&matrix_cuda, size*size*sizeof(float));
    cudaMalloc(&rhs_cuda, size*sizeof(float));
    cudaMalloc(&sol_cuda, size*sizeof(float));
    cudaMalloc(&max_iters_cuda, sizeof(int));
    cudaMalloc(&size_cuda, sizeof(int));
    cudaMalloc(&tol_cuda, sizeof(float));
    cudaMalloc(&r_cuda, size*sizeof(float));
    cudaMalloc(&p_cuda, size*sizeof(float));
    cudaMalloc(&Ap_cuda, size*sizeof(float));
    cudaMemcpy(matrix_cuda, matrix, size*size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rhs_cuda, rhs, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sol_cuda, sol, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(max_iters_cuda, &max_iters, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(size_cuda, &size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tol_cuda, &rel_error, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(r_cuda, rhs, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p_cuda, rhs, size*sizeof(float), cudaMemcpyHostToDevice);


    for(int i = 0; i < serial_trials; i++) {
        long tmp;
        conjugate_gradients_serial(matrix, rhs, sol, size, max_iters, rel_error, &tmp);
        serial_execution_time += tmp;
        memset(sol, 0, sizeof(float) * size);
    }
    for(int i = 0; i < parallel_trials; i++) {
        long tmp;
        conjugate_gradients(matrix_cuda, rhs_cuda, sol_cuda, size, max_iters, rel_error, &tmp);
        parallel_execution_time += tmp;
        memset(sol, 0, sizeof(float) * size);
    }

    std::cout << "check" << std::endl;
    check_cuda("error");
    std::cout << "Serial average execution time: " << (float)serial_execution_time/serial_trials << std::endl;
    std::cout << "Parallel average execution time: " << (float)parallel_execution_time/parallel_trials << std::endl;
    std::cout << "Speedup: " << (float)((float)serial_execution_time/serial_trials)/((float)parallel_execution_time/parallel_trials) << std::endl;
    printf("Finished successfully\n");


}
