
#define SIZE 32

__kernel void conjugate_gradient_kernel(__global const double * __restrict__ A, __global const double * __restrict__ b, __global double * __restrict__ x, unsigned size, __global double * __restrict__ r, __global double * __restrict__ p, __global double* __restrict__ Ap, unsigned max_iters, double squared_tol, double bb)
{
    double alpha, beta, rr, rr_new;
    int num_iters;

    rr = bb;



    for(num_iters = 1; num_iters <= max_iters; num_iters++) {

        //matrix vector multiplication
        for(unsigned row = 0; row < SIZE; row++) {
            Ap[row] = 0.0;
#pragma unroll 2
            for(unsigned col = 0; col < SIZE; col++) {
                Ap[row] += A[col*SIZE+row]*p[col];
            }
        }

        double tmp_dot_result = 0;
        for(unsigned i = 0; i < SIZE; i++) {
            tmp_dot_result += p[i]*Ap[i];
        }
        alpha = rr/tmp_dot_result;

        for(unsigned i = 0; i < SIZE; i++) {
            x[i] += alpha * p[i];
        }

        for(unsigned i = 0; i < SIZE; i++) {
            r[i] += -alpha * Ap[i];
        }

        rr_new = 0;
        for(unsigned i = 0; i < SIZE; i++) {
            rr_new += r[i] * r[i];
        }

        beta = rr_new /  rr;
        rr_new = rr;
        if(rr/bb < squared_tol) {break;}

        for(unsigned i = 0; i < SIZE; i++) {
            p[i] = r[i] + beta*p[i];
        }
    }


}