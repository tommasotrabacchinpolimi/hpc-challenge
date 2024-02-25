
#define UNROLL 8
#define UNROLL_OUTER_LOOP 4
#define UNROLL_INNER_LOOP 4



__kernel void conjugate_gradient_kernel(__global const double * __restrict__ A, __global const double * __restrict__ b, __global double * __restrict__ x, unsigned size, __global double * __restrict__ r, __global double * __restrict__ p, __global double* __restrict__ Ap, unsigned max_iters, double squared_tol, double bb)
{
    double alpha, beta, rr, rr_new;
    int num_iters;

    rr = bb;



    for(num_iters = 1; num_iters <= max_iters; num_iters++) {

        //matrix vector multiplication
#pragma unroll UNROLL_OUTER_LOOP
        for(unsigned row = 0; row < size; row++) {
            Ap[row] = 0.0;
#pragma unroll UNROLL_INNER_LOOP
            for(unsigned col = 0; col < size; col++) {
                Ap[row] += A[col*size+row]*p[col];
            }
        }

        double tmp_dot_result = 0;
#pragma unroll UNROLL
        for(unsigned i = 0; i < size; i++) {
            tmp_dot_result += p[i]*Ap[i];
        }
        alpha = rr/tmp_dot_result;

#pragma unroll UNROLL
        for(unsigned i = 0; i < size; i++) {
            x[i] += alpha * p[i];
        }

#pragma unroll UNROLL
        for(unsigned i = 0; i < size; i++) {
            r[i] += -alpha * Ap[i];
        }

        rr_new = 0;
#pragma unroll UNROLL
        for(unsigned i = 0; i < size; i++) {
            rr_new += r[i] * r[i];
        }

        beta = rr_new /  rr;
        rr_new = rr;
        if(rr/bb < squared_tol) {break;}

#pragma unroll UNROLL
        for(unsigned i = 0; i < size; i++) {
            p[i] = r[i] + beta*p[i];
        }
    }


}