#define ROWS_GROUP_SIZE 64
#define RHS_GROUP_SIZE 64
#define COLUMNS_GROUP_SIZE 64
//#include "../../include/CL/opencl.h"

__kernel void gemv(__global const cl_double * a, __global const cl_double * x, __global cl_double * y, cl_uint ncols, cl_uint nrows)
{
    cl_double rhs_group[RHS_GROUP_SIZE] = {0};
    cl_double row_group[COLUMNS_GROUP_SIZE] = {0};
    cl_double reduce_array[COLUMNS_GROUP_SIZE/2] = {0};
    cl_double sum = 0;
    for(cl_uint i = 0; i < ncols; i += RHS_GROUP_SIZE) {

        for(cl_uint j = 0; j < RHS_GROUP_SIZE && i + j < ncols; j++) {
            rhs_group[j] = x[i + j];
        }

        for(cl_uint j = 0; j < nrows; j += ROWS_GROUP_SIZE) {
            for (cl_uint l = 0; l < ROWS_GROUP_SIZE && j + l < nrows; l++) {
                for(cl_uint k = i ; k < i + RHS_GROUP_SIZE; k+= COLUMNS_GROUP_SIZE) {
                    sum = 0;
                    for(cl_uint t = 0; t < COLUMNS_GROUP_SIZE; t++) {
                        row_group[t] = ((t + k < i + RHS_GROUP_SIZE)?a[(j+l)*ncols + k + t]:0.0) * ((k + t < ncols)?rhs_group[k - i + t]:0.0);
                    }

                    for(cl_uint t = 0; t  < COLUMNS_GROUP_SIZE/2; t++) {
                        reduce_array[t] = row_group[2*t] + row_group[2*t + 1];
                    }

                    for(cl_uint t = 0; t < COLUMNS_GROUP_SIZE/2; t++) {
                        sum += reduce_array[t];
                    }

                    y[j + l] += sum;
                }
            }
        }
    }
}