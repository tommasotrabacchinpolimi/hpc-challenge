#define ROWS_GROUP_SIZE 64
#define RHS_GROUP_SIZE 64
#define COLUMNS_GROUP_SIZE 64
//#include "../../include/CL/opencl.h"

__kernel void gemv(__global const double * a, __global const double * x, __global double * y, int ncols, int nrows)
{
    double rhs_group[RHS_GROUP_SIZE] = {0};
    double row_group[COLUMNS_GROUP_SIZE] = {0};
    double reduce_array[COLUMNS_GROUP_SIZE/2] = {0};
    double sum = 0;
    for(int i = 0; i < ncols; i += RHS_GROUP_SIZE) {

        for(int j = 0; j < RHS_GROUP_SIZE && i + j < ncols; j++) {
            rhs_group[j] = x[i + j];
        }

        for(int j = 0; j < nrows; j += ROWS_GROUP_SIZE) {
            for (int l = 0; l < ROWS_GROUP_SIZE && j + l < nrows; l++) {
                for(int k = i ; k < i + RHS_GROUP_SIZE; k+= COLUMNS_GROUP_SIZE) {
                    sum = 0;
                    for(int t = 0; t < COLUMNS_GROUP_SIZE; t++) {
                        row_group[t] = ((t + k < i + RHS_GROUP_SIZE)?a[(j+l)*ncols + k + t]:0.0) * ((k + t < ncols)?rhs_group[k - i + t]:0.0);
                    }

                    for(int t = 0; t  < COLUMNS_GROUP_SIZE/2; t++) {
                        reduce_array[t] = row_group[2*t] + row_group[2*t + 1];
                    }

                    for(int t = 0; t < COLUMNS_GROUP_SIZE/2; t++) {
                        sum += reduce_array[t];
                    }

                    y[j + l] += sum;
                }
            }
        }
    }
}