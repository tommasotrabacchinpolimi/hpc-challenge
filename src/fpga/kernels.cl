
#define ROWS_GROUP_SIZE 16
#define RHS_GROUP_SIZE 64
#define COLUMNS_GROUP_SIZE 16
#define FIRST_REDUCE_SIZE 8
__kernel void gemv(__global const double * a, __global const double * x, __global double * y, int ncols, int nrows)
{
    double rhs_group[RHS_GROUP_SIZE];
    double row_group[ROWS_GROUP_SIZE];
    double reduce_array[FIRST_REDUCE_SIZE] = {0};
    double sum = 0;
    for(int i = 0; i < ncols; i += RHS_GROUP_SIZE) {
        for(int j = 0; j < RHS_GROUP_SIZE; j++) {
            rhs_group[j] =x[i + j];
        }

        for(int j = 0; j < nrows; j += ROWS_GROUP_SIZE) {
            for(int k = 0; k < ROWS_GROUP_SIZE; k++) {
                for(int l = i; l < i + RHS_GROUP_SIZE; l += COLUMNS_GROUP_SIZE) {
                    for(int t = 0; t < COLUMNS_GROUP_SIZE; t++) {
                        row_group[t] = a[(j+k)*ncols + l + t];
                    }

                    for(int p = 0; p < COLUMNS_GROUP_SIZE; p += 2) {
                        reduce_array[p/2] = row_group[p]*rhs_group[l - i + p] + row_group[p + 1]*rhs_group[l - i + p + 1];
                    }

                    for(int p = 0; p < FIRST_REDUCE_SIZE; p++) {
                        sum += reduce_array[p];
                    }
                    y[j] = sum;
                }
            }
        }

    }
}