#include "matrix.hpp"
#include "utils.hpp"

void free_matrix(matrix m)
{
    int i;
    for(i = 0; i < m.rows; ++i) free(m.vals[i]);
    free(m.vals);
}


matrix make_matrix(int rows, int cols)
{
    int i;
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.vals = (float**)xcalloc(m.rows, sizeof(float*));
    for(i = 0; i < m.rows; ++i){
        m.vals[i] = (float*)xcalloc(m.cols, sizeof(float));
    }
    return m;
}



