#ifndef MATRIX_H
#define MATRIX_H
#include "darknet.hpp"

matrix make_matrix(int rows, int cols);

void free_matrix(matrix m);

#endif
