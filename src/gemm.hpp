#ifndef GEMM_H
#define GEMM_H
#include "activations.hpp"
#include <stdint.h>
#include <stddef.h>

int is_avx();
int is_fma_avx2();

void activate_array_cpu_custom(float *x, const int n, const ACTIVATION a);


void forward_maxpool_layer_avx(float *src, float *dst, int *indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch);


void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc);


#endif
