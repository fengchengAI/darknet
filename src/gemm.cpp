#include "gemm.hpp"
#include "utils.hpp"
#include "im2col.hpp"
#include <stdio.h>
#include <float.h>
#include <stdint.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#define TILE_M 4 // 4 ops
#define TILE_N 16 // AVX2 = 2 ops * 8 floats
#define TILE_K 16 // loop
#define PUT_IN_REGISTER




void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{//矩阵乘法
    //
    /** TA，TA分表表示对AB转置 ，1表示转置
    M是输入A,C的行数，具体含义为卷积核的个数(l.n)；
	N是输入B,C的列数，具体含义为每个卷积核元素个数乘以输入图像的通道数(l.size*l.size*l.c)；
	K是输入A的列数也是B的行数，具体含义为每个输出特征图的元素个数（l.out_w*l.out_h）；
    C=ALPHA*A*B+BETA*C
     */
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}






int is_avx() {
    return 0;
}

int is_fma_avx2() {
    return 0;
}

void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    /*
     T = {0,...,M}
     M = 1;
     N = out_h*out_w;
     K = l.size*l.size*l.c / l.groups;
     ALPHA
     A = l.weights +j*l.nweights / l.groups + T * LDA
     LDA = l.size*l.size*l.c / l.groups;
     B = state.workspace;
     LDB = out_h*out_w;
     BETA
     C = l.output +(i*l.groups + j)*n*m +  T * LDC
     LDC = out_h*out_w;
     ALPHA
     */
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}

void gemm_nn_fast(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    #pragma omp parallel for
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            PUT_IN_REGISTER float A_PART = ALPHA*A[i*lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}


void activate_array_cpu_custom(float *x, const int n, const ACTIVATION a)
{
    int i;
    if (a == LINEAR)
    {
    }
    else if (a == LEAKY)
    {
        for (i = 0; i < n; ++i) {
            x[i] = (x[i]>0) ? x[i] : .1*x[i];
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}




void forward_maxpool_layer_avx(float *src, float *dst, int *indexes, int size, int w, int h, int out_w, int out_h, int c,
    int pad, int stride, int batch)
{
    int b, k;
    const int w_offset = -pad / 2;
    const int h_offset = -pad / 2;

    for (b = 0; b < batch; ++b) {
        #pragma omp parallel for
        for (k = 0; k < c; ++k) {
            int i, j, m, n;
            for (i = 0; i < out_h; ++i) {
                for (j = 0; j < out_w; ++j) {
                    int out_index = j + out_w*(i + out_h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (n = 0; n < size; ++n) {
                        for (m = 0; m < size; ++m) {
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + w*(cur_h + h*(k + b*c));
                            int valid = (cur_h >= 0 && cur_h < h &&
                                cur_w >= 0 && cur_w < w);
                            float val = (valid != 0) ? src[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    dst[out_index] = max;
                    if (indexes) indexes[out_index] = max_i;
                }
            }
        }
    }
}



void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            PUT_IN_REGISTER float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            PUT_IN_REGISTER float A_PART = ALPHA * A[k * lda + i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            PUT_IN_REGISTER float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    /*
     TA, 0
     TB, 0
     M = l.n / l.groups;
     N = out_h*out_w;
     K = l.size*l.size*l.c / l.groups;
     ALPHA
     A = l.weights +j*l.nweights / l.groups;
     LDA = l.size*l.size*l.c / l.groups;
     B = state.workspace;
     LDB = out_h*out_w;
     BETA
     C = l.output +(i*l.groups + j)*n*m;
     LDC = out_h*out_w;
     */
    if (BETA != 1){
        int i, j;
        for(i = 0; i < M; ++i){
            for(j = 0; j < N; ++j){
                C[i*ldc + j] *= BETA;
            }
        }
    }

    is_avx();   // initialize static variable
    if (is_fma_avx2() && !TA && !TB) {
        gemm_nn_fast(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else {
        int t;
        #pragma omp parallel for
        for (t = 0; t < M; ++t) {
            if (!TA && !TB)//1
                gemm_nn(1, N, K, ALPHA, A + t*lda, lda, B, ldb, C + t*ldc, ldc);
            else if (TA && !TB)
                gemm_tn(1, N, K, ALPHA, A + t, lda, B, ldb, C + t*ldc, ldc);
            else if (!TA && TB)
                gemm_nt(1, N, K, ALPHA, A + t*lda, lda, B, ldb, C + t*ldc, ldc);
            else
                gemm_tt(1, N, K, ALPHA, A + t, lda, B, ldb, C + t*ldc, ldc);
        }
    }
}



void init_cpu() {
    is_avx();
    is_fma_avx2();
}