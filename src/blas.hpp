#ifndef BLAS_H
#define BLAS_H
#include "darknet.hpp"



void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void scal_add_cpu(int N, float ALPHA, float BETA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void shortcut_multilayer_cpu(int size, int src_outputs, int batch, int n, int *outputs_of_layers, float **layers_output, float *out, float *in, float *weights, int nweights, WEIGHTS_NORMALIZATION_T weights_normalizion);
void backward_shortcut_multilayer_cpu(int size, int src_outputs, int batch, int n, int *outputs_of_layers,
    float **layers_delta, float *delta_out, float *delta_in, float *weights, float *weight_updates, int nweights, float *in, float **layers_output, WEIGHTS_NORMALIZATION_T weights_normalizion);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);

void add_bias(float *output, float *biases, int batch, int n, int size);
void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

#endif
