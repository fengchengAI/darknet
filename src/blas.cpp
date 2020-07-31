#include "blas.hpp"
#include "utils.hpp"

#include <cmath>
#include <cassert>
#include <cfloat>
#include <cstdlib>
#include <cstring>

static float relu(float src) {
    if (src > 0) return src;
    return 0;
}

void shortcut_multilayer_cpu(int size, int src_outputs, int batch, int n, int *outputs_of_layers, float **layers_output, float *out, float *in, float *weights, int nweights, WEIGHTS_NORMALIZATION_T weights_normalizion)
{
    // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
    const int layer_step = nweights / (n + 1);    // 1 or l.c or (l.c * l.h * l.w)
    int step = 0;
    if (nweights > 0) step = src_outputs / layer_step; // (l.c * l.h * l.w) or (l.w*l.h) or 1

    int id;
    #pragma omp parallel for
    for (id = 0; id < size; ++id) {

        int src_id = id;
        const int src_i = src_id % src_outputs;
        src_id /= src_outputs;
        int src_b = src_id;

        float sum = 1, max_val = -FLT_MAX;
        int i;
        if (weights && weights_normalizion) {
            if (weights_normalizion == SOFTMAX_NORMALIZATION) {
                for (i = 0; i < (n + 1); ++i) {
                    const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (max_val < w) max_val = w;
                }
            }
            const float eps = 0.0001;
            sum = eps;
            for (i = 0; i < (n + 1); ++i) {
                const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                const float w = weights[weights_index];
                if (weights_normalizion == RELU_NORMALIZATION) sum += relu(w);
                else if (weights_normalizion == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
            }
        }

        if (weights) {
            float w = weights[src_i / step];
            if (weights_normalizion == RELU_NORMALIZATION) w = relu(w) / sum;
            else if (weights_normalizion == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

            out[id] = in[id] * w; // [0 or c or (c, h ,w)]
        }
        else out[id] = in[id];

        // layers
        for (i = 0; i < n; ++i) {
            int add_outputs = outputs_of_layers[i];
            if (src_i < add_outputs) {
                int add_index = add_outputs*src_b + src_i;
                int out_index = id;

                float *add = layers_output[i];

                if (weights) {
                    const int weights_index = src_i / step + (i + 1)*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (weights_normalizion == RELU_NORMALIZATION) w = relu(w) / sum;
                    else if (weights_normalizion == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

                    out[out_index] += add[add_index] * w; // [0 or c or (c, h ,w)]
                }
                else out[out_index] += add[add_index];
            }
        }
    }
}

void backward_shortcut_multilayer_cpu(int size, int src_outputs, int batch, int n, int *outputs_of_layers,
    float **layers_delta, float *delta_out, float *delta_in, float *weights, float *weight_updates, int nweights, float *in, float **layers_output, WEIGHTS_NORMALIZATION_T weights_normalizion)
{
    //size = l.outputs * l.batch
    //src_outputs = l.outputs
    //n=1
    // outputs_of_layers = l.input_sizes
    //layers_delta = l.layers_delta
    //delta_out = state.delta
    //delta_in = l.delta

    int id;
    #pragma omp parallel for
    for (id = 0; id < size; ++id) {
        int src_id = id;
        int src_i = src_id % src_outputs;
        src_id /= src_outputs;
        int src_b = src_id;

        int i;
        delta_out[id] += delta_in[id];

        // layers
        for (i = 0; i < n; ++i) {
            int add_outputs = outputs_of_layers[i];
            if (src_i < add_outputs) {
                int add_index = add_outputs*src_b + src_i;

                float *layer_delta = layers_delta[i];
                layer_delta[add_index] += delta_in[id];
            }
        }
    }
}


void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{   // (l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
    //  bn层的mean是对一个通道(即一个卷积核结果)的所有元素进行加和的
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{//(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w)  将output归一化
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f] + .000001f));
            }
        }
    }
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void scal_add_cpu(int N, float ALPHA, float BETA, float *X, int INCX)
{
    int i;
    for (i = 0; i < N; ++i) X[i*INCX] = X[i*INCX] * ALPHA + BETA;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    if (INCX == 1 && ALPHA == 0) {
        memset(X, 0, N * sizeof(float));
    }
    else {
        for (i = 0; i < N; ++i) X[i*INCX] = ALPHA;
    }
}


void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void softmax(float *input, int n, float temp, float *output, int stride)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}



void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{   //forward（net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output）
    // backward(state.delta, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta)
    int i, j, k, b;
    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < h*stride; ++j) {
                for (i = 0; i < w*stride; ++i) {
                    int in_index = b*w*h*c + k*w*h + (j / stride)*w + i / stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if (forward) out[out_index] = scale*in[in_index]; //把一个像素重复stride*stride;
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}
