#include "activations.hpp"

#include <cmath>
#include <iostream>
#include <string>


char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case GELU:
            return "gelu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation(string s)
{
    if (s== "logistic") return LOGISTIC;
    if (s== "swish")  return SWISH;
    if (s== "mish")  return MISH;
    if (s== "normalize_channels")  return NORM_CHAN;
    if (s== "normalize_channels_softmax")  return NORM_CHAN_SOFTMAX;
    if (s== "normalize_channels_softmax_maxval")  return NORM_CHAN_SOFTMAX_MAXVAL;
    if (s== "loggy") return LOGGY;
    if (s== "relu") return RELU;
    if (s== "relu6")  return RELU6;
    if (s== "elu") return ELU;
    if (s== "selu")  return SELU;
    if (s== "gelu")  return GELU;
    if (s== "relie") return RELIE;
    if (s== "plse") return PLSE;
    if (s== "hardtan") return HARDTAN;
    if (s== "lhtan") return LHTAN;
    if (s== "linear") return LINEAR;
    if (s== "ramp") return RAMP;
    if (s== "leaky") return LEAKY;
    if (s== "tanh") return TANH;
    if (s== "stair") return STAIR;
    cerr<<"Couldn't find activation function %s, going with ReLU"<<endl;
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case GELU:
            return gelu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    if (a == LINEAR) {}
    else if (a == LEAKY) {
        #pragma omp parallel for
        for (i = 0; i < n; ++i) {
            x[i] = leaky_activate(x[i]);
        }
    }
    else if (a == LOGISTIC) {
        #pragma omp parallel for
        for (i = 0; i < n; ++i) {
            x[i] = logistic_activate(x[i]);
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}


// https://github.com/digantamisra98/Mish
void activate_array_mish(float *x, const int n, float * activation_input, float * output)
{// activate_array_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);

    const float MISH_THRESHOLD = 20;
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        float x_val = x[i];
        activation_input[i] = x_val;    // store value before activation
        output[i] = x_val * tanh_activate( softplus_activate(x_val, MISH_THRESHOLD) );
        //mish = x * tanh(ln(1 + e^x))
    }
}





/*
** 根据不同的激活函数求取对输入的梯度（导数）
** 输入： x    激活函数接收的输入值
**       a    激活函数类型，包括的激活函数类型见activations.h中枚举类型ACTIVATION的定义
** 输出： 激活函数关于输入x的导数值
*/
float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case RELU6:
            return relu6_gradient(x);
        case NORM_CHAN:
            //return relu_gradient(x);
        case NORM_CHAN_SOFTMAX_MAXVAL:
            //...
        case NORM_CHAN_SOFTMAX:
            printf(" Error: should be used custom NORM_CHAN or NORM_CHAN_SOFTMAX-function for gradient \n");
            exit(0);
            return 0;
        case ELU:
            return elu_gradient(x);
        case SELU:
            return selu_gradient(x);
        case GELU:
            return gelu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}


void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    #pragma omp parallel for
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
}


/*
** 计算激活函数对加权输入的导数，并乘以delta，得到当前层最终的delta（敏感度图）
** 输入： activation_input， 为加权数据，即激活函数的输入
**       n    l.output的维度，即为l.batch * l.out_c * l.out_w * l.out_h（包含整个batch的）
**       delta     当前层敏感度图（与当前成输出x维度一样）
** 说明1： 该函数不但计算了激活函数对于加权输入的导数，还将该导数乘以了之前完成大部分计算的敏感度图delta（对应元素相乘），因此调用改函数之后，将得到该层最终的敏感度图
** 说明2： 这里直接利用输出值求激活函数关于输入的导数值是因为神经网络中所使用的绝大部分激活函数，其关于输入的导数值都可以描述为输出值的函数表达式，
          比如对于Sigmoid激活函数（记作f(x)），其导数值为f(x)'=f(x)*(1-f(x)),因此如果给出y=f(x)，那么f(x)'=y*(1-y)，只需要输出值y就可以了，不需要输入x的值，
** 说明3： 关于l.delta的初值，可能你有注意到在看某一类型网络层的时候，比如卷积层中的backward_convolutional_layer()函数，没有发现在此之前对l.delta赋初值的语句，
**        只是用calloc为其动态分配了内存，这样的l.delta其所有元素的值都为0,那么这里使用*=运算符得到的值将恒为0。是的，如果只看某一层，或者说某一类型的层，的确有这个疑惑，
**        但是整个网络是有很多层的，且有多种类型，一般来说，不会以卷积层为最后一层，而回以COST或者REGION为最后一层，这些层中，会对l.delta赋初值，又由于l.delta是由后
**        网前逐层传播的，因此，当反向运行到某一层时，l.delta的值将都不会为0.
*/void gradient_array_mish(const int n, const float * activation_input, float * delta)
{   //mish = x * tanh(ln(1 + e^x))
    //(l.outputs*l.batch, l.activation_input, l.delta)
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        const float MISH_THRESHOLD = 20.0f;

        float inp = activation_input[i];
        const float sp = softplus_activate(inp, MISH_THRESHOLD);
        const float grad_sp = 1 - exp(-sp);
        const float tsp = tanh(sp);
        const float grad_tsp = (1 - tsp*tsp) * grad_sp;
        const float grad = inp * grad_tsp + tsp;
        delta[i] *= grad;
    }
}
