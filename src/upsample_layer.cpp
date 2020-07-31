#include "upsample_layer.hpp"
#include "utils.hpp"
#include "blas.hpp"


/*
 * 这里的upsample_layer实现是很简单的，
 * 如对于stride都是2
 * 直接将横纵坐标都相邻的重复2个就可以了。
 */
layer make_upsample_layer(int batch, int w, int h, int c, int stride)
{   //只有一个参数，stride都是2
    layer l = { (LAYER_TYPE)0 };
    l.type = UPSAMPLE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w*stride;
    l.out_h = h*stride;
    l.out_c = c;
    if(stride < 0){
        stride = -stride;
        l.reverse=1;
        l.out_w = w/stride;
        l.out_h = h/stride;
    }
    l.stride = stride;
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;
    l.delta = (float*)xcalloc(l.outputs * batch, sizeof(float));
    l.output = (float*)xcalloc(l.outputs * batch, sizeof(float));

    l.forward = forward_upsample_layer;
    l.backward = backward_upsample_layer;

    if(l.reverse) fprintf(stderr, "downsample              %2dx  %4d x%4d x%4d -> %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    else fprintf(stderr, "upsample                %2dx  %4d x%4d x%4d -> %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_upsample_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->out_w = w*l->stride;
    l->out_h = h*l->stride;
    if(l->reverse){
        l->out_w = w/l->stride;
        l->out_h = h/l->stride;
    }
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->h*l->w*l->c;
    l->delta = (float*)xrealloc(l->delta, l->outputs * l->batch * sizeof(float));
    l->output = (float*)xrealloc(l->output, l->outputs * l->batch * sizeof(float));



}

void forward_upsample_layer(const layer l, network_state net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);  //置零
    if(l.reverse){
        upsample_cpu(l.output, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input);
    }else{
        upsample_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output);
    }
}

void backward_upsample_layer(const layer l, network_state state)
{
    if(l.reverse){
        upsample_cpu(l.delta, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, state.delta);
    }else{
        upsample_cpu(state.delta, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta);
    }
}


