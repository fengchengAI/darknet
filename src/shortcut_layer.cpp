#include "shortcut_layer.hpp"
#include "blas.hpp"
#include "utils.hpp"
#include "gemm.hpp"


layer make_shortcut_layer(int batch, int n, int *input_layers, int* input_sizes, int w, int h, int c,
    float **layers_output, float **layers_delta, WEIGHTS_TYPE_T weights_type, WEIGHTS_NORMALIZATION_T weights_normalizion,
    ACTIVATION activation, int train)
{
    /*
    input_layers = layers[i] = index;
    input_sizes = sizes[i] = params.net.layers[index].outputs;
    layers_output[i] = params.net.layers[index].output;
    layers_delta[i] = params.net.layers[index].delta;
    weights_normalizion = NO_NORMALIZATION
    weights_type =  NO_WEIGTHS
    activation =  linear
    */
    fprintf(stderr, "Shortcut Layer: ");
    int i;
    for(i = 0; i < n; ++i) fprintf(stderr, "%d, ", input_layers[i]);

    layer l = { (LAYER_TYPE)0 };
    l.train = train;
    l.type = SHORTCUT;
    l.batch = batch;
    l.activation = activation;  //linear
    l.n = n;// 表示连接的层数
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    l.layers_output = layers_output;
    l.layers_delta = layers_delta;
    l.weights_type = weights_type;
    l.weights_normalizion = weights_normalizion;
    l.learning_rate_scale = 1;  // not necessary

    l.w = l.out_w = w;
    l.h = l.out_h = h;
    l.c = l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.index = l.input_layers[0];

    if (train) l.delta = (float*)xcalloc(l.outputs * batch, sizeof(float));
    l.output = (float*)xcalloc(l.outputs * batch, sizeof(float));

    l.nweights = 0;
    if (l.weights_type == PER_FEATURE) l.nweights = (l.n + 1);
    else if (l.weights_type == PER_CHANNEL) l.nweights = (l.n + 1) * l.c;

    if (l.nweights > 0) {//0
        l.weights = (float*)calloc(l.nweights, sizeof(float));
        float scale = sqrt(2. / l.nweights);
        for (i = 0; i < l.nweights; ++i) l.weights[i] = 1;// +0.01*rand_uniform(-1, 1);// scale*rand_uniform(-1, 1);   // rand_normal();

        if (train) l.weight_updates = (float*)calloc(l.nweights, sizeof(float));
        l.update = update_shortcut_layer;
    }

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;
    if (l.activation == SWISH || l.activation == MISH) l.activation_input = (float*)calloc(l.batch*l.outputs, sizeof(float));



    l.bflops = l.out_w * l.out_h * l.out_c * l.n / 1000000000.;
    if (l.weights_type) l.bflops *= 2;
    fprintf(stderr, " wt = %d, wn = %d, outputs:%4d x%4d x%4d %5.3f BF\n", l.weights_type, l.weights_normalizion, l.out_w, l.out_h, l.out_c, l.bflops);
    return l;
}

void resize_shortcut_layer(layer *l, int w, int h, network *net)
{
    //assert(l->w == l->out_w);
    //assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w*h*l->out_c;
    l->inputs = l->outputs;
    if (l->train) l->delta = (float*)xrealloc(l->delta, l->outputs * l->batch * sizeof(float));
    l->output = (float*)xrealloc(l->output, l->outputs * l->batch * sizeof(float));

    int i;
    for (i = 0; i < l->n; ++i) {
        int index = l->input_layers[i];
        l->input_sizes[i] = net->layers[index].outputs;
        l->layers_output[i] = net->layers[index].output;
        l->layers_delta[i] = net->layers[index].delta;

        assert(l->w == net->layers[index].out_w && l->h == net->layers[index].out_h);
    }

    if (l->activation == SWISH || l->activation == MISH) l->activation_input = (float*)realloc(l->activation_input, l->batch*l->outputs * sizeof(float));



}

void forward_shortcut_layer(const layer l, network_state state)
{ //state.net.layers[l.index]指向连接的第一层的index
    int from_w = state.net.layers[l.index].w;
    int from_h = state.net.layers[l.index].h;
    int from_c = state.net.layers[l.index].c;
// nweights 由weights_type决定，此处为0
//l.n = 1 ，表示只连接了一层
    if (l.nweights == 0 && l.n == 1 && from_w == l.w && from_h == l.h && from_c == l.c) {//1
        int size = l.batch * l.w * l.h * l.c;
        int i;
        #pragma omp parallel for
        for(i = 0; i < size; ++i)
            l.output[i] = state.input[i] + state.net.layers[l.index].output[i];
    }
    else {
        shortcut_multilayer_cpu(l.outputs * l.batch, l.outputs, l.batch, l.n, l.input_sizes, l.layers_output, l.output, state.input, l.weights, l.nweights, l.weights_normalizion);
    }

    if (l.activation == MISH) activate_array_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);
    else activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);//1
}

void backward_shortcut_layer(const layer l, network_state state)
{
    if (l.activation == MISH) gradient_array_mish(l.outputs*l.batch, l.activation_input, l.delta);
    else gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);//1

    backward_shortcut_multilayer_cpu(l.outputs * l.batch, l.outputs, l.batch, l.n, l.input_sizes,
        l.layers_delta, state.delta, l.delta, l.weights, l.weight_updates, l.nweights, state.input, l.layers_output, l.weights_normalizion);

}

void update_shortcut_layer(layer l, int batch, float learning_rate_init, float momentum, float decay)
{
    if (l.nweights > 0) {//0
        float learning_rate = learning_rate_init*l.learning_rate_scale;

        axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
        axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
        scal_cpu(l.nweights, momentum, l.weight_updates, 1);
    }
}
