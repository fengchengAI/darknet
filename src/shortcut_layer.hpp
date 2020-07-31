#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include "layer.hpp"
#include "network.hpp"


layer make_shortcut_layer(int batch, int n, int *input_layers, int* input_sizes, int w, int h, int c,
    float **layers_output, float **layers_delta, WEIGHTS_TYPE_T weights_type, WEIGHTS_NORMALIZATION_T weights_normalizion,
    ACTIVATION activation, int train);
void forward_shortcut_layer(const layer l, network_state state);
void backward_shortcut_layer(const layer l, network_state state);
void update_shortcut_layer(layer l, int batch, float learning_rate_init, float momentum, float decay);
void resize_shortcut_layer(layer *l, int w, int h, network *net);

#endif
