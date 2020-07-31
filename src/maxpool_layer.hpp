#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.hpp"
#include "layer.hpp"
#include "network.hpp"
#include <cfloat>
typedef layer maxpool_layer;

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride_x, int stride_y, int padding, int maxpool_depth, int out_channels, bool antialiasing, bool avgpool, bool train);
void resize_maxpool_layer(maxpool_layer *l, int w, int h);
void forward_maxpool_layer(const maxpool_layer l, network_state state);
void backward_maxpool_layer(const maxpool_layer l, network_state state);



#endif
