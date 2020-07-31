// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.hpp"
#include "layer.hpp"
#include "image.hpp"
#include "data.hpp"
#include <iostream>
#include <vector>
#include <map>
using namespace std;


float get_current_seq_subdivisions(network net);
int get_sequence_value(network net);
float get_current_rate(network net);
int64_t get_current_iteration(network const &net);

int get_current_batch(network const &net);

network make_network(int num);
void forward_network(network net, network_state state);
void backward_network(network net, network_state state);
void update_network(network net);

float train_network(network net, data d);
float train_network_waitkey(network net, data d, int wait_key);

float train_network_datum(network net, float *x, float *y);


float *get_network_output(network net);

int get_network_output_size(network net);

int resize_network(network *net, int w, int h);

float get_network_cost(network net);



#endif
