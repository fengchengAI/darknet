#ifndef PARSER_H
#define PARSER_H
#include "network.hpp"
#include <map>
#include <vector>



network parse_network_cfg(string  filename);
network parse_network_cfg_custom(string filename, int batch, int time_steps);
void save_network(network net, string filename);
void save_weights(network net, string filename);
void save_weights_upto(network net, string filename, int cutoff);
void save_weights_double(network net, string filename);
void load_weights(network &net, string filename);
void load_weights_upto(network &net, string filename, int cutoff);


#endif
