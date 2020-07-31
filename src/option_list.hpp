#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "darknet.hpp"
#include <vector>
#include <map>
#include <sstream>
using namespace std;

map<string,string> read_data_cfg(string filename);
string option_find_str(map<string,string> const &m, string key, string def="");
int option_find_int(map<string,string> const & m, string key, int def);
int option_find_float(map<string,string> const &m , string key, float def);

int* find_mul_int(string a, int &num, bool force );

float* find_mul_float(string a, int &num, bool force);

#endif
