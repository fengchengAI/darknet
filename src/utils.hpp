#ifndef UTILS_H
#define UTILS_H
#include "darknet.hpp"

#include <iostream>
#include <ctime>

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

void *xcalloc(size_t nmemb, size_t size);
void *xrealloc(void *ptr, size_t size);
double what_time_is_it_now();

void replace_image_to_label(string& input_path);
void error(const char *s);
void malloc_error();
void calloc_error();
void realloc_error();

int constrain(float min, float max, float a);

float rand_uniform(float min, float max);
float rand_scale(float s);
int rand_int(int min, int max);
float sum_array(float *a, int n);
float mean_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
float dist_array(float *a, float *b, int n, int sub);

int find_arg_int(vector<string> const &argv, string s, int def);
string find_arg_str(vector<string> const &argv, string s, string def );
float find_arg_float(vector<string> const &argv, string s, float def);
bool find_arg_or(vector<string> const &argv, string s);


unsigned int random_gen();
float random_float();
float rand_uniform_strong(float min, float max);

int int_index(int *a, int val, int n);

#define max_val_cmp(a,b) (((a) > (b)) ? (a) : (b))
#define min_val_cmp(a,b) (((a) < (b)) ? (a) : (b))



#endif
