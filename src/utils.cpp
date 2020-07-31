#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "utils.hpp"
#include <cstdio>
#include <cstdlib>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <assert.h>
#include <float.h>
#include <climits>
#include <sys/time.h>
#include <sys/stat.h>
#include <typeinfo>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
using namespace std ;

#ifndef USE_CMAKE_LIBS
#pragma warning(disable: 4996)
#endif

void *xmalloc(size_t size) {
    void *ptr=malloc(size);
    if(!ptr) {
        malloc_error();
    }
    return ptr;
}

void *xcalloc(size_t nmemb, size_t size) {
    void *ptr=calloc(nmemb,size);
    if(!ptr) {
        calloc_error();
    }
    return ptr;
}

void *xrealloc(void *ptr, size_t size) {
    ptr=realloc(ptr,size);
    if(!ptr) {
        realloc_error();
    }
    return ptr;
}

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

bool find_arg_or(vector<string> const &argv, string s)
{
    auto it = find(argv.begin(), argv.end(), s);
    if (it != argv.end())
        return true;
    else
       return false;
}

int find_arg_int(vector<string> const &argv, string s, int def)
{
    auto it = find(argv.begin(), argv.end(), s);
    if (it != argv.end())
        return stof(*it);
    else
        return def ;
}

string find_arg_str(vector<string> const &argv, string s, string def )
{
    auto it = find(argv.begin(), argv.end(), s);
    if (it != argv.end())
        return *it;
    else
        return def;
}

float find_arg_float(vector<string> const &argv, string s, float def)
{
    auto it = find(argv.begin(), argv.end(), s);
    if (it != argv.end())
        return atof((it->c_str()));
    else
        return def;
}


void replace_image_to_label(string &input_path)
{
    input_path.replace(input_path.find(".jpg"),4,".txt");

    if(input_path.length() > 4) {
        if( input_path.substr(input_path.length()-4,4)!=".txt"){
            cerr<<"Failed to infer label file name (check image extension is supported): "<<input_path<<endl;
        }
    }else{
        clog<<"Label file name is too short: "<<input_path<<endl;
    }
}

void error(const char *s)
{
    perror(s);
    assert(0);
    exit(EXIT_FAILURE);
}

void malloc_error()
{
    fprintf(stderr, "xMalloc error\n");
    exit(EXIT_FAILURE);
}

void calloc_error()
{
    fprintf(stderr, "Calloc error\n");
    exit(EXIT_FAILURE);
}

void realloc_error()
{
    fprintf(stderr, "Realloc error\n");
    exit(EXIT_FAILURE);
}

float sum_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}

float mean_array(float *a, int n)
{
    return sum_array(a,n)/n;
}

float variance_array(float *a, int n)
{
    int i;
    float sum = 0;
    float mean = mean_array(a, n);
    for(i = 0; i < n; ++i) sum += (a[i] - mean)*(a[i]-mean);
    float variance = sum/n;
    return variance;
}

int constrain(float min, float max, float a)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

float dist_array(float *a, float *b, int n, int sub)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; i += sub) sum += pow(a[i]-b[i], 2);
    return sqrt(sum);
}

float mag_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        sum += a[i]*a[i];
    }
    return sqrt(sum);
}

int int_index(int *a, int val, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        if (a[i] == val) return i;
    }
    return -1;
}

int rand_int(int min, int max)
{
    if (max < min){
        /*
        int tmp = min;
        min = max;
        max = tmp;
        */
        return rand_int(max,min);
    }
    int r = (random_gen()%(max - min + 1)) + min;
    return r;
}

float rand_uniform(float min, float max)
{
    if(max < min){
        /*
        float swap = min;
        min = max;
        max = swap;
         */return rand_uniform(max,min);
    }
    return ((float)rand() / RAND_MAX * (max - min)) + min;

}

float rand_scale(float s)
{
    float scale = rand_uniform_strong(1, s);
    if(random_gen()%2) return scale;
    return 1./scale;
}

unsigned int random_gen()
{
    srand((int)time(0));  // 产生随机种子  把0换成NULL也行
    return rand();
}

float random_float()
{
    return ((float) random_gen() / (float)RAND_MAX);
}

float rand_uniform_strong(float min, float max)//返回min，max之间的值
{
    if (max < min) {
        /*
       float swap = min;
       min = max;
       max = swap;
        */return rand_uniform_strong(max,min);
    }
    return (random_float() * (max - min)) + min;
}