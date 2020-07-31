#ifndef DATA_H
#define DATA_H

#include "darknet.hpp"
#include "matrix.hpp"
#include "image.hpp"

#include <iostream>
#include <fstream>
#include <string>
using namespace std ;
static inline float distance_from_edge(int x, int max)
{
    int dx = (max/2) - x;
    if (dx < 0) dx = -dx;
    dx = (max/2) + 1 - dx;
    dx *= 2;
    float dist = (float)dx/max;
    if (dist > 1) dist = 1;
    return dist;
}


data load_data_detection(int n, vector<string>paths, int m, int w, int h, int c, int boxes, int classes, int use_flip, int gaussian_noise, int use_blur, int use_mixup,
    float jitter, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int show_imgs);


vector<box_label> read_boxes(string filename, int &n);



vector<string>get_labels(string filename);
vector<string>get_labels_custom(string filename, int &size);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void array_to_float(float *d, vector<array<float,5> >s, int num );
data concat_data(data d1, data d2);
data concat_datas(data *d, int n);
vector<string> get_paths(string filename);
int custom_atomic_load_int(volatile int* obj);
void custom_atomic_store_int(volatile int* obj, int desr);
void this_thread_sleep_for(int ms_time);
#endif
