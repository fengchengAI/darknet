#ifndef IMAGE_H
#define IMAGE_H
#include "darknet.hpp"
#include "image_opencv.hpp"
#include "box.hpp"

float get_color(int c, int x, int max);
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);

void draw_detections_v3(image im, detection *dets, int num, float thresh, vector<string>names, image **alphabet, int classes, bool ext_output);
void scale_image(image m, float s);

void fill_image(image m, float s);

void normalize_image(image p);

void embed_image(image source, image dest, int dx, int dy);

void constrain_image(image im);
int best_3d_shift_r(image a, image b, int min, int max);

image threshold_image(image im, float thresh);

image collapse_image_layers(image source, int border);
image collapse_images_vert(image *ims, int n);


void save_image(image p, string name);
void show_images(image *ims, int n, char *window);

void show_image(image p, string name);



//LIB_API image make_image(int w, int h, int c);
image make_empty_image(int w, int h, int c);
image float_to_image(int w, int h, int c, float *data);
image copy_image(image p);
image load_image(string filename, int w, int h, int c);

image get_image_layer(image m, int l);



#endif
