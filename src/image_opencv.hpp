#ifndef IMAGE_OPENCV_H
#define IMAGE_OPENCV_H

#include "image.hpp"
#include "matrix.hpp"
#include <opencv2/opencv.hpp>



// declaration
typedef void* mat_cv;

// cv::Mat
cv::Mat load_image_mat_cv(string filename, int flag);
image load_image_cv(string filename, int channels);
int get_width_mat(cv::Mat mat);
int get_height_mat(cv::Mat mat);
void release_mat(mat_cv **mat);



void destroy_all_windows_cv();
int wait_key_cv(int delay);
int wait_until_press_key_cv();
void show_image_cv(image p, string name);



void save_image_cv(image p, string name);
// Draw Detection

// Draw Loss & Accuracy chart
mat_cv* draw_train_chart(string windows_name, float max_img_loss, int max_batches, int number_of_lines, int img_size, int dont_show, string chart_path);
void draw_train_loss(string windows_name, mat_cv* img, int img_size, float avg_loss, float max_img_loss, int current_batch, int max_batches,
    float precision, int draw_precision, string accuracy_name, int dont_show, int mjpeg_port, double time_remaining);

// Data augmentation
image image_data_augmentation(cv::Mat mat, int w, int h,
    int pleft, int ptop, int swidth, int sheight, int flip,
    float dhue, float dsat, float dexp,
    int gaussian_noise, int blur, int num_boxes, vector<array<float, 5> > &truth);

// blend two images with (alpha and beta)
void blend_images_cv(image new_img, float alpha, image old_img, float beta);

void show_opencv_info();

#endif  // OPENCV


