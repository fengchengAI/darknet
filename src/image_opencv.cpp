#ifndef IMAGE_OPENCV_API
#define IMAGE_OPENCV_API

#include "image_opencv.hpp"
#include <iostream>

#include "utils.hpp"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <atomic>
#include <opencv2/opencv.hpp>



//using namespace cv;

using std::cerr;
using std::endl;




#ifndef CV_RGB
#define CV_RGB(r, g, b) cvScalar( (b), (g), (r), 0 )
#endif

#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif

#ifndef CV_AA
#define CV_AA cv::LINE_AA
#endif


// ====================================================================
// cv::Mat
// ====================================================================
    image mat_to_image(cv::Mat mat);
    cv::Mat image_to_mat(image img);


cv::Mat load_image_mat_cv(string filename, int flag)
{
    //cv::Mat *mat_ptr = NULL;

        cv::Mat mat = cv::imread(filename, flag);
        if (mat.empty())
        {
            std::string shrinked_filename = filename;
            if (shrinked_filename.length() > 1024) {
                shrinked_filename.resize(1024);
                shrinked_filename = std::string("name is too long: ") + shrinked_filename;
            }
            cerr << "Cannot load image " << shrinked_filename << std::endl;
            std::ofstream bad_list("bad.list", std::ios::out | std::ios::app);
            bad_list << shrinked_filename << std::endl;
            //if (check_mistakes) getchar();
            cerr << "OpenCV exception: load_image_mat_cv \n";
            exit(1)  ;
        }
        cv::Mat dst;
        if (mat.channels() == 3) cv::cvtColor(mat, dst, cv::COLOR_RGB2BGR);
        else if (mat.channels() == 4) cv::cvtColor(mat, dst, cv::COLOR_RGBA2BGRA);
        else dst = mat;

        //mat_ptr = new cv::Mat(dst);

        return dst;

        cerr << "OpenCV exception: load_image_mat_cv \n";

    //if (mat_ptr) delete mat_ptr;
}
// ----------------------------------------

cv::Mat load_image_mat(string filename, int channels)
{
    int flag = cv::IMREAD_UNCHANGED;
    if (channels == 0) flag = cv::IMREAD_COLOR;
    else if (channels == 1) flag = cv::IMREAD_GRAYSCALE;
    else if (channels == 3) flag = cv::IMREAD_COLOR;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    //flag |= IMREAD_IGNORE_ORIENTATION;    // un-comment it if you want

    cv::Mat mat  = load_image_mat_cv(filename, flag);

    if (!mat.data) {
        return cv::Mat();
    }
    return mat;
}
// ----------------------------------------

image load_image_cv(string filename, int channels)
{
    cv::Mat mat = load_image_mat(filename, channels);

    if (mat.empty()) {
        return make_image(10, 10, channels);
    }
    return mat_to_image(mat);
}
// ----------------------------------------

// ----------------------------------------

 int get_width_mat(cv::Mat mat)
{
    if (!mat.data) {
        cerr << " Pointer is NULL in get_width_mat() \n";
        return 0;
    }
    return mat.cols;
}
// ----------------------------------------

 int get_height_mat(cv::Mat mat)
{
    if (!mat.data) {
        cerr << " Pointer is NULL in get_height_mat() \n";
        return 0;
    }
    return mat.rows;
}
// ----------------------------------------

 void release_mat(mat_cv **mat)
{
    try {
        cv::Mat **mat_ptr = (cv::Mat **)mat;
        if (*mat_ptr) delete *mat_ptr;
        *mat_ptr = NULL;
    }
    catch (...) {
        cerr << "OpenCV exception: release_mat \n";
    }
}



 cv::Mat image_to_mat(image img)
{
    int channels = img.c;
    int width = img.w;
    int height = img.h;
    cv::Mat mat = cv::Mat(height, width, CV_8UC(channels));
    int step = mat.step;

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < img.c; ++c) {
                float val = img.data[c*img.h*img.w + y*img.w + x];
                mat.data[y*step + x*img.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return mat;
}
// ----------------------------------------

image mat_to_image(cv::Mat mat)
{
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    image im = make_image(w, h, c);
    unsigned char *data = mat.data;
    int step = mat.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
            }
        }
    }
    return im;
}


 void destroy_all_windows_cv()
{
    try {
        cv::destroyAllWindows();
    }
    catch (...) {
        cerr << "OpenCV exception: destroy_all_windows_cv \n";
    }
}
// ----------------------------------------

 int wait_key_cv(int delay)
{
    try {
        return cv::waitKey(delay);
    }
    catch (...) {
        cerr << "OpenCV exception: wait_key_cv \n";
    }
    return -1;
}
// ----------------------------------------

 int wait_until_press_key_cv()
{
    return wait_key_cv(0);
}
// ----------------------------------------

// ----------------------------------------

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
// ----------------------------------------

void show_image_cv(image p, string name)
{
    try {
        image copy = copy_image(p);
        constrain_image(copy);

        cv::Mat mat = image_to_mat(copy);
        if (mat.channels() == 3) cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        else if (mat.channels() == 4) cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::imshow(name, mat);
        free_image(copy);
    }
    catch (...) {
        cerr << "OpenCV exception: show_image_cv \n";
    }
}
// ----------------------------------------




// ----------------------------------------

void save_image_cv(image p, string name){
    try {
        image copy = copy_image(p);
        constrain_image(copy);

        cv::Mat mat = image_to_mat(copy);
        if (mat.channels() == 3) cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        else if (mat.channels() == 4) cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
        cv::imwrite(name,mat);
    }
    catch (...) {
        cerr << "OpenCV exception: show_image_cv \n";
    }
 }

 void save_mat_png(cv::Mat img_src, string name)
{/*
    cv::Mat img_rgb;
    if (img_src.channels() >= 3) cv::cvtColor(img_src, img_rgb, cv::COLOR_RGB2BGR);
    stbi_write_png(name, img_rgb.cols, img_rgb.rows, 3, (char *)img_rgb.data, 0);
*/
}

// ====================================================================
// Draw Loss & Accuracy chart
// ====================================================================
 mat_cv* draw_train_chart(string windows_name, float max_img_loss, int max_batches, int number_of_lines, int img_size, int dont_show, string chart_path)
{
    int img_offset = 60;
    int draw_size = img_size - img_offset;
    cv::Mat *img_ptr = new cv::Mat(img_size, img_size, CV_8UC3, CV_RGB(255, 255, 255));
    cv::Mat &img = *img_ptr;
    cv::Point pt1, pt2, pt_text;

    try {
        // load chart from file
        if (!chart_path.empty() && chart_path[0] != '\0') {
            *img_ptr = cv::imread(chart_path);
        }
        else {
            // draw new chart
            char char_buff[100];
            int i;
            // vertical lines
            pt1.x = img_offset; pt2.x = img_size, pt_text.x = 30;
            for (i = 1; i <= number_of_lines; ++i) {
                pt1.y = pt2.y = (float)i * draw_size / number_of_lines;
                cv::line(img, pt1, pt2, CV_RGB(224, 224, 224), 1, 8, 0);
                if (i % 10 == 0) {
                    sprintf(char_buff, "%2.1f", max_img_loss*(number_of_lines - i) / number_of_lines);
                    pt_text.y = pt1.y + 3;

                    cv::putText(img, char_buff, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
                    cv::line(img, pt1, pt2, CV_RGB(128, 128, 128), 1, 8, 0);
                }
            }
            // horizontal lines
            pt1.y = draw_size; pt2.y = 0, pt_text.y = draw_size + 15;
            for (i = 0; i <= number_of_lines; ++i) {
                pt1.x = pt2.x = img_offset + (float)i * draw_size / number_of_lines;
                cv::line(img, pt1, pt2, CV_RGB(224, 224, 224), 1, 8, 0);
                if (i % 10 == 0) {
                    sprintf(char_buff, "%d", max_batches * i / number_of_lines);
                    pt_text.x = pt1.x - 20;
                    cv::putText(img, char_buff, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
                    cv::line(img, pt1, pt2, CV_RGB(128, 128, 128), 1, 8, 0);
                }
            }

            cv::putText(img, "Loss", cv::Point(10, 55), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 255), 1, CV_AA);
            cv::putText(img, "Iteration number", cv::Point(draw_size / 2, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
            char max_batches_buff[100];
            sprintf(max_batches_buff, "in cfg max_batches=%d", max_batches);
            cv::putText(img, max_batches_buff, cv::Point(draw_size - 195, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
            cv::putText(img, "Press 's' to save : chart.png", cv::Point(5, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
        }

        if (!dont_show) {
            printf(" If error occurs - run training with flag: -dont_show \n");
            cv::namedWindow(windows_name, cv::WINDOW_NORMAL);
            cv::moveWindow(windows_name, 0, 0);
            cv::resizeWindow(windows_name, img_size, img_size);
            cv::imshow(windows_name, img);
            cv::waitKey(20);
        }
    }
    catch (...) {
        cerr << "OpenCV exception: draw_train_chart() \n";
    }
    return (mat_cv*)img_ptr;
}
// ----------------------------------------

void draw_train_loss(string windows_name, mat_cv* img_src, int img_size, float avg_loss, float max_img_loss, int current_batch, int max_batches,
    float precision, int draw_precision, string accuracy_name, int dont_show, int mjpeg_port, double time_remaining)
{
    try {
        cv::Mat &img = *(cv::Mat*)img_src;
        int img_offset = 60;
        int draw_size = img_size - img_offset;
        char char_buff[100];
        cv::Point pt1, pt2;
        pt1.x = img_offset + draw_size * (float)current_batch / max_batches;
        pt1.y = draw_size * (1 - avg_loss / max_img_loss);
        if (pt1.y < 0) pt1.y = 1;
        cv::circle(img, pt1, 1, CV_RGB(0, 0, 255), CV_FILLED, 8, 0);

        // precision
        if (draw_precision) {
            static float old_precision = 0;
            static float max_precision = 0;
            static int iteration_old = 0;
            static int text_iteration_old = 0;
            if (iteration_old == 0)
                cv::putText(img, accuracy_name, cv::Point(10, 12), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 0, 0), 1, CV_AA);

	    if (iteration_old != 0){
            	cv::line(img,
                    cv::Point(img_offset + draw_size * (float)iteration_old / max_batches, draw_size * (1 - old_precision)),
                    cv::Point(img_offset + draw_size * (float)current_batch / max_batches, draw_size * (1 - precision)),
                    CV_RGB(255, 0, 0), 1, 8, 0);
	    }

            sprintf(char_buff, "%2.1f%% ", precision * 100);
            cv::putText(img, char_buff, cv::Point(10, 28), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 5, CV_AA);
            cv::putText(img, char_buff, cv::Point(10, 28), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(200, 0, 0), 1, CV_AA);

            if ((std::fabs(old_precision - precision) > 0.1)  || (max_precision < precision) || (current_batch - text_iteration_old) >= max_batches / 10) {
                text_iteration_old = current_batch;
                max_precision = std::max(max_precision, precision);
                sprintf(char_buff, "%2.0f%% ", precision * 100);
                cv::putText(img, char_buff, cv::Point(pt1.x - 30, draw_size * (1 - precision) + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 5, CV_AA);
                cv::putText(img, char_buff, cv::Point(pt1.x - 30, draw_size * (1 - precision) + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(200, 0, 0), 1, CV_AA);
            }
            old_precision = precision;
            iteration_old = current_batch;
        }
        sprintf(char_buff, "current avg loss = %2.4f    iteration = %d    approx. time left = %2.2f hours", avg_loss, current_batch, time_remaining);
        pt1.x = 15, pt1.y = draw_size + 18;
        pt2.x = pt1.x + 800, pt2.y = pt1.y + 20;
        cv::rectangle(img, pt1, pt2, CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
        pt1.y += 15;
        cv::putText(img, char_buff, pt1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 100), 1, CV_AA);

        int k = 0;
        if (!dont_show) {
            cv::imshow(windows_name, img);
            k = cv::waitKey(20);
        }
        static int old_batch = 0;
        if (k == 's' || current_batch == (max_batches - 1) || (current_batch / 100 > old_batch / 100)) {
            old_batch = current_batch;
            save_mat_png(img, "chart.png");
            save_mat_png(img, windows_name);
            cv::putText(img, "- Saved", cv::Point(260, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 0, 0), 1, CV_AA);
        }
        else
            cv::putText(img, "- Saved", cv::Point(260, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 1, CV_AA);

        //if (mjpeg_port > 0) send_mjpeg((mat_cv *)&img, mjpeg_port, 500000, 70);
    }
    catch (...) {
        cerr << "OpenCV exception: draw_train_loss() \n";
    }
}
// ----------------------------------------


// ====================================================================
// Data augmentation
// ====================================================================

image image_data_augmentation(cv::Mat mat, int w, int h,
                              int pleft, int ptop, int swidth, int sheight, int flip,
                              float dhue, float dsat, float dexp,
                              int gaussian_noise, int blur, int num_boxes, vector<array<float, 5> > &truth)
{// 首先裁减，即就是扣图，然后填补，填补元素为图像均值，然后做缩放，再做hue，sat等变换

    cv::Mat out;
    try {
        cv::Mat img = mat;

        // crop
        // Rect(x,y,w,h)实际上就是左下角的x，y值，及Rect的w，h值
        cv::Rect src_rect(pleft, ptop, swidth, sheight);
        cv::Rect img_rect(cv::Point2i(0, 0), img.size());
        cv::Rect new_src_rect = src_rect & img_rect;

        cv::Rect dst_rect(cv::Point2i(std::max<int>(0, -pleft), std::max<int>(0, -ptop)), new_src_rect.size());
        cv::Mat sized;

        if (src_rect.x == 0 && src_rect.y == 0 && src_rect.size() == img.size()) {
            cv::resize(img, sized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        }
        else {
            cv::Mat cropped(src_rect.size(), img.type());
            //cropped.setTo(cv::Scalar::all(0));
            cropped.setTo(cv::mean(img));

            img(new_src_rect).copyTo(cropped(dst_rect));

            // resize
            cv::resize(cropped, sized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        }

        // flip
        if (flip) {
            cv::Mat cropped;
            cv::flip(sized, cropped, 1);    // 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
            sized = cropped.clone();
        }

        // HSV augmentation
        // cv::COLOR_BGR2HSV, cv::COLOR_RGB2HSV, cv::COLOR_HSV2BGR, cv::COLOR_HSV2RGB
        if (dsat != 1 || dexp != 1 || dhue != 0) {
            if (img.channels() >= 3)
            {
                cv::Mat hsv_src;
                cvtColor(sized, hsv_src, cv::COLOR_RGB2HSV);    // RGB to HSV

                std::vector<cv::Mat> hsv;
                cv::split(hsv_src, hsv);

                hsv[1] *= dsat;
                hsv[2] *= dexp;
                hsv[0] += 179 * dhue;

                cv::merge(hsv, hsv_src);

                cvtColor(hsv_src, sized, cv::COLOR_HSV2RGB);    // HSV to RGB (the same as previous)
            }
            else
            {
                sized *= dexp;
            }
        }


        if (blur) {//0
            cv::Mat dst(sized.size(), sized.type());
            if (blur == 1) {
                cv::GaussianBlur(sized, dst, cv::Size(17, 17), 0);
                //cv::bilateralFilter(sized, dst, 17, 75, 75);
            }
            else {
                int ksize = (blur / 2) * 2 + 1;
                cv::Size kernel_size = cv::Size(ksize, ksize);
                cv::GaussianBlur(sized, dst, kernel_size, 0);
            }
            //std::cout << " blur num_boxes = " << num_boxes << std::endl;

            if (blur == 1) {//0
                cv::Rect img_rect(0, 0, sized.cols, sized.rows);
                int t;
                for (t = 0; t < num_boxes; ++t) {
                    box b = array_to_box(truth[t]);
                    if (!b.x) break;
                    int left = (b.x - b.w / 2.)*sized.cols;
                    int width = b.w*sized.cols;
                    int top = (b.y - b.h / 2.)*sized.rows;
                    int height = b.h*sized.rows;
                    cv::Rect roi(left, top, width, height);
                    roi = roi & img_rect;

                    sized(roi).copyTo(dst(roi));
                }
            }
            dst.copyTo(sized);
        }

        if (gaussian_noise) {//0
            cv::Mat noise = cv::Mat(sized.size(), sized.type());
            gaussian_noise = std::min(gaussian_noise, 127);
            gaussian_noise = std::max(gaussian_noise, 0);
            cv::randn(noise, 0, gaussian_noise);  //mean and variance
            cv::Mat sized_norm = sized + noise;
            sized = sized_norm;
        }


        // Mat -> image
        out = sized;
    }
    catch (...) {
        cerr << "OpenCV can't augment image: " << w << " x " << h << " \n";
        out = mat;
    }
    return mat_to_image(out);
}

// blend two images with (alpha and beta)
 void blend_images_cv(image new_img, float alpha, image old_img, float beta)
{
    cv::Mat new_mat(cv::Size(new_img.w, new_img.h), CV_32FC(new_img.c), new_img.data);// , size_t step = AUTO_STEP)
    cv::Mat old_mat(cv::Size(old_img.w, old_img.h), CV_32FC(old_img.c), old_img.data);
    cv::addWeighted(new_mat, alpha, old_mat, beta, 0.0, new_mat);
}




void show_opencv_info()
{
    std::cerr << " OpenCV version: " << CV_VERSION_MAJOR << "." << CV_VERSION_MINOR << std::endl;
}
#endif // OPENCV
