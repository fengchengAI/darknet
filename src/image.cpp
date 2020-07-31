#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "image.hpp"
#include "utils.hpp"
#include "blas.hpp"
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <cstring>
#include "image_opencv.hpp"
#include "box.hpp"
#include <string>
#include <iostream>
using namespace std;

extern int check_mistakes;
//int windows = 0;

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{//selected_detections[i].best_class * 123457 % classes;
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
static float get_pixel_extend(image m, int x, int y, int c)
{
    if (x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
    /*
    if(x < 0) x = 0;
    if(x >= m.w) x = m.w-1;
    if(y < 0) y = 0;
    if(y >= m.h) y = m.h-1;
    */
    if (c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}
static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}
static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}


/*
image get_label_v3(image **characters, string string, int size)
{
    size = size / 10;
    if (size > 7) size = 7;
    image label = make_empty_image(0, 0, 0);
    while (!string.empty()) {
        image l = characters[size][(int)*string];
        image n = tile_images(label, l, -size - 1 + (size + 1) / 2);
        free_image(label);
        label = n;
        ++string;
    }
    image b = border_image(label, label.h*.25);
    free_image(label);
    return b;
}
*/

void draw_box_bw(image a, int x1, int y1, int x2, int y2, float brightness)
{
    //normalize_image(a);
    int i;
    if (x1 < 0) x1 = 0;
    if (x1 >= a.w) x1 = a.w - 1;
    if (x2 < 0) x2 = 0;
    if (x2 >= a.w) x2 = a.w - 1;

    if (y1 < 0) y1 = 0;
    if (y1 >= a.h) y1 = a.h - 1;
    if (y2 < 0) y2 = 0;
    if (y2 >= a.h) y2 = a.h - 1;

    for (i = x1; i <= x2; ++i) {
        a.data[i + y1*a.w + 0 * a.w*a.h] = brightness;
        a.data[i + y2*a.w + 0 * a.w*a.h] = brightness;
    }
    for (i = y1; i <= y2; ++i) {
        a.data[x1 + i*a.w + 0 * a.w*a.h] = brightness;
        a.data[x2 + i*a.w + 0 * a.w*a.h] = brightness;
    }
}

void draw_box_width_bw(image a, int x1, int y1, int x2, int y2, int w, float brightness)
{
    int i;
    for (i = 0; i < w; ++i) {
        float alternate_color = (w % 2) ? (brightness) : (1.0 - brightness);
        draw_box_bw(a, x1 + i, y1 + i, x2 - i, y2 - i, alternate_color);
    }
}

void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if(x1 < 0) x1 = 0;
    if(x1 >= a.w) x1 = a.w-1;
    if(x2 < 0) x2 = 0;
    if(x2 >= a.w) x2 = a.w-1;

    if(y1 < 0) y1 = 0;
    if(y1 >= a.h) y1 = a.h-1;
    if(y2 < 0) y2 = 0;
    if(y2 >= a.h) y2 = a.h-1;

    for(i = x1; i <= x2; ++i){
        a.data[i + y1*a.w + 0*a.w*a.h] = r;
        a.data[i + y2*a.w + 0*a.w*a.h] = r;

        a.data[i + y1*a.w + 1*a.w*a.h] = g;
        a.data[i + y2*a.w + 1*a.w*a.h] = g;

        a.data[i + y1*a.w + 2*a.w*a.h] = b;
        a.data[i + y2*a.w + 2*a.w*a.h] = b;
    }
    for(i = y1; i <= y2; ++i){
        a.data[x1 + i*a.w + 0*a.w*a.h] = r;
        a.data[x2 + i*a.w + 0*a.w*a.h] = r;

        a.data[x1 + i*a.w + 1*a.w*a.h] = g;
        a.data[x2 + i*a.w + 1*a.w*a.h] = g;

        a.data[x1 + i*a.w + 2*a.w*a.h] = b;
        a.data[x2 + i*a.w + 2*a.w*a.h] = b;
    }
}

void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for(i = 0; i < w; ++i){
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}




// Creates array of detections with prob > thresh and fills best_class for them
detection_with_class* get_actual_detections(detection *dets, int dets_num, float thresh, int& selected_detections_num, vector<string>names)
{//返回满足一定阈值的box信息和类所（detection_with_class），并传递满足的数量到selected_detections_num
    int selected_num = 0;
    //当一个预测box有大于thresh的类别概率时，认为这个box是合法的，selected_num+1

    detection_with_class* result_arr = (detection_with_class*)xcalloc(dets_num, sizeof(detection_with_class));
    int i;
    for (i = 0; i < dets_num; ++i) {
        int best_class = -1;
        float best_class_prob = thresh;
        int j;
        for (j = 0; j < dets[i].classes; ++j) {//选择一个box的最大prob的类别
            bool show = names[j].find("dont_show")!=string::npos;//对有些类别进行忽略。所以正常情况下show非零

            if (dets[i].prob[j] > best_class_prob && !show) {
                best_class = j;
                best_class_prob = dets[i].prob[j];
            }
        }
        if (best_class >= 0) {
            result_arr[selected_num].det = dets[i];
            result_arr[selected_num].best_class = best_class;
            ++selected_num;
        }
    }

    selected_detections_num = selected_num;
    return result_arr;
}

// compare to sort detection** by bbox.x
int compare_by_lefts(const void *a_ptr, const void *b_ptr) {
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    const float delta = (a->det.bbox.x - a->det.bbox.w/2) - (b->det.bbox.x - b->det.bbox.w/2);
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

// compare to sort detection** by best_class probability
int compare_by_probs(const void *a_ptr, const void *b_ptr) {
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    float delta = a->det.prob[a->best_class] - b->det.prob[b->best_class];
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

void draw_detections_v3(image im, detection *dets, int num, float thresh, vector<string>names, image **alphabet, int classes, bool ext_output)
{
    static int frame_id = 0;
    frame_id++;

    int selected_detections_num;  //满足一定阈值的detections_box数量
    detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, selected_detections_num, names);

    // text output// 将selected_detections中的detection按照detection中的box左边排序
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);

    int i;
    for (i = 0; i < selected_detections_num; ++i) {
        const int best_class = selected_detections[i].best_class;
        //printf("%s: %.0f%%", names[best_class],    selected_detections[i].det.prob[best_class] * 100);
        if (ext_output)
            printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
                round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2)*im.w),
                round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2)*im.h),
                round(selected_detections[i].det.bbox.w*im.w), round(selected_detections[i].det.bbox.h*im.h));
        else
            printf("\n");
        int j;
        for (j = 0; j < classes; ++j) {
            if (selected_detections[i].det.prob[j] > thresh && j != best_class) {
                //printf("%s: %.0f%%", names[j], selected_detections[i].det.prob[j] * 100);

                if (ext_output)
                    printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
                        round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2)*im.w),
                        round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2)*im.h),
                        round(selected_detections[i].det.bbox.w*im.w), round(selected_detections[i].det.bbox.h*im.h));
                else
                    printf("\n");
            }
        }
    }

    // image output

    // 将具有某类最大概率的detection放在前面
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_probs);
    for (i = 0; i < selected_detections_num; ++i) {
            int width = im.h * .006;  //坐标框宽度
            if (width < 1)
                width = 1;

            //printf("%d %s: %.0f%%\n", i, names[selected_detections[i].best_class], prob*100);
            int offset = selected_detections[i].best_class * 123457 % classes;
            float red = get_color(2, offset, classes);
            float green = get_color(1, offset, classes);
            float blue = get_color(0, offset, classes);
            float rgb[3];  // 为边框的颜色，因为是多类别的，所以对每个类的边框不一样

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = selected_detections[i].det.bbox;
            //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

            int left = (b.x - b.w / 2.)*im.w;
            int right = (b.x + b.w / 2.)*im.w;
            int top = (b.y - b.h / 2.)*im.h;
            int bot = (b.y + b.h / 2.)*im.h;

            if (left < 0) left = 0;
            if (right > im.w - 1) right = im.w - 1;
            if (top < 0) top = 0;
            if (bot > im.h - 1) bot = im.h - 1;

            if (im.c == 1) {
                draw_box_width_bw(im, left, top, right, bot, width, 0.8);    // 1 channel Black-White
            }
            else {
                draw_box_width(im, left, top, right, bot, width, red, green, blue); // 3 channels RGB
            }
            /*
            if (alphabet) {//0\

                string labelstr = "";
                labelstr = labelstr + names[selected_detections[i].best_class];

                int j;
                for (j = 0; j < classes; ++j) {
                    if (selected_detections[i].det.prob[j] > thresh && j != selected_detections[i].best_class) {

                        labelstr = labelstr + ", " + names[j];

                    }
                }
                image label = get_label_v3(alphabet, labelstr, (im.h*.03));
                draw_label(im, top + width, left, label, rgb);
                free_image(label);
            }
             */
            if (selected_detections[i].det.mask) {//0
                image mask = float_to_image(14, 14, 1, selected_detections[i].det.mask);
                image resized_mask = resize_image(mask, b.w*im.w, b.h*im.h);
                image tmask = threshold_image(resized_mask, .5);
                embed_image(tmask, im, left, top);
                free_image(mask);
                free_image(resized_mask);
                free_image(tmask);
            }
    }
    free(selected_detections);
}




void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

image collapse_image_layers(image source, int border)
{
    int h = source.h;
    h = (h+border)*source.c - border;
    image dest = make_image(source.w, h, 1);
    int i;
    for(i = 0; i < source.c; ++i){
        image layer = get_image_layer(source, i);
        int h_offset = i*(source.h+border);
        embed_image(layer, dest, 0, h_offset);
        free_image(layer);
    }
    return dest;
}

void constrain_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] < 0) im.data[i] = 0;
        if(im.data[i] > 1) im.data[i] = 1;
    }
}

void normalize_image(image p)
{
    int i;
    float min = 9999999;
    float max = -999999;

    for(i = 0; i < p.h*p.w*p.c; ++i){
        float v = p.data[i];
        if(v < min) min = v;
        if(v > max) max = v;
    }
    if(max - min < .000000001){
        min = 0;
        max = 1;
    }
    for(i = 0; i < p.c*p.w*p.h; ++i){
        p.data[i] = (p.data[i] - min)/(max-min);
    }
}

image copy_image(image p)
{
    image copy = p;
    copy.data = (float*)xcalloc(p.h * p.w * p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

void show_image(image p, string name)
{
    show_image_cv(p, name);
}


image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = (float*)xcalloc(h * w * c, sizeof(float));
    return out;
}

image float_to_image(int w, int h, int c, float *data)
{
    image out = make_empty_image(w,h,c);
    out.data = data;
    return out;
}



void scale_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] *= s;
}

image crop_image(image im, int dx, int dy, int w, int h)
{//dx表示要从原始图片左边删除的宽度，dy表示要从原始图片上边删除的宽度，
//w,h为缩放后的目的大小

    image cropped = make_image(w, h, im.c);
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int r = j + dy;
                int c = i + dx;
                float val = 0;
                r = constrain(0, im.h-1, r);  //r 限制在min，max中
                c = constrain(0, im.w-1, r);
                if (r >= 0 && r < im.h && c >= 0 && c < im.w) {
                    val = get_pixel(im, c, r, k);
                }
                set_pixel(cropped, i, j, k, val);
            }
        }
    }
    return cropped;
}

int best_3d_shift_r(image a, image b, int min, int max)
{
    if(min == max) return min;
    int mid = floor((min + max) / 2.);
    image c1 = crop_image(b, 0, mid, b.w, b.h);
    image c2 = crop_image(b, 0, mid+1, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 10);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 10);
    free_image(c1);
    free_image(c2);
    if(d1 < d2) return best_3d_shift_r(a, b, min, mid);
    else return best_3d_shift_r(a, b, mid+1, max);
}



void fill_image(image m, float s)
{
    int i;
    for (i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}



image letterbox_image(image im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w / im.w) < ((float)h / im.h)) {
        new_w = w;
        new_h = (im.h * w) / im.w;
    }
    else {
        new_h = h;
        new_w = (im.w * h) / im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    image boxed = make_image(w, h, im.c);
    fill_image(boxed, .5);
    //int i;
    //for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
    embed_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
    free_image(resized);
    return boxed;
}


image threshold_image(image im, float thresh)
{
    int i;
    image t = make_image(im.w, im.h, im.c);
    for(i = 0; i < im.w*im.h*im.c; ++i){
        t.data[i] = im.data[i]>thresh ? 1 : 0;
    }
    return t;
}





image resize_image(image im, int w, int h)
{
    if (im.w == w && im.h == h) return copy_image(im);

    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}


image load_image(string filename, int w, int h, int c)
{
    image out = load_image_cv(filename, c);


    if((h && w) && (h != out.h || w != out.w)){
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}



image get_image_layer(image m, int l)
{
    image out = make_image(m.w, m.h, 1);
    int i;
    for(i = 0; i < m.h*m.w; ++i){
        out.data[i] = m.data[i+l*m.h*m.w];
    }
    return out;
}


image collapse_images_vert(image *ims, int n)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    w = ims[0].w;
    h = (ims[0].h + border) * n - border;
    c = ims[0].c;
    if(c != 3 || !color){
        w = (w+border)*c - border;
        c = 1;
    }

    image filters = make_image(w, h, c);
    int i,j;
    for(i = 0; i < n; ++i){
        int h_offset = i*(ims[0].h+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color){
            embed_image(copy, filters, 0, h_offset);
        }
        else{
            for(j = 0; j < copy.c; ++j){
                int w_offset = j*(ims[0].w+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, w_offset, h_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    return filters;
}



void show_images(image *ims, int n, char *window)
{
    image m = collapse_images_vert(ims, n);
    /*
       int w = 448;
       int h = ((float)m.h/m.w) * 448;
       if(h > 896){
       h = 896;
       w = ((float)m.w/m.h) * 896;
       }
       image sized = resize_image(m, w, h);
     */
    normalize_image(m);
    save_image(m, window);
    show_image(m, window);
    free_image(m);
}

void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}

void save_image(image im, string name)
{
    save_image_cv(im,name);

}