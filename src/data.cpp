#include "data.hpp"
#include "utils.hpp"
#include "image.hpp"
#include "box.hpp"
#include <darknet.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>
#include <opencv2/opencv.hpp>
//#include <mutex>
#include <thread>
#include <atomic>
#include <ctime>
using std::cerr;
using std::endl;
using namespace std;
extern int check_mistakes;

#define NUMCHARS 37

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;


vector<string> get_paths(string filename){
    vector<string> result;
    ifstream isf(filename);
    if(isf.is_open()){
        string buf;
        while (getline(isf,buf)){
            result.push_back(buf);
        }
        isf.close();
    }
    return result;

}


vector<string>get_sequential_paths(vector<string>paths, int n, int m, int mini_batch, int augment_speed)
{// n 为一次加载量 == net.batch * net.subdivisions * ngpus;
//  m为图片总量

    int speed = rand_int(1, augment_speed);
    if (speed < 1) speed = 1;
    vector<string>sequentia_paths(n) ;
    int i;
    //pthread_mutex_lock(&mutex);
    //printf("n = %d, mini_batch = %d \n", n, mini_batch);
    unsigned int *start_time_indexes = (unsigned int *)xcalloc(mini_batch, sizeof(unsigned int));
    for (i = 0; i < mini_batch; ++i) {
        start_time_indexes[i] = random_gen() % m;
        //printf(" start_time_indexes[i] = %u, ", start_time_indexes[i]);
    }

    for (i = 0; i < n; ++i) {
        do {
            int time_line_index = i % mini_batch;
            unsigned int index = start_time_indexes[time_line_index] % m;
            start_time_indexes[time_line_index] += speed;
            //speed表示两个批次的间隔，eg：当speed为2时；
            //如第一批为2，3，4,...
            //则第二批为4，5，6,...

            //int index = random_gen() % m;
            sequentia_paths[i] = paths[index];
            //if(i == 0) printf("%s\n", paths[index]);
            //printf(" index = %u - grp: %s \n", index, paths[index]);
            if (sequentia_paths[i].length() <= 4) clog<<" Very small path to the image: , "<<sequentia_paths[i]<<endl;
        } while (sequentia_paths[i].empty());
    }
    free(start_time_indexes);
    //pthread_mutex_unlock(&mutex);
    return sequentia_paths;
}

vector<string>get_random_paths(vector<string>paths, int n, int m)
{//随即选n个图片路经
    vector<string>random_paths(n);
    int i;
    //pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        int index = random_gen() % m;
        random_paths[i] = paths[index];
        if (random_paths[i].length() <= 4) clog<<" Very small path to the image: "<< random_paths[i]<<endl;
    }
    //pthread_mutex_unlock(&mutex);
    return random_paths;
}



vector<box_label> read_boxes(string filename, int &n)
{

    vector<box_label> boxes;
    FILE *file = fopen(filename.c_str(), "r");
    if (!file) {
        clog<<"Can't open label file. (This can be normal only if you use MSCOCO): "<<filename<<endl;
        //file_error(filename);
        FILE* fw = fopen("bad.list", "a");
        fwrite(filename.c_str(), sizeof(char), filename.length(), fw);
        char *new_line = "\n";
        fwrite(new_line, sizeof(char), strlen(new_line), fw);
        fclose(fw);
        if (check_mistakes) {
            printf("\n Error in read_boxes() \n");
            getchar();
        }

        n = 0;
        return boxes;
    }
    float x, y, h, w;
    int id;
    int count = 0;
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        boxes.push_back(box_label{0});
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;

        ++count;
    }
    fclose(file);
    n = count;
    return boxes;
}

void randomize_boxes(vector<box_label> b, int n)
{//打乱label_box的出现顺序
    int i;
    for(i = 0; i < n; ++i){
        box_label swap = b[i];
        int index = random_gen()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

void correct_boxes(vector<box_label>boxes, int n, float dx, float dy, float sx, float sy, bool flip)
{
    //int swidth = ow - pleft - pright;
    //int sheight = oh - ptop - pbot;
    //float dx =  pleft/swight
    //float dy = ptop/sheight
    //float sx = ow/swidth;
    //float sy = oh/sheight ;
    int i;
    for(i = 0; i < n; ++i){
        if(boxes[i].x == 0 && boxes[i].y == 0) {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        if ((boxes[i].x + boxes[i].w / 2) < 0 || (boxes[i].y + boxes[i].h / 2) < 0 ||
            (boxes[i].x - boxes[i].w / 2) > 1 || (boxes[i].y - boxes[i].h / 2) > 1)
        {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        if(flip){
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        boxes[i].left =  constrain(0, 1, boxes[i].left);//boxes[i].left加逼在min,max中
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}


int fill_truth_detection(string path, int num_boxes, vector<array<float, 5> > &truth, int classes, bool flip, float dx, float dy, float sx, float sy, int net_w, int net_h)
{
    //filename, boxes, truth, classes, flip, dx, dy, 1. / sx, 1. / sy, w, h
    //float *truth = (float*)xcalloc(5 * boxes, sizeof(float));
    //float dx =  pleft/swight
    //float dy = ptop/sheight
    //float sx = ow/swidth;
    //float sy = oh/sheight ;
    replace_image_to_label(path);// 将训练的图像路经换成label

    int count = 0;
    int i;
    int init_min = INT_MAX;
    vector<box_label> boxes = read_boxes(path, count); //读取label文件，并获取坐标框值
    //int min_w_h = 0;
    float lowest_w = 1.F / net_w;
    float lowest_h = 1.F / net_h;
    randomize_boxes(boxes, count);//打乱label_box的顺序，
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);//因为图像进行了缩放，所以gt_box也进行处理。 这里会对boxes进行修改
    if (count > num_boxes) count = num_boxes;  //一张最多处理num_boxes的box

    float x, y, w, h;
    int id;
    int sub = 0;

    for (i = 0; i < count; ++i) {
        x = boxes[i].x;
        y = boxes[i].y;
        w = boxes[i].w;
        h = boxes[i].h;
        id = boxes[i].id;

        if (id >= classes) {
            ++sub;
            continue;
        }
        if ((w < lowest_w || h < lowest_h)) {

            ++sub;
            continue;
        }
        if (x == 999999 || y == 999999) {
            ++sub;
            continue;
        }

        //为什么x,y,w,h都是小于1的 ，在gt_label时就是这样吗？
        // 是的！ 在用作者提供的yolo_mark工具标注的时候，就是[0,1]之间爱你
        // 并且id为[0, num-1]
        if (x <= 0 || x > 1 || y <= 0 || y > 1) {

            ++sub;
            continue;
        }


        truth[(i-sub)][0] = x;
        truth[(i-sub)][1] = y;
        truth[(i-sub)][2] = w;
        truth[(i-sub)][3] = h;
        truth[(i-sub)][4] = id;

        init_min = min(init_min, (int)min(w*net_w,h*net_h));
    }
    //free(boxes);
    return init_min;
}



vector<string> get_labels_custom(string filename, int &size)
{
        vector<string> result = get_labels(filename);
        size = result.size();
        return result;
}

vector<string>get_labels(string filename)
{
    vector<string> result;
    std::ifstream ifs;
    ifs.open(filename, ios::in);
    if (!ifs.is_open()) {
        cout << "文件打开失败！" << endl;
        ifs.close();
        return result;
    }else
    {
        string buf;
        while (getline(ifs, buf)) {
            result.push_back(buf);
        }

        return result;
    }
}

void free_data(data d)
{
    if(!d.shallow){
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        free(d.X.vals);
        free(d.y.vals);
    }
}

void blend_truth(float *new_truth, int boxes, vector<array<float,5 > >&old_truth)
{
    const int t_size = 4 + 1;
    int count_new_truth = 0;
    int t;
    for (t = 0; t < boxes; ++t) {
        float x = new_truth[t*(4 + 1)];
        if (!x) break;
        count_new_truth++;

    }// truth存储了两个图片的label，
    for (t = count_new_truth; t < boxes; ++t) {
        float *new_truth_ptr = new_truth + t*t_size;
        array<float,5 > old_truth_ptr = old_truth[t - count_new_truth];
        float x = old_truth_ptr[0];
        if (!x) break;

        new_truth_ptr[0] = old_truth_ptr[0];
        new_truth_ptr[1] = old_truth_ptr[1];
        new_truth_ptr[2] = old_truth_ptr[2];
        new_truth_ptr[3] = old_truth_ptr[3];
        new_truth_ptr[4] = old_truth_ptr[4];
    }
    //printf("\n was %d bboxes, now %d bboxes \n", count_new_truth, t);
}


void blend_truth_mosaic(float *new_truth, int boxes, vector<array<float,5> >&old_truth, int w, int h, float cut_x, float cut_y, int i_mixup,
    int left_shift, int right_shift, int top_shift, int bot_shift)
{
    const int t_size = 4 + 1;
    int count_new_truth = 0;
    int t;
    for (t = 0; t < boxes; ++t) {//记录在new_truth中几经有几个值了，后面的新近来的truth会根在后面
        float x = new_truth[t*(4 + 1)];
        if (!x) break;
        count_new_truth++;

    }
    int new_t = count_new_truth;
    for (t = count_new_truth; t < boxes; ++t) {
        float *new_truth_ptr = new_truth + new_t*t_size;
        new_truth_ptr[0] = 0;
        array<float,5> old_truth_ptr = old_truth[t - count_new_truth];
        float x = old_truth_ptr[0];
        if (!x) break;

        float xb = old_truth_ptr[0];
        float yb = old_truth_ptr[1];
        float wb = old_truth_ptr[2];
        float hb = old_truth_ptr[3];



        // shift 4 images
        if (i_mixup == 0) {//右上
            xb = xb - (float)(w - cut_x - right_shift) / w;
            yb = yb - (float)(h - cut_y - bot_shift) / h;
        }
        if (i_mixup == 1) { //左上
            xb = xb + (float)(cut_x - left_shift) / w;
            yb = yb - (float)(h - cut_y - bot_shift) / h;
        }
        if (i_mixup == 2) {
            xb = xb - (float)(w - cut_x - right_shift) / w;
            yb = yb + (float)(cut_y - top_shift) / h;
        }
        if (i_mixup == 3) {
            xb = xb + (float)(cut_x - left_shift) / w;
            yb = yb + (float)(cut_y - top_shift) / h;
        }

        int left = (xb - wb / 2)*w;
        int right = (xb + wb / 2)*w;
        int top = (yb - hb / 2)*h;
        int bot = (yb + hb / 2)*h;

        // fix out of bound
        if (left < 0) {
            float diff = (float)left / w;
            xb = xb - diff / 2;
            wb = wb + diff;
        }

        if (right > w) {
            float diff = (float)(right - w) / w;
            xb = xb - diff / 2;
            wb = wb - diff;
        }

        if (top < 0) {
            float diff = (float)top / h;
            yb = yb - diff / 2;
            hb = hb + diff;
        }

        if (bot > h) {
            float diff = (float)(bot - h) / h;
            yb = yb - diff / 2;
            hb = hb - diff;
        }

        left = (xb - wb / 2)*w;
        right = (xb + wb / 2)*w;
        top = (yb - hb / 2)*h;
        bot = (yb + hb / 2)*h;

        // leave only within the image
        if(left >= 0 && right <= w && top >= 0 && bot <= h &&
            wb > 0 && wb < 1 && hb > 0 && hb < 1 &&
            xb > 0 && xb < 1 && yb > 0 && yb < 1)
        {
            new_truth_ptr[0] = xb;
            new_truth_ptr[1] = yb;
            new_truth_ptr[2] = wb;
            new_truth_ptr[3] = hb;
            new_truth_ptr[4] = old_truth_ptr[4];
            new_t++;
        }
    }
}


/*
** 可以参考，看一下对图像进行jitter处理的各种效果:
** https://github.com/vxy10/ImageAugmentation
** 从所有训练图片中，随机读取n张，并对这n张图片进行数据增强，同时矫正增强后的数据标签信息。最终得到的图片的宽高为w,h（原始训练集中的图片尺寸不定），也就是网络能够处理的图片尺寸，
** 数据增强包括：对原始图片进行宽高方向上的插值缩放（两方向上缩放系数不一定相同），下面称之为缩放抖动；随机抠取或者平移图片（位置抖动）；
** 在hsv颜色空间增加噪声（颜色抖动）；左右水平翻转，不含旋转抖动。
** 输入：n         一个线程读入的图片张数（详见函数内部注释）
**       paths     所有训练图片所在路径集合，是一个二维数组，每一行对应一张图片的路径（将在其中随机取n个）
**       m         paths的行数，也即训练图片总数
**       w         网络能够处理的图的宽度（也就是输入图片经过一系列数据增强、变换之后最终输入到网络的图的宽度）
**       h         网络能够处理的图的高度（也就是输入图片经过一系列数据增强、变换之后最终输入到网络的图的高度）
**       c         用来指定训练图片的通道数（默认为3，即RGB图）
**       boxes     每张训练图片最大处理的矩形框数（图片内可能含有更多的物体，即更多的矩形框，那么就在其中随机选择boxes个参与训练，具体执行在fill_truth_detection()函数中）
**       classes   类别总数，本函数并未用到（fill_truth_detection函数其实并没有用这个参数）
**       use_flip  是否使用水平翻转
**       use_mixup 是否使用mixup数据增强
**       jitter    这个参数为缩放抖动系数，就是图片缩放抖动的剧烈程度，越大，允许的抖动范围越大（所谓缩放抖动，就是在宽高上插值缩放图片，宽高两方向上缩放的系数不一定相同）
**       hue       颜色（hsv颜色空间）数据增强参数：色调（取值0度到360度）偏差最大值，实际色调偏差为-hue~hue之间的随机值
**       saturation 颜色（hsv颜色空间）数据增强参数：色彩饱和度（取值范围0~1）缩放最大值，实际为范围内的随机值
**       exposure  颜色（hsv颜色空间）数据增强参数：明度（色彩明亮程度，0~1）缩放最大值，实际为范围内的随机值
**       mini_batch      和目标跟踪有关，这里不关注
**       track           和目标跟踪有关，这里不关注
**       augment_speed   和目标跟踪有关，这里不关注
**       letter_box 是否进行letter_box变换
**       show_imgs
** 返回：data类型数据，包含一个线程读入的所有图片数据（含有n张图片）
** 说明：最后四个参数用于数据增强，主要对原图进行缩放抖动，位置抖动（平移）以及颜色抖动（颜色值增加一定噪声），抖动一定程度上可以理解成对图像增加噪声。
**       通过对原始图像进行抖动，实现数据增强。最后三个参数具体用法参考本函数内调用的random_distort_image()函数
** 说明2：从此函数可以看出，darknet对训练集中图片的尺寸没有要求，可以是任意尺寸的图片，因为经该函数处理（缩放/裁剪）之后，
**       不管是什么尺寸的照片，都会统一为网络训练使用的尺寸
*/
data load_data_detection(int n, vector<string>paths, int m, int w, int h, int c, int boxes, int classes, int use_flip, int use_gaussian_noise, int use_blur, int use_mixup,
    float jitter, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int show_imgs)
{     //(a.n, a.paths, a.m, a.w, a.h, a.c, a.num_boxes, a.classes, a.flip, a.gaussian_noise, a.blur, a.mixup, a.jitter,
    //a.hue, a.saturation, a.exposure, a.mini_batch, a.track, a.augment_speed, a.letter_box, a.show_imgs)

    const int random_index = random_gen();
    c = c ? c : 3;

    if (use_mixup == 2) {  //0
        printf("\n cutmix=1 - isn't supported for Detector \n");
        exit(0);
    }
    if (use_mixup == 3 && letter_box) {  	// mixup 和 letter_box策略不能共存  //0
        printf("\n Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters \n");
        exit(0);
    }
    if (random_gen() % 2 == 0) use_mixup = 0;
    int i;
    int *cut_x = NULL, *cut_y = NULL;  //
    /*
     * mixup = 3 时，
     * 对于输入的一个batch的待测图片images，我们将其和随机抽取的图片进行融合，融合比例为lam，得到混合张量inputs；
     * 1：第1步中图片融合的比例lam是[0,1]之间的随机实数，符合beta分布，相加时两张图对应的每个像素值直接相加，
     *    即 inputs = lam*images + (1-lam)*images_random；
     * 2：将1中得到的混合张量inputs传递给model得到输出张量outpus，
     * 3： 随后计算损失函数时，我们针对两个图片的标签分别计算损失函数，然后按照比例lam进行损失函数的加权求和，
     *  即loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)；
        作者：Pascal
        链接：https://www.zhihu.com/question/308572298/answer/585140274
     */
    if (use_mixup == 3){ //1
        cut_x = (int*)calloc(n, sizeof(int));
        cut_y = (int*)calloc(n, sizeof(int));
        const float min_offset = 0.2; // 20%
        for (i = 0; i < n; ++i) {  //上下左右裁减比例,如：cut_x[i] = 20% 则意味着一个图保留左边20% ，另个图保留右边８0%
            // 这里可以参考
            cut_x[i] = rand_int(w*min_offset, w*(1 - min_offset));
            cut_y[i] = rand_int(h*min_offset, h*(1 - min_offset));
        }
    }

    // 初始化为0,清空内存中之前的旧值
    data d{0};
    d.shallow = 0;
    // 一次读入的图片张数：d.X中每行就是一张图片的数据，因此d.X.cols等于h*w*3
    // n = net.batch * net.subdivisions * ngpus，net中的subdivisions这个参数暂时还没搞懂有什么用，
    // 从parse_net_option()函数可知，net.batch = net.batch / net.subdivision，等号右边的那个batch就是
    // 网络配置文件.cfg中设置的每个batch的图片数量，但是不知道为什么多了subdivision这个参数？总之，
    // net.batch * net.subdivisions又得到了在网络配置文件中设定的batch值，然后乘以ngpus，是考虑多个GPU实现数据并行，
    // 一次读入多个batch的数据，分配到不同GPU上进行训练。在load_threads()函数中，又将整个的n仅可能均匀的划分到每个线程上，
    // 也就是总的读入图片张数为n = net.batch * net.subdivisions * ngpus，但这些图片不是一个线程读完的，而是分配到多个线程并行读入，
    // 因此本函数中的n实际不是总的n，而是分配到该线程上的n，比如总共要读入128张图片，共开启8个线程读数据，那么本函数中的n为16,而不是总数128

    d.X.rows = n;
    //d.X为一个matrix类型数据，其中d.X.vals是其具体数据，是指针的指针（即为二维数组），此处先为第一维动态分配内存
    d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*c;

    //float r_scale = 0;
    float dhue = 0, dsat = 0, dexp = 0, flip = 0, blur = 0;
    int gaussian_noise = 0;
    // d.y存储了所有读入照片的标签信息，每条标签包含5条信息：类别，以及矩形框的x,y,w,h
    // boxes为一张图片最多能够处理（参与训练）的矩形框的数（如果图片中的矩形框数多于这个数，那么随机挑选boxes个，这个参数仅在parse_region以及parse_detection中出现，好奇怪？
    // 在其他网络解析函数中并没有出现）。同样，d.y是一个matrix，make_matrix会指定y的行数和列数，同时会为其第一维动态分配内存

    d.y = make_matrix(n, 5*boxes);  //x,y,w,h,c
    int i_mixup = 0;
    // 当use_mixup==0 时，没有拼接数据增强，只有当use_mixup==3时才有，在cfg中，指定为3
    for (i_mixup = 0; i_mixup <= use_mixup; i_mixup++) { //use_mixup等于3或者0，也就是这个循环执行一次或者四次
        // 当use_mixup　等于三时，实际上会加载４×batch的图像，每四个图像生成一个图像。　

        vector<string>random_paths;
        if (track) random_paths = get_sequential_paths(paths, n, m, mini_batch, augment_speed);//0
        else random_paths = get_random_paths(paths, n, m);

        for (i = 0; i < n; ++i) {
            //float *truth = (float*)xcalloc(5 * boxes, sizeof(float));
            vector<array<float, 5> > truth;
            string filename = random_paths[i];
            //读入原始的图片
            int flag = (c >= 3);
            //mat_cv *src;
            cv::Mat src ;
            src = load_image_mat_cv(filename, flag);
            if (!src.data) {
                if (check_mistakes) {
                    printf("\n Error in load_data_detection() - OpenCV \n");
                    getchar();
                }
                continue;
            }

            int oh = get_height_mat(src);
            int ow = get_width_mat(src);

            int dw = (ow*jitter); //抖动
            int dh = (oh*jitter);

            if (i_mixup || !track)//1  //即当i_mixup不等于零时
            {
//
                //r_scale = random_float();

                dhue = rand_uniform_strong(-hue, hue);  // 生成一个介于-hue, hue之间的值
                dsat = rand_scale(saturation);  //可大可小
                dexp = rand_scale(exposure);

                flip = use_flip ? random_gen() % 2 : 0;  //use_flip = 1  //flip为随即正负

                if (use_blur) {  //0
                    /*
                    int tmp_blur = rand_int(0, 2);  // 0 - disable, 1 - blur background, 2 - blur the whole image
                    if (tmp_blur == 0) blur = 0;
                    else if (tmp_blur == 1) blur = 1;
                    else blur = use_blur;
                    */
                    blur =  rand_int(0, 2);
                }

                if (use_gaussian_noise && rand_int(0, 1) == 1) gaussian_noise = use_gaussian_noise;  //0
                else gaussian_noise = 0;
            }

            int pleft =rand_uniform_strong(-dw, dw);  //从左边裁去的,
            // 注意这里为什么可以是负值，因为pleft在image_data_augmentation使用时是作为左下角从cv::Rect的x值的，
            // 同理ptop ， 当把pleft和ptop作为image_data_augmentation的入口，表示要裁剪区域的左下角，这样当pleft为负值时，即表示不裁剪
            int pright = rand_uniform_strong(-dw, dw);
            int ptop = rand_uniform_strong(-dh, dh);
            int pbot = rand_uniform_strong(-dh, dh);

            if (letter_box)//0
            {
                float img_ar = (float)ow / (float)oh; //原始图像宽高比
                float net_ar = (float)w / (float)h; //输入到网络要求的图像宽高比
                float result_ar = img_ar / net_ar; //两者求比值来判断如何进行letter_box缩放
                //printf(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if (result_ar > 1)  // sheight - should be increased
                {
                    float oh_tmp = ow / net_ar;
                    float delta_h = (oh_tmp - oh) / 2;
                    ptop = ptop - delta_h;
                    pbot = pbot - delta_h;
                    //printf(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                }
                else  // swidth - should be increased
                {
                    float ow_tmp = oh * net_ar;
                    float delta_w = (ow_tmp - ow) / 2;
                    pleft = pleft - delta_w;
                    pright = pright - delta_w;
                    //printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);
                }
            }

            // 以下步骤就是执行了letter_box变换
            int swidth = ow - pleft - pright;
            int sheight = oh - ptop - pbot;

            float sx = (float)swidth / ow;
            float sy = (float)sheight / oh;

            float dx =(float)pleft /(float)swidth ;
            float dy = (float)ptop / (float)sheight;

            //填写truth信息 ，因为图像发生了缩放之类的操作，所以gt_box也会变化，左边信息会传递到truth中, min_w_h为最小的bbox尺寸
            int min_w_h = fill_truth_detection(filename, boxes, truth, classes, flip, dx, dy, 1. / sx, 1. / sy, w, h);

            if ((min_w_h / 8) < blur && blur > 1) blur = min_w_h / 8;   //0 // disable blur if one of the objects is too small
            //事实上pleft, ptop, swidth, sheight分别为要裁减矩阵的左上角顶点及其w，h
            image ai = image_data_augmentation(src, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp,
                gaussian_noise, blur, boxes, truth);

            if (use_mixup == 0) {//0
                d.X.vals[i] = ai.data;
                array_to_float(d.y.vals[i], truth, boxes); // 将truth的数据写进d.y.vals[i]中
            }
            else if (use_mixup == 1) {//0
                if (i_mixup == 0) {
                    d.X.vals[i] = ai.data;
                    //memcpy(d.y.vals[i], truth, 5 * boxes * sizeof(float));
                    array_to_float(d.y.vals[i], truth, boxes);
                }
                else if (i_mixup == 1) {
                    image old_img = make_empty_image(w, h, c);
                    old_img.data = d.X.vals[i];

                    blend_images_cv(ai, 0.5, old_img, 0.5);
                    blend_truth(d.y.vals[i], boxes, truth);
                    free_image(old_img);
                    d.X.vals[i] = ai.data;
                }

            }
            else if (use_mixup == 3) {//1
                if (i_mixup == 0) {
                    image tmp_img = make_image(w, h, c);
                    d.X.vals[i] = tmp_img.data;
                }

                if (flip) {//1
                    int tmp = pleft;
                    pleft = pright;
                    pright = tmp;
                }
                // 因为是从４个图片中各裁剪一部分，然后拼接成一个图片，
                const int left_shift = min_val_cmp(cut_x[i], max_val_cmp(0, (-pleft*w / ow)));  //返回最小值
                const int top_shift = min_val_cmp(cut_y[i], max_val_cmp(0, (-ptop*h / oh)));  //这几个shift一般较小

                const int right_shift = min_val_cmp((w - cut_x[i]), max_val_cmp(0, (-pright*w / ow)));
                const int bot_shift = min_val_cmp(h - cut_y[i], max_val_cmp(0, (-pbot*h / oh)));


                int k, x, y;
                for (k = 0; k < c; ++k) {
                    for (y = 0; y < h; ++y) {
                        int j = y*w + k*w*h;
                        if (i_mixup == 0 && y < cut_y[i]) { //第一个图，保留左边的cut_x[i]
                            int j_src = (w - cut_x[i] - right_shift) + (y + h - cut_y[i] - bot_shift)*w + k*w*h;
                            memcpy(&d.X.vals[i][j + 0], &ai.data[j_src], cut_x[i] * sizeof(float));
                            //给左下位置存放图片，为什么是左下，左是因为存放地址的起始地址为d.X.vals[i][j + 0]，为什么是下，因为y < cut_y[i]
                            //
                        }
                        if (i_mixup == 1 && y < cut_y[i]) {
                            int j_src = left_shift + (y + h - cut_y[i] - bot_shift)*w + k*w*h;
                            memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src], (w-cut_x[i]) * sizeof(float));
                            //给右下位置存放图片
                        }
                        if (i_mixup == 2 && y >= cut_y[i]) {//0
                            int j_src = (w - cut_x[i] - right_shift) + (top_shift + y - cut_y[i])*w + k*w*h;
                            memcpy(&d.X.vals[i][j + 0], &ai.data[j_src], cut_x[i] * sizeof(float));
                            //给左上位置存放图片
                        }
                        if (i_mixup == 3 && y >= cut_y[i]) {//0
                            int j_src = left_shift + (top_shift + y - cut_y[i])*w + k*w*h;
                            memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src], (w - cut_x[i]) * sizeof(float));
                            //给右上位置存放图片
                        }
                    }
                }
                //将truth值写进d.y.vals[i]，同时纠正truth  //truth的纠正结果将续在d.y.vals[i]后
                blend_truth_mosaic(d.y.vals[i], boxes, truth, w, h, cut_x[i], cut_y[i], i_mixup, left_shift, right_shift, top_shift, bot_shift);

                free_image(ai);
                ai.data = d.X.vals[i];
            }


            if (show_imgs && i_mixup == use_mixup)   // delete i_mixup  //show_imgs = 0 //0
            {
                image tmp_ai = copy_image(ai);
                char buff[1000];
                sprintf(buff, "aug_%d_%d_%d", random_index, i, random_gen());
                int t;
                for (t = 0; t < boxes; ++t) {
                    box b = float_to_box(d.y.vals[i] + t*(4 + 1));
                    if (!b.x) break;
                    int left = (b.x - b.w / 2.)*ai.w;
                    int right = (b.x + b.w / 2.)*ai.w;
                    int top = (b.y - b.h / 2.)*ai.h;
                    int bot = (b.y + b.h / 2.)*ai.h;
                    draw_box_width(tmp_ai, left, top, right, bot, 1, 150, 100, 50); // 3 channels RGB
                }

                save_image(tmp_ai, buff);
                if (show_imgs == 1) {
                    show_image(tmp_ai, buff);
                    wait_until_press_key_cv();
                }
                printf("\nYou use flag -show_imgs, so will be saved aug_...jpg images. Click on window and press ESC button \n");
                free_image(tmp_ai);
            }

            //release_mat(&src);
            //free(truth);
        }
        //if (random_paths) free(random_paths);
    }


    return d;
}




void get_next_batch(data d, int n, int offset, float *X, float *y)

{   //会返回一个小batch
    // n = batch, offset = i*batch,
    // float* X = (float*)xcalloc(batch * d.X.cols, sizeof(float));
    //    float* y = (float*)xcalloc(batch * d.y.cols, sizeof(float));
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}
void array_to_float(float *d, vector<array<float,5> >s, int num ){
    for(int i = 0 ; i < num; i++){
        for (int j = 0 ; j< 5; j++){
            d[i*5+j] = s[i][j];
        }
    }
}