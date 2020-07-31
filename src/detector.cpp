
#include "darknet.hpp"
#include "network.hpp"
#include "utils.hpp"
#include "parser.hpp"
#include "box.hpp"
#include "option_list.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <map>
#include "image_opencv.hpp"
using namespace std;
#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int (*__compar_fn_t)(const void*, const void*);
#ifdef __USE_GNU
typedef __compar_fn_t comparison_fn_t;
#endif
#endif


bool check_mistakes = false;

void train_detector(string datacfg, string cfgfile, string weightfile, int *gpus, int ngpus, int clear, int dont_show, bool calc_map, int mjpeg_port, int show_imgs, int benchmark_layers, string chart_path)
{

    map<string,string> options = read_data_cfg(datacfg);
    string name_list = option_find_str(options, "names", "data/names.list");
    string train_images = option_find_str(options, "train", "data/train.txt");
    string valid_images = option_find_str(options, "valid", train_images);
    string backup_directory = option_find_str(options, "backup", "/backup/");

    network net_map;

    srand(time(0));

    float avg_loss = -1;
    network* nets = (network*)xcalloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int k;
    for (k = 0; k < ngpus; ++k) {
        srand(seed);

        nets[k] = parse_network_cfg(cfgfile);
        nets[k].benchmark_layers = benchmark_layers;
        if (!weightfile.empty()) {
            load_weights(nets[k], weightfile);
        }
        if (clear) {
            nets[k].seen = 0;
            nets[k].cur_iteration = 0;
        }
        nets[k].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    const int actual_batch_size = net.batch * net.subdivisions;
    if (actual_batch_size == 1) {
        printf("\n Error: You set incorrect value batch=1 for Training! You should set batch=64 subdivision=64 \n");
        getchar();
    }
    else if (actual_batch_size < 8) {
        printf("\n Warning: You set batch=%d lower than 64! It is recommended to set batch=64 subdivision=64 \n", actual_batch_size);
    }

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train;

    layer l = net.layers[net.n - 1];
    //jitter , classes , random 为什么不定义在net中，而是定义在yolo中，yolo中并没有用到这个
    int classes = l.classes;
    float jitter = l.jitter;
    vector<string> path  = get_paths (train_images);
    int train_images_num = path.size();
    //char **paths = (char **)list_to_array(plist);


    const int init_w = net.w;
    const int init_h = net.h;
    const int init_b = net.batch;
    int iter_save, iter_save_last, iter_map;
    iter_save = get_current_iteration(net);
    iter_save_last = get_current_iteration(net);
    iter_map = get_current_iteration(net);
    float mean_average_precision = -1;
    float best_map = mean_average_precision;

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.paths = path;
    args.n = imgs;  //一次加载的数量
    args.m = train_images_num;
    args.classes = classes;
    args.flip = net.flip;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    net.num_boxes = args.num_boxes;
    net.train_images_num = train_images_num;
    //args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 64;    // 16 or 64

    args.angle = net.angle;
    args.gaussian_noise = net.gaussian_noise;
    args.blur = net.blur;
    args.mixup = net.mixup;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.letter_box = net.letter_box;
    if (dont_show && show_imgs) show_imgs = 2;  // dont_show = 0 ; show_imgs = 0
    args.show_imgs = show_imgs;  //SO  show_imgs = 0

    args.threads = 6 * ngpus;   // 3 for - Amazon EC2 Tesla V100: p3.2xlarge (8 logical cores) - p3.16xlarge
    //args.threads = 12 * ngpus;    // Ryzen 7 2700X (16 logical cores)
    mat_cv* img = NULL;
    float max_img_loss = 5;
    int number_of_lines = 100;
    int img_size = 1000;
    //char windows_name[100];
    string windows_name = "chart_yolov4.png";
    //sprintf(windows_name, "chart_%s.png", "yolov4");

    //Draw Loss & Accuracy chart
    //img = draw_train_chart(windows_name, max_img_loss, net.max_batches, number_of_lines, img_size, dont_show, chart_path);
    //printf(" imgs = %d \n", imgs);

    //pthread_t load_thread = load_data(args);//多线程加载


    //n张图片以及图片上的truth box会被加载到buffer.X,buffer.y里面去

    int count = 0;
    double time_remaining, avg_time = -1, alpha_time = 0.01;

    //while(i*imgs < N*120){
    while (get_current_iteration(net) < net.max_batches) { //max_batches = 500500
        train = load_data_detection(args.n, args.paths, args.m, args.w, args.h, args.c, args.num_boxes, args.classes, args.flip, args.gaussian_noise, args.blur, args.mixup, args.jitter, args.hue, args.saturation, args.exposure, args.mini_batch, args.track, args.augment_speed, args.letter_box, args.show_imgs);

        //每subdivisions*batch个图片记为一个iteration
        if (l.random && count++ % 10 == 0) { //l.random决定是否多尺度训练,如果要的话每训练10个batch进行一下下面的操作,也就是随即缩放`// random==1
            float rand_coef = 1.4;
            if (l.random != 1.0) rand_coef = l.random; //l.random = 1
            cout<<"Resizing, random_coef = "<<rand_coef<<endl;
            float random_val = rand_scale(rand_coef);    // x or 1/x
            int dim_w = roundl(random_val*init_w / net.resize_step + 1) * net.resize_step;  //resize_step :32
            int dim_h = roundl(random_val*init_h / net.resize_step + 1) * net.resize_step;
            if (random_val < 1 && (dim_w > init_w || dim_h > init_h)) dim_w = init_w, dim_h = init_h;

            int max_dim_w = roundl(rand_coef*init_w / net.resize_step + 1) * net.resize_step;
            int max_dim_h = roundl(rand_coef*init_h / net.resize_step + 1) * net.resize_step;

            // at the beginning (check if enough memory) and at the end (calc rolling mean/variance)
            if (avg_loss < 0 || get_current_iteration(net) > net.max_batches - 100) {  //max_batches = 500500
                dim_w = max_dim_w;
                dim_h = max_dim_h;
            }

            if (dim_w < net.resize_step) dim_w = net.resize_step;
            if (dim_h < net.resize_step) dim_h = net.resize_step;
            int dim_b = (init_b * max_dim_w * max_dim_h) / (dim_w * dim_h);
            int new_dim_b = (int)(dim_b * 0.8);
            if (new_dim_b > init_b) dim_b = new_dim_b;


            args.w = dim_w;
            args.h = dim_h;

            int k;

            //pthread_join(load_thread, 0);
            //train = buffer;
            free_data(train);
            //load_thread = load_data(args);

            for (k = 0; k < ngpus; ++k) {
                resize_network(nets + k, dim_w, dim_h);  //多处度训练
            }
            net = nets[0];
        }
        double time = clock();

        //args.n数量的图像由args.threads个子线程加载完成,该线程会退出
        //pthread_join(load_thread, 0);
        //加载完成的args.n张图像会存入到args.d中
        //train = buffer;


        const double load_time = clock() - time;
        printf("Loaded: %lf seconds", load_time);
        if (load_time > 0.1 && avg_loss > 0) printf(" - performance bottleneck on CPU or Disk HDD/SSD");
        printf("\n");

        time = clock();
        float loss = 0;

        loss = train_network(net, train);

        if (avg_loss < 0 || avg_loss != avg_loss) avg_loss = loss;    // if(-inf or nan)
        avg_loss = avg_loss*.9 + loss*.1;

        const int iteration = get_current_iteration(net);
        //i = get_current_batch(net);

        int calc_map_for_each = 4 * train_images_num / (net.batch * net.subdivisions);  // calculate mAP for each 4 Epochs
        calc_map_for_each = std::max(calc_map_for_each, 100);
        int next_map_calc = iter_map + calc_map_for_each;
        next_map_calc = std::max(next_map_calc, net.burn_in);
        //next_map_calc = fmax(next_map_calc, 400);
        if (calc_map) {//0
            printf("\n (next mAP calculation at %d iterations) ", next_map_calc);
            if (mean_average_precision > 0) printf("\n Last accuracy mAP@0.5 = %2.2f %%, best = %2.2f %% ", mean_average_precision * 100, best_map * 100);
        }

        printf("\n %d: %f, %f avg loss, %f rate, %lf seconds, %d images, %f hours left\n", iteration, loss, avg_loss, get_current_rate(net), (what_time_is_it_now() - time), iteration*imgs, avg_time);

        int draw_precision = 0;

        time_remaining = (net.max_batches - iteration)*(clock() - time + load_time) / 60 / 60;
        // set initial value, even if resume training from 10000 iteration
        if (avg_time < 0) avg_time = time_remaining;
        else avg_time = alpha_time * time_remaining + (1 -  alpha_time) * avg_time;
        draw_train_loss(windows_name, img, img_size, avg_loss, max_img_loss, iteration, net.max_batches, mean_average_precision, draw_precision, "mAP%", dont_show, mjpeg_port, avg_time);

        //if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
        //if (i % 100 == 0) {
        if (iteration >= (iter_save + 1000) || iteration % 1000 == 0) { //每1000代保存一次全重
            iter_save = iteration;

            //char buff[256];
            string buff = backup_directory+"/yolov4_"+to_string(iteration)+".weights";
            //sprintf(buff, "%s/%s_%d.weights", backup_directory, base, iteration);
            save_weights(net, buff);
        }
        //实际上上面和下面可以任选一个
        if (iteration >= (iter_save_last + 100) || (iteration % 100 == 0 && iteration > 1)) {
            iter_save_last = iteration;

            //char buff[256];
            //sprintf(buff, "%s/%s_last.weights", backup_directory, base);
            string buff = backup_directory+"/yolov4_last.weights";

            save_weights(net, buff);
        }

        //这里要相当注意,train指针指向的空间来自于buffer,而buffer中的空间来自于load_data函数
        //后续逻辑中动态分配的空间,而在train被赋值为buffer以后,在下一次load_data逻辑中会
        //再次动态分配,这里一定要记得释放前一次分配的,否则指针将脱钩,内存泄漏不可避免

        free_data(train);
    }//训练结束
    //char buff[256];
    string buff = backup_directory+"/yolov4_final.weights";

    //sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);

    release_mat(&img);
    destroy_all_windows_cv();

    // free memory
    //pthread_join(load_thread, 0);
    //free_data(buffer);

    //free_load_threads(&args);

    for (k = 0; k < ngpus; ++k) free_network(nets[k]);
    free(nets);
    //free_network(net);

    if (calc_map) {//0
        net_map.n = 0;
        free_network(net_map);
    }
}

void test_detector(string datacfg, string cfgfile, string weightfile, string filename, float thresh,
    float hier_thresh, bool dont_show, bool ext_output, bool save_labels, string outfile, bool letter_box, bool benchmark_layers)
{/*
datacfg  = cfg/coco.data
cfgfile = cfg/yolov4.cfg
weightfile = yolov4.weights
filename = data/dog.jpg
thresh = .25
*/ 
    map<string,string> options = read_data_cfg(datacfg);
    string name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char tmp [200];
    realpath(name_list.c_str(),tmp);
    name_list = string(tmp);
    vector<string>names = get_labels_custom(name_list, names_size); //get_labels(name_list);

    //image **alphabet = load_alphabet();//在data文件价下有默认图片  //似乎并没有用到这个
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    if (!weightfile.empty()) {
        load_weights(net, weightfile);
    }

    net.benchmark_layers = benchmark_layers;  //好像没有用到

    fuse_conv_batchnorm(net);

    //calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        cerr<<"Error: in the file "<<name_list<<" of name "<<names_size<<" that isn't equal to classes= "<<net.layers[net.n - 1].classes<< " in the files "<<cfgfile<<endl;
        if (net.layers[net.n - 1].classes > names_size)
        system("pause");
    }
    srand(2222222);
    char buff[256];
    string input = buff;

    ofstream ofs;
    if (!outfile.empty()) {
        ofs.open(outfile);
        if(!ofs.is_open()) {
          cerr<<"fopen failed"<<endl;
        }
        string tmp = "[\n";
        //ofs.write(reinterpret_cast<char*>(&tmp[0]),sizeof(char)*tmp.length());
        ofs<<tmp<<flush;
    }
    float nms = .45;    // 0.4F
    while (1) {
        if (!filename.empty()) {
            input = filename;
        }
        else {
            printf("Enter Image Path: ");
            cout<<"Enter Image Path: \"";
            cin>>input;
            if (input.empty()) break;
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;
        if(letter_box) sized = letterbox_image(im, net.w, net.h); // 0
        else sized = resize_image(im, net.w, net.h);
        layer l = net.layers[net.n - 1];

        float *X = sized.data;

        //time= what_time_is_it_now();
        double time = clock();
        network_predict(net, X);

        //network_predict_image(&net, im); letterbox = 1;
        //printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)clock() - time) / 1000);
        //printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));
        cout<<input<<": Predicted in "<<((double)clock() - time) / 1000<< " milli-seconds."<<endl;

        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, nboxes, letter_box);

        if (nms) {
             diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms); //yolo4是这个
        }

        //im与names有什么关系？没有关系的，只是作者在names中添加了关键字“donot_show”以忽略该类别
        //这里只是在nms的基础上，选择最合适的然后在im上画坐标框
        draw_detections_v3(im, dets, nboxes, thresh, names, nullptr, l.classes, ext_output);
        save_image(im, "predictions");
        if (!dont_show) {
            show_image(im, "predictions");
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        if (filename.empty()) break;
    }

    free_network(net);
}



void run_detector(int argc, vector<string>const &argv)
{
    bool dont_show = find_arg_or(argv, "-dont_show");
    bool benchmark = find_arg_or( argv, "-benchmark");
    bool benchmark_layers = find_arg_or( argv, "-benchmark_layers");
    //if (benchmark_layers) benchmark = 1;
    if (benchmark) dont_show = true;
    bool show = find_arg_or( argv, "-show");
    bool letter_box = find_arg_or( argv, "-letter_box");
    bool calc_map = find_arg_or( argv, "-map");  //计算map(精度)
    int map_points = find_arg_int( argv, "-points", 0);
    check_mistakes = find_arg_or(argv, "-check_mistakes");
    bool show_imgs = find_arg_or( argv, "-show_imgs");
    int mjpeg_port = find_arg_int(argv, "-mjpeg_port", -1);
    int json_port = find_arg_int(argv, "-json_port", -1);
    string http_post_host = find_arg_str(argv, "-http_post_host", "");
    int time_limit_sec = find_arg_int( argv, "-time_limit_sec", 0);
    string out_filename = find_arg_str(argv, "-out_filename", "");
    string outfile = find_arg_str( argv, "-out", "");
    string prefix = find_arg_str( argv, "-prefix", "");
    float thresh = find_arg_float( argv, "-thresh", .25);    // 0.24
    float iou_thresh = find_arg_float( argv, "-iou_thresh", .5);    // 0.5 for mAP
    float hier_thresh = find_arg_float(argv, "-hier", .5);
    int cam_index = find_arg_int( argv, "-c", 0);  //使用相机
    int frame_skip = find_arg_int(argv, "-s", 0);
    int num_of_clusters = find_arg_int(argv, "-num_of_clusters", 5);
    int width = find_arg_int( argv, "-width", -1);
    int height = find_arg_int( argv, "-height", -1);
    // extended output in test mode (output of rect bound coords)
    // and for recall mode (extended output table-like format with results for best_class fit)
    bool ext_output = find_arg_or( argv, "-ext_output");  //打印坐标框信息（printf  ）
    bool save_labels = find_arg_or(argv, "-save_labels");
    string chart_path = find_arg_str( argv, "-chart", "");
    if (argc < 4) {
        cerr<<"usage: "<< argv[0]<<" "<<argv[1]<<"[train/test/valid/demo/map] [data] [cfg] [weights (optional)]"<<endl;
        return;
    }
    string gpu_list = find_arg_str( argv, "-gpus", "");
    //-gpus 0,1,2,3
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    //gpu_index = -1;
    //gpu = gpu_index;
    gpus = &gpu;
    ngpus = 1;
    bool clear = find_arg_or( argv, "-clear");//是否从零开始训练

    string datacfg = argv[3];
    char tmp [200];
    realpath(datacfg.c_str(),tmp);
    datacfg = string(tmp);
    string cfg = argv[4];
    realpath(cfg.c_str(),tmp);
    cfg = string(tmp);
    string weights = (argc > 5) ? argv[5] : 0;

    if (weights.length()> 0)
        if (weights[weights.length() - 1] == 0x0d) weights[weights.length() - 1] = 0;
    realpath(weights.c_str(),tmp);
    weights = string(tmp);
    string filename = (argc > 6) ? argv[7] : "";
    realpath(filename.c_str(),tmp);
    filename = string(tmp);
    if (argv[2]=="test") test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers);
    else if (argv[2]=="train") train_detector(datacfg, cfg, weights, gpus, ngpus, clear, dont_show, calc_map, mjpeg_port, show_imgs, benchmark_layers, chart_path);

}