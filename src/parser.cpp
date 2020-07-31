#include "batchnorm_layer.hpp"
#include "blas.hpp"
#include "convolutional_layer.hpp"
#include "maxpool_layer.hpp"
#include "option_list.hpp"
#include "parser.hpp"
#include "route_layer.hpp"
#include "shortcut_layer.hpp"
#include "utils.hpp"
#include "upsample_layer.hpp"
#include "yolo_layer.hpp"
#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace std;



vector<pair<string,map<string,string>>> read_cfg(string filename);


LAYER_TYPE string_to_layer_type(string  type)
{

    if (type== "[shortcut]") return SHORTCUT;
    if (type== "[scale_channels]")  return SCALE_CHANNELS;
    if (type== "[sam]")  return SAM;
    if (type== "[crop]") return CROP;
    if (type== "[cost]") return COST;
    if (type== "[detection]") return DETECTION;
    if (type== "[region]") return REGION;
    if (type== "[yolo]")  return YOLO;
    if (type== "[Gaussian_yolo]")  return GAUSSIAN_YOLO;
    if (type== "[local]") return LOCAL;
    if (type== "[conv]"
            || type== "[convolutional]") return CONVOLUTIONAL;
    if (type== "[activation]") return ACTIVE;
    if (type== "[net]"
            || type== "[network]") return NETWORK;
    if (type== "[crnn]") return CRNN;
    if (type== "[gru]") return GRU;
    if (type== "[lstm]") return LSTM;
    if (type== "[conv_lstm]")  return CONV_LSTM;
    if (type== "[rnn]") return RNN;
    if (type== "[conn]"
            || type== "[connected]") return CONNECTED;
    if (type== "[max]"
            || type== "[maxpool]") return MAXPOOL;
    if (type== "[local_avg]"
        || type== "[local_avgpool]")  return LOCAL_AVGPOOL;
    if (type== "[reorg3d]") return REORG;
    if (type== "[reorg]")  return REORG_OLD;
    if (type== "[avg]"
            || type== "[avgpool]") return AVGPOOL;
    if (type== "[dropout]") return DROPOUT;
    if (type== "[lrn]"
            || type== "[normalization]") return NORMALIZATION;
    if (type== "[batchnorm]") return BATCHNORM;
    if (type== "[soft]"
            || type== "[softmax]") return SOFTMAX;
    if (type== "[route]") return ROUTE;
    if (type== "[upsample]")  return UPSAMPLE;
    if (type== "[empty]")  return EMPTY;
    return BLANK;
}



typedef struct {  //主要在parser网络上使用，parser一层接着一层，比如第一层为卷积层，第二层也是，在构建这两层时，都会传递size_params，然后更新这些参数
//如经过第一层后，将inputs更新为上一层的输出，然后传递到下一层
    int batch;
    int inputs;  //net网络输入
    int h,w,c; //输入图像大小
    int index;  //当前活跃的层数
    int time_steps;
    int train;
    //network net ;
}size_params;

convolutional_layer parse_convolutional(map<string,string>const & options, size_params const & params, network const & net)
{
    int n = option_find_int(options, "filters",1);
    int groups = option_find_int(options, "groups", 1);
    int size = option_find_int(options, "size",1);
    int stride = -1;
    //int stride = option_find_int(options, "stride",1);
    int stride_x = option_find_int(options, "stride_x", -1); //yolo4.cfg 中没有这个
    int stride_y = option_find_int(options, "stride_y", -1);
    if (stride_x < 1 || stride_y < 1) {
        stride = option_find_int(options, "stride", 1);
        if (stride_x < 1) stride_x = stride;
        if (stride_y < 1) stride_y = stride;
    }
    else {
        stride = option_find_int(options, "stride", 1);
    }
    int dilation = option_find_int(options, "dilation", 1);
    int antialiasing = option_find_int(options, "antialiasing", 0);
    if (size == 1) dilation = 1;
    bool pad = option_find_int(options, "pad",0);
    int padding = option_find_int(options, "padding",0);  ////yolo4.cfg 中没有这个，全部"pad=1"
    if(pad) padding = size/2;

    string activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int assisted_excitation = option_find_int(options, "assisted_excitation", 0);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int(options, "batch_normalize", 0);
    int cbn = option_find_int(options, "cbn", 0);
    if (cbn) batch_normalize = 2;
    int binary = option_find_int(options, "binary", 0);
    int xnor = option_find_int(options, "xnor", 0);
    int use_bin_output = option_find_int(options, "bin_output", 0);
    int sway = option_find_int(options, "sway", 0);
    int rotate = option_find_int(options, "rotate", 0);
    int stretch = option_find_int(options, "stretch", 0);
    int stretch_sway = option_find_int(options, "stretch_sway", 0);
    if ((sway + rotate + stretch + stretch_sway) > 1) {
        printf(" Error: should be used only 1 param: sway=1, rotate=1 or stretch=1 in the [convolutional] layer \n");
        exit(0);
    }
    int deform = sway || rotate || stretch || stretch_sway;
    if (deform && size == 1) {
        printf(" Error: params (sway=1, rotate=1 or stretch=1) should be used only with size >=3 in the [convolutional] layer \n");
        exit(0);
    }

    convolutional_layer layer = make_convolutional_layer(batch,1,h,w,c,n,groups,size,stride_x,stride_y,dilation,padding,activation, batch_normalize, binary, xnor, net.adam, use_bin_output, params.index, antialiasing,
                                                         nullptr, assisted_excitation, deform, params.train);
    layer.flipped = option_find_int(options, "flipped", 0);
    layer.dot = option_find_int(options, "dot", 0);
    layer.sway = sway;
    layer.rotate = rotate;
    layer.stretch = stretch;
    layer.stretch_sway = stretch_sway;
    layer.angle = option_find_int(options, "angle", 15);

    if(net.adam){
        layer.B1 = net.B1;
        layer.B2 = net.B2;
        layer.eps = net.eps;
    }

    return layer;
}

layer parse_yolo(map<string,string> const & options, size_params const & params)
{
    /*
    mask = 6,7,8
    anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
    classes=80
    num=9
    jitter=.3
    ignore_thresh = .7
    truth_thresh = 1
    random=1
    scale_x_y = 1.05
    iou_thresh=0.213
    cls_normalizer=1.0
    iou_normalizer=0.07
    iou_loss=ciou
    nms_kind=greedynms
    beta_nms=0.6
    */
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;
    string a = option_find_str(options, "mask", "");
    int* mask = find_mul_int(a, num, false);
    int max_boxes = option_find_int(options, "max", 90);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
    if (l.outputs != params.inputs) {
        printf("Error: l.outputs == params.inputs \n");
        printf("filters= in the [convolutional]-layer doesn't correspond to classes= or mask= in [yolo]-layer \n");
        exit(EXIT_FAILURE);
    }
    //assert(l.outputs == params.inputs);

    string cpc = option_find_str(options, "counters_per_class", "");
   // l.classes_multipliers = get_classes_multipliers(cpc, classes);
    l.classes_multipliers = NULL;


    l.label_smooth_eps = option_find_float(options, "label_smooth_eps", 0.0f);
    l.scale_x_y = option_find_int(options, "scale_x_y", 1);
    l.max_delta = option_find_float(options, "max_delta", FLT_MAX);   // set 10
    l.iou_normalizer = option_find_float(options, "iou_normalizer", 0.75);
    l.cls_normalizer = option_find_float(options, "cls_normalizer", 1);
    string iou_loss = option_find_str(options, "iou_loss", "mse");   //  "iou");

    if (iou_loss=="mse") l.iou_loss = MSE;
    else if (iou_loss=="giou")  l.iou_loss = GIOU;
    else if (iou_loss== "diou")  l.iou_loss = DIOU;
    else if (iou_loss=="ciou")  l.iou_loss = CIOU;  //YOLO中是这个
    else l.iou_loss = IOU;
    clog<<"[yolo] params: iou loss: "<<iou_loss<<"iou_norm: "<<l.iou_normalizer<<" cls_norm: "<<l.cls_normalizer<<" scale_x_y: "<<l.scale_x_y<<endl;


    string iou_thresh_kind_str = option_find_str(options, "iou_thresh_kind", "iou");
    if (iou_thresh_kind_str== "iou")  l.iou_thresh_kind = IOU;
    else if (iou_thresh_kind_str =="giou")  l.iou_thresh_kind = GIOU;
    else if (iou_thresh_kind_str== "diou") l.iou_thresh_kind = DIOU;
    else if (iou_thresh_kind_str== "ciou")  l.iou_thresh_kind = CIOU;
    else {
        cerr<<" Wrong iou_thresh_kind = "<< iou_thresh_kind_str<<endl;
        l.iou_thresh_kind = IOU;
    }

    l.beta_nms = option_find_float(options, "beta_nms", 0.6);
    string nms_kind = option_find_str(options, "nms_kind", "default");
    if (nms_kind== "default")  l.nms_kind = DEFAULT_NMS;
    else {
        if (nms_kind== "greedynms")  l.nms_kind = GREEDY_NMS;  //yolo是这个！
        else if (nms_kind== "diounms")  l.nms_kind = DIOU_NMS;
        else l.nms_kind = DEFAULT_NMS;
        cerr<<"nms_kind: "<<nms_kind<<" "<<l.nms_kind<<" beta"<<l.beta_nms<<endl;

    }

    l.jitter = option_find_float(options, "jitter", .2);
    l.focal_loss = option_find_int(options, "focal_loss", 0);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.iou_thresh = option_find_float(options, "iou_thresh", 1); // recommended to use iou_thresh=0.213 in [yolo]
    l.random = option_find_float(options, "random", 0);

    string map_file = option_find_str(options, "map", "");
    //if (!map_file.empty()) l.map = read_map(map_file);
    a = option_find_str(options, "anchors", "");
    char * c = const_cast<char *>(a.c_str());
    if (c) {
        int len = a.length();
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (c[i] == ',') ++n;
        }
        for (i = 0; i < n && i < total*2; ++i) {
            float bias = atof(c);
            l.biases[i] = bias;//将anchor存储到l.biases
            c = strchr(c, ',') + 1;
        }
    }
    return l;
}

maxpool_layer parse_maxpool(map<string,string> const &options, size_params const &params)
{
    //max只有stride和size参数。
    int stride = option_find_int(options, "stride",1);
    int stride_x = option_find_int(options, "stride_x", stride);
    int stride_y = option_find_int(options, "stride_y", stride);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int(options, "padding", size-1);
    int maxpool_depth = option_find_int(options, "maxpool_depth", 0);
    int out_channels = option_find_int(options, "out_channels", 1);
    bool antialiasing = option_find_int(options, "antialiasing", false);
    const bool avgpool = false;

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) cerr<<"Layer before [maxpool] layer must output image.";

    maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, avgpool, params.train);
    return layer;
}


layer parse_shortcut(map<string,string>const &options, size_params const &params, network const &net)
{ //  from=-3
  //  activation=linear

    string activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    string weights_type_str = option_find_str(options, "weights_type", "none");
    WEIGHTS_TYPE_T weights_type = NO_WEIGHTS;
    if(weights_type_str == "per_feature"  || weights_type_str==  "per_layer") weights_type = PER_FEATURE;
    else if (weights_type_str == "per_channel") weights_type = PER_CHANNEL;
    else if (weights_type_str != "none") {
       // printf("Error: Incorrect weights_type = %s \n Use one of: none, per_feature, per_channel \n", weights_type_str);
        cerr<<"Error: Incorrect weights_type = "<<weights_type_str<<endl<<" Use one of: none, per_feature, per_channel"<<endl;
        exit(0);
    }
    string weights_normalizion_str = option_find_str(options, "weights_normalizion", "none");
    WEIGHTS_NORMALIZATION_T weights_normalizion = NO_NORMALIZATION;
    if (weights_normalizion_str== "relu" || weights_normalizion_str =="avg_relu") weights_normalizion = RELU_NORMALIZATION;
    else if (weights_normalizion_str == "softmax") weights_normalizion = SOFTMAX_NORMALIZATION;
    else if (weights_type_str!= "none")  {
        cerr<<"Error: Incorrect weights_normalizion = "<<weights_normalizion_str<<" \n Use one of: none, relu, softmax" <<endl;
        exit(0);
    }

    vector<int>shortcut_from ;
    string l = option_find_str(options, "from","");
    if (l.empty()) error("Route Layer must specify input layers: from = ...");

    string::iterator  tmp = l.begin();
    while (tmp!=l.end()){
        if (*tmp==' '){
            l.erase(tmp);
        } else tmp++;
    }
    istringstream ist(l);
    string buf ;

    while( getline(ist,buf,',')){
        shortcut_from.push_back(stoi(buf));
    }
    int n = shortcut_from.size();
    //事实上yolo4的shortcut只有一层，即-3层，所以n=1

    int* layers = (int*)calloc(n, sizeof(int));
    int* sizes = (int*)calloc(n, sizeof(int));
    float **layers_output = (float **)calloc(n, sizeof(float *));
    float **layers_delta = (float **)calloc(n, sizeof(float *));

    int i ;
    for (i = 0; i < n; ++i) {
        int index = shortcut_from[i];
        if (index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net.layers[index].outputs;
        layers_output[i] = net.layers[index].output;
        layers_delta[i] = net.layers[index].delta;
    }



    layer s = make_shortcut_layer(params.batch, n, layers, sizes, params.w, params.h, params.c, layers_output, layers_delta,
         weights_type, weights_normalizion, activation, params.train);



    for (i = 0; i < n; ++i) {
        int index = layers[i];
        assert(params.w == net.layers[index].out_w && params.h == net.layers[index].out_h);

        if (params.w != net.layers[index].out_w || params.h != net.layers[index].out_h || params.c != net.layers[index].out_c)
            fprintf(stderr, " (%4d x%4d x%4d) + (%4d x%4d x%4d) \n",
                params.w, params.h, params.c, net.layers[index].out_w, net.layers[index].out_h, net.layers[index].out_c);
    }

    return s;
}






layer parse_upsample(map<string,string>const &options, size_params params, network const &net)
{
    //upsample中只有一个参数
    int stride = option_find_int(options, "stride", 2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float(options, "scale", 1.0);
    return l;
}

route_layer parse_route(map<string,string>const &options, size_params const &params, network const & net)
{  //只有一个参数，layers；

    string l = option_find_str(options, "layers" ,"");
    if(l.empty()) cerr<<"Route Layer must specify input layers"<<endl;
    vector<int> lay ;
    string::iterator  c = l.begin();
    while (c!=l.end()){
        if (*c==' '){
            l.erase(c);
        } else c++;
    }
    istringstream ist(l);
    string buf ;

    while( getline(ist,buf,',')){
        lay.push_back(stoi(buf));
    }
    int n = lay.size();

    int i;
    int* layers = (int*)xcalloc(n, sizeof(int));
    int* sizes = (int*)xcalloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index =lay[i];
        if(index < 0) index = params.index + index;  //存放layer层的index
        layers[i] = index;
        sizes[i] = net.layers[index].outputs;
    }
    int batch = params.batch;

    int groups = option_find_int(options, "groups", 1);
    int group_id = option_find_int(options, "group_id", 0);

    route_layer layer = make_route_layer(batch, n, layers, sizes, groups, group_id);

    convolutional_layer first = net.layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = layers[i];
        convolutional_layer next = net.layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            layer.out_c += next.out_c;
        }else{
            fprintf(stderr, " The width and height of the input layers are different. \n");
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }
    layer.out_c = layer.out_c / layer.groups;

    layer.w = first.w;
    layer.h = first.h;
    layer.c = layer.out_c;

    if (n > 3) fprintf(stderr, " \t    ");
    else if (n > 1) fprintf(stderr, " \t            ");
    else fprintf(stderr, " \t\t            ");

    fprintf(stderr, "           ");
    if (layer.groups > 1) fprintf(stderr, "%d/%d", layer.group_id, layer.groups);
    else fprintf(stderr, "   ");
    fprintf(stderr, " -> %4d x%4d x%4d \n", layer.out_w, layer.out_h, layer.out_c);

    return layer;
}

learning_rate_policy get_policy(string s)
{
    if (s=="random" )return RANDOM;
    if (s=="poly") return POLY;
    if (s=="constant") return CONSTANT;
    if(s=="step" )return STEP;
    if (s=="exp") return EXP;
    if(s== "sigmoid" )return SIG;
    if (s== "steps") return STEPS;
    if (s== "sgdr") return SGDR;

    cerr<<"Couldn't find policy "<<s<<", going with constant"<<endl;
    return CONSTANT;
}

void parse_net_options(map<string,string>const &options, network &net)
{//net赋值
    net.max_batches = option_find_int(options, "max_batches", 0);
    net.batch = option_find_int(options, "batch",1);
    net.learning_rate = option_find_float(options, "learning_rate", .001);
    net.learning_rate_min = option_find_float(options, "learning_rate_min", .00001);
    net.batches_per_cycle = option_find_int(options, "sgdr_cycle", net.max_batches);
    net.batches_cycle_mult = option_find_int(options, "sgdr_mult", 2);
    net.momentum = option_find_float(options, "momentum", .9);
    net.decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net.time_steps = option_find_int(options, "time_steps",1);
    net.track = option_find_int(options, "track", 0);
    net.augment_speed = option_find_int(options, "augment_speed", 2);
    net.init_sequential_subdivisions = net.sequential_subdivisions = option_find_int(options, "sequential_subdivisions", subdivs);
    if (net.sequential_subdivisions > subdivs) net.init_sequential_subdivisions = net.sequential_subdivisions = subdivs;
    net.try_fix_nan = option_find_int(options, "try_fix_nan", 0);
    net.batch /= subdivs;
    net.batch *= net.time_steps;
    net.subdivisions = subdivs;

    net.seen = 0;
    net.cur_iteration = 0;
    net.loss_scale = option_find_float(options, "loss_scale", 1.0);
    net.dynamic_minibatch = option_find_int(options, "dynamic_minibatch", 0);
    net.optimized_memory = option_find_int(options, "optimized_memory", 0);
    net.workspace_size_limit = (size_t)1024*1024 * option_find_int(options, "workspace_size_limit_MB", 1024);  // 1024 MB by default

    net.adam = option_find_int(options, "adam", 0);
    if(net.adam){
        net.B1 = option_find_float(options, "B1", .9);
        net.B2 = option_find_float(options, "B2", .999);
        net.eps = option_find_float(options, "eps", .000001);
    }

    net.h = option_find_int(options, "height",0);
    net.w = option_find_int(options, "width",0);
    net.c = option_find_int(options, "channels",0);
    net.inputs = option_find_int(options, "inputs", net.h * net.w * net.c);
    net.max_crop = option_find_int(options, "max_crop",net.w*2);
    net.min_crop = option_find_int(options, "min_crop",net.w);
    net.flip = option_find_int(options, "flip", 1);
    net.blur = option_find_int(options, "blur", 0);
    net.gaussian_noise = option_find_int(options, "gaussian_noise", 0);
    net.mixup = option_find_int(options, "mixup", 0);
    int cutmix = option_find_int(options, "cutmix", 0);
    int mosaic = option_find_int(options, "mosaic", 0);
    if (mosaic && cutmix) net.mixup = 4;
    else if (cutmix) net.mixup = 2;
    else if (mosaic) net.mixup = 3;
    net.letter_box = option_find_int(options, "letter_box", 0);
    net.label_smooth_eps = option_find_float(options, "label_smooth_eps", 0.0f);
    net.resize_step = option_find_int(options, "resize_step", 32);
    net.attention = option_find_int(options, "attention", 0);
    net.adversarial_lr = option_find_float(options, "adversarial_lr", 0);

    net.angle = option_find_float(options, "angle", 0.0f);
    net.aspect = option_find_float(options, "aspect", 1.0);
    net.saturation = option_find_float(options, "saturation", 1.0);
    net.exposure = option_find_float(options, "exposure", 1.0);
    net.hue = option_find_float(options, "hue", 0.0);
    net.power = option_find_float(options, "power", 4.0);

    if(!net.inputs && !(net.h && net.w && net.c)) error("No input parameters supplied");

    string policy_s = option_find_str(options, "policy", "constant");
    net.policy = get_policy(policy_s);
    net.burn_in = option_find_int(options, "burn_in", 0);

    if(net.policy == STEP){
        net.step = option_find_int(options, "step", 1);
        net.scale = option_find_float(options, "scale", 1.0f);
    } else if (net.policy == STEPS || net.policy == SGDR){
        string l = option_find_str(options, "steps","");
        string p = option_find_str(options, "scales","");
        string s = option_find_str(options, "seq_scales","");


        if(net.policy == STEPS && (l.empty() || p.empty()))
            cerr<<"STEPS policy must have steps and scales in cfg file"<<endl;
        int steps_nums = 0;

        if (!p.empty()) {
            net.scales = find_mul_float(p,steps_nums,0);
        }if (!l.empty()) {
            net.steps = find_mul_int(l,steps_nums,1);

        }if (!s.empty()) {
            net.seq_scales = find_mul_float(s,steps_nums,1);

        }

    } else if (net.policy == EXP){
        net.gamma = option_find_float(options, "gamma", 1.0);
    } else if (net.policy == SIG){
        net.gamma = option_find_float(options, "gamma", 1.0);
        net.step = option_find_int(options, "step", 1);
    } else if (net.policy == POLY || net.policy == RANDOM){
        //net.power = option_find(options, "power", 1);
    }

}


void set_train_only_bn(network net)
{
    int train_only_bn = 0;
    int i;
    for (i = net.n - 1; i >= 0; --i) {
        if (net.layers[i].train_only_bn) train_only_bn = net.layers[i].train_only_bn;  // set l.train_only_bn for all previous layers
        if (train_only_bn) {
            net.layers[i].train_only_bn = train_only_bn;

            if (net.layers[i].type == CONV_LSTM) {
                net.layers[i].wf->train_only_bn = train_only_bn;
                net.layers[i].wi->train_only_bn = train_only_bn;
                net.layers[i].wg->train_only_bn = train_only_bn;
                net.layers[i].wo->train_only_bn = train_only_bn;
                net.layers[i].uf->train_only_bn = train_only_bn;
                net.layers[i].ui->train_only_bn = train_only_bn;
                net.layers[i].ug->train_only_bn = train_only_bn;
                net.layers[i].uo->train_only_bn = train_only_bn;
                if (net.layers[i].peephole) {
                    net.layers[i].vf->train_only_bn = train_only_bn;
                    net.layers[i].vi->train_only_bn = train_only_bn;
                    net.layers[i].vo->train_only_bn = train_only_bn;
                }
            }
            else if (net.layers[i].type == CRNN) {
                net.layers[i].input_layer->train_only_bn = train_only_bn;
                net.layers[i].self_layer->train_only_bn = train_only_bn;
                net.layers[i].output_layer->train_only_bn = train_only_bn;
            }
        }
    }
}

network parse_network_cfg(string filename)
{
    return parse_network_cfg_custom(filename, 0, 0);
}

network parse_network_cfg_custom(string filename, int batch, int time_steps)
{//读取yolo.cfg
    // batch = 1
    // time_steps =1
    //在测试和训练的时候都是1
   // list *sections = read_cfg(filename);
    vector<pair<string,map<string,string>>> network_cfg = read_cfg(filename);

    if(network_cfg.empty()) cerr<<"Config file has no sections"<<endl;
    network net = make_network(network_cfg.size()-1); //创建空间,创建了seen，cur_iteration, layers 的大小
    net.gpu_index = -1;
    size_params params; //

    if (batch > 0) params.train = 0;    // allocates memory for Detection only
    else params.train = 1;     //为什么这里设置成了  arams.train  = 1 ，       // allocates memory for Detection & Training

    parse_net_options(network_cfg[0].second, net);

    //初始化params为网络输入图片信息
    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    if (batch > 0) net.batch = batch;
    if (time_steps > 0) net.time_steps = time_steps;
    if (net.batch < 1) net.batch = 1;
    if (net.time_steps < 1) net.time_steps = 1;
    if (net.batch < net.time_steps) net.batch = net.time_steps;
    params.batch = net.batch;
    params.time_steps = net.time_steps;
    //params.net = net;
    printf("mini_batch = %d, batch = %d, time_steps = %d, train = %d \n", net.batch, net.batch * net.subdivisions, net.time_steps, params.train);

    int avg_outputs = 0;
    int avg_counter = 0;
    float bflops = 0;
    size_t workspace_size = 0;
    size_t max_inputs = 0;
    size_t max_outputs = 0;

    int count = 0;
    clog<<"layer filters size/strd(dil) input output"<<endl;
    for (int i = 1 ; i < network_cfg.size() ; i ++ ){

        fprintf(stderr, "%4d ", count);
        params.index = count;
        //cout<<"parse "<<count<<"layer network " ;
        layer l = { (LAYER_TYPE)0 };
        LAYER_TYPE lt = string_to_layer_type(network_cfg[i].first);
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(network_cfg[i].second, params, net);
        }else if (lt == YOLO) {
            l = parse_yolo(network_cfg[i].second, params);
            l.keep_delta_gpu = 1;
        }else if(lt == MAXPOOL){
            l = parse_maxpool(network_cfg[i].second, params);
        }else if(lt == ROUTE){
            l = parse_route(network_cfg[i].second, params, net);
            int k;
            for (k = 0; k < l.n; ++k) {
                net.layers[l.input_layers[k]].use_bin_output = 0;
                net.layers[l.input_layers[k]].keep_delta_gpu = 1;
            }
        }else if (lt == UPSAMPLE) {
            l = parse_upsample(network_cfg[i].second, params, net);
        }else if(lt == SHORTCUT){
            l = parse_shortcut(network_cfg[i].second, params, net);
            net.layers[count - 1].use_bin_output = 0;
            net.layers[l.index].use_bin_output = 0;
            net.layers[l.index].keep_delta_gpu = 1;
        }
        else if (lt == EMPTY) {
            layer empty_layer = {(LAYER_TYPE)0};
            empty_layer.out_w = params.w;
            empty_layer.out_h = params.h;
            empty_layer.out_c = params.c;
            l = empty_layer;
            l.output = net.layers[count - 1].output;
            l.delta = net.layers[count - 1].delta;

        }else{
            //fprintf(stderr, "Type not recognized: %s\n", s->type);
            cerr<<"Type not recognized: "<<to_string(lt)<<endl;
        }


        l.clip = option_find_float(network_cfg[i].second, "clip", 0.0);
        l.dynamic_minibatch = net.dynamic_minibatch;
        l.onlyforward = option_find_int(network_cfg[i].second, "onlyforward", 0);
        l.dont_update = option_find_int(network_cfg[i].second, "dont_update", 0);
        l.burnin_update = option_find_int(network_cfg[i].second, "burnin_update", 0);
        l.stopbackward = option_find_int(network_cfg[i].second, "stopbackward", 0);
        l.train_only_bn = option_find_int(network_cfg[i].second, "train_only_bn", 0);
        l.dontload = option_find_int(network_cfg[i].second, "dontload", 0);
        l.dontloadscales = option_find_int(network_cfg[i].second, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float(network_cfg[i].second, "learning_rate", 1.0);
        //option_unused(network_cfg[i].second);
        net.layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if (l.inputs > max_inputs) max_inputs = l.inputs;
        if (l.outputs > max_outputs) max_outputs = l.outputs;
        //free_section(s);
        //n = n->next;
        ++count;

        if (l.antialiasing) {
            params.h = l.input_layer->out_h;
            params.w = l.input_layer->out_w;
            params.c = l.input_layer->out_c;
            params.inputs = l.input_layer->outputs;
        }
        else {
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }

        if (l.bflops > 0) bflops += l.bflops;  //maxpool层会产生bflops

        if (l.w > 1 && l.h > 1) {
            avg_outputs += l.outputs;
            avg_counter++;
        }
    }
    //free_list(sections);



    set_train_only_bn(net); // set l.train_only_bn for all required layers

    net.outputs = get_network_output_size(net);
    net.output = get_network_output(net);
    avg_outputs = avg_outputs / avg_counter;
    fprintf(stderr, "Total BFLOPS %5.3f \n", bflops);
    fprintf(stderr, "avg_outputs = %d \n", avg_outputs);

        if (workspace_size) {
            net.workspace = (float*)xcalloc(1, workspace_size);
        }


    LAYER_TYPE lt = net.layers[net.n - 1].type;
    if ((net.w % 32 != 0 || net.h % 32 != 0) && (lt == YOLO || lt == REGION || lt == DETECTION)) {
        printf("\n Warning: width=%d and height=%d in cfg-file must be divisible by 32 for default networks Yolo v1/v2/v3!!! \n\n",
            net.w, net.h);
    }
    return net;
}




vector<pair<string,map<string,string>>> read_cfg(string filename){
    vector<pair<string,map<string,string>>> network_cfg ;
    ifstream ifs;
    ifs.open(filename, ios::in);
    if (!ifs.is_open()) {
        cout << "文件打开失败！" << endl;
        ifs.close();
        return network_cfg;

    }else
    {
        string buf;
        int num = 0;
        map<string,string> tmp_map;
        while (getline(ifs, buf)) {
            if(buf[0]!='#'){

                string tmp_name ;
                auto c = buf.begin();
                while (c!=buf.end()){
                    if (*c==' ' ){
                        buf.erase(c);
                    } else c++;
                }
                if (buf[0]=='['){
                    num++;
                    tmp_name = buf;
                    //tmp_map.empty();
                    network_cfg.push_back(make_pair(tmp_name,tmp_map));
                } else {
                    //network_cfg[num-1].second["type"] =tmp_name.substr(1, tmp_name.length()-2);

                    network_cfg[num-1].second[buf.substr(0, buf.find('='))] = buf.substr(buf.find('=') + 1, buf.length() - buf.find('='));
                }
            }
        }
    }       ifs.close();
    return    network_cfg ;

}


void save_shortcut_weights(layer l, ofstream &ofs)
{

    int i;
    for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weight_updates[i]);
    printf(" l.nweights = %d - update \n", l.nweights);
    for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    printf(" l.nweights = %d \n\n", l.nweights);

    int num = l.nweights;
    ofs.write((char*)&l.weights,sizeof(float)*l.nweights);

}

void save_convolutional_weights(layer l, ofstream &ofs)
{
    if(l.binary){
        //save_convolutional_weights_binary(l, fp);
        //return;
    }

   // int num = l.nweights;

    ofs.write((char*)&l.biases,sizeof(float)*l.n);
    if (l.batch_normalize){
        ofs.write((char*)&l.scales,sizeof(float)*l.n);
        ofs.write((char*)&l.rolling_mean,sizeof(float)*l.n);
        ofs.write((char*)&l.rolling_variance,sizeof(float)*l.n);

    }
    ofs.write((char*)&l.weights,sizeof(float)*l.nweights);

    //if(l.adam){
    //    fwrite(l.m, sizeof(float), num, fp);
    //    fwrite(l.v, sizeof(float), num, fp);
    //}
}




void save_weights_upto(network net, string filename, int cutoff)
{

    //fprintf(stderr, "Saving weights to %s\n", filename);
    clog<<"Saving weights to"<<filename<<endl;
    ofstream ofs;
    ofs.open(filename, ios::binary);
    if (ofs.is_open()) {
        cout << "文件打开失败！" << endl;
        ofs.close();
        exit(0);
    }else{
//这里删去了版本信息

        net.seen = get_current_iteration(net) * net.batch * net.subdivisions; // remove this line, when you will save to weights-file both: seen & cur_iteration
        ofs.write((char*)&net.seen,sizeof(uint64_t));

        int i;
        for(i = 0; i < net.n && i < cutoff; ++i){
            layer l = net.layers[i];
            if (l.type == CONVOLUTIONAL && l.share_layer == NULL) {
                save_convolutional_weights(l, ofs);
            } if (l.type == SHORTCUT && l.nweights > 0) {
                save_shortcut_weights(l, ofs);
            }
        }
        ofs.close();
    }



}
void save_weights(network net, string filename)
{
    save_weights_upto(net, filename, net.n);
}

void transpose_matrix(float *a, int rows, int cols)
{
    float* transpose = (float*)xcalloc(rows * cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}


void load_convolutional_weights(layer &l, ifstream &ifs)
{

    if(l.binary){
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    int num = l.nweights;
    ifs.read((char*)l.biases,  l.n*sizeof(float));
    //if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.biases - l.index = %d \n", l.index);
    //fread(l.weights, sizeof(float), num, fp); // as in connected layer
    if (l.batch_normalize && (!l.dontloadscales)){
        ifs.read((char*)l.scales,  l.n*sizeof(float));

       // read_bytes = fread(l.scales[0], sizeof(float), l.n, fp);
      //  if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.scales - l.index = %d \n", l.index);

        ifs.read((char*)l.rolling_mean,  l.n*sizeof(float));
        ifs.read((char*)l.rolling_variance,  l.n*sizeof(float));

        //if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.rolling_mean - l.index = %d \n", l.index);
        //read_bytes = fread(l.rolling_variance, sizeof(float), l.n, fp);
        //if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.rolling_variance - l.index = %d \n", l.index);
        if(0){
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
        if(0){
            fill_cpu(l.n, 0, l.rolling_mean, 1);
            fill_cpu(l.n, 0, l.rolling_variance, 1);
        }
    }
    //rolling_variance
    ifs.read((char*)l.weights,  num*sizeof(float));

    //read_bytes = fread(l.weights, sizeof(float), num, fp);
    //if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", l.index);
    //if(l.adam){
    //    fread(l.m, sizeof(float), num, fp);
    //    fread(l.v, sizeof(float), num, fp);
    //}
    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l.flipped) {//转置
        transpose_matrix(l.weights, (l.c/l.groups)*l.size*l.size, l.n);
    }
    //if (l.binary) binarize_weights(l.weights, l.n, (l.c/l.groups)*l.size*l.size, l.weights);

}

void load_shortcut_weights(layer &l, ifstream &ifs)
{
    int num = l.nweights;
    //int read_bytes;
    //read_bytes = fread(l.weights, sizeof(float), num, fp);
    ifs.read((char *)(l.weights),sizeof(float)*num);
    //if (read_bytes > 0 && read_bytes < num) printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", l.index);
    //for (int i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    //printf(" read_bytes = %d \n\n", read_bytes);

}
void load_weights(network &net, string filename)
{
    load_weights_upto(net, filename, net.n);
}

void load_weights_upto(network &net, string filename, int cutoff)
{
    clog<<"Loading weights from "<< filename<<endl;
    ifstream ifs;
    ifs.open(filename, ios::binary);
    if (!ifs.is_open()) {
        cout << "文件打开失败！" << endl;
        ifs.close();
        exit(0);
    }else{

        int major;
        int minor;
        int revision;
        ifs.read((char*)&major, sizeof(int));
        ifs.read((char*)&minor, sizeof(int));
        ifs.read((char*)&revision, sizeof(int));
        if ((major * 10 + minor) >= 2) {
            printf("\n u_int64_t");
            u_int64_t iseen = 0;
            //fread(&iseen, sizeof(uint64_t), 1, fp);
            ifs.read((char*)&iseen, sizeof(u_int64_t));

            net.seen = iseen;
        }
        else {
            printf("\n seen 32");
            u_int32_t iseen = 0;
            ifs.read((char*)&iseen, sizeof(u_int32_t));
            net.seen = iseen;
        }
        net.cur_iteration = get_current_batch(net);
        cerr<<"trained: "<<static_cast<float >(net.seen / 1000) <<" K-images ("<<static_cast<float >(net.seen / 64000)<< " Kilo-batches_64)"<<endl;
        int i;
        for( i = 0; i < net.n && i < cutoff; ++i){
            layer l = net.layers[i];  //TODO net.layers[i] 索引  目前加载一次会改变net
            if (l.dontload) continue;
            if(l.type == CONVOLUTIONAL && l.share_layer == NULL){
                load_convolutional_weights(l, ifs);
            }
            if (l.type == SHORTCUT && l.nweights > 0) {
                load_shortcut_weights(l, ifs);
            }
            if (ifs.eof()) break;
        }
        clog<<"Done! Loaded %d layers from weights-file "<<i<< endl;
        ifs.close();
    }
}
