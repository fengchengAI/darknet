#ifndef DARKNET_API
#define DARKNET_API


#if defined(DEBUG) && !defined(_CRTDBG_MAP_ALLOC)
#define _CRTDBG_MAP_ALLOC
#endif

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cassert>
//#include <pthread.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cfloat>
#include <climits>
#include <array>
using namespace std;
#ifndef LIB_API
#ifdef LIB_EXPORTS
#if defined(_MSC_VER)
#define LIB_API __declspec(dllexport)
#else
#define LIB_API __attribute__((visibility("default")))
#endif
#else

#define LIB_API
#endif
#endif

#define SECRET_NUM -1234

 enum UNUSED_ENUM_TYPE{ UNUSED_DEF_VAL } ;



extern int gpu_index;

// option_list.h
struct metadata {
    int classes;
    char **names;
} ;



// activations.h
 enum ACTIVATION {
     LOGISTIC,
     RELU,
     RELU6,
     RELIE,
     LINEAR,
     RAMP,
     TANH,
     PLSE,
     LEAKY,
     ELU,
     LOGGY,
     STAIR,
     HARDTAN,
     LHTAN,
     SELU,
     GELU,
     SWISH,
     MISH,
     NORM_CHAN,
     NORM_CHAN_SOFTMAX,
     NORM_CHAN_SOFTMAX_MAXVAL
 };
// parser.h
 enum IOU_LOSS{
    IOU, GIOU, MSE, DIOU, CIOU
} ;

// parser.h
 enum NMS_KIND{
    DEFAULT_NMS, GREEDY_NMS, DIOU_NMS, CORNERS_NMS
} ;

// parser.h
 enum YOLO_POINT{
    YOLO_CENTER = 1 << 0, YOLO_LEFT_TOP = 1 << 1, YOLO_RIGHT_BOTTOM = 1 << 2
} ;

// parser.h
 enum WEIGHTS_TYPE_T{
    NO_WEIGHTS, PER_FEATURE, PER_CHANNEL
} ;

// parser.h
 enum WEIGHTS_NORMALIZATION_T{
    NO_NORMALIZATION, RELU_NORMALIZATION, SOFTMAX_NORMALIZATION
} ;

// image.h
 enum IMTYPE{
    PNG, BMP, TGA, JPG
} ;

// activations.h
 enum BINARY_ACTIVATION {
    MULT, ADD, SUB, DIV
} ;

// layer.h
 enum LAYER_TYPE{
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    LOCAL_AVGPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    SCALE_CHANNELS,
    SAM,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CONV_LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    GAUSSIAN_YOLO,
    ISEG,
    REORG,
    REORG_OLD,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    EMPTY,
    BLANK
} ;

// layer.h
 enum COST_TYPE{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} ;

// layer.h
 struct update_args {
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} ;

// layer.h
struct layer {
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void(*forward)   (struct layer, struct network_state);
    void(*backward)  (struct layer, struct network_state);
    void(*update)    (struct layer, int, float, float, float);
    void(*forward_gpu)   (struct layer, struct network_state);
    void(*backward_gpu)  (struct layer, struct network_state);
    void(*update_gpu)    (struct layer, int, float, float, float, float);
    layer *share_layer;
    bool train;
    bool avgpool; //amxpool层指定为0
    bool batch_normalize;
    int shortcut;
    int batch;
    int dynamic_minibatch;
    int forced;
    bool flipped;
    int inputs;  //h * w * c;
    int outputs; //out_h * out_w * out_c;
    int nweights;  //卷积层为权重信息的个数   (c / groups) * n * size * size;
    int nbiases;
    int extra;

    int truths;  // < 根据region_layer.c判断，这个变量表示一张图片含有的真实值的个数，对于检测模型来说，一个真实的标签含有5个值，
    // 包括类型对应的编号以及定位矩形框用到的w,h,x,y四个参数，且在darknet中固定每张图片最大处理30个矩形框，（可查看max_boxes参数），
    // 因此，在region_layer.c的make_region_layer()函数中赋值为30*5.

    int h, w, c;  // 该层输入的图片的宽，高，通道数（一般在各网络层构建函数中赋值，比如make_connected_layer()）
    int out_h, out_w, out_c;// 该层输出图片的高、宽、通道数（一般在各网络层构建函数中赋值，比如make_connected_layer()）
    int n; // 对于卷积层，该参数表示卷积核个数，等于out_c，其值由网络配置文件指定；对于region_layer层，该参数等于配置文件中的num值
    // (该参数通过make_region_layer()函数赋值，在parser.c中调用的make_region_layer()函数)，
    // 可以在darknet/cfg文件夹下执行命令：grep num *.cfg便可以搜索出所有设置了num参数的网络，这里面包括yolo.cfg等，其值有
    // 设定为3,5,2的，该参数就是Yolo论文中的B，也就是一个cell中预测多少个box。
    int max_boxes; // 每张图片最多含有的标签矩形框数（参看：data.c中的load_data_detection()，其输入参数boxes就是指这个参数），
    // 什么意思呢？就是每张图片中最多打了max_boxes个标签物体，模型预测过程中，可能会预测出很多的物体，但实际上，
    // 图片中打上标签的真正存在的物体最多就max_boxes个，预测多出来的肯定存在false positive，需要滤出与筛选，
    // 可参看region_layer.c中forward_region_layer()函数的第二个for循环中的注释
    int groups;
    int group_id;
    int size;
    int side;
    int stride;
    int stride_x;
    int stride_y;
    int dilation; //卷积层默认为1
    int antialiasing; //抗锯齿标志，如果为真强行设置所有的步长为1
    int maxpool_depth;
    int out_channels;
    int reverse;  //在upsample_layer中有
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;// shortcut层表示上层的index
    int scale_wh;
    int binary;
    int xnor;
    int peephole;
    int use_bin_output;
    int keep_delta_gpu;
    int optimized_memory;
    int steps;
    int state_constrain;
    int hidden;
    int truth;
    float smooth;
    float dot;
    int deform;
    int sway;
    int rotate;
    int stretch;
    int stretch_sway;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int focal_loss;
    float *classes_multipliers;  //yolo层有这个,每个类别的权重不一样
    float label_smooth_eps;
    int noloss;
    int softmax;
    int classes;
    // 物体类别种数，一个训练好的网络，只能检测指定所有物体类别中的物体，比如yolo9000.cfg，设置该值为9418，
    // 也就是该网络训练好了之后可以检测9418种物体。该参数由网络配置文件指定。目前在作者给的例子中,
    // 有设置该值的配置文件大都是检测模型，纯识别的网络模型没有设置该值，我想是因为检测模型输出的一般会为各个类别的概率，
    // 所以需要知道这个种类数目，而识别的话，不需要知道某个物体属于这些所有类的具体概率，因此可以不知道。


    int coords;  //yolo层使用，默认4
    // 这个参数一般用在检测模型中，且不是所有层都有这个参数，一般在检测模型最后一层有，比如region_layer层，该参数的含义
    // 是定位一个物体所需的参数个数，一般为4个，包括物体所在矩形框中心坐标x,y两个参数以及矩形框长宽w,h两个参数，
    // 可以在darknet/cfg文件夹下，执行grep coords *.cfg，会搜索出所有使用该参数的模型，并可看到该值都设置为4
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;  //yolo层是yolo.cfg中sum的值

    float bflops; // 在maxpool中有此记录
    //卷积层中 2.0 * l.nweights * l.out_h*l.out_w) / 1000000000

    bool adam;
    float B1;
    float B2;
    float eps;

    int t;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;


    float random;  // 1 训练阶段，在yolo层指定
    float ignore_thresh;  //yolo层参数，指定0.7 
    float truth_thresh;//yolo层参数，默认1 
    float iou_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;
    int assisted_excitation;

    int onlyforward; // 标志参数，当值为1那么当前层只执行前向传播
    int stopbackward;  // 标志参数，用来强制停止反向传播过程（值为1则停止反向传播），参看network.c中的backward_network()函数

    int train_only_bn;
    int dont_update;
    int burnin_update;
    bool dontload;  //在加载数据模型是，是否不加载
    int dontsave;
    bool dontloadscales;
    int numload;
    float temperature; // 温度参数，softmax层特有参数，在parse_softmax()函数中赋值，由网络配置文件指定，如果未指定，则使用默认值1（见parse_softmax()）
    float probability; // dropout概率，即舍弃概率，相应的1-probability为保留概率（具体的使用可参见forward_dropout_layer()），在make_dropout_layer()中赋值，
    // 其值由网络配置文件指定，如果网络配置文件未指定，则取默认值0.5（见parse_dropout()）

    float dropblock_size_rel;
    int dropblock_size_abs;
    int dropblock;
    float scale;  //upsample有这个

    int receptive_w;
    int receptive_h;
    int receptive_w_scale;
    int receptive_h_scale;

    char  * cweights;
    int   * indexes;
    // 维度为l.out_h * l.out_w * l.out_c * l.batch，可知包含整个batch输入图片的输出，一般在构建具体网络层时动态分配内存（比如make_maxpool_layer()中）。
    // 目前仅发现其用在在最大池化层中。该变量存储的是索引值，并与当前层所有输出元素一一对应，表示当前层每个输出元素的值是上一层输出中的哪一个元素值（存储的索引值是
    // 在上一层所有输出元素（包含整个batch）中的索引），因为对于最大池化层，每一个输出元素的值实际是上一层输出（也即当前层输入）某个池化区域中的最大元素值，indexes就是记录
    // 这些局部最大元素值在上一层所有输出元素中的总索引。记录这些值有什么用吗？当然有，用于反向传播过程计算上一层敏感度值，详见backward_maxpool_layer()以及forward_maxpool_layer()函数。

    int   * input_layers;  //route层参数，route输入层的index  //还有shortcut中
    int   * input_sizes;  ///route层参数，route输入层的输入维度，即index所在层的输出维度
    float **layers_output;//shortcut参数  存储要连接层的输出
    float **layers_delta;//shortcut参数
    WEIGHTS_TYPE_T weights_type;  //shortcut参数
    WEIGHTS_NORMALIZATION_T weights_normalizion;  //shortcut参数
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;  //实际上就是一个值
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float *concat;
    float *concat_delta;

    float *binary_weights;
    float *biases;  // 当前层所有偏置，对于卷积层，维度l.n*sizeof(float)，每个卷积核有一个偏置,积层的biases是经过bn层后才加的；对于全连接层，维度等于单张输入图片对应的元素个数即outputs，一般在各网络构建函数中动态分配内存（比如make_connected_layer()
    float *bias_updates;// 当前层所有偏置更新值，对于卷积层，维度l.n，每个卷积核有一个偏置；对于全连接层，维度为outputs。所谓权重系数更新值，就是梯度下降中与步长相乘的那项，也即误差对偏置的导数，
    // 一般在各网络构建函数中动态分配内存（比如make_connected_layer())

    float *scales;  //bn层的参数，n * sizeof(float)  //n 为通道数，即为卷积核的数量，回答会对经过bn层后的每个元素按通道乘以这个
    float *scale_updates;

    float *weights;  //卷积层为权重信息, //当前层所有权重系数（连接当前层和上一层的系数，但记在当前层上），对于卷积层，维度为l.n*l.c*l.size*l.size，即卷积核个数乘以卷积核尺寸再乘以输入通道数（各个通道上的权重系数独立不一样）；
    float *weight_updates;// 当前层所有权重系数更新值，对于卷积层维度为l.n*l.c*l.size*l.size；对于全连接层，维度为单张图片输入与输出元素个数之积inputs*outputs，
    // 所谓权重系数更新值，就是梯度下降中与步长相乘的那项，也即误差对权重的导数，一般在各网络构建函数中动态分配内存（比如make_connected_layer()


    float scale_x_y; //这个在yolo层对centet_x 和centet_y 的加权上
    float max_delta;
    float uc_normalizer;
    float iou_normalizer;  //0.07
    float cls_normalizer;  //1.0
    IOU_LOSS iou_loss;
    IOU_LOSS iou_thresh_kind;
    NMS_KIND nms_kind;  //GREEDY_NMS 
    float beta_nms;
    YOLO_POINT yolo_point;

    char *align_bit_weights_gpu;
    float *mean_arr_gpu;
    float *align_workspace_gpu;
    float *transposed_align_workspace_gpu;
    int align_workspace_size;

    char *align_bit_weights;
    float *mean_arr;  //xnor需要用到的参数
    int align_bit_weights_size;
    int lda_align; //xnor需要用到
    int new_lda;
    int bit_align;

    float *col_image;
    float * delta;
    // 存储每一层的敏感度图：包含所有输出元素的敏感度值（整个batch所有图片）。所谓敏感度，即误差函数关于当前层每个加权输入的导数值，
    // 关于敏感度图这个名称，其实就是梯度，可以参考https://www.zybuluo.com/hanbingtao/note/485480。
    // 元素个数为l.batch * l.outputs（其中l.outputs = l.out_h * l.out_w * l.out_c），
    // 对于卷积神经网络，在make_convolutional_layer()动态分配内存，按行存储，可视为l.batch行，l.outputs列，
    // 即batch中每一张图片，对应l.delta中的一行，而这一行，又可以视作有l.out_c行，l.out_h*l.out_c列，
    // 其中每小行对应一张输入图片的一张输出特征图的敏感度。一般在构建具体网络层时动态分配内存（比如make_maxpool_layer()中）。

    float * output;
    // 存储该层所有的输出，维度为l.out_h * l.out_w * l.out_c * l.batch，可知包含整个batch输入图片的输出，一般在构建具体网络层时动态分配内存（比如make_maxpool_layer()中）。
    // 按行存储：每张图片按行铺排成一大行，图片间再并成一行。

    float * activation_input;  //存储进行激活函数前的数据
    int delta_pinned;
    int output_pinned;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;  //bn层
    float * variance;  //bn层

    float * mean_delta;  //bn层
    float * variance_delta;  //bn层

    float * rolling_mean;  //bn层
    float * rolling_variance;  //bn层

    float * x;  //在卷积bn层中，会将进入bn层的数据copy在这里，（bn输入）
    float * x_norm;  //在卷积bn层中，会将进入bn层后（即归一化）的数据copy在这里，（bn输出）

    float * m;
    float * v;

    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;

    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float *stored_h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *stored_c_cpu;
    float *dc_cpu;

    float *binary_input;
    uint32_t *bin_re_packed_input;  //xnor需要用到
    char *t_bit_input; //xnor需要用到

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *vo;
    struct layer *uf;
    struct layer *wf;
    struct layer *vf;
    struct layer *ui;
    struct layer *wi;
    struct layer *vi;
    struct layer *ug;
    struct layer *wg;

    //tree *softmax_tree;

    size_t workspace_size; // l.out_h*l.out_w*l.size*l.size*(l.c / l.groups)*sizeof(float);


    //#ifdef GPU
    int *indexes_gpu;

    float *z_gpu;
    float *r_gpu;
    float *h_gpu;
    float *stored_h_gpu;

    float *temp_gpu;
    float *temp2_gpu;
    float *temp3_gpu;

    float *dh_gpu;
    float *hh_gpu;
    float *prev_cell_gpu;
    float *prev_state_gpu;
    float *last_prev_state_gpu;
    float *last_prev_cell_gpu;
    float *cell_gpu;
    float *f_gpu;
    float *i_gpu;
    float *g_gpu;
    float *o_gpu;
    float *c_gpu;
    float *stored_c_gpu;
    float *dc_gpu;

    // adam
    float *m_gpu;
    float *v_gpu;
    float *bias_m_gpu;
    float *scale_m_gpu;
    float *bias_v_gpu;
    float *scale_v_gpu;

    float * combine_gpu;
    float * combine_delta_gpu;

    float * forgot_state_gpu;
    float * forgot_delta_gpu;
    float * state_gpu;
    float * state_delta_gpu;
    float * gate_gpu;
    float * gate_delta_gpu;
    float * save_gpu;
    float * save_delta_gpu;
    float * concat_gpu;
    float * concat_delta_gpu;

    float *binary_input_gpu;
    float *binary_weights_gpu;
    float *bin_conv_shortcut_in_gpu;
    float *bin_conv_shortcut_out_gpu;

    float * mean_gpu;
    float * variance_gpu;
    float * m_cbn_avg_gpu;
    float * v_cbn_avg_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * col_image_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;
    float * weight_deform_gpu;
    float * weight_change_gpu;

    float * weights_gpu16;
    float * weight_updates_gpu16;

    float * biases_gpu;
    float * bias_updates_gpu;
    float * bias_change_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;
    float * scale_change_gpu;

    float * input_antialiasing_gpu;
    float * output_gpu;
    float * activation_input_gpu;
    float * loss_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * drop_blocks_scale;
    float * drop_blocks_scale_gpu;
    float * squared_gpu;
    float * norms_gpu;

    float *gt_gpu;
    float *a_avg_gpu;

    int *input_sizes_gpu;
    float **layers_output_gpu;
    float **layers_delta_gpu;

//#endif  // GPU
};


// network.h
 enum learning_rate_policy{
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM, SGDR
} ;

// network.h
 struct network {
    int n;   // 网络总层数（make_network()时赋值）
    int batch; //mini_batch
    uint64_t seen;// 目前已经读入的图片张数（网络已经处理的图片张数）在训练阶段每次加一个小batch
    int cur_iteration;  // 当前的iteration，每subdivisions个小batch为一个iteration
    float loss_scale; //1.0 默认
    int *t; // ？？
    float epoch;  // 到目前为止训练了整个数据集的次数
    int subdivisions; //方便训练，在训练时每次计算一个mini_batch,然后进行subdivisions次记为一个it，再进行反向传播。
    float *output;
    learning_rate_policy policy;  //steps//学习率衰减方法 不是默认
    bool benchmark_layers;  //TO DO
    vector<layer> layers;
    float learning_rate;  //0.01 不是默认
    float learning_rate_min;  //.00001默认
    float learning_rate_max;
    int batches_per_cycle;   //500500 默认
    int batches_cycle_mult;  //2
    float momentum;  //0.949  不是默认
    float decay;  //0.0005  不是默认
    float gamma;
    float scale;  //STEP
    float power; //4.0 默认
    int time_steps;  //1 默认
    int step;  //STEP
    int max_batches;  //500500 不是默认
    int num_boxes;  //在yolo层表示一个图片中最大box数量
    int train_images_num;  //训练图片的总数
    float* seq_scales;  //policy为STEPS和SGDR时对应的参数
    float* scales;  //policy为STEPS和SGDR时对应的参数
    int* steps;  // policy为STEPS和SGDR时对应的参数
    int num_steps;  //policy为STEPS和SGDR时对应的参数
    int burn_in;  //1000不是默认
    int cudnn_half;

    bool adam;  //0 默认
    float B1;
    float B2;
    float eps;

     int inputs;         // 一张输入图片的元素个数，如果网络配置文件中未指定，则默认等于net->h * net->w * net->c，在parse_net_options()中赋值
     int outputs;
     // 一张输入图片对应的输出元素个数，对于一些网络，可由输入图片的尺寸及相关参数计算出，比如卷积层，可以通过输入尺寸以及跨度、核大小计算出；
     // 对于另一些尺寸，则需要通过网络配置文件指定，如未指定，取默认值1，比如全连接层

    int truths;
    int notruth;
    int h, w, c; // 输入图像大小  688，608，3
    int max_crop;  //w*2  默认
    int min_crop;  //w
    float max_ratio;
    float min_ratio;
    int center;
    int flip; // horizontal flip 50% probability augmentaiont for classifier training (default = 1)
    int gaussian_noise;//0 默认
    int blur;  //0 默认
    int mixup ; //3 非默认，因为检测到了mosaic=1  数据增强方式，
    float label_smooth_eps;  //0.00f 默认
    int resize_step;  //32默认
    int attention;  //0 默认
    int adversarial;     //0 默认
    float adversarial_lr;  //0 默认
    bool letter_box;  //0 默认
    float angle;  //0 ，非默认
    float aspect;  //1.0 默认
    float exposure;  //1.5非默认
    float saturation ;  // 1.5非默认
    float hue;  //  0.1
    int random;
    bool track; //目标跟踪参数  0   默认
    int augment_speed;  //2   默认
    int sequential_subdivisions;  //1 默认
    int init_sequential_subdivisions;  //1 默认
    int current_subdivision;  //{0,1,2,...,subdivisions}
    int try_fix_nan;  //0默认

    int gpu_index;
    //tree *hierarchy;

    float *input;
    //中间变量，用来暂存某层网络的输入（包含一个 batch 的输入，比如某层网络完成前向，
     //将其输出赋给该变量，作为下一层的输入，可以参看 network.c 中的forward_network()

    float *truth;  	// 中间变量，与上面的 input 对应，用来暂存 input 数据对应的标签数据（真实数据）

     // 中间变量，用来暂存某层网络的敏感度图（反向传播处理当前层时，用来存储上一层的敏
     //感度图，因为当前层会计算部分上一层的敏感度图，可以参看 network.c 中的 backward_network() 函数
    float *delta;

     // 网络的工作空间, 指的是所有层中占用运算空间最大的那个层的 workspace_size,
     // 因为实际上在 GPU 或 CPU 中某个时刻只有一个层在做前向或反向运算
    float *workspace;

    bool train;  //
    int index;  	// 标志参数，当前网络的活跃层
    float *cost;  	//每一层的损失，只有[yolo]层有值
    float clip;

//#ifdef GPU
    //float *input_gpu;
    //float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;

    float *input_state_gpu;
    float *input_pinned_cpu;
    int input_pinned_cpu_flag;

    float **input_gpu;
    float **truth_gpu;
    float **input16_gpu;
    float **output16_gpu;
    size_t *max_input16_size;
    size_t *max_output16_size;
    int wait_stream;

    float *global_delta_gpu;
    float *state_delta_gpu;
    size_t max_delta_gpu_size;
//#endif  // GPU
    int optimized_memory;  //0 默认
    bool dynamic_minibatch;  // 0 默认
    size_t workspace_size_limit;  //1024*1024 * 1024
} ;
//*input的存储顺序为行，列，通道，batch
//同理output
// network.h


 struct network_state { //只要是在网络传播的时候用到，
    float *truth;  //在yolo层有用，训练阶段有用   sizeof (5*max_boxs)  为什么是5，新为训练阶段只需要知道box的坐标框和类别id
    float *input;  //当前传播层的input，这个input在每次传播一层就会更新，将上一层输出作为输入
    float *delta;
    float *workspace;  //相当于缓存区，比如在卷积的时候将输入图像进行一定的变形然后存储在这里
    bool train;
    int index;  //当前阶段，如在forward中，第一层为1，...
    network net;
} ;

// image.h
 struct image {
    int w;
    int h;
    int c;
    float *data;
} ;



// box.h
 struct box {
    float x, y, w, h;
} ;

// box.h
 struct boxabs {
    float left, right, top, bot;
} ;

// box.h
 struct dxrep {
    float dt, db, dl, dr;
} ;

// box.h
 struct ious {
    float iou, giou, diou, ciou;
    dxrep dx_iou;
    dxrep dx_giou;
} ;


// box.h
 struct detection{//这里在nms的时候用到了
    box bbox;
    int classes;  //类别数
    float *prob; //classes 的概率
    float *mask;  //没有用到，因为传到detection的时候，已经是预测且经过阈值判断的结果了
    float objectness;  //可见概率，（大于阈值）
    int sort_class;  //用来记录类别，
    float *uc; // Gaussian_YOLOv3 - tx,ty,tw,th uncertainty
    int points; // bit-0 - center, bit-1 - top-left-corner, bit-2 - bottom-right-corner
} ;
/*
// network.c -batch inference
 struct det_num_pair {
    int num;
    detection *dets;
} det_num_pair, *pdet_num_pair;
*/
// matrix.h
 struct matrix {
    int rows, cols;
    float **vals;  //注意这是“**float”
    /*
    所以就是指向指针的指针
    如对data.y的创建 ：在make_matrix(int rows, int cols)
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.vals = (float**)xcalloc(m.rows, sizeof(float*));
    for(i = 0; i < m.rows; ++i){
         m.vals[i] = (float*)xcalloc(m.cols, sizeof(float));}
    */
     };

// data.h
 struct data {  //在训练阶段传输的是这个，即读取img,box到x,y中，这是刚开始的训练构造阶段，接下来会用networkstate来进行forward
     // shallow是指深层释放X,y中vals的内存还是浅层释放X,y中vals（注意是X,y的vals元素，不是X,y本身，X,y本身是万不能用free释放的，因为其连指针都不是）的内存
     // X,y的vals是一个二维数组，如果浅层释放，即直接释放free(vals)，这种释放并不会真正释放二维数组的内容，因为二维数组实际是指针的指针，
     // 这种直接释放只是释放了用于存放第二层指针变量的内存，也就导致二维数组的每一行将不再能通过二维数组名来访问了（因此这种释放，
     // 需要先将数据转移，使得有其他指针能够指向这些数据块，不能就会造成内存溢出了）；
     // 深层释放，是循环逐行释放为每一行动态分配的内存，然后再释放vals，释放完后，整个二维数组包括第一层存储的第二维指针变量以及实际数据将都不再存在，所有数据都被清空。
     // 详细可查看free_data()以及free_matrix()函数。
    int w, h;
    matrix X;  //X.rows = n; n为一次加载量  //X.vals = (float**)xcalloc(d.X.rows, sizeof(float*))  //X.cols = h*w*c;
    matrix y;  //rows = n ，cols = 5*boxnums  // vals = (float**)xcalloc(m.rows, sizeof(float*));
    int shallow;  //data.c中的concat_data()函数中可以看出shallow设为1时只能进行数据的浅拷贝和浅释放
    int *num_boxes;
    box **boxes;
};

// data.h
enum data_type{
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
} ;

// data.h
 struct load_args {//在训练阶段用到，加载数据
    int threads;  //线程  64
    vector<string>paths;  //训练图像路经
    char *path;
    int n; //net.batch * net.subdivisions * ngpus   //一次加载的数量
    int m; //训练图片数量（总的）
    char **labels;
    int h,w,c; //== net.h,w,c

    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;  //90 默认  一张图片最多的box数量
    int min, max, size;
    int classes;  //80
    int background;
    int scale;
    int center;
    int coords;
    int mini_batch; //目标跟踪参数
    bool track;  //目标跟踪参数
    int augment_speed;  //目标跟踪参数
    int letter_box;  //0
    int show_imgs;  //0 默认
    int dontuse_opencv;
    float jitter;  //0.3
    bool flip;  //1
    int gaussian_noise;  //0
    bool blur;  //0
    int mixup;  //0
    float label_smooth_eps;
    float angle;
    float aspect;
    float saturation; //1.5
    float exposure;  //1.5
    float hue;  //0.1
    data *d;
    image *im;
    image *resized;
    data_type type;  //训练阶段加载的数据类型，这里为DETECTION_DATA
    //tree *hierarchy;
} ;

// data.h
 struct box_label {
    int id;
    float x, y, w, h;
    float left, right, top, bottom;
    // 坐标中有x,y,w,h，而left, right, top, bottom是根据x,y,w,h计算的
} ;
// -----------------------------------------------------


LIB_API void free_network(network net);



LIB_API void diounms_sort(detection *dets, int total, int classes, float thresh, NMS_KIND nms_kind, float beta1);

// network.h
LIB_API float *network_predict(network net, float *input);
LIB_API detection *get_network_boxes(network const &net, int w, int h, float thresh, float hier, int *map, int relative, int &num, int letter);
LIB_API void free_detections(detection *dets, int n);
LIB_API void fuse_conv_batchnorm(network& net);
LIB_API void test_detector(string , string , string ,string filename, float thresh,
    float hier_thresh, bool dont_show, bool ext_output, bool save_labels, string outfile, bool letter_box, bool benchmark_layers);
LIB_API image resize_image(image im, int w, int h);

LIB_API image letterbox_image(image im, int w, int h);
LIB_API void rgbgr_image(image im);
LIB_API image make_image(int w, int h, int c);
LIB_API void free_image(image m);
LIB_API image crop_image(image im, int dx, int dy, int w, int h);

// layer.h
LIB_API void free_layer_custom(layer l, int keep_cudnn_desc);
LIB_API void free_layer(layer l);

// data.c
LIB_API void free_data(data d);
LIB_API pthread_t load_data(load_args args);
LIB_API void free_load_threads(void *ptr);
LIB_API void *load_thread(void *ptr);


// utils.h
// gemm.h
LIB_API void init_cpu();


#endif  // DARKNET_API
