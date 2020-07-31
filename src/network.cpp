#include "darknet.hpp"
#include "network.hpp"
#include "image.hpp"
#include "data.hpp"
#include "utils.hpp"
#include "blas.hpp"
#include "convolutional_layer.hpp"
#include "batchnorm_layer.hpp"
#include "maxpool_layer.hpp"
#include "route_layer.hpp"
#include "shortcut_layer.hpp"
#include "yolo_layer.hpp"
#include "upsample_layer.hpp"

int64_t get_current_iteration(network const &net)
{
    return net.cur_iteration;
}

int get_current_batch(network const &net)
{
    int batch_num = (net.seen)/(net.batch*net.subdivisions);
    return batch_num;
}


float get_current_seq_subdivisions(network net)
{
    int sequence_subdivisions = net.init_sequential_subdivisions;

    if (net.num_steps > 0)
    {
        int batch_num = get_current_batch(net);
        int i;
        for (i = 0; i < net.num_steps; ++i) {
            if (net.steps[i] > batch_num) break;
            sequence_subdivisions *= net.seq_scales[i];
        }
    }
    if (sequence_subdivisions < 1) sequence_subdivisions = 1;
    if (sequence_subdivisions > net.subdivisions) sequence_subdivisions = net.subdivisions;
    return sequence_subdivisions;
}

int get_sequence_value(network net)
{
    int sequence = 1;
    if (net.sequential_subdivisions != 0) sequence = net.subdivisions / net.sequential_subdivisions;
    if (sequence < 1) sequence = 1;
    return sequence;
}

float get_current_rate(network net)
{
    int batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net.burn_in) return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
    switch (net.policy) {
        case CONSTANT:
            return net.learning_rate;
        case STEP:
            return net.learning_rate * pow(net.scale, batch_num/net.step);
        case STEPS:
            rate = net.learning_rate;
            for(i = 0; i < net.num_steps; ++i){
                if(net.steps[i] > batch_num) return rate;
                rate *= net.scales[i];
                //if(net.steps[i] > batch_num - 1 && net.scales[i] > 1) reset_momentum(net);
            }
            return rate;
        case EXP:
            return net.learning_rate * pow(net.gamma, batch_num);
        case POLY:
            return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
            //if (batch_num < net.burn_in) return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
            //return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
        case RANDOM:
            return net.learning_rate * pow(rand_uniform(0,1), net.power);
        case SIG:
            return net.learning_rate * (1./(1.+exp(net.gamma*(batch_num - net.step))));
        case SGDR:
        {
            int last_iteration_start = 0;
            int cycle_size = net.batches_per_cycle;
            while ((last_iteration_start + cycle_size) < batch_num)
            {
                last_iteration_start += cycle_size;
                cycle_size *= net.batches_cycle_mult;
            }
            rate = net.learning_rate_min +
                0.5*(net.learning_rate - net.learning_rate_min)
                * (1. + cos((float)(batch_num - last_iteration_start)*3.14159265 / cycle_size));

            return rate;
        }
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net.learning_rate;
    }
}

    network make_network(int n)
{
    network net = {0};
    net.n = n;
    net.layers.resize(net.n);

    return net;
}

void forward_network(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta && state.train){
            scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
            //将delta清零
        }
        l.forward(l, state);
        state.input = l.output;
        /*
        if (i==(3)){
            for(int ij = 0;ij<l.outputs;ij++){
                cout<<l.output[ij]<<endl;
            }
        }*/
    }
}

void update_network(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}

float *get_network_output(network net)
{

    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].output;
}

float get_network_cost(network net)
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    return sum/count;
}


void backward_network(network net, network_state state)
{
    int i;
    float *original_input = state.input;
    float *original_delta = state.delta;
    state.workspace = net.workspace;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{//重点，state.delta指向网络的前一层。
            layer prev = net.layers[i-1];
            state.input = prev.output;
            state.delta = prev.delta;
        }
        layer l = net.layers[i];
        if (l.stopbackward) break;
        if (l.onlyforward) continue;
        l.backward(l, state);//在各个层实现的backward中，对state.delta进行了赋值，因为state.delta是一个指针，所以可以修改值，
        //除了yolo层是直接delta 等于具体的一个值外
        //其他层都是state.delta += XXX
        //这里体现了训练网络是，正向传递subdivision次，然后再反向传递subdivision次，此时的的误差项就是叠加的
        //还体现了第二点，即shortcut层反向传播是会将某层至少有两次误差项；
        //如x1传递到x2，再传递到x3，而x1shortcut连接到x3
        //则在x3的backward中就会对x1的delta进行赋值，而在x2的backward中还会对x1的delta进行赋值，此时的结果就算两个delta相加
    }
}

float train_network_datum(network net, float *x, float *y)
{

    network_state state={0};
    net.seen += net.batch;
    state.index = 0;
    state.net = net;
    state.input = x;
    state.delta = 0;
    state.truth = y;
    state.train = 1;
    forward_network(net, state);
    backward_network(net, state);
    float error = get_network_cost(net);
    //if(((*net.seen)/net.batch)%net.subdivisions == 0) update_network(net);
    return error;
}


float train_network(network net, data d)
{
    return train_network_waitkey(net, d, 0);
}

float train_network_waitkey(network net, data d, int wait_key)
{ // X.rows  = net.batch * net.subdivisions * ngpus
    assert(d.X.rows % net.batch == 0);
    int batch = net.batch;
    int n = d.X.rows / batch;  // = subdivisions
    float* X = (float*)xcalloc(batch * d.X.cols, sizeof(float));
    float* y = (float*)xcalloc(batch * d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, X, y);
        net.current_subdivision = i;
        float err = train_network_datum(net, X, y);  // 该函数包括了 forward_network(net, state)和 backward_network(net, state);
        sum += err;
        if(wait_key) wait_key_cv(5);
    }
    net.cur_iteration += 1;  //每个subdivisions是一个iteration，每个batch为一个subdivision
    //这里体现了batch = subdivision*batch
    //这是是进行了一个大batch，然后更新
    //留心backward_network(net, state)会发现其关于l.delta的计算都是累加的，
    update_network(net);

    free(X);
    free(y);
    return (float)sum/(n*batch);
}


int resize_network(network *net, int w, int h) // 训练阶段出现的
{//通过变化输入图像的大小，来改变卷积，池化等层的输入输出尺寸，//
    //主要变化为：w，h，out_w，out_h, outputs, inputs, delta 的大小

    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        //printf(" (resize %d: layer = %d) , ", i, l.type);
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if (l.type == LOCAL_AVGPOOL) {
            resize_maxpool_layer(&l, w, h);
        }else if (l.type == YOLO) {
            resize_yolo_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if (l.type == SHORTCUT) {
            resize_shortcut_layer(&l, w, h, net);
        }else if (l.type == UPSAMPLE) {
            resize_upsample_layer(&l, w, h);
        }else{
            fprintf(stderr, "Resizing type %d \n", (int)l.type);
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        inputs = l.outputs;
        net->layers[i] = l;//??
        if(l.type != DROPOUT)
        {
            w = l.out_w;
            h = l.out_h;
        }
        //if(l.type == AVGPOOL) break;
    }


    free(net->workspace);
    net->workspace = (float*)xcalloc(1, workspace_size);

    //fprintf(stderr, " Done!\n");
    return 0;
}

int get_network_output_size(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}



float *network_predict(network net, float *input)
{

    network_state state = {0};
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network(net, state);
    float *out = get_network_output(net);
    return out;
}

int num_detections(network const & net, float thresh)
{
    int i;
    int s = 0;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == YOLO) {

            s += yolo_num_detections(l, thresh);
            /*
            for(int ij = 0;ij<l.outputs;ij++){
                cout<<l.output[ij]<<endl;
            }
            */
        }
    }
    return s;
}


detection *make_network_boxes(network const &net, float thresh, int &num)
{
    layer l = net.layers[net.n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    num = nboxes;
    detection* dets = (detection*)xcalloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = (float*)xcalloc(l.classes, sizeof(float));
        // tx,ty,tw,th uncertainty
        dets[i].uc = (float*)xcalloc(4, sizeof(float)); // Gaussian_YOLOv3
        if (l.coords > 4) {//0
            dets[i].mask = (float*)xcalloc(l.coords - 4, sizeof(float));
        }
    }
    return dets;
}


void fill_network_boxes(network const &net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter)
{
    int prev_classes = -1;
    int j;
    for (j = 0; j < net.n; ++j) {
        layer l = net.layers[j];
        if (l.type == YOLO) {//返回objectness > thresh的box数量，并将box坐标等信息存到dets中
            int count = get_yolo_detections(l, w, h, net.w, net.h, thresh, map, relative, dets, letter);
            dets += count;
            if (prev_classes < 0) prev_classes = l.classes;
            else if (prev_classes != l.classes) {
                printf(" Error: Different [yolo] layers have different number of classes = %d and %d - check your cfg-file! \n",
                    prev_classes, l.classes);
            }
        }

    }
}


detection *get_network_boxes(network const &net, int w, int h, float thresh, float hier, int *map, int relative, int &num, int letter)
{//w,h 为原始图片的大小，在缩放前的大小
    detection *dets = make_network_boxes(net, thresh, num);  //根据阈值计算了大于thresh的数量，然后根据该数量创建dets大小，并且传值到num
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets, letter);//将objectness > thresh的box坐标等信息存到dets中
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        free(dets[i].prob);
        if (dets[i].uc) free(dets[i].uc);
        if (dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}



void free_network(network net)
{
    int i;
    for (i = 0; i < net.n; ++i) {
        free_layer(net.layers[i]);
    }
    //free(net.layers);

    free(net.seq_scales);
    free(net.scales);
    free(net.steps);
    //free(net.seen);
    //free(net.cur_iteration);


    free(net.workspace);
}

static float relu(float src) {
    if (src > 0) return src;
    return 0;
}

static float lrelu(float src) {
    const float eps = 0.001;
    if (src > eps) return src;
    return eps;
}

void fuse_conv_batchnorm(network  &net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer &l = net.layers[j];

        if (l.type == CONVOLUTIONAL) {

            if (l.share_layer != NULL) {
                l.batch_normalize = 0;
            }

            if (l.batch_normalize) {
                int f;
                for (f = 0; f < l.n; ++f)
                {
                    l.biases[f] = l.biases[f] - (double)l.scales[f] * l.rolling_mean[f] / (sqrt((double)l.rolling_variance[f] + .00001));

                    const size_t filter_size = l.size*l.size*l.c / l.groups;
                    int i;
                    for (i = 0; i < filter_size; ++i) {
                        int w_index = f*filter_size + i;

                        l.weights[w_index] = (double)l.weights[w_index] * l.scales[f] / (sqrt((double)l.rolling_variance[f] + .00001));
                    }
                }

                free_convolutional_batchnorm(l);
                l.batch_normalize = 0;

            }
        }
        else  if (l.type == SHORTCUT && l.weights && l.weights_normalizion)
        {
            if (l.nweights > 0) {
                //cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
                int i;
                for (i = 0; i < l.nweights; ++i) printf(" w = %f,", l.weights[i]);
                printf(" l.nweights = %d, j = %d \n", l.nweights, j);
            }

            // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
            const int layer_step = l.nweights / (l.n + 1);    // 1 or l.c or (l.c * l.h * l.w)

            int chan, i;
            for (chan = 0; chan < layer_step; ++chan)
            {
                float sum = 1, max_val = -FLT_MAX;

                if (l.weights_normalizion == SOFTMAX_NORMALIZATION) {
                    for (i = 0; i < (l.n + 1); ++i) {
                        int w_index = chan + i * layer_step;
                        float w = l.weights[w_index];
                        if (max_val < w) max_val = w;
                    }
                }

                const float eps = 0.0001;
                sum = eps;

                for (i = 0; i < (l.n + 1); ++i) {
                    int w_index = chan + i * layer_step;
                    float w = l.weights[w_index];
                    if (l.weights_normalizion == RELU_NORMALIZATION) sum += lrelu(w);
                    else if (l.weights_normalizion == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
                }

                for (i = 0; i < (l.n + 1); ++i) {
                    int w_index = chan + i * layer_step;
                    float w = l.weights[w_index];
                    if (l.weights_normalizion == RELU_NORMALIZATION) w = lrelu(w) / sum;
                    else if (l.weights_normalizion == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;
                    l.weights[w_index] = w;
                }
            }

            l.weights_normalizion = NO_NORMALIZATION;


        }
        else {
            //printf(" Fusion skip layer type: %d \n", l.type);
        }
    }
}

