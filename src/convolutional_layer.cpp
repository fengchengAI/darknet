#include "convolutional_layer.hpp"
#include "utils.hpp"
#include "batchnorm_layer.hpp"
#include "im2col.hpp"
#include "col2im.hpp"
#include "blas.hpp"
#include "gemm.hpp"
#include "box.hpp"

inline int convolutional_out_height(convolutional_layer const & l)
{
    return (l.h + 2*l.pad - l.size) / l.stride_y + 1;
}

inline int convolutional_out_width(convolutional_layer const & l)
{
    return (l.w + 2*l.pad - l.size) / l.stride_x + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.output);
}



size_t get_workspace_size32(layer l){

    if (l.xnor) {
        size_t re_packed_input_size = l.c * l.w * l.h * sizeof(float);
        size_t workspace_size = (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);
        if (workspace_size < re_packed_input_size) workspace_size = re_packed_input_size;
        return workspace_size;
    }
    return (size_t)l.out_h*l.out_w*l.size*l.size*(l.c / l.groups)*sizeof(float);
}

size_t get_workspace_size16(layer l) {
    return 0;
    //if (l.xnor) return (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);
    //return (size_t)l.out_h*l.out_w*l.size*l.size*l.c * sizeof(float);
}

size_t get_convolutional_workspace_size(layer l) {
    size_t workspace_size = get_workspace_size32(l);
    size_t workspace_size16 = get_workspace_size16(l);
    if (workspace_size16 > workspace_size) workspace_size = workspace_size16;
    return workspace_size;
}



void free_convolutional_batchnorm(convolutional_layer &l)
{
    if (!l.share_layer) {
        if (l.scales)          free(l.scales),            l.scales = NULL;
        if (l.scale_updates)   free(l.scale_updates),     l.scale_updates = NULL;
        if (l.mean)            free(l.mean),              l.mean = NULL;
        if (l.variance)        free(l.variance),          l.variance = NULL;
        if (l.mean_delta)      free(l.mean_delta),        l.mean_delta = NULL;
        if (l.variance_delta)  free(l.variance_delta),    l.variance_delta = NULL;
        if (l.rolling_mean)    free(l.rolling_mean),      l.rolling_mean = NULL;
        if (l.rolling_variance) free(l.rolling_variance),  l.rolling_variance = NULL;
        if (l.x)               free(l.x),                 l.x = NULL;
        if (l.x_norm)          free(l.x_norm),            l.x_norm = NULL;

    }
}

convolutional_layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, bool xnor, int adam, int use_bin_output, int index, int antialiasing, convolutional_layer *share_layer, int assisted_excitation, int deform, int train)
{   //steps 固定为1
    int total_batch = batch*steps;
    int i;
    convolutional_layer l = { (LAYER_TYPE)0 };
    l.type = CONVOLUTIONAL;
    l.train = train;

    if (xnor) groups = 1;   // disable groups for XNOR-net
    if (groups < 1) groups = 1;

    const int blur_stride_x = stride_x;
    const int blur_stride_y = stride_y;
    l.antialiasing = antialiasing;  //0
    if (antialiasing) {
        stride_x = stride_y = l.stride = l.stride_x = l.stride_y = 1; // use stride=1 in host-layer
    }

    l.deform = deform;  //0
    l.assisted_excitation = assisted_excitation;  //0
    l.share_layer = share_layer;  //0
    l.index = index;
    l.h = h;
    l.w = w;
    l.c = c;
    l.groups = groups;
    l.n = n;
    l.binary = binary;  //0
    l.xnor = xnor;  //0
    l.use_bin_output = use_bin_output;  //0
    l.batch = batch;
    l.steps = steps;
    l.stride = stride_x;
    l.stride_x = stride_x;
    l.stride_y = stride_y;
    l.dilation = dilation;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;
    l.learning_rate_scale = 1;
    l.nweights = (c / groups) * n * size * size;

    if (l.share_layer) {
        if (l.size != l.share_layer->size || l.nweights != l.share_layer->nweights || l.c != l.share_layer->c || l.n != l.share_layer->n) {
            printf(" Layer size, nweights, channels or filters don't match for the share_layer");
            getchar();
        }

        l.weights = l.share_layer->weights;
        l.weight_updates = l.share_layer->weight_updates;

        l.biases = l.share_layer->biases;
        l.bias_updates = l.share_layer->bias_updates;
    }
    else {
        l.weights = (float*)xcalloc(l.nweights, sizeof(float));
        l.biases = (float*)xcalloc(n, sizeof(float));

        if (train) {
            l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
            l.bias_updates = (float*)xcalloc(n, sizeof(float));
        }
    }

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/groups));
    if (l.activation == NORM_CHAN || l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL) {
        for (i = 0; i < l.nweights; ++i) l.weights[i] = 1;   // rand_normal();
    }
    else {
        for (i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
        // rand_normal(); 初始化权重信息。
    }
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
    l.activation = activation;

    l.output = (float*)xcalloc(total_batch*l.outputs, sizeof(float));
    if (train) l.delta = (float*)xcalloc(total_batch*l.outputs, sizeof(float));

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;

    if(batch_normalize){
        if (l.share_layer) {
            l.scales = l.share_layer->scales;
            l.scale_updates = l.share_layer->scale_updates;
            l.mean = l.share_layer->mean;
            l.variance = l.share_layer->variance;
            l.mean_delta = l.share_layer->mean_delta;
            l.variance_delta = l.share_layer->variance_delta;
            l.rolling_mean = l.share_layer->rolling_mean;
            l.rolling_variance = l.share_layer->rolling_variance;
        }
        else {
            l.scales = (float*)xcalloc(n, sizeof(float));
            for (i = 0; i < n; ++i) {
                l.scales[i] = 1;
            }
            if (train) {
                l.scale_updates = (float*)xcalloc(n, sizeof(float));//注意这里的n，表示是对通道求均值的
                l.mean = (float*)xcalloc(n, sizeof(float));
                l.variance = (float*)xcalloc(n, sizeof(float));

                l.mean_delta = (float*)xcalloc(n, sizeof(float));
                l.variance_delta = (float*)xcalloc(n, sizeof(float));
            }
            l.rolling_mean = (float*)xcalloc(n, sizeof(float));
            l.rolling_variance = (float*)xcalloc(n, sizeof(float));
        }

        if (train) {
            l.x = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
            l.x_norm = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
        }
    }

    if (l.activation == SWISH || l.activation == MISH) l.activation_input = (float*)calloc(total_batch*l.outputs, sizeof(float));

    if(adam){
        l.adam = true;
        l.m = (float*)xcalloc(l.nweights, sizeof(float));
        l.v = (float*)xcalloc(l.nweights, sizeof(float));
        l.bias_m = (float*)xcalloc(n, sizeof(float));
        l.scale_m = (float*)xcalloc(n, sizeof(float));
        l.bias_v = (float*)xcalloc(n, sizeof(float));
        l.scale_v = (float*)xcalloc(n, sizeof(float));
    }

    l.workspace_size = get_convolutional_workspace_size(l);

    //fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    l.bflops = (2.0 * l.nweights * l.out_h*l.out_w) / 1000000000.;
    if (l.xnor) l.bflops = l.bflops / 32;
    if (l.xnor && l.use_bin_output) fprintf(stderr, "convXB");
    else if (l.xnor) fprintf(stderr, "convX ");
    else if (l.share_layer) fprintf(stderr, "convS ");
    else if (l.assisted_excitation) fprintf(stderr, "convAE");
    else fprintf(stderr, "conv  ");

    if (groups > 1) fprintf(stderr, "%5d/%4d ", n, groups);
    else           fprintf(stderr, "%5d      ", n);

    if (stride_x != stride_y) fprintf(stderr, "%2dx%2d/%2dx%2d ", size, size, stride_x, stride_y);
    else {
        if (dilation > 1) fprintf(stderr, "%2d x%2d/%2d(%1d)", size, size, stride_x, dilation);
        else             fprintf(stderr, "%2d x%2d/%2d   ", size, size, stride_x);
    }

    fprintf(stderr, "%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

    //fprintf(stderr, "%5d/%2d %2d x%2d /%2d(%d)%4d x%4d x%4d  -> %4d x%4d x%4d %5.3f BF\n", n, groups, size, size, stride, dilation, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.nweights; ++j){
            l.weights[i*l.nweights + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}


void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    int total_batch = l->batch*l->steps;
    int old_w = l->w;
    int old_h = l->h;
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;


    l->output = (float*)xrealloc(l->output, total_batch * l->outputs * sizeof(float));
    if (l->train) {
        l->delta = (float*)xrealloc(l->delta, total_batch * l->outputs * sizeof(float));

        if (l->batch_normalize) {
            l->x = (float*)xrealloc(l->x, total_batch * l->outputs * sizeof(float));
            l->x_norm = (float*)xrealloc(l->x_norm, total_batch * l->outputs * sizeof(float));
        }
    }

    if (l->activation == SWISH || l->activation == MISH) l->activation_input = (float*)realloc(l->activation_input, total_batch*l->outputs * sizeof(float));
    l->workspace_size = get_convolutional_workspace_size(*l);


}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{//（l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w）
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

/*
** 计算每个卷积核的偏置更新值，所谓偏置更新值，就是bias = bias - alpha * bias_update中的bias_update
** 输入： bias_updates     当前层所有偏置的更新值，维度为l.n（即当前层卷积核的个数）
**       delta            当前层的敏感度图（即l.delta）
**       batch            一个batch含有的图片张数（即l.batch）
**       n                当前层卷积核个数（即l.n）
**       k                当前层输入特征图尺寸（即l.out_w*l.out_h）
** 原理：当前层的敏感度图l.delta是误差函数对加权输入的导数，也就是偏置更新值，只是其中每l.out_w*l.out_h个元素都对应同一个
**      偏置，因此需要将其加起来，得到的和就是误差函数对当前层各偏置的导数（l.delta的维度为l.batch*l.n*l.out_h*l.out_w,
**      可理解成共有l.batch行，每行有l.n*l.out_h*l.out_w列，而这一大行又可以理解成有l.n，l.out_h*l.out_w列，这每一小行就
**      对应同一个卷积核也即同一个偏置）
*/
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
	// 遍历batch中每张输入图片
    // 注意，最后的偏置更新值是所有输入图片的总和（多张图片无非就是重复一张图片的操作，求和即可）。
    // 总之：一个卷积核对应一个偏置更新值，该偏置更新值等于batch中所有输入图片累积的偏置更新值，
    // 而每张图片也需要进行偏置更新值求和（因为每个卷积核在每张图片多个位置做了卷积运算，这都对偏置更新值有贡献）以得到每张图片的总偏置更新值。
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}


void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i, j;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);//output清零


    //l.n为输出通道数，l.size为卷积核大小，l.c为输入通道数
    int m = l.n / l.groups;
    int k = l.size*l.size*l.c / l.groups;
    int n = out_h*out_w;

    static int u = 0;  //???
    u++;

    for(i = 0; i < l.batch; ++i)
    {
        for (j = 0; j < l.groups; ++j)
        {
            float *a = l.weights +j*l.nweights / l.groups;
            float *b = state.workspace;//
            float *c = l.output +(i*l.groups + j)*n*m;


            //printf(" l.index = %d - FP32 \n", l.index);
            float *im = state.input + (i*l.groups + j)*(l.c / l.groups)*l.h*l.w;
            if (l.size == 1) {
                b = im;
            }
            else {//1
                //将输入图片进行一定变形后，放在b中
                im2col_cpu_ext(im,   // input
                    l.c / l.groups,     // input channels
                    l.h, l.w,           // input size (h, w)
                    l.size, l.size,     // kernel size (h, w)
                    l.pad * l.dilation, l.pad * l.dilation,       // padding (h, w)
                    l.stride_y, l.stride_x, // stride (h, w)
                    l.dilation, l.dilation, // dilation (h, w)
                    b);                 // output


            }
            //根据a，b计算c
            gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
            /*
             0,
             0,
             m = l.n / l.groups;
             n = out_h*out_w;
             k = l.size*l.size*l.c / l.groups;
             1
             a = l.weights +j*l.nweights / l.groups;
             k = l.size*l.size*l.c / l.groups;
             b = state.workspace;
             n = out_h*out_w;
             1
             c = l.output +(i*l.groups + j)*n*m;
             n = out_h*out_w;
             */


        }
    }


    if(l.batch_normalize){//1
        forward_batchnorm_layer(l, state);
    }
    else {
        add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

    //卷积为mish，在最后的卷积层中为leaky
    //short_cut为linear
    if (l.activation == MISH) activate_array_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);//1
    else activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);

}


/*
** 卷积神经网络反向传播核心函数
** 主要流程：1） 调用gradient_array()计算当前层l所有输出元素关于加权输入的导数值（也即激活函数关于输入的导数值），
**             并乘上上一次调用backward_convolutional_layer()还没计算完的l.delta，得到当前层最终的敏感度图；
**          2） 如果网络进行了BN，则backward_batchnorm_layer。
**          3） 如果网络没有进行BN，则直接调用 backward_bias()计算当前层所有卷积核的偏置更新值；
**          4） 依次调用im2col_cpu()，gemm_nt()函数计算当前层权重系数更新值；
**          5） 如果上一层的delta已经动态分配了内存，则依次调用gemm_tn(), col2im_cpu()计算上一层的敏感度图（并未完成所有计算，还差一个步骤）；
** 强调：每次调用本函数会计算完成当前层的敏感度计算，同时计算当前层的偏置、权重更新值，除此之外，还会计算上一层的敏感度图，但是要注意的是，
**      并没有完全计算完，还差一步：乘上激活函数对加权输入的导数值。这一步在下一次调用本函数时完成。
*/
void backward_convolutional_layer(convolutional_layer l, network_state state)
{
    int i, j;
    int m = l.n / l.groups;
	// 卷积核个数，考虑到分组卷积

	// 每一个卷积核元素个数
    int n = l.size*l.size*l.c / l.groups;
	// 每张输出特征图的元素个数：out_w，out_h是输出特征图的宽高
    int k = l.out_w*l.out_h;

	// 计算当前层激活函数对加权输入的导数值并乘以l.delta相应元素，从而彻底完成当前层敏感度图的计算，得到当前层的敏感度图l.delta。
    // l.output存储了该层网络的所有输出：l.n*l.out_w*l.out_h*l.batch；
    // l.delta就是l.output中每一个元素关于激活函数函数输入的导数值，
    // 注意，这里直接利用输出值求得激活函数关于输入的导数值是因为神经网络中所使用的绝大部分激活函数关于输入的导数值都可以描述为输出值的函数表达式，
    // 比如对于Sigmoid激活函数（记作f(x)），其导数值为f(x)'=f(x)*(1-f(x)),因此如果给出y=f(x)，那么f(x)'=y*(1-y)，只需要输出值y就可以了，不需要输入x的值，
    // l.delta是一个一维数组，长度为l.batch * l.outputs，
    // 再强调一次：gradient_array类函数()不单单是完成激活函数对输入的求导运算，还完成计算当前层敏感度图的最后一步：l.delta中每个元素乘以激活函数对输入的导数（注意gradient_arry中使用的是*=运算符）。
    // 每次调用backward_convolutional_laye时，都会完成当前层敏感度图的计算，同时会计算上一层的敏感度图，但对于上一层，其敏感度图并没有完全计算完成，还差一步，
    // 需要等到下一次调用backward_convolutional_layer()时来完成，
    //实际上卷积网络的前一层误差项是需要当前层的误差项、上层到当前层的权重信息，上层激活函数的导数三者卷积或乘积得到的。该操作分两步：
    //第一步，在backward_batchnorm_layer函数中完成当前层的误差项与上层激活函数的导数的成绩关系
    //第二步，在
    // if (state.delta) {...
    //                gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);...}
    // 中完成与上层到当前层的权重信息的操作
    if (l.activation == MISH) gradient_array_mish(l.outputs*l.batch, l.activation_input, l.delta);
    else gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);  //

    if (l.batch_normalize) {
		// 进行bn层的反向传播，此后的l.delta是对激活函数前数据的导数。
        backward_batchnorm_layer(l, state);
    }
    else {
		// 计算偏置的更新值：每个卷积核都有一个偏置，偏置的更新值也即误差函数对偏置的导数，这个导数的计算很简单，实际所有的导数已经求完了，都存储在l.delta中，
        // 接下来只需把l.delta中对应同一个卷积核的项加起来就可以（卷积核在图像上逐行逐列跨步移动做卷积，每个位置处都有一个输出，共有l.out_w*l.out_h个，
        // 这些输出都与同一个偏置关联，因此将l.delta中对应同一个卷积核的项加起来即得误差函数对这个偏置的导数）
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

	// 遍历batch中的每张照片，对于l.delta来说，每张照片是分开存的，因此其维度会达到：l.batch*l.n*l.out_w*l.out_h，
    // 对于l.weights,l.weight_updates以及上面提到的l.bias,l.bias_updates，是将所有照片对应元素叠加起来
    // （循环的过程就是叠加的过程，注意gemm()这系列函数含有叠加效果，不是覆盖输入C的值，而是叠加到之前的C上），
    // 因此l.weights与l.weight_updates维度为l.n*l.size*l.size，l.bias与l.bias_updates的维度为l.n，都与l.batch无关
    for (i = 0; i < l.batch; ++i) {
        for (j = 0; j < l.groups; ++j) {
			float *a = l.delta + (i*l.groups + j)*m*k;
			// net.workspace充当一个临时工作空间的作用，存储临时所需要的计算参数，比如每层单张图片重排后的结果（这些参数马上就会参与卷积运算），一旦用完，就会被马上更新（因此该变量的值的更新频率比较大）
            float *b = state.workspace;

            float *c = l.weight_updates + j*l.nweights / l.groups;

			// 进入本函数之前，在backward_network()函数中，若当前层为第l层，net.input此时已经是第l-1层的输出
			// 注意反向传播从后往前来看
            float *im = state.input + (i*l.groups + j)* (l.c / l.groups)*l.h*l.w;


			// 下面两步：im2col_cpu()与gemm()是为了计算当前层的权重更新值（其实也就是误差函数对当前层权重的导数）
			// 将多通道二维图像net.input变成按一定存储规则排列的数组b，以方便、高效地进行矩阵（卷积）计算，详细查看该函数注释（比较复杂），
			// im2col_cpu_ext每次仅处理net.input（包含整个batch）中的一张输入图片（对于第一层，则就是读入的图片，对于之后的层，这些图片都是上一层的输出，通道数等于上一层卷积核个数）。
			// 最终重排的b为l.c * l.size * l.size行，l.out_h * l.out_w列。
			// 你会发现在前向forward_convolutional_layer()函数中，也为每层的输入进行了重排，但是很遗憾的是，并没有一个l.workspace把每一层的重排结果保存下来，而是统一存储到net.workspace中，
			// 并被不断擦除更新，那为什么不保存呢？保存下来不是省掉一大笔额外重复计算开销？原因有两个：1）net.workspace中只存储了一张输入图片的重排结果，所以重排下张图片时，马上就会被擦除，
			// 当然你可能会想，那为什么不弄一个l.worspaces将每层所有输入图片的结果保存呢？这引出第二个原因；2）计算成本是降低了，但存储空间需求急剧增加，想想每一层都有l.batch张图，且每张都是多通道的，
			// 重排后其元素个数还会增多，这个存储量搁谁都受不了，如果一个batch有128张图，输入图片尺寸为400*400，3通道，网络有16层（假设每层输入输出尺寸及通道数都一样），那么单单为了存储这些重排结果，
			// 就需要128*400*400*3*16*4/1024/1024/1024 = 3.66G，所以为了权衡，只能重复计算！
            im2col_cpu_ext(
                    im,                 // input
                    l.c / l.groups,     // input channels
                    l.h, l.w,           // input size (h, w)
                    l.size, l.size,     // kernel size (h, w)
                    l.pad * l.dilation, l.pad * l.dilation,       // padding (h, w)
                    l.stride_y, l.stride_x, // stride (h, w)
                    l.dilation, l.dilation, // dilation (h, w)
                    b);                // output

			// 下面计算当前层的权重更新值，所谓权重更新值就是weight = weight - alpha * weight_update中的weight_update，

			// a为l.delta的一大行。l.delta为本层所有输出元素激活函数前数据的导数集合,
			// l.delta按行存储，共有l.batch行，l.out_c * l.out_h * l.out_w列，
			// 即l.delta中每行包含一张图的所有输出图，故这么一大行，又可以视作有l.out_c（l.out_c=l.n）小行，l.out_h*l*out_w小列，而一次循环就是处理l.delta的一大行，
			// 故可以将a视作l.out_c行，l.out_h*l*out_w列的矩阵；
			// b为单张输入图像经过im2col_cpu重排后的图像数据；
			// c为输出，按行存储，可视作有l.n行，l.c*l.size*l.size列（l.c是输入图像的通道数，l.n是卷积核个数），
			// 由上可知：
			// a: (l.out_c) * (l.out_h*l*out_w)
			// b: (l.c * l.size * l.size) * (l.out_h * l.out_w)
			// c: (l.n) * (l.c*l.size*l.size)（注意：l.n = l.out_c）
			// 故要进行a * b + c计算，必须对b进行转置（否则行列不匹配），因故调用gemm_nt()函数
            gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

			// 接下来，用当前层的敏感度图l.delta以及权重l.weights（还未更新）来获取上一层网络的敏感度图，BP算法的主要流程就是依靠这种层与层之间敏感度反向递推传播关系来实现。
			// 在network.c的backward_network()中，会从最后一层网络往前遍循环历至第一层，而每次开始遍历某一层网络之前，都会更新net.input为这一层网络前一层的输出
			// 同时更新net.delta为上一层的delta，因此，这里的net.delta是当前层前一层的敏感度图。
			// 已经强调很多次了，再说一次：下面得到的上一层的敏感度并不完整，完整的敏感度图是损失函数对上一层的加权输入的导数，
			// 而这里得到的敏感度图是损失函数对上一层输出值的导数，还差乘以一个输出值也即激活函数对加权输入的导数。
            if (state.delta) {
				// 当前层还未更新的权重
                a = l.weights + j*l.nweights / l.groups;

				// 每次循环仅处理一张输入图
                b = l.delta + (i*l.groups + j)*m*k;

				// net.workspace和上面一样，还是一张输入图片的重排，不同的是，此处我们只需要这个容器，而里面存储的值我们并不需要，在后面的处理过程中，
				// 会将其中存储的值一一覆盖掉（尺寸维持不变，还是(l.c * l.size * l.size) * (l.out_h * l.out_w）
                c = state.workspace;

                // 相比上一个gemm，此处的a对应上一个的c,b对应上一个的a，c对应上一个的b，即此处a,b,c的行列分别为：
				// a: (l.n) * (l.c*l.size*l.size)，表示当前层所有权重系数
				// b: (l.out_c) * (l.out_h*l*out_w)（注意：l.n = l.out_c），表示当前层的敏感度图
				// c: (l.c * l.size * l.size) * (l.out_h * l.out_w)，表示上一层的敏感度图（其元素个数等于上一层网络单张输入图片的所有输出元素个数），
				// 此时要完成a * b + c计算，必须对a进行转置（否则行列不匹配），因故调用gemm_tn()函数。
				// 此操作含义是用：用当前层还未更新的权重值对敏感度图做卷积，得到包含上一层所有敏感度信息的矩阵，但这不是上一层最终的敏感度图，
				// 因为此时的c，也即net.workspace的尺寸为(l.c * l.size * l.size) * (l.out_h * l.out_w)，明显不是上一层的输出尺寸l.c*l.w*l.h，
				// 接下来还需要调用col2im_cpu()函数将其恢复至l.c*l.w*l.h（可视为l.c行，l.w*l.h列），这才是上一层的敏感度图（实际还差一个环节，
				// 这个环节需要等到下一次调用backward_convolutional_layer()才完成：将net.delta中每个元素乘以激活函数对加权输入的导数值）。
				// 完成gemm这一步，如col2im_cpu()中注释，是考虑了多个卷积核导致的一对多关系（上一层的一个输出元素会流入到下一层多个输出元素中），
				// 接下来调用col2im_cpu()则是考虑卷积核重叠（步长较小）导致的一对多关系。
                gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

				// 对c也即state.workspace进行重排，得到的结果存储在state.delta中，注意是state.delta
				// 整个循环结束后，net.delta的总尺寸为l.batch * l.h * l.w * l.c，这就是上一层网络整个batch的敏感度图，可视为有l.batch行，l.h*l.w*l.c列，
				// 每行存储了一张输入图片所有输出特征图的敏感度
				// col2im_cpu()函数中会调用col2im_add_pixel()函数，该函数中使用了+=运算符，也即该函数要求输入的net.delta的初始值为0,而在gradient_array()中注释到l.delta的元素是不为0（也不能为0）的，
				// 看上去是矛盾的，实则不然，gradient_array()使用的l.delta是当前层的敏感度图，而在col2im_cpu()使用的net.delta是上一层的敏感度图
				// 当前层l.delta之所以不为0,是因为从后面层反向传播过来的，对于上一层，显然还没有反向传播到那，因此net.delta的初始值都是为0的（注意，每一层在构建时，就为其delta动态分配了内存，
                col2im_cpu_ext(
                        state.workspace,        // input
                        l.c / l.groups,         // input channels (h, w)
                        l.h, l.w,               // input size (h, w)
                        l.size, l.size,         // kernel size (h, w)
                        l.pad * l.dilation, l.pad * l.dilation,           // padding (h, w)
                        l.stride_y, l.stride_x,     // stride (h, w)
                        l.dilation, l.dilation, // dilation (h, w)
                        state.delta + (i*l.groups + j)* (l.c / l.groups)*l.h*l.w); // output (delta)
        }
     }
    }
}
void update_convolutional_layer(convolutional_layer l, int batch, float learning_rate_init, float momentum, float decay)
{
    float learning_rate = learning_rate_init*l.learning_rate_scale;
    //float momentum = a.momentum;
    //float decay = a.decay;
    //int batch = a.batch;

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);  //weight_updates = weight_updates + (-decay*batch)*weights
    axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1); //weights = weights + (weight_updates*learning_rate / batch)
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);  //weight_updates = weight_updates * momentum

    //bn层更新
    axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);
    if (l.scales) {
        axpy_cpu(l.n, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }
}



image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c / l.groups;
    return float_to_image(w, h, c, l.weights + i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for (i = 0; i < l.n; ++i) {
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for (i = 0; i < l.n; ++i) {
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = (image *)xcalloc(l.n, sizeof(image));
    int i;
    for (i = 0; i < l.n; ++i) {
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
        char buff[256];
        sprintf(buff, "filter%d", i);
        save_image(weights[i], buff);
        */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}


void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean: -mean;
        }
    }
}
