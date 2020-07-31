
#include "network.hpp"
#include "parser.hpp"
#include "blas.hpp"
#include "utils.hpp"

// ./darknet nightmare cfg/extractor.recon.cfg ~/trained/yolo-coco.conv frame6.png -reconstruct -iters 500 -i 3 -lambda .1 -rate .01 -smooth 2


void calculate_loss(float *output, float *delta, int n, float thresh)
{
    int i;
    float mean = mean_array(output, n);
    float var = variance_array(output, n);
    for(i = 0; i < n; ++i){
        if(delta[i] > mean + thresh*sqrt(var)) delta[i] = output[i];
        else delta[i] = 0;
    }
}





