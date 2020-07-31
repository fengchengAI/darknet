#include "darknet.hpp"
#include <iostream>
#include <string>
#include <vector>
#include "parser.hpp"
#include "image_opencv.hpp"
using namespace std;


extern void run_detector(int argc, vector<string> const &argv);


int main(int argc, char **argv)
{
    vector<string> arg;
	for (int i = 0; i < argc; ++i) {
		arg.push_back(argv[i]);
	}

    if(argc < 2){
        cerr<<"usage: "<<argv[0]<<" <function>"<<endl;
        return 0;
    }

    init_cpu();
    show_opencv_info();

    if (arg[1]=="detector"){
        run_detector(argc, arg);
    } else {
        cerr<< "Not an option: "<<argv[1]<<endl;;
    }
    return 0;
}
