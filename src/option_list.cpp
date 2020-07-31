#include <string>
#include "option_list.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "utils.hpp"


using namespace std ;

map<string,string> read_data_cfg(string filename)
{
    map<string,string> result ;
    ifstream ifs;
    ifs.open(filename, ios::in);
    if (!ifs.is_open()) {
        cout << "文件打开失败！" << endl;
        ifs.close();
        return result;
    }else
    {
        string buf;
        while (getline(ifs, buf)) {
            auto c = buf.begin();
            while (c!=buf.end()){
                if (*c==' ' ){
                    buf.erase(c);
                } else c++;
            }
            result[ buf.substr(0,buf.find('='))] =  buf.substr(buf.find('=')+1,buf.length()-buf.find('='));

        }
    }       ifs.close();
            return result;
}

string option_find_str(map<string,string> const &m, string key, string def)
{
    auto search = m.find(key);
    if (search != m.end()) {

        return search->second;
    }
    return def;
}
int option_find_int(map<string,string> const & m, string key, int def){
    string result = option_find_str(m, key);
    if (!result.empty())
        return stoi(result);
    else return def ;
}
int option_find_float(map<string,string> const &m, string key, float def){
    string result = option_find_str(m, key);
    if (!result.empty())
        return stof(result);
    else return def ;
}

int* find_mul_int(string a, int &num, bool force )
{

    if (a[a.length() - 1]==',') a.erase(a.length() - 1);
    if (!force){  // num==0,表示按照查找的大小。不等于零则是固定大小为num
        num = 0;
        for (char i : a){
            if (i==',') num++;
        }
        if (num!=0) num++;
    }

    int* result = (int*)xcalloc(num, sizeof(int));
    istringstream iss(a);
    string buf;
    int index = 0;

    while( getline(iss,buf,',')){
        result[index++] = stoi(buf);
    }
    return result;
}

float* find_mul_float(string a, int &num, bool force)
{  /*
 *  parse_net_options函数中 force 为０　
 *  解析ｍａｓｋ时，ｆｏｒｃｅ为１
 */
    if (a[a.length() - 1]==',') a.erase(a.length() - 1);
    if (!force){  // num==0,表示按照查找的大小。不等于零则是固定大小为num
        num = 0;
        for (char i : a){
            if (i==',') num++;
        }
        if (num!=0) num++;
    }
    float* result = new float (num);
    istringstream iss(a);
    string buf;
    int index = 0;

    while( getline(iss,buf,',')){
        result[index++] = stof(buf);
    }
    return result;
}