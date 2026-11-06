#include "model_loader.h"
#include "inference/kan_engine.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
std::vector<float> read_input(const std::string& s) {
    std::stringstream stream(s);
    std::string tok;
    std::vector<float> x;
    while(std::getline(stream,tok,',')) {
        std::size_t n=0;
        const float v=std::stof(tok,&n);
        if(n!=tok.size()||!std::isfinite(v)) throw std::invalid_argument("bad numeric input");
        x.push_back(v);
    }
    return x;
}
}
int main(int argc,char** argv) {
  try {
    std::string path,input;
    for(int i=1;i<argc;++i) {
      const std::string a=argv[i];
      if(a=="--help") {std::cout<<"kan_legacy_infer --model legacy.json --input 0.4,-0.2\n";return 0;}
      if(i+1>=argc) throw std::invalid_argument("argument value missing");
      const std::string v=argv[++i];
      if(a=="--model") path=v;
      else if(a=="--input") input=v;
      else throw std::invalid_argument("unsupported argument: "+a);
    }
    if(path.empty()||input.empty()) throw std::invalid_argument("require --model and --input");
    const auto result=KANEngine(loadKANModelFromJSON(path)).infer(read_input(input));
    std::cout<<std::setprecision(std::numeric_limits<float>::max_digits10);
    for(std::size_t j=0;j<result.size();++j) {
       if(j) std::cout<<',';
       std::cout<<result[j];
    }
    std::cout<<'\n';
    return 0;
  } catch(const std::exception& e) {std::cerr<<"legacy KAN error: "<<e.what()<<'\n';return 2;}
}
