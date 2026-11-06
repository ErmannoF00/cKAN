#include "kan/model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void usage() {
    std::cout << "kan_infer --model PATH (--input x,y,... | --batch inputs.csv)"
              << " [--engine baseline|shared|packed] [--output predictions.csv]"
              << " [--repeat N]\n";
}
float parse(const std::string& raw) {
    const auto first=raw.find_first_not_of(" \t\r\n");
    if(first==std::string::npos) throw std::invalid_argument("empty CSV field");
    const auto last=raw.find_last_not_of(" \t\r\n");
    const auto s=raw.substr(first,last-first+1);
    std::size_t used=0;
    const float x=std::stof(s,&used);
    if(used!=s.size() || !std::isfinite(x)) throw std::invalid_argument("bad number: "+s);
    return x;
}
std::vector<float> parse_row(const std::string& row,std::size_t dim) {
    std::stringstream ss(row);
    std::string token;
    std::vector<float> x;
    while(std::getline(ss,token,',')) {
        if(token.empty()) throw std::invalid_argument("empty CSV field");
        x.push_back(parse(token));
    }
    if(x.size()!=dim) throw std::invalid_argument("CSV dimensions mismatch");
    return x;
}
std::size_t posint(const std::string& x) {
    if(x.empty() || x.find_first_not_of("0123456789")!=std::string::npos)
        throw std::invalid_argument("bad --repeat argument");
    const auto n=std::stoull(x);
    if(n<1 || n>100000000ULL) throw std::invalid_argument("repeat outside [1,1e8]");
    return static_cast<std::size_t>(n);
}
}
int main(int argc,char** argv) {
    try {
        std::string path,input,batch,output,engine="baseline";
        std::size_t repeat=1;
        for(int i=1;i<argc;++i) {
            const std::string a=argv[i];
            if(a=="--help" || a=="-h") { usage(); return 0; }
            if(i+1>=argc) throw std::invalid_argument("missing value after " + a);
            const std::string v=argv[++i];
            if(a=="--model") path=v;
            else if(a=="--input") input=v;
            else if(a=="--batch") batch=v;
            else if(a=="--output") output=v;
            else if(a=="--engine") engine=v;
            else if(a=="--repeat") repeat=posint(v);
            else throw std::invalid_argument("unknown option: " + a);
        }
        if(path.empty() || (input.empty()==batch.empty())) {
            usage(); throw std::invalid_argument("select a model and exactly one of input/batch");
        }
        const auto model=kan::load_model(path);
        kan::Executor exe(model,kan::parse_engine(engine));
        const auto dim=model.layers.front().inputs;
        std::vector<std::vector<float>> samples;
        if(!input.empty()) samples.push_back(parse_row(input,dim));
        else {
            std::ifstream in(batch);
            if(!in) throw std::runtime_error("cannot read CSV: " + batch);
            std::string line;
            while(std::getline(in,line)) {
                if(line.empty()) continue;
                if(samples.size()>=1000000) throw std::runtime_error("CSV sample limit exceeded");
                samples.push_back(parse_row(line,dim));
            }
            if(samples.empty()) throw std::runtime_error("no samples in CSV");
        }
        std::ofstream file;
        if(!output.empty()) { file.open(output); if(!file) throw std::runtime_error("cannot write: "+output); }
        std::ostream& os=output.empty()?std::cout:file;
        os<<std::setprecision(std::numeric_limits<float>::max_digits10);
        const auto workspace_base=exe.make_workspace();
        auto workspace=workspace_base;
        std::vector<float> scores(model.layers.back().outputs);
        const auto t0=std::chrono::steady_clock::now();
        // Repeat is intended to measure compute; write each output only on the final repetition.
        for(std::size_t r=0;r<repeat;++r) {
            for(const auto& sample:samples) {
                exe.forward_into(sample.data(),scores.data(),workspace);
                if(r+1!=repeat) continue;
                for(std::size_t j=0;j<scores.size();++j) {
                    if(j) os<<',';
                    os<<scores[j];
                }
                os<<'\n';
            }
        }
        const auto t1=std::chrono::steady_clock::now();
        if(!os.good()) throw std::runtime_error("write error");
        if(repeat>1)
            std::cerr<<"compute_and_output_us_per_sample="
                     <<std::chrono::duration<double,std::micro>(t1-t0).count()/double(samples.size()*repeat)
                     <<" (last batch includes CSV writes)\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"cKAN error: "<<e.what()<<'\n';
        return 2;
    }
}
