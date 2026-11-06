#include "kan/model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using clock_type=std::chrono::steady_clock;
kan::Model random_model(int I,int O,int K,int seed) {
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_real_distribution<float> dist(-0.1f,0.1f);
    kan::Layer l;
    l.inputs=static_cast<std::size_t>(I); l.outputs=static_cast<std::size_t>(O);
    l.knots.resize(static_cast<std::size_t>(K));
    // Perturbed but guaranteed monotone nonuniform knots.
    for(int k=0;k<K;++k) l.knots[static_cast<std::size_t>(k)]=-3.0f+static_cast<float>(k)*6.0f/float(K-1)
        + ((k>0 && k<K-1)?float(k%3)*0.007f:0.f);
    l.bias.resize(static_cast<std::size_t>(O));
    l.skip.resize(static_cast<std::size_t>(I*O));
    l.coefficients.resize(static_cast<std::size_t>(I*O*K));
    for(auto* v : {&l.bias,&l.skip,&l.coefficients})
        for(auto& val:*v) val=dist(rng);
    return kan::Model{{std::move(l)}};
}
struct Stats {double prep_us, median_ns, p95_ns, checksum; std::size_t packing_bytes;};
Stats measure(const kan::Model& model,kan::Engine e,int batch,int reps,int rounds,int seed) {
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_real_distribution<float> dist(-3.5f,3.5f);
    const auto I=model.layers.front().inputs,O=model.layers.back().outputs;
    std::vector<float> input(static_cast<std::size_t>(batch)*I);
    std::vector<float> output(O);
    for(float& x:input) x=dist(rng);
    std::vector<double> prep_samples;
    prep_samples.reserve(static_cast<std::size_t>(rounds));
    for(int r=0;r<rounds;++r) {
        const auto t_start=clock_type::now();
        kan::Executor candidate(model,e);
        const auto t_end=clock_type::now();
        if(candidate.parameters()==0) throw std::runtime_error("empty candidate");
        prep_samples.push_back(std::chrono::duration<double,std::micro>(t_end-t_start).count());
    }
    std::sort(prep_samples.begin(),prep_samples.end());
    const double prep=prep_samples[prep_samples.size()/2];
    kan::Executor ex(model,e);
    auto ws=ex.make_workspace();
    double checksum=0;
    // Untimed warmup; all engines get identical input stream and warmup passes.
    for(int r=0;r<2;++r)
        for(int s=0;s<batch;++s) {
            ex.forward_into(&input[static_cast<std::size_t>(s)*I],output.data(),ws);
            checksum+=output[0];
        }
    std::vector<double> blocks;
    blocks.reserve(static_cast<std::size_t>(rounds));
    for(int r=0;r<rounds;++r) {
        const auto t0=clock_type::now();
        for(int iter=0;iter<reps;++iter)
            for(int s=0;s<batch;++s) {
                ex.forward_into(&input[static_cast<std::size_t>(s)*I],output.data(),ws);
                checksum+=output[0];
            }
        const auto t1=clock_type::now();
        blocks.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/double(reps*batch));
    }
    std::sort(blocks.begin(),blocks.end());
    const auto n=blocks.size();
    return {prep,blocks[n/2],blocks[std::min(n-1,(n*95+99)/100-1)],checksum,ex.packed_bytes()};
}
}
int main(int argc,char** argv) {
    try {
        std::string out="results/bench_host.csv", model_file;
        int reps=5,rounds=9,batch=32;
        bool quick=false;
        for(int i=1;i<argc;++i) {
            const std::string a=argv[i];
            if(a=="--quick") {quick=true;continue;}
            if(i+1>=argc) throw std::invalid_argument("missing benchmark flag value");
            const std::string v=argv[++i];
            if(a=="--output") out=v;
            else if(a=="--model") model_file=v;
            else if(a=="--reps") reps=std::stoi(v);
            else if(a=="--rounds") rounds=std::stoi(v);
            else if(a=="--batch") batch=std::stoi(v);
            else throw std::invalid_argument("unknown flag: "+a);
        }
        if(reps<1||rounds<3||batch<1||reps>100000||rounds>100||batch>4096)
            throw std::invalid_argument("invalid benchmark dimensions");
        std::ofstream file(out);
        if(!file) throw std::runtime_error("cannot write output: "+out);
        file<<"inputs,outputs,knots,batch,engine,parameters,packing_bytes,prepare_us,median_ns_per_sample,p95_block_ns_per_sample,checksum,layers,geometry\n";
        file<<std::fixed<<std::setprecision(6);
        std::vector<kan::Model> models;
        if(!model_file.empty()) models.push_back(kan::load_model(model_file));
        else {
            const std::vector<int> widths=quick?std::vector<int>{8,32}:std::vector<int>{8,32,128};
            const std::vector<int> knots=quick?std::vector<int>{8,16}:std::vector<int>{8,16,32};
            for(int w:widths) for(int k:knots) models.push_back(random_model(w,w,k,w*123+k));
        }
        for(const auto& model:models) {
            const auto I=model.layers.front().inputs, O=model.layers.back().outputs;
            const auto K=model.layers.front().knots.size();
            std::string geometry=std::to_string(I);
            for(const auto& l:model.layers) geometry+='x'+std::to_string(l.outputs);
            for(const auto e:{kan::Engine::Baseline,kan::Engine::Shared,kan::Engine::Packed}) {
                const auto s=measure(model,e,batch,reps,rounds,702);
                kan::Executor ex(model,e);
                file<<I<<','<<O<<','<<K<<','<<batch<<','<<kan::engine_name(e)<<','
                    <<ex.parameters()<<','<<s.packing_bytes<<','<<s.prep_us<<','
                    <<s.median_ns<<','<<s.p95_ns<<','<<s.checksum<<','
                    <<model.layers.size()<<','<<geometry<<'\n';
                std::cout<<geometry<<" K0="<<K<<" "<<kan::engine_name(e)
                         <<" "<<s.median_ns<<" ns/sample\n";
            }
        }
        if(!file.good()) throw std::runtime_error("benchmark CSV write failed");
        std::cout<<"Wrote "<<out<<'\n';
    } catch(const std::exception& e) {std::cerr<<"benchmark error: "<<e.what()<<'\n'; return 1;}
}
