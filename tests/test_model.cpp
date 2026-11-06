#include "kan/model.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool condition,const char* text) { if(!condition) throw std::runtime_error(text); }
bool exact(float a,float b) { return std::memcmp(&a,&b,sizeof(float))==0; }
kan::Layer make(std::size_t I,std::size_t O,std::vector<float> knots, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-0.5f,0.5f);
    kan::Layer l;
    l.inputs=I; l.outputs=O; l.knots=std::move(knots);
    l.bias.resize(O); l.skip.resize(I*O); l.coefficients.resize(I*O*l.knots.size());
    for(auto* arr:{&l.bias,&l.skip,&l.coefficients})
        for(auto& x:*arr) x=dist(rng);
    return l;
}
template<typename F> void should_fail(F f,const char* case_name) {
    try { f(); } catch(const std::exception&) { return; }
    throw std::runtime_error(std::string("expected failure: ")+case_name);
}
void test_example() {
    auto m=kan::load_model(std::string(KAN_PROJECT_ROOT)+"/models/example.kan");
    for(auto engine:{kan::Engine::Baseline,kan::Engine::Shared,kan::Engine::Packed}) {
        kan::Executor ex(m,engine);
        const auto y=ex.forward({0.4f,-0.2f});
        check(y.size()==1 && std::abs(y[0]-1.0f)<1e-6f,"example result 1.0");
        // Spline clamps, skip does not.
        const auto z=ex.forward({10.0f,0.0f});
        check(std::abs(z[0]-11.0f)<1e-6f,"unclamped skip");
    }
}
void test_random() {
    std::mt19937 rng(101);
    const std::vector<std::vector<float>> grids={{-2.0f,0.0f,0.6f,3.0f},
        {-1.f,-0.999f,0.0f,0.01f,1.0f}, {-2.f,-1.f,0.f,1.f,2.f}};
    int checked=0;
    for(int trial=0;trial<80;++trial) {
        const int I=1+(trial%13),O=1+(trial%17);
        kan::Model m;
        m.layers.push_back(make(I,O,grids[static_cast<std::size_t>(trial)%grids.size()],rng));
        if(trial%4==0) m.layers.push_back(make(O,3,{-4.f,0.f,4.f},rng));
        kan::Executor ref(m,kan::Engine::Baseline),shared(m,kan::Engine::Shared),packed(m,kan::Engine::Packed);
        std::uniform_real_distribution<float> dist(-3.f,3.f);
        auto workspace=packed.make_workspace();
        for(int t=0;t<40;++t) {
            std::vector<float> x(static_cast<std::size_t>(I));
            for(auto& v:x) v=dist(rng);
            if(t==0) std::fill(x.begin(),x.end(),m.layers.front().knots.front());
            if(t==1) std::fill(x.begin(),x.end(),m.layers.front().knots.back());
            if(t==2) std::fill(x.begin(),x.end(),0.0f);
            if(t==3) std::fill(x.begin(),x.end(),10000.0f);
            if(t==4) std::fill(x.begin(),x.end(),-10000.0f);
            const auto a=ref.forward(x),b=shared.forward(x),c=packed.forward(x);
            check(a.size()==b.size()&&b.size()==c.size(),"shape mismatch");
            for(std::size_t n=0;n<a.size();++n) {
                check(exact(a[n],b[n])&&exact(a[n],c[n]),"backend not float32 bitwise identical");
                ++checked;
            }
            std::vector<float> y(a.size());
            packed.forward_into(x.data(),y.data(),workspace);
            for(std::size_t n=0;n<a.size();++n) check(exact(a[n],y[n]),"workspace mismatch");
        }
    }
    std::cout << "randomized exact comparisons: "<<checked<<'\n';
}
void test_roundtrip() {
    std::mt19937 rng(11);
    kan::Model m; m.layers.push_back(make(2,7,{-4.0f,-0.2f,0.1f,1.3f,7.0f},rng));
    m.layers.push_back(make(7,2,{-2.0f,1.0f,5.0f},rng));
    const std::string path="ckan_roundtrip_test.kan";
    kan::save_model(m,path);
    const auto reload=kan::load_model(path);
    std::remove(path.c_str());
    for(std::size_t l=0;l<m.layers.size();++l) {
        const auto& a=m.layers[l]; const auto& b=reload.layers[l];
        check(a.knots==b.knots && a.bias==b.bias && a.skip==b.skip &&
              a.coefficients==b.coefficients,"float32 roundtrip mismatch");
    }
}
void test_invalid() {
    std::mt19937 rng(4);
    auto m=kan::Model{{make(2,3,{-1.0f,0.0f,1.0f},rng)}};
    auto bad=m;
    bad.layers[0].knots[1]=bad.layers[0].knots[0];
    should_fail([&]{bad.validate();},"duplicate knots");
    bad=m; bad.layers[0].coefficients.pop_back();
    should_fail([&]{bad.validate();},"array dimension");
    bad=m; bad.layers[0].skip[0]=std::numeric_limits<float>::quiet_NaN();
    should_fail([&]{bad.validate();},"NaN weight");
    bad=m; bad.layers.push_back(make(5,2,{-1.f,0.f,1.f},rng));
    should_fail([&]{bad.validate();},"mismatched layers");
    kan::Executor ex(m,kan::Engine::Shared);
    should_fail([&]{ex.forward({1.f});},"input shape");
    should_fail([&]{ex.forward({std::numeric_limits<float>::infinity(),0.f});},"nonfinite input");
    const std::string path="ckan_bad_test.kan";
    const std::vector<std::string> broken={
        "CKAN 999\nLAYERS 1", "CKAN 1\nLAYERS 1\nLAYER -1 1 3",
        "CKAN 1\nLAYERS 1\nLAYER 2 1 3\nKNOTS -1 0 1\nBIAS 0\nSKIP 1 1\nCOEFF 0 0 0 0 nan 0",
        "CKAN 1\nLAYERS 1\nLAYER 2 1 3\nKNOTS -1 0 1\nBIAS 0\nSKIP 1 1\nCOEFF 0 0 0 0 1 0 junk",
        "CKAN 1\nLAYERS 1\nLAYER 2 1 3\nKNOTS -1 0 1\nBIAS 0\nSKIP 1 1\nCOEFF 0 0 0 0 1"
    };
    for(const auto& text:broken) {
        std::ofstream out(path); out<<text; out.close();
        should_fail([&]{(void)kan::load_model(path);},"malformed file");
    }
    std::remove(path.c_str());
}
}
int main() {
    try {test_example();test_random();test_roundtrip();test_invalid();}
    catch(const std::exception& e) {std::cerr<<"test FAILED: "<<e.what()<<'\n';return 1;}
    std::cout<<"All cKAN C++ tests PASS\n";
}
