#include "inference/kan_engine.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
constexpr std::size_t kMaxDim=4096;
constexpr std::size_t kMaxLayers=64;
constexpr std::size_t kMaxKnots=256;
constexpr std::size_t kMaxElements=4000000;
void need(bool v,const char* msg) { if(!v) throw std::invalid_argument(msg); }
bool finite(float x) { return std::isfinite(static_cast<double>(x)); }
float apply_spline(const SplineFunction& f,float x) {
    const auto& t=f.knots;
    const auto& c=f.coefficients;
    if(x<=t.front()) return c.front();
    if(x>=t.back()) return c.back();
    const auto upper=std::upper_bound(t.begin(),t.end(),x);
    const std::size_t k=static_cast<std::size_t>(upper-t.begin()-1);
    // Match original legacy arithmetic: float32 interpolation, not CKAN 1 double.
    const float a=(x-t[k])/(t[k+1]-t[k]);
    return (1.0f-a)*c[k]+a*c[k+1];
}
}
void validateKANModel(const KANModel& m) {
    need(m.input_dim>0 && static_cast<std::size_t>(m.input_dim)<=kMaxDim,"legacy input_dim out of range");
    need(!m.layers.empty() && m.layers.size()<=kMaxLayers,"legacy layer count out of range");
    std::size_t dim=static_cast<std::size_t>(m.input_dim);
    std::size_t total=0;
    for(const auto& l:m.layers) {
        need(!l.W1.empty() && l.W1.size()<=kMaxDim,"legacy W1 hidden size invalid");
        const std::size_t h=l.W1.size();
        need(l.phi.size()==h,"legacy phi count must equal hidden width");
        need(!l.W2.empty() && l.W2.size()<=kMaxDim,"legacy W2 output width invalid");
        for(const auto& row:l.W1) {
            need(row.size()==dim,"legacy W1 row has incorrect input width");
            for(float v:row) need(finite(v),"legacy W1 has nonfinite weight");
        }
        for(const auto& row:l.W2) {
            need(row.size()==h,"legacy W2 row has incorrect hidden width");
            for(float v:row) need(finite(v),"legacy W2 has nonfinite weight");
        }
        for(const auto& spline:l.phi) {
            need(spline.knots.size()>=2 && spline.knots.size()<=kMaxKnots,"legacy knot count invalid");
            need(spline.knots.size()==spline.coefficients.size(),"legacy spline coefficient size invalid");
            for(std::size_t i=0;i<spline.knots.size();++i) {
                need(finite(spline.knots[i])&&finite(spline.coefficients[i]),"legacy spline nonfinite value");
                if(i) need(spline.knots[i]>spline.knots[i-1],"legacy knot positions must strictly increase");
            }
            need(spline.knots.size()<=kMaxElements-total,"legacy model size exceeds limit");
            total+=spline.knots.size();
        }
        need(dim<=kMaxElements/h,"legacy W1 parameter limit exceeded");
        const std::size_t a=h*dim;
        need(a<=kMaxElements-total,"legacy W1 parameter limit exceeded"); total+=a;
        const std::size_t o=l.W2.size();
        need(o<=kMaxElements/h,"legacy W2 parameter limit exceeded");
        const std::size_t b=o*h;
        need(b<=kMaxElements-total,"legacy W2 parameter limit exceeded"); total+=b;
        dim=o;
    }
}
KANEngine::KANEngine(KANModel model): model_(std::move(model)) {validateKANModel(model_);}
std::vector<float> KANEngine::infer(const std::vector<float>& input) const {
    if(input.size()!=static_cast<std::size_t>(model_.input_dim))
        throw std::invalid_argument("legacy input width mismatch");
    for(float x:input) if(!finite(x)) throw std::invalid_argument("legacy nonfinite input");
    std::vector<float> state=input;
    for(const auto& layer:model_.layers) {
        std::vector<float> hidden(layer.W1.size());
        for(std::size_t h=0;h<layer.W1.size();++h) {
            float v=0.0f;
            for(std::size_t i=0;i<state.size();++i) v+=layer.W1[h][i]*state[i];
            hidden[h]=apply_spline(layer.phi[h],v);
        }
        std::vector<float> output(layer.W2.size());
        for(std::size_t o=0;o<layer.W2.size();++o) {
            float v=0.0f;
            for(std::size_t h=0;h<hidden.size();++h) v+=layer.W2[o][h]*hidden[h];
            output[o]=v;
        }
        for(float v:output) if(!finite(v)) throw std::overflow_error("nonfinite legacy intermediate");
        state=std::move(output);
    }
    return state;
}
