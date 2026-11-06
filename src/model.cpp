#include "kan/model.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace kan {
namespace {
constexpr std::size_t max_dims = 4096;
constexpr std::size_t max_coeff = 4000000;
constexpr std::size_t max_layers = 64;
constexpr std::size_t max_knots = 256;

void require(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }
bool finite(float x) { return std::isfinite(static_cast<double>(x)); }
std::size_t checked_mul(std::size_t x, std::size_t y, const char* msg) {
    require(!y || x <= std::numeric_limits<std::size_t>::max()/y, msg);
    return x*y;
}
struct Segment { std::size_t k; double alpha; };
Segment locate(const std::vector<float>& knots, float value) {
    // Saturate only the spline argument, preserving original x in linear skip.
    const double v = std::max<double>(knots.front(), std::min<double>(knots.back(), value));
    const auto it = std::upper_bound(knots.begin(), knots.end(), v);
    const std::size_t k = it == knots.begin() ? 0 :
        std::min<std::size_t>(static_cast<std::size_t>(it-knots.begin()-1), knots.size()-2);
    const double a = (v-static_cast<double>(knots[k])) /
        (static_cast<double>(knots[k+1])-static_cast<double>(knots[k]));
    return {k,a};
}
float parse_float(const std::string& token) {
    // Classic text formatting from std::stof is locale sensitive; force finite tokens
    // and exactly consumed numeric text. No silent NaN, Inf or truncation.
    std::size_t used=0;
    float value=0;
    try { value = std::stof(token,&used); }
    catch (const std::exception&) { throw std::runtime_error("invalid/nonrepresentable float: " + token); }
    require(used==token.size() && finite(value), "invalid/nonfinite float");
    return value;
}
std::string take(std::istream& in, const char* what) {
    std::string s;
    if (!(in >> s)) throw std::runtime_error(std::string("missing ") + what);
    return s;
}
void expect(std::istream& in, const char* expected) {
    const auto s=take(in,expected);
    if(s!=expected) throw std::runtime_error(std::string("expected ")+expected+", got "+s);
}
std::size_t read_uint(std::istream& in, std::size_t lo, std::size_t hi) {
    const auto token=take(in,"integer");
    require(!token.empty() && token.find_first_not_of("0123456789")==std::string::npos, "invalid unsigned field");
    std::size_t used=0, n=0;
    try { n=static_cast<std::size_t>(std::stoull(token,&used)); }
    catch(const std::exception&) { throw std::runtime_error("integer overflow"); }
    require(used==token.size() && n>=lo && n<=hi,"integer out of range");
    return n;
}
std::vector<float> read_floats(std::istream& in, std::size_t n) {
    std::vector<float> v;
    v.reserve(n);
    for(std::size_t i=0;i<n;++i) v.push_back(parse_float(take(in,"float")));
    return v;
}
void write_values(std::ostream& o, const char* name, const std::vector<float>& values) {
    o << name;
    for (float x : values) o << ' ' << x;
    o << '\n';
}
void validate_input(const float* x, std::size_t n) {
    for(std::size_t i=0;i<n;++i)
        if(!finite(x[i])) throw std::invalid_argument("nonfinite input");
}
}
void Layer::validate() const {
    require(inputs>=1 && inputs<=max_dims && outputs>=1 && outputs<=max_dims, "layer dimensions out of range");
    require(knots.size()>=2 && knots.size()<=max_knots,"knot count out of range");
    const auto edges=checked_mul(inputs,outputs,"layer edge count overflow");
    const auto coeff=checked_mul(edges,knots.size(),"layer coefficient count overflow");
    require(coeff<=max_coeff,"layer coefficient limit exceeded");
    require(bias.size()==outputs && skip.size()==edges && coefficients.size()==coeff,"incorrect array lengths");
    for(std::size_t k=0;k<knots.size();++k) {
        require(finite(knots[k]),"nonfinite knot");
        if(k) require(knots[k]>knots[k-1],"knots must be strictly increasing");
    }
    for(float v:bias) require(finite(v),"nonfinite bias");
    for(float v:skip) require(finite(v),"nonfinite skip");
    for(float v:coefficients) require(finite(v),"nonfinite coefficient");
}
void Model::validate() const {
    require(!layers.empty() && layers.size()<=max_layers,"model layer count out of range");
    std::size_t total=0;
    for(std::size_t l=0;l<layers.size();++l) {
        layers[l].validate();
        if(l) require(layers[l].inputs==layers[l-1].outputs,"disconnected layers");
        const auto n=layers[l].coefficients.size();
        require(n<=max_coeff-total,"global coefficient count exceeded");
        total+=n;
    }
}
std::vector<float> Layer::forward(const std::vector<float>& x) const {
    Model m; m.layers.push_back(*this);
    return Executor(std::move(m),Engine::Baseline).forward(x);
}
std::vector<float> Model::forward(const std::vector<float>& x) const {
    return Executor(*this,Engine::Baseline).forward(x);
}
std::vector<std::vector<float>> Model::forward_batch(const std::vector<std::vector<float>>& x) const {
    Executor executor(*this,Engine::Baseline);
    std::vector<std::vector<float>> result;
    result.reserve(x.size());
    for(const auto& sample:x) result.push_back(executor.forward(sample));
    return result;
}
Model load_model(const std::string& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("could not open model: "+path);
    in.imbue(std::locale::classic());
    expect(in,"CKAN");
    require(read_uint(in,1,1)==1,"unsupported file version");
    expect(in,"LAYERS");
    const auto n_layers=read_uint(in,1,max_layers);
    Model model;
    std::size_t remaining_coeff=max_coeff;
    for(std::size_t l=0;l<n_layers;++l) {
        expect(in,"LAYER");
        Layer v;
        v.inputs=read_uint(in,1,max_dims);
        v.outputs=read_uint(in,1,max_dims);
        const auto K=read_uint(in,2,max_knots);
        const auto edges=checked_mul(v.inputs,v.outputs,"dimension overflow");
        const auto count=checked_mul(edges,K,"coefficient overflow");
        require(count<=remaining_coeff,"global coefficient limit exceeded");
        remaining_coeff-=count;
        expect(in,"KNOTS"); v.knots=read_floats(in,K);
        expect(in,"BIAS"); v.bias=read_floats(in,v.outputs);
        expect(in,"SKIP"); v.skip=read_floats(in,edges);
        expect(in,"COEFF"); v.coefficients=read_floats(in,count);
        model.layers.push_back(std::move(v));
    }
    std::string extra;
    require(!(in>>extra),"unexpected trailing token");
    model.validate();
    return model;
}
void save_model(const Model& model, const std::string& path) {
    model.validate();
    std::ofstream out(path);
    if(!out) throw std::runtime_error("cannot write model: " + path);
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    out << "CKAN 1\nLAYERS " << model.layers.size() << '\n';
    for(const auto& l:model.layers) {
        out << "LAYER " << l.inputs << ' ' << l.outputs << ' ' << l.knots.size() << '\n';
        write_values(out,"KNOTS",l.knots);
        write_values(out,"BIAS",l.bias);
        write_values(out,"SKIP",l.skip);
        write_values(out,"COEFF",l.coefficients);
    }
    require(out.good(),"model write failed");
}
Engine parse_engine(const std::string& name) {
    if(name=="baseline") return Engine::Baseline;
    if(name=="shared") return Engine::Shared;
    if(name=="packed") return Engine::Packed;
    throw std::invalid_argument("unknown engine: "+name);
}
const char* engine_name(Engine e) {
    switch(e) {
        case Engine::Baseline: return "baseline";
        case Engine::Shared: return "shared";
        case Engine::Packed: return "packed";
    }
    return "unknown";
}
Executor::Executor(Model model, Engine engine) : model_(std::move(model)), engine_(engine) {
    model_.validate();
    plan_.resize(model_.layers.size());
    if(engine_!=Engine::Packed) return;
    for(std::size_t l=0;l<model_.layers.size();++l) {
        const auto& lay=model_.layers[l];
        auto& plan=plan_[l];
        const auto I=lay.inputs,O=lay.outputs,K=lay.knots.size();
        plan.packed_coeff.resize(I*K*O);
        plan.packed_skip.resize(I*O);
        for(std::size_t i=0;i<I;++i)
            for(std::size_t o=0;o<O;++o) {
                plan.packed_skip[i*O+o]=lay.skip[o*I+i];
                for(std::size_t k=0;k<K;++k)
                    plan.packed_coeff[(i*K+k)*O+o]=lay.coefficients[(o*I+i)*K+k];
            }
    }
}
std::size_t Executor::parameters() const {
    std::size_t p=0;
    for(const auto& l:model_.layers) p+=l.knots.size()+l.bias.size()+l.skip.size()+l.coefficients.size();
    return p;
}
std::size_t Executor::packed_bytes() const {
    std::size_t bytes=0;
    for(const auto& p:plan_) bytes+=sizeof(float)*(p.packed_coeff.size()+p.packed_skip.size());
    return bytes;
}
void Executor::eval_layer(const Layer& l,const PlanLayer& p,const float* x,float* out,std::vector<double>& sums) const {
    const auto I=l.inputs,O=l.outputs,K=l.knots.size();
    // Each output accumulates in double; addition order per output remains i=0,...,I-1.
    if(engine_==Engine::Baseline) {
        for(std::size_t o=0;o<O;++o) {
            double sum=static_cast<double>(l.bias[o]);
            for(std::size_t i=0;i<I;++i) {
                const Segment s=locate(l.knots,x[i]);
                const auto base=(o*I+i)*K+s.k;
                const double c0=l.coefficients[base],c1=l.coefficients[base+1];
                const double interp=(1.0-s.alpha)*c0+s.alpha*c1;
                sum+=(static_cast<double>(l.skip[o*I+i])*static_cast<double>(x[i])+interp);
            }
            out[o]=static_cast<float>(sum);
        }
        return;
    }
    if(engine_==Engine::Shared) {
        if(sums.size()<O) sums.resize(O);
        for(std::size_t o=0;o<O;++o) sums[o]=l.bias[o];
        for(std::size_t i=0;i<I;++i) {
            const Segment s=locate(l.knots,x[i]);
            for(std::size_t o=0;o<O;++o) {
                const auto base=(o*I+i)*K+s.k;
                const double c0=l.coefficients[base],c1=l.coefficients[base+1];
                const double interp=(1.0-s.alpha)*c0+s.alpha*c1;
                sums[o]+=(static_cast<double>(l.skip[o*I+i])*static_cast<double>(x[i])+interp);
            }
        }
        for(std::size_t o=0;o<O;++o) out[o]=static_cast<float>(sums[o]);
        return;
    }
    if(sums.size()<O) sums.resize(O);
    for(std::size_t o=0;o<O;++o) sums[o]=l.bias[o];
    for(std::size_t i=0;i<I;++i) {
        const Segment s=locate(l.knots,x[i]);
        const float* c0=&p.packed_coeff[(i*K+s.k)*O];
        const float* c1=&p.packed_coeff[(i*K+s.k+1)*O];
        const float* skip=&p.packed_skip[i*O];
        for(std::size_t o=0;o<O;++o) {
            const double interp=(1.0-s.alpha)*static_cast<double>(c0[o])+s.alpha*static_cast<double>(c1[o]);
            sums[o]+=(static_cast<double>(skip[o])*static_cast<double>(x[i])+interp);
        }
    }
    for(std::size_t o=0;o<O;++o) out[o]=static_cast<float>(sums[o]);
}
std::vector<float> Executor::forward(const std::vector<float>& x) const {
    std::vector<float> y;
    forward_into(x,y);
    return y;
}
Workspace Executor::make_workspace() const {
    Workspace w;
    std::size_t maxwidth=0;
    for(const auto& l:model_.layers)
        maxwidth=std::max(maxwidth,std::max(l.inputs,l.outputs));
    w.a.resize(maxwidth);
    w.b.resize(maxwidth);
    w.sums.resize(maxwidth);
    return w;
}
void Executor::forward_into(const std::vector<float>& x, std::vector<float>& y) const {
    const auto in_dim=model_.layers.front().inputs;
    if(x.size()!=in_dim) throw std::invalid_argument("input dimension mismatch");
    y.resize(model_.layers.back().outputs);
    auto workspace=make_workspace();
    forward_into(x.data(),y.data(),workspace);
}
void Executor::forward_into(const float* x,float* y) const {
    auto workspace=make_workspace();
    forward_into(x,y,workspace);
}
void Executor::forward_into(const float* x,float* y,Workspace& w) const {
    const auto in_dim=model_.layers.front().inputs;
    validate_input(x,in_dim);
    std::size_t maxwidth=0;
    for(const auto& l:model_.layers)
        maxwidth=std::max(maxwidth,std::max(l.inputs,l.outputs));
    if(w.a.size()<maxwidth || w.b.size()<maxwidth || w.sums.size()<maxwidth)
        throw std::invalid_argument("workspace too small; use make_workspace");
    std::copy(x,x+in_dim,w.a.begin());
    float* src=w.a.data();
    float* dst=w.b.data();
    for(std::size_t n=0;n<model_.layers.size();++n) {
        eval_layer(model_.layers[n],plan_[n],src,dst,w.sums);
        std::swap(src,dst);
    }
    std::copy(src,src+model_.layers.back().outputs,y);
}
}
