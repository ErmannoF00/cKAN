#include "model_loader.h"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using boost::property_tree::ptree;
void require(bool x,const char* msg) { if(!x) throw std::runtime_error(msg); }
float scalar(const ptree& p) {
    require(p.empty(),"expected JSON scalar, got collection");
    const std::string raw=p.data();
    require(!raw.empty() && raw.find_first_not_of("-+0123456789.eE")==std::string::npos,"expected numeric JSON value");
    std::size_t used=0;
    float v=0;
    try { v=std::stof(raw,&used); }
    catch(...) { throw std::runtime_error("invalid float in legacy JSON"); }
    require(used==raw.size() && std::isfinite(static_cast<double>(v)),"invalid/nonfinite float");
    return v;
}
std::vector<float> vector_of_float(const ptree& node) {
    require(node.data().empty() && !node.empty(),"expected nonempty JSON array");
    std::vector<float> out;
    for(const auto& it:node) {
        require(it.first.empty(),"expected JSON array, not object");
        out.push_back(scalar(it.second));
        require(out.size()<=4096,"oversized JSON row");
    }
    return out;
}
std::vector<std::vector<float>> matrix(const ptree& node) {
    require(node.data().empty()&&!node.empty(),"expected nonempty matrix array");
    std::vector<std::vector<float>> out;
    for(const auto& x:node) {
        require(x.first.empty(),"expected matrix array");
        out.push_back(vector_of_float(x.second));
        require(out.size()<=4096,"oversized matrix rows");
    }
    return out;
}
}
KANModel loadKANModelFromJSON(const std::string& filepath) {
    // Reject unusually large files before materializing a DOM in memory.
    namespace fs=std::filesystem;
    const auto bytes=fs::file_size(filepath);
    require(bytes<=32ULL*1024ULL*1024ULL,"legacy JSON size exceeds 32 MiB");
    ptree root;
    boost::property_tree::read_json(filepath,root);
    KANModel m;
    m.name=root.get<std::string>("name","");
    m.input_dim=root.get<int>("input_dim");
    const auto& layers=root.get_child("layers");
    require(layers.data().empty()&&!layers.empty(),"expected nonempty layers array");
    for(const auto& entry:layers) {
        require(entry.first.empty(),"expected layers array");
        const auto& p=entry.second;
        KANLayer l;
        l.W1=matrix(p.get_child("W1"));
        l.W2=matrix(p.get_child("W2"));
        const auto& spline_array=p.get_child("phi");
        require(spline_array.data().empty()&&!spline_array.empty(),"expected phi array");
        for(const auto& f:spline_array) {
            require(f.first.empty(),"expected phi array");
            const auto kind=f.second.get_optional<std::string>("type");
            if(kind) require(*kind=="spline","unsupported legacy phi type");
            SplineFunction spline;
            spline.knots=vector_of_float(f.second.get_child("knots"));
            spline.coefficients=vector_of_float(f.second.get_child("coefficients"));
            l.phi.push_back(std::move(spline));
            require(l.phi.size()<=4096,"oversized phi array");
        }
        m.layers.push_back(std::move(l));
        require(m.layers.size()<=64,"oversized legacy layer count");
    }
    validateKANModel(m);
    return m;
}
