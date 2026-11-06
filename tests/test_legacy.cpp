#include "inference/kan_engine.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
namespace {
void check(bool b,const char* msg) { if(!b) throw std::runtime_error(msg); }
template<class F> void fails(F&& f) {try {f();} catch(const std::exception&) {return;} throw std::runtime_error("invalid legacy model accepted");}
KANModel simple() {
    KANModel m;
    m.name="legacy-fixture";
    m.input_dim=2;
    m.layers.push_back({{{0.8f,-0.2f},{1.0f,0.5f}},
       {{{-1.f,0.f,1.f},{0.f,0.5f,1.f}},{{-1.f,0.f,1.f},{1.f,0.5f,0.f}}},
       {{1.f,-0.5f}}});
    return m;
}
}
int main() {
  try {
    const auto m=simple();
    const auto y=KANEngine(m).infer({0.4f,-0.2f});
    check(y.size()==1 && std::abs(y[0]-0.505f)<1e-6f,"legacy 1-layer result wrong");
    auto m2=m;
    KANLayer second;
    second.W1={{2.f}};
    second.phi={{{-2.f,0.f,2.f},{-2.f,0.f,2.f}}};
    second.W2={{3.f}};
    m2.layers.push_back(second);
    const auto z=KANEngine(m2).infer({0.4f,-0.2f});
    check(z.size()==1&&std::abs(z[0]-3.03f)<1e-5f,"legacy multilayer result wrong");
    fails([&] { (void)KANEngine(m).infer({1.f}); });
    fails([&] { (void)KANEngine(m).infer({std::numeric_limits<float>::quiet_NaN(),1.f}); });
    auto bad=m; bad.layers[0].phi[0].knots[1]=-1.f;
    fails([&] { (void)KANEngine(bad); });
    bad=m; bad.layers[0].W2[0].pop_back();
    fails([&] { (void)KANEngine(bad); });
    bad=m; bad.layers[0].W1[0][0]=std::numeric_limits<float>::infinity();
    fails([&] { (void)KANEngine(bad); });
    std::cout<<"Legacy W2*phi(W1*x) and validation PASS\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
