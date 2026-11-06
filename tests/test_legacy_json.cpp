#include "inference/kan_engine.h"
#include "model_loader.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
namespace {
void check(bool b,const char* s) {if(!b) throw std::runtime_error(s);}
void reject_json(const std::string& data) {
    const std::string path="ckan_legacy_invalid.json";
    {std::ofstream f(path); f<<data;}
    bool threw=false;
    try {(void)loadKANModelFromJSON(path);} catch(const std::exception&) {threw=true;}
    std::remove(path.c_str());
    check(threw,"invalid JSON accepted");
}
}
int main() {
  try {
    const auto m=loadKANModelFromJSON(std::string(KAN_PROJECT_ROOT)+"/models/dummy_model.json");
    const auto y=KANEngine(m).infer({0.4f,-0.2f});
    check(y.size()==1,"wrong output width");
    check(std::abs(y.front()-0.505f)<1e-6f,"legacy JSON model output mismatch");
    reject_json("not-json");
    reject_json(R"({"name":"bad","input_dim":2,"layers":[{"W1":[[1,2]],"W2":[[1]],"phi":[{"type":"bogus","knots":[-1,1],"coefficients":[0,1]}]}]})");
    reject_json(R"({"name":"bad","input_dim":2,"layers":[{"W1":[[1,2]],"W2":[[1]],"phi":[{"knots":[1,-1],"coefficients":[0,1]}]}]})");
    reject_json(R"({"name":"bad","input_dim":2,"layers":[{"W1":[[1]],"W2":[[1]],"phi":[{"knots":[-1,1],"coefficients":[0,1]}]}]})");
    reject_json(R"({"name":"bad","input_dim":2,"layers":[{"W1":[["nan",2]],"W2":[[1]],"phi":[{"knots":[-1,1],"coefficients":[0,1]}]}]})");
    std::cout<<"Legacy JSON loader, dimensional checks, format rejection PASS\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
