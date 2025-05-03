#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

struct SplineFunction {
    std::vector<float> knots;
    std::vector<float> coefficients;
};

struct KANLayer {
    std::vector<std::vector<float>> W1;
    std::vector<SplineFunction> phi;
    std::vector<std::vector<float>> W2;
};

struct KANModel {
    std::string name;
    int input_dim;
    std::vector<KANLayer> layers;
};

KANModel loadKANModelFromJSON(const std::string& filepath);

#endif // MODEL_LOADER_H
