#include "model_loader.h"
#include <fstream>

using json = nlohmann::json;

KANModel loadKANModelFromJSON(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open model file: " + filepath);
    }

    json j;
    file >> j;

    KANModel model;
    model.name = j["name"];
    model.input_dim = j["input_dim"];

    for (const auto& jlayer : j["layers"]) {
        KANLayer layer;

        layer.W1 = jlayer["W1"].get<std::vector<std::vector<float>>>();
        layer.W2 = jlayer["W2"].get<std::vector<std::vector<float>>>();

        for (const auto& jphi : jlayer["phi"]) {
            SplineFunction phi;
            phi.knots = jphi["knots"].get<std::vector<float>>();
            phi.coefficients = jphi["coefficients"].get<std::vector<float>>();
            layer.phi.push_back(phi);
        }

        model.layers.push_back(layer);
    }

    return model;
}
