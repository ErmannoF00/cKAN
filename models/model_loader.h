#pragma once
#include <cstddef>
#include <string>
#include <vector>

// Original cKAN model family: W2 * phi(W1 * x), NOT edge-wise CKAN 1.
struct SplineFunction {
    std::vector<float> knots;
    std::vector<float> coefficients;
};
struct KANLayer {
    std::vector<std::vector<float>> W1; // hidden x input
    std::vector<SplineFunction> phi;    // hidden splines
    std::vector<std::vector<float>> W2; // output x hidden
};
struct KANModel {
    std::string name;
    int input_dim = 0;
    std::vector<KANLayer> layers;
};
void validateKANModel(const KANModel& model);
// JSON loader requires Boost.PropertyTree headers at build time.
KANModel loadKANModelFromJSON(const std::string& filepath);
