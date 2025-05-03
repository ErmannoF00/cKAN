#include "inference/kan_engine.h"
#include <stdexcept>
#include <cmath>

KANEngine::KANEngine(const KANModel& model) : model_(model) {}

std::vector<float> KANEngine::infer(const std::vector<float>& input) {
    if (model_.layers.empty()) throw std::runtime_error("Model has no layers");

    const KANLayer& layer = model_.layers[0];

    // Step 1: W1 * x
    std::vector<float> wx = matVecMul(layer.W1, input);

    // Step 2: Apply spline φ
    for (size_t i = 0; i < wx.size(); ++i) {
        wx[i] = applySpline(layer.phi[i], wx[i]);
    }

    // Step 3: W2 * φ(wx)
    return matVecMul(layer.W2, wx);
}

std::vector<float> KANEngine::matVecMul(const std::vector<std::vector<float>>& W, const std::vector<float>& x) {
    std::vector<float> out(W.size(), 0.0f);
    for (size_t i = 0; i < W.size(); ++i)
        for (size_t j = 0; j < x.size(); ++j)
            out[i] += W[i][j] * x[j];
    return out;
}

float KANEngine::applySpline(const SplineFunction& phi, float x) {
    const auto& k = phi.knots;
    const auto& c = phi.coefficients;
    if (k.size() < 2 || k.size() != c.size()) return 0.0f;

    for (size_t i = 0; i < k.size() - 1; ++i) {
        if (x >= k[i] && x <= k[i + 1]) {
            float t = (x - k[i]) / (k[i + 1] - k[i]);
            return (1 - t) * c[i] + t * c[i + 1]; // linear interpolation
        }
    }
    return x < k.front() ? c.front() : c.back();
}
