#ifndef KAN_ENGINE_H
#define KAN_ENGINE_H

#include "model_loader.h"
#include <vector>

class KANEngine {
public:
    explicit KANEngine(const KANModel& model);
    std::vector<float> infer(const std::vector<float>& input);

private:
    KANModel model_;

    float applySpline(const SplineFunction& phi, float x);
    std::vector<float> matVecMul(const std::vector<std::vector<float>>& W, const std::vector<float>& x);
};

#endif // KAN_ENGINE_H
