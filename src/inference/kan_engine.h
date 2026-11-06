#pragma once
#include "model_loader.h"
#include <vector>

// Legacy JSON network, deliberately separate from kan::Executor/CKAN 1.
class KANEngine {
public:
    explicit KANEngine(KANModel model);
    // Executes every layer: W2 * phi(W1 * x), no hidden activation between layers.
    std::vector<float> infer(const std::vector<float>& input) const;
    const KANModel& model() const noexcept { return model_; }
private:
    KANModel model_;
};
