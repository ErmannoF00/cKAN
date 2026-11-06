#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kan {
// CKAN 1: all stored parameters are IEEE float32, data layout [out][in][knot].
struct Layer {
    std::size_t inputs = 0, outputs = 0;
    std::vector<float> knots, bias, skip, coefficients;
    void validate() const;
    std::vector<float> forward(const std::vector<float>& x) const;
};
struct Model {
    std::vector<Layer> layers;
    void validate() const;
    std::vector<float> forward(const std::vector<float>& x) const;
    std::vector<std::vector<float>> forward_batch(const std::vector<std::vector<float>>& x) const;
};
Model load_model(const std::string& path);
void save_model(const Model& model, const std::string& path);

enum class Engine { Baseline, Shared, Packed };
Engine parse_engine(const std::string& name);
const char* engine_name(Engine e);
// Cached execution plan: all three backends use the same canonical model and arithmetic.
// Baseline   : per-edge upper_bound; original layout [out][in][knot]
// Shared     : one upper_bound per input; original layout
// Packed     : one upper_bound per input; transposed [in][knot][out] for contiguous
//              output-lane access. Preparation cost and extra memory are explicit.
struct Workspace {
    std::vector<float> a, b;
    std::vector<double> sums;
};
class Executor {
public:
    explicit Executor(Model model, Engine engine);
    std::vector<float> forward(const std::vector<float>& x) const;
    void forward_into(const std::vector<float>& x, std::vector<float>& y) const;
    void forward_into(const float* x, float* y) const;  // caller-provided arrays
    Workspace make_workspace() const;
    void forward_into(const float* x, float* y, Workspace& workspace) const;
    std::size_t parameters() const;
    std::size_t packed_bytes() const;
    std::size_t layer_count() const { return model_.layers.size(); }
    const Model& model() const { return model_; }
    Engine engine() const { return engine_; }
private:
    struct PlanLayer {
        std::vector<float> packed_coeff; // [in][knot][out]
        std::vector<float> packed_skip;  // [in][out]
    };
    Model model_;
    Engine engine_;
    std::vector<PlanLayer> plan_;
    void eval_layer(const Layer& layer, const PlanLayer& plan,
                    const float* x, float* out, std::vector<double>& sums) const;
};
}
