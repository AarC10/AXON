#include "nn/Linear.h"

Linear::Linear(int in_features, int out_features, bool bias)
    : in_features(in_features), out_features(out_features), use_bias(bias) {}

Tensor Linear::forward(const Tensor &input) { return {}; }

std::vector<std::shared_ptr<Tensor>> Linear::parameters() { return {}; }
