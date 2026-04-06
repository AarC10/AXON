#include "optimizers/RMSProp.h"

#include <stdexcept>

RMSProp::RMSProp(std::vector<std::shared_ptr<Tensor>> parameters, const float learning_rate, const float decay_rate,
                 const float epsilon)
    : Optimizer(std::move(parameters)), learningRate(learning_rate), decayRate(decay_rate), epsilonValue(epsilon) {}

void RMSProp::step() {
}
