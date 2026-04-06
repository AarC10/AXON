#include "optimizers/SGDWithMomentum.h"

#include <stdexcept>

SGDWithMomentum::SGDWithMomentum(std::vector<std::shared_ptr<Tensor>> parameters, const float learning_rate,
                                 const float momentum)
    : Optimizer(std::move(parameters)), learningRate(learning_rate), momentumValue(momentum) {}

void SGDWithMomentum::step() {
}
