#include "optimizers/SGD.h"

#include <stdexcept>

SGD::SGD(std::vector<std::shared_ptr<Tensor>> parameters, const float learning_rate)
    : Optimizer(std::move(parameters)), learningRate(learning_rate) {}

void SGD::step() {
}
