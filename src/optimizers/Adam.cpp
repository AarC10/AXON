#include "optimizers/Adam.h"

#include <stdexcept>

Adam::Adam(std::vector<std::shared_ptr<Tensor>> parameters, const float learning_rate, const float beta1,
           const float beta2, const float epsilon)
    : Optimizer(std::move(parameters)), learningRate(learning_rate), beta1Value(beta1), beta2Value(beta2),
      epsilonValue(epsilon) {}

void Adam::step() {
}
