#include "optimizers/SGD.h"

SGD::SGD(std::vector<Tensor> parameters, const float learning_rate)
    : Optimizer(std::move(parameters)), learningRate(learning_rate) {}

void SGD::step() {
    for (Tensor& parameter : trackedParameters) {
        if (!parameter || !parameter->has_grad()) {
            continue;
        }

        Tensor update = parameter->grad() * learningRate;
        parameter -= update;
    }
}
