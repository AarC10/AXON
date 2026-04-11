#include "optimizers/SGDWithMomentum.h"

SGDWithMomentum::SGDWithMomentum(std::vector<Tensor> parameters, const float learning_rate,
                                 const float momentum)
    : Optimizer(std::move(parameters)), learningRate(learning_rate), momentumValue(momentum) {
    velocities.reserve(trackedParameters.size());
    for (const auto& parameter : trackedParameters) {
        velocities.push_back(parameter ? TensorImpl::zeros(parameter->get_shape()) : Tensor());
    }
}

void SGDWithMomentum::step() {
    for (std::size_t i = 0; i < trackedParameters.size(); ++i) {
        Tensor& parameter = trackedParameters[i];
        if (!parameter || !parameter->has_grad()) {
            continue;
        }

        velocities[i] *= TensorImpl::full(velocities[i]->get_shape(), momentumValue);
        velocities[i] += parameter->grad() * learningRate;
        parameter -= velocities[i];
    }
}
