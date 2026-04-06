#include "optimizers/RMSProp.h"

#include <stdexcept>

RMSProp::RMSProp(std::vector<std::shared_ptr<Tensor>> parameters, const float learning_rate, const float decay_rate,
                 const float epsilon)
    : Optimizer(std::move(parameters)), learningRate(learning_rate), decayRate(decay_rate), epsilonValue(epsilon) {
    squaredGradientAvg.reserve(parameters.size());
    for (const auto parameter : parameters) {
        squaredGradientAvg.push_back(parameter ? Tensor::zeros(parameter->get_shape()) : Tensor());
    }
}

void RMSProp::step() {
    for (std::size_t i = 0; i < squaredGradientAvg.size(); ++i) {
        const std::shared_ptr<Tensor>& parameter = trackedParameters[i];
        if (!parameter || !parameter->has_grad()) {
            continue;
        }

        const Tensor& grad = parameter->grad();
        squaredGradientAvg[i] *= Tensor::full(grad.get_shape(), decayRate);
        squaredGradientAvg[i] += (grad * grad) * Tensor::full(grad.get_shape(), 1.0f - decayRate);

        Tensor denominator = squaredGradientAvg[i].sqrt() + Tensor::full(grad.get_shape(), epsilonValue);
        Tensor update = (grad / denominator) * Tensor::full(grad.get_shape(), learningRate);
        *parameter -= update;
    }
}
