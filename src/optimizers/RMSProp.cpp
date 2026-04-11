#include "optimizers/RMSProp.h"

RMSProp::RMSProp(std::vector<Tensor> parameters, const float learning_rate, const float decay_rate,
                 const float epsilon)
    : Optimizer(std::move(parameters)), learningRate(learning_rate), decayRate(decay_rate), epsilonValue(epsilon) {
    squaredGradientAvg.reserve(trackedParameters.size());
    for (const auto& parameter : trackedParameters) {
        squaredGradientAvg.push_back(parameter ? TensorImpl::zeros(parameter->get_shape()) : Tensor());
    }
}

void RMSProp::step() {
    for (std::size_t i = 0; i < squaredGradientAvg.size(); ++i) {
        Tensor& parameter = trackedParameters[i];
        if (!parameter || !parameter->has_grad()) {
            continue;
        }

        const Tensor& grad = parameter->grad();
        squaredGradientAvg[i] *= TensorImpl::full(grad->get_shape(), decayRate);
        squaredGradientAvg[i] += (grad * grad) * TensorImpl::full(grad->get_shape(), 1.0f - decayRate);

        Tensor denominator = sqrt(squaredGradientAvg[i]) + TensorImpl::full(grad->get_shape(), epsilonValue);
        Tensor update = (grad / denominator) * TensorImpl::full(grad->get_shape(), learningRate);
        parameter -= update;
    }
}
