#include "optimizers/Adam.h"

#include <cmath>
#include <stdexcept>

Adam::Adam(std::vector<std::shared_ptr<Tensor>> parameters, const float learning_rate, const float beta1,
           const float beta2, const float epsilon)
    : Optimizer(std::move(parameters)), learningRate(learning_rate), beta1Value(beta1), beta2Value(beta2),
      epsilonValue(epsilon) {
    firstMomentEstimates.reserve(trackedParameters.size());
    secondMomentEstimates.reserve(trackedParameters.size());
    for (const auto &parameter : trackedParameters) {
        firstMomentEstimates.push_back(parameter ? Tensor::zeros(parameter->get_shape()) : Tensor());
        secondMomentEstimates.push_back(parameter ? Tensor::zeros(parameter->get_shape()) : Tensor());
    }
}

void Adam::step() {
    ++stepCount;

    const float firstMomentBiasCorrection = 1.0f - std::pow(beta1Value, static_cast<float>(stepCount));
    const float secondMomentBiasCorrection = 1.0f - std::pow(beta2Value, static_cast<float>(stepCount));

    for (std::size_t i = 0; i < trackedParameters.size(); ++i) {
        const std::shared_ptr<Tensor> &parameter = trackedParameters[i];
        if (!parameter || !parameter->has_grad()) {
            continue;
        }

        const Tensor &gradient = parameter->grad();
        firstMomentEstimates[i] *= Tensor::full(gradient.get_shape(), beta1Value);
        firstMomentEstimates[i] += gradient * Tensor::full(gradient.get_shape(), 1.0f - beta1Value);

        secondMomentEstimates[i] *= Tensor::full(gradient.get_shape(), beta2Value);
        secondMomentEstimates[i] += (gradient * gradient) * Tensor::full(gradient.get_shape(), 1.0f - beta2Value);

        Tensor firstMomentHat = firstMomentEstimates[i] / Tensor::full(gradient.get_shape(), firstMomentBiasCorrection);
        Tensor secondMomentHat = secondMomentEstimates[i] / Tensor::full(gradient.get_shape(), secondMomentBiasCorrection);
        Tensor denominator = secondMomentHat.sqrt() + Tensor::full(gradient.get_shape(), epsilonValue);
        Tensor update = (firstMomentHat / denominator) * Tensor::full(gradient.get_shape(), learningRate);

        *parameter -= update;
    }
}
