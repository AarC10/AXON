#include "optimizers/AdamW.h"

#include <cmath>

AdamW::AdamW(std::vector<Tensor> parameters, const float learning_rate, const float beta1, const float beta2,
             const float epsilon, const float weight_decay)
    : Optimizer(std::move(parameters)), learningRate(learning_rate), beta1Value(beta1), beta2Value(beta2),
      epsilonValue(epsilon), weightDecay(weight_decay) {
    firstMomentEstimates.reserve(trackedParameters.size());
    secondMomentEstimates.reserve(trackedParameters.size());
    for (const auto &parameter : trackedParameters) {
        firstMomentEstimates.push_back(parameter ? TensorImpl::zeros(parameter->get_shape()) : Tensor());
        secondMomentEstimates.push_back(parameter ? TensorImpl::zeros(parameter->get_shape()) : Tensor());
    }
}

void AdamW::step() {
    ++stepCount;

    const float firstMomentBiasCorrection = 1.0f - std::pow(beta1Value, static_cast<float>(stepCount));
    const float secondMomentBiasCorrection = 1.0f - std::pow(beta2Value, static_cast<float>(stepCount));

    for (std::size_t i = 0; i < trackedParameters.size(); ++i) {
        Tensor &parameter = trackedParameters[i];
        if (!parameter || !parameter->has_grad()) {
            continue;
        }

        const Tensor &gradient = parameter->grad();
        firstMomentEstimates[i] *= TensorImpl::full(gradient->get_shape(), beta1Value);
        firstMomentEstimates[i] += gradient * TensorImpl::full(gradient->get_shape(), 1.0f - beta1Value);

        secondMomentEstimates[i] *= TensorImpl::full(gradient->get_shape(), beta2Value);
        secondMomentEstimates[i] += (gradient * gradient) * TensorImpl::full(gradient->get_shape(), 1.0f - beta2Value);

        Tensor firstMomentHat =
            firstMomentEstimates[i] / TensorImpl::full(gradient->get_shape(), firstMomentBiasCorrection);
        Tensor secondMomentHat =
            secondMomentEstimates[i] / TensorImpl::full(gradient->get_shape(), secondMomentBiasCorrection);
        Tensor denominator = sqrt(secondMomentHat) + TensorImpl::full(gradient->get_shape(), epsilonValue);
        Tensor adaptiveUpdate = (firstMomentHat / denominator) * TensorImpl::full(gradient->get_shape(), learningRate);
        Tensor weightDecayUpdate = parameter * TensorImpl::full(parameter->get_shape(), learningRate * weightDecay);

        parameter -= adaptiveUpdate;
        parameter -= weightDecayUpdate;
    }
}
