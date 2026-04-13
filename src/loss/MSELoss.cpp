#include "loss/MSELoss.h"

#include <stdexcept>

Tensor MSELoss::forward(const Tensor &prediction, const Tensor &target) {
    if (prediction->get_shape() != target->get_shape()) {
        throw std::invalid_argument("MSELoss: prediction and target shapes must match");
    }

    const int elementCount = prediction->nelem();
    if (elementCount == 0) {
        throw std::invalid_argument("MSELoss: cannot compute loss over an empty tensor");
    }

    Tensor difference = prediction - target;
    Tensor squaredDifference = difference * difference;

    float sumOfSquares = 0.0f;
    for (int i = 0; i < elementCount; ++i) {
        sumOfSquares += squaredDifference->at(i);
    }

    const float meanSquaredError = sumOfSquares / static_cast<float>(elementCount);
    Tensor loss = TensorImpl::full(std::vector<int>{1}, meanSquaredError,
                                   prediction->get_require_grad() || target->get_require_grad());

    if (loss->get_require_grad()) {
        loss->set_gradient_func(
            [prediction, target, elementCount](const Tensor &grad) {
                const float upstream = grad->at(0);
                const float scale = (2.0f * upstream) / static_cast<float>(elementCount);

                if (prediction->get_require_grad()) {
                    Tensor prediction_grad = TensorImpl::zeros(prediction->get_shape());
                    for (int i = 0; i < elementCount; ++i) {
                        prediction_grad->at(i) = scale * (prediction->at(i) - target->at(i));
                    }
                    prediction->grad() =
                        prediction->has_grad() ? prediction->grad() + prediction_grad : prediction_grad;
                }

                if (target->get_require_grad()) {
                    Tensor target_grad = TensorImpl::zeros(target->get_shape());
                    for (int i = 0; i < elementCount; ++i) {
                        target_grad->at(i) = scale * (target->at(i) - prediction->at(i));
                    }
                    target->grad() = target->has_grad() ? target->grad() + target_grad : target_grad;
                }
            },
            {prediction, target});
    }

    return loss;
}
