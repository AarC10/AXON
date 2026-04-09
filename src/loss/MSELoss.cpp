#include "loss/MSELoss.h"

#include <stdexcept>

Tensor MSELoss::forward(const Tensor &prediction, const Tensor &target) {
    if (prediction.get_shape() != target.get_shape()) {
        throw std::invalid_argument("MSELoss: prediction and target shapes must match");
    }

    const int elementCount = prediction.nelem();
    if (elementCount == 0) {
        throw std::invalid_argument("MSELoss: cannot compute loss over an empty tensor");
    }

    Tensor difference = prediction - target;
    Tensor squaredDifference = difference * difference;

    float sumOfSquares = 0.0f;
    for (int i = 0; i < elementCount; ++i) {
        sumOfSquares += squaredDifference[i];
    }

    const float meanSquaredError = sumOfSquares / static_cast<float>(elementCount);
    return Tensor(std::vector<float>{meanSquaredError}, std::vector<int>{1});
}
