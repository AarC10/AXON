#include "loss/CrossEntropyLoss.h"

#include <cmath>
#include <stdexcept>

Tensor CrossEntropyLoss::forward(const Tensor &prediction, const Tensor &target) {
    if (prediction.ndim() != 2) {
        throw std::invalid_argument("CrossEntropyLoss: prediction must be a 2D tensor of shape [N, C]");
    }
    if (target.ndim() != 1) {
        throw std::invalid_argument("CrossEntropyLoss: target must be a 1D tensor of class indices");
    }

    const int batchSize = prediction.size(0);
    const int classCount = prediction.size(1);
    if (target.size(0) != batchSize) {
        throw std::invalid_argument("CrossEntropyLoss: batch dimension of prediction and target must match");
    }
    if (batchSize == 0) {
        throw std::invalid_argument("CrossEntropyLoss: cannot compute loss over an empty batch");
    }

    float negativeLogLikelihoodSum = 0.0f;
    for (int sampleIndex = 0; sampleIndex < batchSize; ++sampleIndex) {
      // Findlargest logit to apply log sum exp for num stability
        float maxLogit = prediction.at({sampleIndex, 0});
        for (int classIndex = 1; classIndex < classCount; ++classIndex) {
            const float logit = prediction.at({sampleIndex, classIndex});
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }

        float shiftedExpSum = 0.0f;
        for (int classIndex = 0; classIndex < classCount; ++classIndex) {
            shiftedExpSum += std::exp(prediction.at({sampleIndex, classIndex}) - maxLogit);
        }
        const float logSumExp = maxLogit + std::log(shiftedExpSum);

        const int targetClass = static_cast<int>(target.at({sampleIndex}));
        if (targetClass < 0 || targetClass >= classCount) {
            throw std::out_of_range("CrossEntropyLoss: target class index is out of range");
        }

        // reduces to logSumExp - prediction[target]
        negativeLogLikelihoodSum += logSumExp - prediction.at({sampleIndex, targetClass});
    }

    const float meanNegativeLogLikelihood = negativeLogLikelihoodSum / static_cast<float>(batchSize);
    return Tensor(std::vector<float>{meanNegativeLogLikelihood}, std::vector<int>{1});
}
