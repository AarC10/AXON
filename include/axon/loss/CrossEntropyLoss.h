#ifndef AXON_CROSSENTROPYLOSS_H
#define AXON_CROSSENTROPYLOSS_H

#include "core/Tensor.h"

/**
 * @brief Cross-Entropy loss combining log softmax and negative log likelihood
 */
class CrossEntropyLoss {
  public:
    /** @brief Constructs a CrossEntropyLoss instance */
    CrossEntropyLoss() = default;

    /**
     * @brief Computes the cross-entropy loss between raw logits and class targets
     * @param prediction Predicted logits tensor (unnormalized scores)
     * @param target Ground-truth class label tensor
     * @return Scalar loss tensor
     */
    Tensor forward(const Tensor &prediction, const Tensor &target);
};

#endif // AXON_CROSSENTROPYLOSS_H
