#ifndef AXON_CROSSENTROPYLOSS_H
#define AXON_CROSSENTROPYLOSS_H

#include "core/Tensor.h"
#include "loss/Loss.h"

/**
 * @brief Cross-Entropy loss combining log softmax and negative log likelihood
 */
class CrossEntropyLoss : public Loss {
  public:
    /** @brief Constructs a CrossEntropyLoss instance */
    CrossEntropyLoss() = default;

    /** @copydoc Loss::forward */
    Tensor forward(const Tensor &prediction, const Tensor &target) override;
};

#endif // AXON_CROSSENTROPYLOSS_H
