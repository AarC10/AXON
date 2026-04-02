#ifndef AXON_MSELOSS_H
#define AXON_MSELOSS_H

#include "core/Tensor.h"

/**
 * @brief Mean Squared Error loss
 */
class MSELoss {
  public:
    /** @brief Constructs an MSELoss instance */
    MSELoss() = default;

    /**
     * @brief Computes the mean squared error between prediction and target
     * @param prediction Predicted tensor
     * @param target Ground-truth tensor
     * @return Scalar loss tensor
     */
    Tensor forward(const Tensor &prediction, const Tensor &target);
};

#endif // AXON_MSELOSS_H
