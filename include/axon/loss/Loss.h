#ifndef AXON_LOSS_H
#define AXON_LOSS_H

#include "core/TensorImpl.h"

/**
 * @brief Abstract base class for loss functions
 */
class Loss {
  public:
    /**
     * @brief Calculates the scalar loss between a prediction and a target
     * @param prediction Predicted tensor
     * @param target Ground-truth tensor
     * @return Scalar loss tensor
     */
    virtual Tensor forward(const TensorImpl &prediction, const TensorImpl &target) = 0;
};

#endif // AXON_LOSS_H
