#ifndef AXON_MSELOSS_H
#define AXON_MSELOSS_H

#include "core/TensorImpl.h"
#include "loss/Loss.h"

/**
 * @brief Mean Squared Error loss
 */
class MSELoss : public Loss {
  public:
    /** @brief Constructs an MSELoss instance */
    MSELoss() = default;

    /** @copydoc Loss::forward */
    Tensor forward(const Tensor &prediction, const Tensor &target) override;
};

#endif // AXON_MSELOSS_H
