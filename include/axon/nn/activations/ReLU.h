#ifndef AXON_RELU_H
#define AXON_RELU_H

#include "core/Tensor.h"
#include "nn/Module.h"

/**
 * @brief Rectified Linear Unit activation
 */
class ReLU : public Module {
  public:
    /** @brief Constructs a ReLU activation module */
    ReLU() = default;

    /** @copydoc Module::forward */
    Tensor forward(const Tensor &input) override;
};

#endif // AXON_RELU_H
