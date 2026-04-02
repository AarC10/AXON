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

    /**
     * @brief Applies the ReLU activation elementwise
     * @param input Input tensor
     * @return Output tensor with negative values zeroed
     */
    Tensor forward(const Tensor &input) override;
};

#endif // AXON_RELU_H
