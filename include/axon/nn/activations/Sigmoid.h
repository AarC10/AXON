#ifndef AXON_SIGMOID_H
#define AXON_SIGMOID_H

#include "core/Tensor.h"
#include "nn/Module.h"

/**
 * @brief Sigmoid activation
 */
class Sigmoid : public Module {
  public:
    /** @brief Constructs a Sigmoid activation module */
    Sigmoid() = default;

    /** @copydoc Module::forward */
    Tensor forward(const Tensor &input) override;
};

#endif // AXON_SIGMOID_H
