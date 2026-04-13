#ifndef AXON_MODULE_H
#define AXON_MODULE_H

#include "core/TensorImpl.h"

#include <vector>

/**
 * @brief Abstract base class for all neural network modules
 */
class Module {
  public:
    /** @brief Constructs a module */
    Module() = default;

    /** @brief Destroys the module */
    virtual ~Module() = default;

    /**
     * @brief Computes the forward pass of the module
     * @param input Input tensor
     * @return Output tensor
     */
    virtual Tensor forward(const Tensor &input) = 0;

    /**
     * @brief Returns all learnable parameters of the module
     * @return Vector of parameter tensors
     */
    virtual std::vector<Tensor> parameters();

    /**
     * @brief Zeroes the gradients of all learnable parameters
     */
    virtual void zero_grad();
};

#endif // AXON_MODULE_H
