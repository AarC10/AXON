#ifndef AXON_MODULE_H
#define AXON_MODULE_H

#include <memory>
#include <vector>

#include "core/Tensor.h"

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
     * @return Vector of shared pointers to parameter tensors
     */
    virtual std::vector<std::shared_ptr<Tensor>> parameters();

    /**
     * @brief Zeroes the gradients of all learnable parameters
     */
    virtual void zero_grad();
};

#endif // AXON_MODULE_H
