#ifndef AXON_OPTIMIZER_H
#define AXON_OPTIMIZER_H

#include "core/TensorImpl.h"

#include <memory>
#include <utility>
#include <vector>

/**
 * @brief Abstract base class for all parameter optimizers
 */
class Optimizer {
  public:
    /**
     * @brief Constructs an optimizer over a set of learnable parameters
     * @param parameters Tensors that should be updated by step()
     */
    explicit Optimizer(std::vector<Tensor> parameters) : trackedParameters(std::move(parameters)) {}

    /** @brief Destroys the optimizer */
    virtual ~Optimizer() = default;

    /**
     * @brief Performs a single optimization step, updating all tracked parameters
     */
    virtual void step() = 0;

    /**
     * @brief Zeroes the gradients of all tracked parameters
     */
    virtual void zero_grad() {
        for (const Tensor &parameter : trackedParameters) {
            if (parameter) {
                parameter->zero_grad();
            }
        }
    }

  protected:
    std::vector<Tensor> trackedParameters;
};

#endif // AXON_OPTIMIZER_H
