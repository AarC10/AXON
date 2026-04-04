#ifndef AXON_OPTIMIZER_H
#define AXON_OPTIMIZER_H

/**
 * @brief Abstract base class for all parameter optimizers
 */
class Optimizer {
  public:
    /** @brief Destroys the optimizer */
    virtual ~Optimizer() = default;

    /**
     * @brief Performs a single optimization step, updating all tracked parameters
     */
    virtual void step() = 0;

    /**
     * @brief Zeroes the gradients of all tracked parameters
     */
    virtual void zero_grad() = 0;
};

#endif // AXON_OPTIMIZER_H
