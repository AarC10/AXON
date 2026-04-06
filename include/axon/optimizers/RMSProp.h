#ifndef AXON_RMSPROP_H
#define AXON_RMSPROP_H

#include "optimizers/Optimizer.h"

class RMSProp : public Optimizer {
  public:
    /**
     * @brief Constructs an RMSProp optimizer instance
     * @param parameters Tensors tracked by the optimizer
     * @param learning_rate Per-step learning rate
     * @param decay_rate Running average decay factor for squared gradients
     * @param epsilon Stability constant for denominator updates
     */
    RMSProp(std::vector<std::shared_ptr<Tensor>> parameters, float learning_rate = 1e-3f, float decay_rate = 0.9f,
            float epsilon = 1e-8f);

    /** @brief Performs one optimization step */
    void step() override;

  private:
    float learningRate;
    float decayRate;
    float epsilonValue;
};

#endif // AXON_RMSPROP_H
