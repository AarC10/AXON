#ifndef AXON_SGDWITHMOMENTUM_H
#define AXON_SGDWITHMOMENTUM_H

#include "optimizers/Optimizer.h"

class SGDWithMomentum : public Optimizer {
  public:
    /**
     * @brief Constructs an SGD with momentum optimizer instance
     * @param parameters Tensors tracked by the optimizer
     * @param learning_rate Per-step learning rate
     * @param momentum Momentum coefficient
     */
    SGDWithMomentum(std::vector<Tensor> parameters, float learning_rate = 1e-3f, float momentum = 0.9f);

    /** @brief Performs one optimization step */
    void step() override;

  private:
    std::vector<Tensor> velocities;
    float learningRate;
    float momentumValue;
};

#endif // AXON_SGDWITHMOMENTUM_H
