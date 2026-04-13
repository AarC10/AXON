#ifndef AXON_ADAM_H
#define AXON_ADAM_H

#include "optimizers/Optimizer.h"

class Adam : public Optimizer {
  public:
    /**
     * @brief Constructs an Adam optimizer instance
     * @param parameters Tensors tracked by the optimizer
     * @param learning_rate Per-step learning rate
     * @param beta1 Exponential decay for first moment estimates
     * @param beta2 Exponential decay for second moment estimates
     * @param epsilon Stability constant for denominator updates
     */
    Adam(std::vector<Tensor> parameters, float learning_rate = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
         float epsilon = 1e-8f);

    /** @brief Performs one optimization step */
    void step() override;

  private:
    std::vector<Tensor> firstMomentEstimates;
    std::vector<Tensor> secondMomentEstimates;
    int stepCount = 0;

    float learningRate;
    float beta1Value;
    float beta2Value;
    float epsilonValue;
};

#endif // AXON_ADAM_H
