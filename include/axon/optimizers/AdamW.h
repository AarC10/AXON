#ifndef AXON_ADAMW_H
#define AXON_ADAMW_H

#include "optimizers/Optimizer.h"

class AdamW : public Optimizer {
  public:
    /**
     * @brief Constructs an AdamW optimizer instance
     * @param parameters Tensors tracked by the optimizer
     * @param learning_rate Per-step learning rate
     * @param beta1 Exponential decay for first moment estimates
     * @param beta2 Exponential decay for second moment estimates
     * @param epsilon Stability constant for denominator updates
     * @param weight_decay Decoupled weight decay coefficient
     */
    AdamW(std::vector<std::shared_ptr<Tensor>> parameters, float learning_rate = 1e-3f, float beta1 = 0.9f,
          float beta2 = 0.999f, float epsilon = 1e-8f, float weight_decay = 1e-2f);

    /** @brief Performs one optimization step */
    void step() override;

  private:
    float learningRate;
    float beta1Value;
    float beta2Value;
    float epsilonValue;
    float weightDecay;
};

#endif // AXON_ADAMW_H
