#ifndef AXON_SGD_H
#define AXON_SGD_H

#include "optimizers/Optimizer.h"

class SGD : public Optimizer {
  public:
    /**
     * @brief Constructs an SGD optimizer instance
     * @param parameters Tensors tracked by the optimizer
     * @param learning_rate Per-step learning rate
     */
    explicit SGD(std::vector<Tensor> parameters, float learning_rate = 1e-3f);

    /** @brief Performs one optimization step */
    void step() override;

  private:
    float learningRate;
};

#endif // AXON_SGD_H
