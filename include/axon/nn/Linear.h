#ifndef AXON_LINEAR_H
#define AXON_LINEAR_H

#include "core/TensorImpl.h"
#include "nn/Module.h"

#include <vector>

/**
 * @brief Fully-connected linear layer
 */
class Linear : public Module {
  public:
    /**
     * @brief Constructs a linear layer
     * @param in_features Number of input features
     * @param out_features Number of output features
     * @param bias Whether to include a learnable bias term
     */
    Linear(int in_features, int out_features, bool bias = true);

    /** @copydoc Module::forward */
    Tensor forward(const Tensor &input) override;

    /** @copydoc Module::parameters */
    std::vector<Tensor> parameters() override;

  private:
    int in_features;
    int out_features;
    bool use_bias;

    Tensor weight;
    Tensor bias_tensor;
};

#endif // AXON_LINEAR_H
