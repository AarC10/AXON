#ifndef AXON_LINEAR_H
#define AXON_LINEAR_H

#include <memory>
#include <vector>

#include "core/Tensor.h"
#include "nn/Module.h"

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

    /**
     * @brief Computes the linear forward pass
     * @param input Input tensor of shape (*, in_features)
     * @return Output tensor of shape (*, out_features)
     */
    Tensor forward(const Tensor &input) override;

    /**
     * @brief Returns all learnable parameters (weight and optional bias)
     * @return Vector of shared pointers to parameter tensors
     */
    std::vector<std::shared_ptr<Tensor>> parameters() override;

  private:
    int in_features;
    int out_features;
    bool use_bias;

    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias_tensor;
};

#endif // AXON_LINEAR_H
