#ifndef AXON_LINEAR_H
#define AXON_LINEAR_H

#include "core/TensorImpl.h"
#include "nn/Module.h"

#include <memory>
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
    TensorImpl forward(const TensorImpl &input) override;

    /** @copydoc Module::parameters */
    std::vector<std::shared_ptr<TensorImpl>> parameters() override;

  private:
    int in_features;
    int out_features;
    bool use_bias;

    std::shared_ptr<TensorImpl> weight;
    std::shared_ptr<TensorImpl> bias_tensor;
};

#endif // AXON_LINEAR_H
