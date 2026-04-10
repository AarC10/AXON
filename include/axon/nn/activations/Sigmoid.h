#ifndef AXON_SIGMOID_H
#define AXON_SIGMOID_H

#include "core/TensorImpl.h"
#include "nn/Module.h"

/**
 * @brief Sigmoid activation
 */
class Sigmoid : public Module {
  public:
    /** @brief Constructs a Sigmoid activation module */
    Sigmoid() = default;

    /** @copydoc Module::forward */
    TensorImpl forward(const TensorImpl &input) override;
};

#endif // AXON_SIGMOID_H
