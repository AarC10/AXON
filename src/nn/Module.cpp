#include "nn/Module.h"

#include <memory>
#include <vector>

std::vector<std::shared_ptr<TensorImpl>> Module::parameters() { return {}; }

void Module::zero_grad() {
    for (const std::shared_ptr<TensorImpl> &parameter : parameters()) {
        if (parameter) {
            parameter->zero_grad();
        }
    }
}
