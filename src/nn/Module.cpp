#include "nn/Module.h"

#include <vector>

std::vector<Tensor> Module::parameters() { return {}; }

void Module::zero_grad() {
    for (const Tensor &parameter : parameters()) {
        if (parameter) {
            parameter->zero_grad();
        }
    }
}
