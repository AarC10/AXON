#include "nn/Module.h"

std::vector<std::shared_ptr<TensorImpl>> Module::parameters() { return {}; }

void Module::zero_grad() {}
