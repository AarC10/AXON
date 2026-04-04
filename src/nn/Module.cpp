#include "nn/Module.h"

std::vector<std::shared_ptr<Tensor>> Module::parameters() { return {}; }

void Module::zero_grad() {}
