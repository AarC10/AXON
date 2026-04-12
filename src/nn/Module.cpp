#include "nn/Module.h"

std::vector<Tensor> Module::parameters() { return {}; }

void Module::zero_grad() {}
