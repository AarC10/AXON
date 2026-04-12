#include "nn/activations/ReLU.h"

#include <stdexcept>

Tensor ReLU::forward(const Tensor &input) {
    if (!input) {
        throw std::invalid_argument("ReLU::forward received a null input tensor");
    }

    Tensor output = TensorImpl::zeros(input->get_shape(), input->get_require_grad());

    for (int i = 0; i < input->nelem(); ++i) {
        output->at(i) = input->at(i) > 0.0f ? input->at(i) : 0.0f;
    }

    if (output->get_require_grad()) {
        output->set_gradient_func(
            [input_tensor = input](const Tensor &grad) {
                Tensor input_grad = TensorImpl::zeros(input_tensor->get_shape());

                for (int i = 0; i < input_tensor->nelem(); ++i) {
                    input_grad->at(i) = input_tensor->at(i) > 0.0f ? grad->at(i) : 0.0f;
                }

                input_tensor->grad() = input_tensor->has_grad() ? input_tensor->grad() + input_grad : input_grad;
            },
            {input});
    }

    return output;
}
