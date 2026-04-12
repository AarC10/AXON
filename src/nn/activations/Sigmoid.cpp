#include "nn/activations/Sigmoid.h"

#include <cmath>
#include <stdexcept>

Tensor Sigmoid::forward(const Tensor &input) {
    if (!input) {
        throw std::invalid_argument("Sigmoid::forward received a null input tensor");
    }

    Tensor output = TensorImpl::zeros(input->get_shape(), input->get_require_grad());

    for (int i = 0; i < input->nelem(); ++i) {
        output->at(i) = 1.0f / (1.0f + std::exp(-input->at(i)));
    }

    if (output->get_require_grad()) {
        output->set_gradient_func(
            [input_tensor = input](const Tensor &grad) {
                Tensor input_grad = TensorImpl::zeros(input_tensor->get_shape());

                for (int i = 0; i < input_tensor->nelem(); ++i) {
                    const float y = 1.0f / (1.0f + std::exp(-input_tensor->at(i)));
                    input_grad->at(i) = grad->at(i) * y * (1.0f - y);
                }

                input_tensor->grad() = input_tensor->has_grad() ? input_tensor->grad() + input_grad : input_grad;
            },
            {input});
    }

    return output;
}
