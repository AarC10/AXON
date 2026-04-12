#include "nn/activations/ReLU.h"

#include <memory>

Tensor ReLU::forward(const TensorImpl &input) {
    Tensor input_tensor = std::const_pointer_cast<TensorImpl>(input.shared_from_this());
    Tensor output = TensorImpl::zeros(input.get_shape(), input.get_require_grad());

    for (int i = 0; i < input.nelem(); ++i) {
        output->at(i) = input.at(i) > 0.0f ? input.at(i) : 0.0f;
    }

    if (output->get_require_grad()) {
        output->set_gradient_func(
            [input_tensor](const Tensor &grad) {
                Tensor input_grad = TensorImpl::zeros(input_tensor->get_shape());

                for (int i = 0; i < input_tensor->nelem(); ++i) {
                    input_grad->at(i) = input_tensor->at(i) > 0.0f ? grad->at(i) : 0.0f;
                }

                input_tensor->grad() = input_tensor->has_grad() ? input_tensor->grad() + input_grad : input_grad;
            },
            {input_tensor});
    }

    return output;
}
