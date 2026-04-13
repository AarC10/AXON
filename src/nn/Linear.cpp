#include "nn/Linear.h"

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

Linear::Linear(int in_features, int out_features, bool bias)
    : in_features(in_features), out_features(out_features), use_bias(bias) {
    if (in_features <= 0 || out_features <= 0) {
        throw std::invalid_argument("Linear layer dimensions must be positive");
    }

    static std::mt19937 generator(std::random_device{}());
    const float limit = std::sqrt(1.0f / static_cast<float>(in_features));
    std::uniform_real_distribution<float> distribution(-limit, limit);

    std::vector<float> weight_data(in_features * out_features);
    for (float &value : weight_data) {
        value = distribution(generator);
    }

    weight = TensorImpl::from_data(weight_data, {out_features, in_features}, true);

    if (use_bias) {
        bias_tensor = TensorImpl::zeros({out_features}, true);
    }
}

Tensor Linear::forward(const Tensor &input) {
    if (!input) {
        throw std::invalid_argument("Linear::forward received a null input tensor");
    }

    const std::vector<int> &input_shape = input->get_shape();
    const bool is_vector_input = input_shape.size() == 1;
    const bool is_batched_input = input_shape.size() == 2;

    if (!is_vector_input && !is_batched_input) {
        throw std::invalid_argument("Linear::forward expects a 1D or 2D input tensor");
    }

    const int last_dimension = input_shape.back();
    if (last_dimension != in_features) {
        throw std::invalid_argument("Linear::forward input feature dimension does not match layer in_features");
    }

    const int batch_size = is_vector_input ? 1 : input_shape[0];
    const std::vector<int> output_shape =
        is_vector_input ? std::vector<int>{out_features} : std::vector<int>{batch_size, out_features};

    Tensor output = TensorImpl::zeros(output_shape, input->get_require_grad() || weight->get_require_grad() ||
                                                        (use_bias && bias_tensor && bias_tensor->get_require_grad()));

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int out_feature = 0; out_feature < out_features; ++out_feature) {
            float sum = use_bias ? bias_tensor->at(out_feature) : 0.0f;

            for (int in_feature = 0; in_feature < in_features; ++in_feature) {
                const int input_index = is_vector_input ? in_feature : batch * in_features + in_feature;
                sum += input->at(input_index) * weight->at(out_feature * in_features + in_feature);
            }

            const int output_index = is_vector_input ? out_feature : batch * out_features + out_feature;
            output->at(output_index) = sum;
        }
    }

    if (!output->get_require_grad()) {
        return output;
    }

    std::vector upstream_inputs{input, weight};
    if (use_bias && bias_tensor) {
        upstream_inputs.push_back(bias_tensor);
    }

    output->set_gradient_func(
        [input_tensor = input, weight = this->weight, bias = this->bias_tensor, batch_size, is_vector_input,
         use_bias = this->use_bias, in_features = this->in_features,
         out_features = this->out_features](const Tensor &grad) {
            if (input_tensor->get_require_grad()) {
                Tensor input_grad = TensorImpl::zeros(input_tensor->get_shape());

                for (int batch = 0; batch < batch_size; ++batch) {
                    for (int in_feature = 0; in_feature < in_features; ++in_feature) {
                        float sum = 0.0f;

                        for (int out_feature = 0; out_feature < out_features; ++out_feature) {
                            const int grad_index = is_vector_input ? out_feature : batch * out_features + out_feature;
                            sum += grad->at(grad_index) * weight->at(out_feature * in_features + in_feature);
                        }

                        const int input_index = is_vector_input ? in_feature : batch * in_features + in_feature;
                        input_grad->at(input_index) = sum;
                    }
                }

                input_tensor->grad() = input_tensor->has_grad() ? input_tensor->grad() + input_grad : input_grad;
            }

            if (weight->get_require_grad()) {
                Tensor weight_grad = TensorImpl::zeros(weight->get_shape());

                for (int out_feature = 0; out_feature < out_features; ++out_feature) {
                    for (int in_feature = 0; in_feature < in_features; ++in_feature) {
                        float sum = 0.0f;

                        for (int batch = 0; batch < batch_size; ++batch) {
                            const int grad_index = is_vector_input ? out_feature : batch * out_features + out_feature;
                            const int input_index = is_vector_input ? in_feature : batch * in_features + in_feature;
                            sum += grad->at(grad_index) * input_tensor->at(input_index);
                        }

                        weight_grad->at(out_feature * in_features + in_feature) = sum;
                    }
                }

                weight->grad() = weight->has_grad() ? weight->grad() + weight_grad : weight_grad;
            }

            if (use_bias && bias && bias->get_require_grad()) {
                Tensor bias_grad = TensorImpl::zeros(bias->get_shape());

                for (int out_feature = 0; out_feature < out_features; ++out_feature) {
                    float sum = 0.0f;

                    for (int batch = 0; batch < batch_size; ++batch) {
                        const int grad_index = is_vector_input ? out_feature : batch * out_features + out_feature;
                        sum += grad->at(grad_index);
                    }

                    bias_grad->at(out_feature) = sum;
                }

                bias->grad() = bias->has_grad() ? bias->grad() + bias_grad : bias_grad;
            }
        },
        upstream_inputs);

    return output;
}

std::vector<Tensor> Linear::parameters() {
    std::vector layer_parameters{weight};
    if (use_bias && bias_tensor) {
        layer_parameters.push_back(bias_tensor);
    }

    return layer_parameters;
}
